"""Reproduce the RCCL failure that kills kimi pipeline parallelism, without paying for vLLM.

vLLM hands a pipeline stage off with ``get_pp_group().irecv_tensor_dict(...)``, which reaches
``torch.distributed.irecv`` on the PP process group. That group has size 4 (four PP stages), but a
point-to-point op on it needs a communicator holding only the two ranks involved, so RCCL builds a
fresh 2-rank one on first use -- the thing torch warns about as "an unbatched P2P op (send/recv) was
called on this ProcessGroup with size 4 ... a new 2-rank NCCL communicator to be created".

In job 590270 that creation failed with ``ncclInternalError: Internal check failed`` while every
collective was healthy, which is why this probe does BOTH: an allreduce over the world group to show
the fabric works at all, then the P2P chain over a size-4 group that straddles nodes. A run where
allreduce passes and P2P fails is the kimi bug; a run where both pass clears the env under test.

Layout mirrors the real one. With 2 nodes x 4 ranks the group is [0, 1, L, L+1], so the chain covers
an intra-node hop and a cross-node hop, and the group stays size 4 -- at size 2 the group's own
communicator would serve the P2P directly and the lazy path this exists to test never runs.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def report(key: str, value: object) -> None:
    """One machine-greppable line per fact; the sbatch verdict keys off these."""
    print(f"{key}={value}", flush=True)


def pp_group_ranks(world_size: int, local_world_size: int) -> list[int]:
    """Two ranks from the first node and two from the second: size 4, straddling the fabric."""
    if world_size < 4:
        raise ValueError(f"need at least 4 ranks to build a size-4 group, got {world_size}")
    if local_world_size < 2 or world_size < 2 * local_world_size:
        raise ValueError(f"need >= 2 nodes of >= 2 ranks, got world={world_size} local={local_world_size}")
    return [0, 1, local_world_size, local_world_size + 1]


def occupy_memory(device: torch.device, fraction: float, rank: int):
    """Hold ``fraction`` of free GPU memory, so communicators are built on a nearly-full device.

    An empty-GPU probe is not a fair model of the failing run. vLLM had ~37 GiB of weights and ~67
    GiB of KV cache resident per GPU when the pipeline handoff created its communicator, and CXI's
    memory registration draws on a pool that large allocations have already eaten into -- probe
    590216 failed with ``cxil_map: write error``, which is libcxi's registration ioctl, not a
    transport error. Returns the ballast so the caller keeps it alive; None when disabled.
    """
    if fraction <= 0.0:
        return None
    free, total = torch.cuda.mem_get_info(device)
    want = int(free * fraction)
    ballast = torch.empty(want, dtype=torch.uint8, device=device)
    if rank == 0:
        report("fill_gib", round(want / (1 << 30), 2))
        report("free_after_fill_gib", round(torch.cuda.mem_get_info(device)[0] / (1 << 30), 2))
        report("total_gib", round(total / (1 << 30), 2))
    return ballast


def run_allreduce(device: torch.device, rank: int, world_size: int) -> None:
    tensor = torch.full((1024, 1024), float(rank + 1), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    expected = float(world_size * (world_size + 1) // 2)
    got = tensor[0, 0].item()
    if rank == 0:
        report("allreduce_expected", expected)
        report("allreduce_got", got)
        report("allreduce_correct", got == expected)
    if got != expected:
        raise ValueError(f"allreduce wrong: expected {expected}, got {got}")


def run_p2p_chain(group, ranks: list[int], device: torch.device, rank: int, elements: int) -> None:
    """Forward a payload down the group as a LINE, the way a PP stage forwards hidden states.

    A line, not a ring. Creating the lazy 2-rank communicator is itself blocking and collective over
    exactly that pair, so a ring in which every rank issues its irecv first deadlocks on communicator
    creation alone -- rank 0 waits on comm{0,last} while the last rank waits on comm{last-1,last} --
    and reports a TCPStore rendezvous timeout that has nothing to do with the fabric. That is a bug
    in the test, not in RCCL, and this probe hit it before the wraparound came out. vLLM's pipeline
    is a line: stage 0 only sends, the last stage only receives, the middle both.

    Non-blocking isend/irecv on purpose: that is the exact torch entry point vLLM uses
    (``irecv_tensor_dict``), and the blocking form takes a different path inside the process group.
    """
    position = ranks.index(rank)
    payload = torch.full((elements,), float(rank), device=device)

    if position > 0:
        src = ranks[position - 1]
        inbox = torch.empty((elements,), device=device)
        dist.irecv(inbox, src=src, group=group).wait()
        torch.cuda.synchronize()
        got = inbox[0].item()
        if got != float(src):
            raise ValueError(f"rank {rank} expected {src} from src {src}, got {got}")
    if position < len(ranks) - 1:
        dist.isend(payload, dst=ranks[position + 1], group=group).wait()
        torch.cuda.synchronize()


def tp_group_ranks(rank: int, local_world_size: int) -> list[int]:
    """The ranks sharing this rank's node: kimi runs tensor parallelism inside a node, pipeline across."""
    node = rank // local_world_size
    return list(range(node * local_world_size, (node + 1) * local_world_size))


def run_sustained(
    wide_group,
    gloo_group,
    tp_group,
    ranks: list[int],
    tp_ranks: list[int],
    device: torch.device,
    rank: int,
    elements: int,
    allgather_elements: int,
    seconds: float,
) -> int:
    """Drive TP allgather, gloo metadata handoff and NCCL pipeline P2P together, for a while.

    The one-shot stages above only prove a communicator can be BUILT. Kimi got past that: every arm
    reported its KV cache, served 80 requests, and only then stopped -- job 590379 with its pipeline
    peer vanishing mid ``recv_object`` (gloo), jobs 590380/590381 with a ``_ALLGATHER_BASE`` wedged
    past the 600 s watchdog. So the interesting question is not whether the fabric comes up but
    whether it keeps making progress under the traffic a real decode loop generates.

    All three shapes run in one round because that is how they run in vLLM, and it is their overlap
    that is untested: a lazy 2-rank P2P communicator in flight while a separate group runs a
    collective. ``allgather_elements`` defaults to the width observed in the failure so the message
    sizes are the ones that actually wedged.

    Returns the number of completed rounds, so a hang that the timeout cuts short still reports how
    far it got. Correctness is checked every round: the failure mode this machine has already shown
    once is a silently WRONG answer, not an exception.

    EVERY rank runs this, not just the pipeline members: the TP allgather is collective over all four
    node-local ranks, so a loop entered by only the two that happen to sit in the pipeline group would
    hang on the first round. For the same reason the stop decision is rank 0's and is broadcast --
    letting each rank test its own clock independently desynchronises them at the deadline, and the
    rank that leaves first strands the rest inside a collective.
    """
    position = ranks.index(rank) if rank in ranks else -1
    tp_position = tp_ranks.index(rank)
    # bfloat16: the dtype the model actually moves, so the byte counts match the wedged collective.
    contribution = torch.full((allgather_elements,), float(tp_position + 1), dtype=torch.bfloat16, device=device)
    gathered = torch.empty((allgather_elements * len(tp_ranks),), dtype=torch.bfloat16, device=device)
    payload = torch.full((elements,), float(rank), device=device)
    inbox = torch.empty((elements,), device=device)
    meta = torch.empty((1,), dtype=torch.int64)
    keep_going = torch.ones((1,), dtype=torch.int64, device=device)

    deadline = time.monotonic() + seconds
    rounds = 0
    while True:
        if rank == 0:
            keep_going.fill_(1 if time.monotonic() < deadline else 0)
        dist.broadcast(keep_going, src=0)
        if keep_going.item() == 0:
            break

        dist.all_gather_into_tensor(gathered, contribution, group=tp_group)
        torch.cuda.synchronize()
        # Check the LAST contributor's slice: a partial-fabric failure leaves the local rank's own
        # slice right and a remote one zeroed, so reading slot 0 would call a broken gather clean.
        last = gathered[allgather_elements * (len(tp_ranks) - 1)].item()
        if last != float(len(tp_ranks)):
            raise ValueError(f"allgather wrong at round {rounds}: expected {float(len(tp_ranks))}, got {last}")

        # The gloo hop first, then the NCCL one: vLLM sends the tensor metadata over the CPU group
        # before the hidden states, and it was the gloo recv that raised "Connection closed by peer".
        # A rank outside the pipeline group (position -1) does the TP half of the round only.
        if position > 0:
            dist.recv(meta, src=ranks[position - 1], group=gloo_group)
            if meta.item() != rounds:
                raise ValueError(f"metadata wrong at round {rounds}: got {meta.item()}")
            dist.irecv(inbox, src=ranks[position - 1], group=wide_group).wait()
            torch.cuda.synchronize()
            if inbox[0].item() != float(ranks[position - 1]):
                raise ValueError(f"payload wrong at round {rounds}: got {inbox[0].item()}")
        if 0 <= position < len(ranks) - 1:
            dist.send(torch.tensor([rounds], dtype=torch.int64), dst=ranks[position + 1], group=gloo_group)
            dist.isend(payload, dst=ranks[position + 1], group=wide_group).wait()
            torch.cuda.synchronize()

        rounds += 1
        # Progress on stdout is what distinguishes "wedged at round 3" from "wedged at round 40000".
        if rank == 0 and rounds % 200 == 0:
            report("sustained_rounds", rounds)
    return rounds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=1 << 20, help="payload floats per P2P hop")
    parser.add_argument("--timeout-seconds", type=int, default=90, help="collective timeout, per group")
    parser.add_argument(
        "--fill-fraction",
        type=float,
        default=0.0,
        help="occupy this fraction of free GPU memory BEFORE building groups, so the "
        "communicator is created under the memory pressure a loaded model imposes",
    )
    parser.add_argument(
        "--sustain-seconds",
        type=float,
        default=180.0,
        help="how long to drive mixed TP/PP traffic; 0 skips the sustained stage. The "
        "default covers the ~2.5 min kimi survived after its first request",
    )
    parser.add_argument(
        "--allgather-elements",
        type=int,
        default=14680064,
        help="per-rank TP allgather width; the default is the NumelIn of the "
        "_ALLGATHER_BASE that wedged in jobs 590380 and 590381",
    )
    args = parser.parse_args()

    rank = env_int("RANK", 0)
    world_size = env_int("WORLD_SIZE", 1)
    local_rank = env_int("LOCAL_RANK", 0)
    local_world_size = env_int("LOCAL_WORLD_SIZE", world_size)

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    timeout = datetime.timedelta(seconds=args.timeout_seconds)

    # A finite timeout is the point: the vLLM symptom was a wedge, and a wedged probe teaches
    # nothing that a raised DistBackendError does not teach faster.
    dist.init_process_group(backend="nccl", timeout=timeout)

    if rank == 0:
        report("world_size", world_size)
        report("local_world_size", local_world_size)
        report("torch_version", torch.__version__)
        report("nccl_version", ".".join(str(v) for v in torch.cuda.nccl.version()))

    stage = "allreduce"
    ballast = None
    try:
        # Before any group is built: the failure under investigation happens at communicator
        # CREATION, so the pressure has to be in place by then, not added afterwards.
        ballast = occupy_memory(device, args.fill_fraction, rank)
        run_allreduce(device, rank, world_size)
        if rank == 0:
            report("stage_allreduce", "PASS")

        stage = "group_create"
        wide = pp_group_ranks(world_size, local_world_size)
        pair = [0, local_world_size]
        if rank == 0:
            report("pp_group_ranks", ",".join(str(r) for r in wide))
            report("pair_group_ranks", ",".join(str(r) for r in pair))
        # Every rank must call new_group, members or not -- it is collective over the world group,
        # and both groups must be built before either is used so the two stages cannot interleave.
        wide_group = dist.new_group(ranks=wide, timeout=timeout)
        pair_group = dist.new_group(ranks=pair, timeout=timeout)
        # Built here with the others: new_group is collective over the world group, so every group
        # the run will ever need has to be created before any of them is used.
        gloo_group = dist.new_group(ranks=wide, backend="gloo", timeout=timeout)
        tp_ranks = tp_group_ranks(rank, local_world_size)
        tp_groups = [
            dist.new_group(ranks=list(range(n * local_world_size, (n + 1) * local_world_size)), timeout=timeout)
            for n in range(world_size // local_world_size)
        ]
        tp_group = tp_groups[rank // local_world_size]
        if rank == 0:
            report("stage_group_create", "PASS")

        # The discriminating pair of stages. torch only mints a lazy 2-rank communicator when the
        # group is WIDER than the pair actually exchanging data; on a size-2 group the group's own
        # communicator serves the P2P directly. So size-4 FAIL + size-2 PASS means pipeline
        # parallelism is fine at pp=2 and only the lazy sub-communicator is broken -- which is the
        # difference between "kimi needs a different topology" and "kimi cannot run here at all".
        # size-2 runs FIRST, deliberately. It is the stage expected to pass, and a failed collective
        # leaves the world group in a state the next stage cannot be trusted through -- so ordering
        # the likely-failing stage last is what guarantees both verdicts are actually recorded.
        stage = "p2p_size2"
        if rank in pair:
            run_p2p_chain(pair_group, pair, device, rank, args.elements)
        dist.barrier()
        if rank == 0:
            report("stage_p2p_size2", "PASS")

        stage = "p2p_size4"
        if rank in wide:
            run_p2p_chain(wide_group, wide, device, rank, args.elements)
        dist.barrier()
        if rank == 0:
            report("stage_p2p_size4", "PASS")

        # Last, because it is the long one and because everything above is a precondition for it.
        stage = "sustained"
        if args.sustain_seconds > 0.0:
            rounds = run_sustained(
                wide_group,
                gloo_group,
                tp_group,
                wide,
                tp_ranks,
                device,
                rank,
                args.elements,
                args.allgather_elements,
                args.sustain_seconds,
            )
            dist.barrier()
            if rank == 0:
                report("sustained_rounds_total", rounds)
                report("stage_sustained", "PASS")
        if rank == 0:
            report("verdict", "PASS")
    except Exception as exc:  # noqa: BLE001 -- the failure IS the measurement; report it, do not raise
        # EVERY rank reports its own failure, not just rank 0. Under memory pressure the cross-node
        # allreduce came back as zeros on the SECOND node while rank 0 computed the right answer and
        # printed PASS -- so a verdict keyed on rank 0 alone called a silent-wrong-answer run green.
        report(f"stage_{stage}", "FAIL")
        report("failed_stage", stage)
        report("failed_rank", rank)
        report("failure", repr(exc))
        report("verdict", "FAIL")
        traceback.print_exc()
        del ballast
        dist.destroy_process_group()
        return 1

    del ballast  # referenced here so the fill cannot be collected before the last communicator
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
