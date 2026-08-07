# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Single-job cluster launcher: ONE SLURM allocation, MPI rank -> role (vLLM head/worker, judge, driver)."""
import math
import socket
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

VLLM_HEAD = "vllm_head"
VLLM_WORKER = "vllm_worker"
JUDGE = "judge"
#: A rank that runs the optimizer itself instead of serving a model -- the traditional
#: (deterministic, no-vLLM) track. It hosts no server, so it is ready the moment it exists.
OPTIMIZER = "optimizer"

RAY_PORT = 6379

#: Poll cadence for the collective readiness/liveness loop.
POLL_INTERVAL = 5.0


@dataclass(frozen=True)
class RankRole:
    """The role a single rank plays: endpoint index, and this endpoint's head rank.

    For a :data:`JUDGE` rank ``endpoint`` is its JUDGE RANK -- its index into the ``judge_urls``
    the driver assembles, which is what an agent worker names when it grades there. Judges are
    ordered by MPI rank in both places, so the two agree by construction."""
    role: str
    endpoint: int
    head_rank: int
    is_driver: bool


def expected_world(inference_endpoints: int, nodes_per_vllm: int, judge_nodes: int) -> int:
    """Nodes a launch needs: I*K inference + J judge (the driver co-locates on rank 0)."""
    return inference_endpoints * nodes_per_vllm + judge_nodes


def expected_traditional_world(optimizer_nodes: int, judge_nodes: int) -> int:
    """Nodes a traditional (no-vLLM) launch needs: O optimizer + J judge."""
    return optimizer_nodes + judge_nodes


def plan_traditional_roles(world_size: int, optimizer_nodes: int, judge_nodes: int) -> List[RankRole]:
    """Per-rank roles for the TRADITIONAL track: O optimizer ranks then J judge ranks, no inference.

    A deterministic optimizer emits the same artifact every run, so there is no model to serve --
    but the judge still scores the result, so the judge role is kept exactly as the agent track has
    it. Rank 0 is the driver, co-located on the first optimizer node.
    """
    if optimizer_nodes < 1 or judge_nodes < 1:
        raise ValueError(f"optimizer_nodes ({optimizer_nodes}) and judge_nodes ({judge_nodes}) must both be >= 1")
    need = expected_traditional_world(optimizer_nodes, judge_nodes)
    if world_size != need:
        raise ValueError(f"world size {world_size} != O + J = {need} (optimizer_nodes={optimizer_nodes} "
                         f"+ judge_nodes={judge_nodes}); allocate exactly {need} nodes "
                         f"(srun -N {need} --ntasks-per-node=1)")
    roles: List[RankRole] = []
    for r in range(world_size):
        if r < optimizer_nodes:
            roles.append(RankRole(OPTIMIZER, -1, -1, is_driver=(r == 0)))
        else:
            roles.append(RankRole(JUDGE, r - optimizer_nodes, -1, is_driver=False))
    return roles


def plan_roles(world_size: int, inference_endpoints: int, nodes_per_vllm: int, judge_nodes: int) -> List[RankRole]:
    """The per-rank role table for the whole allocation; raises ValueError on a bad shape or size mismatch."""
    if inference_endpoints < 1 or nodes_per_vllm < 1 or judge_nodes < 1:
        raise ValueError(f"inference_endpoints ({inference_endpoints}), nodes_per_vllm ({nodes_per_vllm}) and "
                         f"judge_nodes ({judge_nodes}) must all be >= 1")
    need = expected_world(inference_endpoints, nodes_per_vllm, judge_nodes)
    if world_size != need:
        raise ValueError(f"world size {world_size} != I*K + J = {need} "
                         f"(inference_endpoints={inference_endpoints} x nodes_per_vllm={nodes_per_vllm} "
                         f"+ judge_nodes={judge_nodes}); allocate exactly {need} nodes "
                         f"(srun -N {need} --ntasks-per-node=1)")
    vllm_total = inference_endpoints * nodes_per_vllm
    roles: List[RankRole] = []
    for r in range(world_size):
        if r < vllm_total:
            endpoint = r // nodes_per_vllm
            head_rank = endpoint * nodes_per_vllm
            role = VLLM_HEAD if r == head_rank else VLLM_WORKER
            roles.append(RankRole(role, endpoint, head_rank, is_driver=(r == 0)))
        else:
            roles.append(RankRole(JUDGE, r - vllm_total, -1, is_driver=False))
    return roles


def assemble_urls(gathered: Sequence[dict], vllm_port: int, judge_port: int) -> Tuple[List[str], List[str]]:
    """Build the ordered (vllm_urls, judge_urls) from the allgathered rank identities.

    ``judge_urls`` is in MPI-rank order, which is exactly the ``endpoint`` index
    :func:`plan_roles` handed each judge -- so ``judge_urls[j]`` is the judge running with
    ``serve --rank j``, and a worker's round-robin index is the rank that judge expects."""
    heads = sorted((g for g in gathered if g["role"] == VLLM_HEAD), key=lambda g: g["endpoint"])
    judges = sorted((g for g in gathered if g["role"] == JUDGE), key=lambda g: g["rank"])
    vllm_urls = [f"http://{g['hostname']}:{vllm_port}/v1" for g in heads]
    judge_urls = [f"http://{g['hostname']}:{judge_port}" for g in judges]
    return vllm_urls, judge_urls


def vllm_command(model: str, port: int, tensor_parallel: int, pipeline_parallel: int,
                 extra: Sequence[str]) -> List[str]:
    """The vllm serve argv for an endpoint head; pipeline_parallel > 1 turns on the ray executor."""
    cmd = [
        "vllm", "serve", model, "--host", "0.0.0.0", "--port",
        str(port), "--tensor-parallel-size",
        str(tensor_parallel)
    ]
    if pipeline_parallel > 1:
        cmd += ["--pipeline-parallel-size", str(pipeline_parallel), "--distributed-executor-backend", "ray"]
    return cmd + list(extra)


def endpoint_hostport(url: str) -> Tuple[str, int]:
    """``(host, port)`` from a base URL, defaulting the scheme so a bare ``host:port`` parses."""
    parsed = urllib.parse.urlparse(url if "//" in url else "http://" + url)
    return parsed.hostname, parsed.port


def wait_ready(urls: Sequence[str], timeout: float, log: Callable[[str], None]) -> bool:
    """Block until every URL's host:port accepts a TCP connection or timeout elapses; True iff all reachable."""
    deadline = time.monotonic() + timeout
    pending = list(urls)
    while pending and time.monotonic() < deadline:
        still: List[str] = []
        for url in pending:
            host, port = endpoint_hostport(url)
            try:
                socket.create_connection((host, port), timeout=5).close()
            except OSError:
                still.append(url)
        if still:
            log(f"[launch] waiting on {len(still)}/{len(urls)} endpoint(s)...")
            time.sleep(5)
        pending = still
    return not pending


def popen(cmd: Sequence[str], log: Callable[[str], None]) -> subprocess.Popen:
    """Spawn a long-lived server/ray process (non-blocking); the caller tears it down."""
    log("[launch] $ " + " ".join(str(c) for c in cmd))
    return subprocess.Popen(list(cmd))


def start_inference(me: RankRole, nodes_per_vllm: int, model: str, vllm_port: int, gpus_per_node: int, head_host: str,
                    vllm_extra: Sequence[str], log: Callable[[str], None]) -> List[subprocess.Popen]:
    """Bring this inference rank up: vllm serve alone for K==1, else ray head/workers then vllm serve over ray."""
    procs: List[subprocess.Popen] = []
    use_ray = nodes_per_vllm > 1
    if use_ray and me.role == VLLM_WORKER:
        # --block keeps this process (and its ray node) alive until teardown kills it
        join = f"until ray start --address={head_host}:{RAY_PORT} --num-gpus={gpus_per_node} --block; do sleep 2; done"
        procs.append(popen(["bash", "-c", join], log))
        return procs
    if use_ray and me.role == VLLM_HEAD:
        procs.append(
            popen(["ray", "start", "--head", f"--port={RAY_PORT}", f"--num-gpus={gpus_per_node}", "--block"], log))
        # Popen returns once 'ray start --block' backgrounds; wait for the GCS before vllm attaches
        if not wait_ready([f"http://127.0.0.1:{RAY_PORT}"], 120.0, log):
            log("[launch] ray GCS not up after 120s; vllm serve may fail to attach")
    if me.role == VLLM_HEAD:
        cmd = vllm_command(model,
                           vllm_port,
                           tensor_parallel=gpus_per_node,
                           pipeline_parallel=nodes_per_vllm,
                           extra=vllm_extra)
        procs.append(popen(cmd, log))
    return procs


def start_judge(judge_port: int, judge_rank: int, serve_extra: Sequence[str], log: Callable[[str],
                                                                                            None]) -> subprocess.Popen:
    """Run hpcagent-bench serve through THIS interpreter, so the judge subprocess uses the launcher's venv/image.

    ``--rank`` is passed EXPLICITLY rather than left for the judge to read out of ``$PMI_RANK`` /
    ``$SLURM_PROCID``: the judge's identity is its index into ``judge_urls`` (0..J-1), which is NOT
    the MPI world rank (the vLLM ranks come first), and a server that infers its own identity from
    whatever the environment happens to hold is the ambient-configuration bug this check exists to
    catch. The launcher owns the mapping; the judge is told."""
    cmd = [
        sys.executable, "-m", "hpcagent_bench", "serve", "--host", "0.0.0.0", "--port",
        str(judge_port), "--rank",
        str(judge_rank)
    ]
    return popen(cmd + list(serve_extra), log)


def teardown(procs: Sequence[subprocess.Popen], me: RankRole, nodes_per_vllm: int, log: Callable[[str], None]) -> None:
    """Stop this rank's server/ray processes (SIGTERM, grace wait, SIGKILL, ray stop); never raises."""
    try:
        for proc in reversed(list(procs)):
            try:
                proc.terminate()
            except OSError:
                pass
        for proc in reversed(list(procs)):
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except OSError:
                    pass
        if nodes_per_vllm > 1 and me.role in (VLLM_HEAD, VLLM_WORKER):
            subprocess.run(["ray", "stop"], timeout=30, check=False)
    except BaseException as exc:  # noqa: BLE001 -- teardown must never escape launch()'s finally
        log(f"[launch] teardown error (ignored): {exc}")


def settle_rounds(ready_timeout: float, poll_interval: float = POLL_INTERVAL) -> int:
    """Number of settle rounds for ready_timeout; non-finite normalizes to a long wait, floored at 2."""
    if not math.isfinite(ready_timeout):
        ready_timeout = 24 * 3600.0
    return max(2, int(ready_timeout // poll_interval))


def rank_status(me: RankRole, procs: Sequence[subprocess.Popen], vllm_port: int, judge_port: int, hostname: str,
                rank: int) -> Dict[str, str]:
    """This rank's {"kind", "detail"} for the settle loop: dead / ready / pending, probed locally."""
    for proc in procs:
        rc = proc.poll()
        if rc not in (None, 0):
            return {"kind": "dead", "detail": f"rank {rank} ({me.role}) on {hostname}: server exited rc={rc}"}
    # Neither hosts a server of its own: a vLLM worker is driven by its head, and an optimizer rank
    # runs the deterministic optimizer in-process. Probing a port would strand them as "pending".
    if me.role in (VLLM_WORKER, OPTIMIZER):
        return {"kind": "ready", "detail": ""}
    port = judge_port if me.role == JUDGE else vllm_port
    try:
        socket.create_connection(("127.0.0.1", port), timeout=5).close()
        return {"kind": "ready", "detail": ""}
    except OSError:
        return {"kind": "pending", "detail": f"rank {rank} ({me.role}) on {hostname}: port {port} not bound yet"}


def launch(*,
           inference_endpoints: int,
           nodes_per_vllm: int,
           judge_nodes: int,
           model: str,
           optimizer_nodes: int = 0,
           run_driver: Callable[[List[str], List[str]], int],
           vllm_port: int = 8000,
           judge_port: int = 8800,
           gpus_per_node: int = 4,
           ready_timeout: float = 1800.0,
           vllm_extra: Sequence[str] = (),
           serve_extra: Sequence[str] = (),
           log: Callable[[str], None] = print) -> int:
    """Bootstrap the whole allocation and drive one run (collective across all ranks); rc broadcast from rank 0."""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()

    # ``inference_endpoints == 0`` selects the TRADITIONAL track: optimizer ranks instead of vLLM
    # ranks, judge role unchanged. Everything downstream -- the allgather, the settle loop, the
    # driver's rc broadcast, teardown -- is shared with the agent track rather than duplicated.
    traditional = inference_endpoints == 0
    try:
        roles = (plan_traditional_roles(world, optimizer_nodes, judge_nodes) if traditional else plan_roles(
            world, inference_endpoints, nodes_per_vllm, judge_nodes))
    except ValueError as exc:
        if rank == 0:
            log(f"[launch] bad allocation shape: {exc}")
        return 2
    me = roles[rank]
    hostname = socket.gethostname()
    gathered = comm.allgather({"rank": rank, "role": me.role, "endpoint": me.endpoint, "hostname": hostname})

    # a spawn failure must not skip the collectives below and deadlock the others
    procs: List[subprocess.Popen] = []
    spawn_error = ""
    try:
        if me.role in (VLLM_HEAD, VLLM_WORKER):
            head_host = next(g["hostname"] for g in gathered if g["rank"] == me.head_rank)
            procs = start_inference(me, nodes_per_vllm, model, vllm_port, gpus_per_node, head_host, vllm_extra, log)
        elif me.role == JUDGE:
            procs = [start_judge(judge_port, me.endpoint, serve_extra, log)]
    except BaseException as exc:  # noqa: BLE001 -- a spawn failure must NOT skip the collectives below and deadlock
        spawn_error = f"rank {rank} ({me.role}) on {hostname}: {exc}"
        log(f"[launch] {spawn_error}")

    # each round every rank allgathers its state, so all break on the same round; bounded by a
    # round count (not a per-rank clock) so nobody strands the others in the next allgather
    rounds_budget = settle_rounds(ready_timeout)
    failures: List[str] = []
    pending: List[str] = []
    for attempt in range(rounds_budget):
        if attempt:
            time.sleep(POLL_INTERVAL)
        mine = ({
            "kind": "dead",
            "detail": spawn_error
        } if spawn_error else rank_status(me, procs, vllm_port, judge_port, hostname, rank))
        statuses = comm.allgather(mine)
        failures = [s["detail"] for s in statuses if s["kind"] == "dead"]
        pending = [s["detail"] for s in statuses if s["kind"] == "pending"]
        if failures or not pending:
            break
        if me.is_driver:
            log(f"[launch] waiting on {len(pending)}/{len(statuses)} rank(s) "
                f"(round {attempt + 1}/{rounds_budget})...")

    rc = 0
    try:
        if me.is_driver:
            if failures:
                log(f"[launch] {len(failures)} rank(s) failed to start their server; aborting:")
                for err in failures:
                    log(f"  {err}")
                rc = 4
            elif pending:
                log(f"[launch] {len(pending)} rank(s) not ready within {ready_timeout:.0f}s; aborting:")
                for stuck in pending:
                    log(f"  {stuck}")
                rc = 3
            else:
                vllm_urls, judge_urls = assemble_urls(gathered, vllm_port, judge_port)
                log(f"[launch] {inference_endpoints} vLLM endpoint(s) x {nodes_per_vllm} node(s), "
                    f"{judge_nodes} judge(s)")
                log(f"[launch] vllm_urls={vllm_urls}")
                log(f"[launch] judge_urls={judge_urls}")
                # confirm the driver can reach each endpoint across the fabric (bound != reachable)
                if wait_ready(vllm_urls + judge_urls, min(60.0, ready_timeout), log):
                    rc = run_driver(vllm_urls, judge_urls) or 0
                else:
                    log("[launch] endpoints bound but not reachable from the driver; aborting")
                    rc = 3
    except BaseException as exc:  # noqa: BLE001 -- a driver crash must still reach the barrier + release the servers
        log(f"[launch] driver failed: {exc}")
        rc = 1
    finally:
        comm.Barrier()  # driver done (or crashed/aborted) -> release the server ranks so nobody hangs
        teardown(procs, me, nodes_per_vllm, log)
    # reached by all ranks: guards above catch everything, so nobody skips this final collective
    return comm.bcast(rc, root=0)
