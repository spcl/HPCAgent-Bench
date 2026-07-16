# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Single-job cluster launcher: ONE SLURM allocation, MPI rank -> role.

``optarena launch`` runs under ``srun --mpi=pmix --ntasks-per-node=1`` across the
whole allocation. Each rank owns one node and derives its ROLE from its rank and the
``(inference_endpoints I, nodes_per_vllm K, judge_nodes J)`` counts -- no per-node
config, because the cluster is homogeneous (Daint: every node is 4x GH200) so a
rank-order slice of the nodelist needs no hardware pinning:

* ranks ``[0, I*K)``          -- inference. Consecutive groups of ``K`` ranks form one
  vLLM endpoint; the group's local-rank-0 is the HEAD (runs ``vllm serve``), the other
  ``K-1`` are ray workers (``K>1`` only -- a model too big for one node). Each endpoint
  is one URL the agents round-robin over.
* ranks ``[I*K, I*K + J)``    -- judge. Each runs ``optarena serve`` (the HTTP oracle).
* rank ``0``                  -- ALSO the agent DRIVER (co-located on endpoint-0's head).
  The agent loop is an HTTP client -- GPU-idle -- so it rides the vLLM node without
  disturbing the CPU-bound judge timings.

So the allocation is ``N = I*K + J`` nodes. The ranks ``allgather`` their hostnames, the
driver assembles the vLLM + judge URL lists in rank order and hands them to the static
pipeline (:func:`optarena.harness.pipeline.run_static`), where worker ``w`` binds to
``vllm_urls[w % I]`` (think) + ``judge_urls[w % J]`` (grade). Two barriers bound the run:
one after every rank has launched its server, one after the driver finishes (holding the
server ranks alive meanwhile); then every rank tears its server down and they exit
together.

``vllm`` is assumed on ``PATH`` (a site module / venv); the launcher only orchestrates
placement, it does not provision vLLM. The pure planning helpers (:func:`plan_roles`,
:func:`assemble_urls`) carry no MPI dependency so they unit-test without a cluster;
:func:`launch` imports ``mpi4py`` lazily.
"""
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

RAY_PORT = 6379

#: Poll cadence for the collective readiness/liveness loop: each round every rank re-reports whether
#: its server bound its port or died, so a death at any point in startup aborts within one round.
POLL_INTERVAL = 5.0


@dataclass(frozen=True)
class RankRole:
    """The role a single rank plays. ``endpoint`` is the vLLM endpoint index (0-based)
    for inference ranks, ``-1`` for judge ranks; ``head_rank`` is the rank of this
    endpoint's head (so a worker can find the head to join its ray cluster)."""
    role: str
    endpoint: int
    head_rank: int
    is_driver: bool


def expected_world(inference_endpoints: int, nodes_per_vllm: int, judge_nodes: int) -> int:
    """Nodes a launch needs: ``I*K`` inference + ``J`` judge (the driver co-locates on
    rank 0, so it costs no extra node)."""
    return inference_endpoints * nodes_per_vllm + judge_nodes


def plan_roles(world_size: int, inference_endpoints: int, nodes_per_vllm: int, judge_nodes: int) -> List[RankRole]:
    """The per-rank role table for the whole allocation.

    Raises :class:`ValueError` on a nonsensical shape (a non-positive count) or when the
    allocation size does not match ``I*K + J`` -- a misconfigured job should fail loudly
    at rank 0, not silently under- or over-subscribe roles.
    """
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
            roles.append(RankRole(JUDGE, -1, -1, is_driver=False))
    return roles


def assemble_urls(gathered: Sequence[dict], vllm_port: int, judge_port: int) -> Tuple[List[str], List[str]]:
    """Build the ordered ``(vllm_urls, judge_urls)`` from the allgathered rank identities.

    vLLM URLs are the endpoint HEADS ordered by endpoint id (so ``vllm_urls[e]`` is
    endpoint ``e``); judge URLs are the judge ranks in rank order. Each entry is a
    ``{rank, role, endpoint, hostname}`` dict.
    """
    heads = sorted((g for g in gathered if g["role"] == VLLM_HEAD), key=lambda g: g["endpoint"])
    judges = sorted((g for g in gathered if g["role"] == JUDGE), key=lambda g: g["rank"])
    vllm_urls = [f"http://{g['hostname']}:{vllm_port}/v1" for g in heads]
    judge_urls = [f"http://{g['hostname']}:{judge_port}" for g in judges]
    return vllm_urls, judge_urls


def vllm_command(model: str, port: int, tensor_parallel: int, pipeline_parallel: int,
                 extra: Sequence[str]) -> List[str]:
    """The ``vllm serve`` argv for an endpoint head. ``tensor_parallel`` shards each layer
    across a node's GPUs (intra-node); ``pipeline_parallel`` (= ``K``) splits layers across
    the endpoint's nodes (inter-node) and, when ``> 1``, turns on the ray executor -- the
    head + its workers must already have joined one ray cluster."""
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
    """Block until every URL's ``host:port`` accepts a TCP connection, or ``timeout`` s
    elapse. vLLM/uvicorn bind the port only once initialization is far enough along to
    serve, so a successful connect is a good readiness proxy without needing a per-service
    health path. Returns True iff all became reachable."""
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
    """Bring this inference rank up. For a single-node endpoint (``K == 1``) the head just
    runs ``vllm serve`` (tensor-parallel over its GPUs, no ray). For a multi-node endpoint
    the head starts a ray head and the workers retry-join it (``--block`` keeps ray attached
    to the process we own, so teardown reaps it); the head then serves with pipeline
    parallelism ``K`` over the formed cluster. ``vllm serve`` itself waits until the ray
    cluster has all ``K*gpus_per_node`` GPUs, so a worker joining slightly late is fine."""
    procs: List[subprocess.Popen] = []
    use_ray = nodes_per_vllm > 1
    if use_ray and me.role == VLLM_WORKER:
        # Retry until the head's ray is up; on success --block keeps this process (and the
        # ray node it registered) alive until we kill it at teardown.
        join = f"until ray start --address={head_host}:{RAY_PORT} --num-gpus={gpus_per_node} --block; do sleep 2; done"
        procs.append(popen(["bash", "-c", join], log))
        return procs
    if use_ray and me.role == VLLM_HEAD:
        procs.append(
            popen(["ray", "start", "--head", f"--port={RAY_PORT}", f"--num-gpus={gpus_per_node}", "--block"], log))
        # 'ray start --block' backgrounds under Popen and returns at once, so the GCS may not
        # be up yet; wait for it before vllm's ray executor initializes against it.
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


def start_judge(judge_port: int, serve_extra: Sequence[str], log: Callable[[str], None]) -> subprocess.Popen:
    """Run ``optarena serve`` (the HTTP oracle) through THIS interpreter, so the judge
    subprocess uses the same venv/image the launcher runs in."""
    cmd = [sys.executable, "-m", "optarena", "serve", "--host", "0.0.0.0", "--port", str(judge_port)]
    return popen(cmd + list(serve_extra), log)


def teardown(procs: Sequence[subprocess.Popen], me: RankRole, nodes_per_vllm: int, log: Callable[[str], None]) -> None:
    """Stop this rank's server/ray processes: SIGTERM (reverse order), reap with a grace
    window, SIGKILL any straggler, then ``ray stop`` on a multi-node endpoint's nodes.

    Never propagates: it runs in :func:`launch`'s ``finally`` right before the final barrier +
    bcast, so an exception escaping here would make this rank skip those collectives and deadlock
    every other rank. ``ray stop`` also reaps the ray child of a worker's ``bash --block`` wrapper,
    which a bare SIGTERM to the wrapper may not."""
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
    """Number of settle rounds for a ``ready_timeout``, from the rank-identical timeout so every rank
    agrees. A non-finite timeout (argparse's ``type=float`` accepts ``inf``/``nan``) normalizes to a
    long finite wait instead of crashing ``int()``. Floored at 2 so at least one poll runs AFTER a
    grace sleep -- a server that dies just after spawn is still alive at the first (immediate) probe.
    """
    if not math.isfinite(ready_timeout):
        ready_timeout = 24 * 3600.0
    return max(2, int(ready_timeout // poll_interval))


def rank_status(me: RankRole, procs: Sequence[subprocess.Popen], vllm_port: int, judge_port: int, hostname: str,
                rank: int) -> Dict[str, str]:
    """This rank's ``{"kind", "detail"}`` for the settle loop: ``dead`` (a server exited non-zero --
    fatal, abort the run), ``ready`` (serving port accepts a local connection; a ray WORKER has no
    port of its own so it is ready once alive), or ``pending`` (still coming up). Each rank probes its
    OWN 127.0.0.1 port, so a mid-startup death is caught the next round with no cross-node lookup.
    """
    for proc in procs:
        rc = proc.poll()
        if rc not in (None, 0):
            return {"kind": "dead", "detail": f"rank {rank} ({me.role}) on {hostname}: server exited rc={rc}"}
    if me.role == VLLM_WORKER:
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
           run_driver: Callable[[List[str], List[str]], int],
           vllm_port: int = 8000,
           judge_port: int = 8800,
           gpus_per_node: int = 4,
           ready_timeout: float = 1800.0,
           vllm_extra: Sequence[str] = (),
           serve_extra: Sequence[str] = (),
           log: Callable[[str], None] = print) -> int:
    """Bootstrap the whole allocation and drive one run. Collective across all ranks.

    ``run_driver(vllm_urls, judge_urls) -> int`` is invoked on rank 0 ONLY, once every
    endpoint is reachable; its return code is broadcast so every rank exits with it. A
    shape mismatch (world != ``I*K + J``) returns ``2`` from every rank without starting
    anything.
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()

    try:
        roles = plan_roles(world, inference_endpoints, nodes_per_vllm, judge_nodes)
    except ValueError as exc:
        if rank == 0:
            log(f"[launch] bad allocation shape: {exc}")
        return 2
    me = roles[rank]
    hostname = socket.gethostname()
    gathered = comm.allgather({"rank": rank, "role": me.role, "endpoint": me.endpoint, "hostname": hostname})

    # Guard the spawn: a failure (e.g. vllm not on PATH) must NOT let this rank skip the collectives
    # below and deadlock the others. Catch it; the loop's allgather then reports it as dead.
    procs: List[subprocess.Popen] = []
    spawn_error = ""
    try:
        if me.role in (VLLM_HEAD, VLLM_WORKER):
            head_host = next(g["hostname"] for g in gathered if g["rank"] == me.head_rank)
            procs = start_inference(me, nodes_per_vllm, model, vllm_port, gpus_per_node, head_host, vllm_extra, log)
        elif me.role == JUDGE:
            procs = [start_judge(judge_port, serve_extra, log)]
    except BaseException as exc:  # noqa: BLE001 -- a spawn failure must NOT skip the collectives below and deadlock
        spawn_error = f"rank {rank} ({me.role}) on {hostname}: {exc}"
        log(f"[launch] {spawn_error}")

    # Collective readiness + liveness loop: each round every rank allgathers its own state, so a
    # death anywhere/anytime aborts fast and every rank breaks on the SAME round (the decision reads
    # only the shared allgather result -- no rank skips a collective and strands the others).
    # Bound by a round COUNT, not each rank's wall clock: a per-rank monotonic deadline could trip a
    # round apart, leaving one rank blocked forever in the next allgather. settle_rounds derives the
    # count from the shared ready_timeout so every rank runs identically. spawn_error counts as dead.
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
                # Ports are bound on their own nodes; confirm the driver can REACH each across the
                # fabric (bound-but-unreachable = a DNS/Slingshot problem). Driver-local, no
                # collective. Constant first in min() so a non-finite ready_timeout still yields 60.0.
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
    # Reached by ALL ranks: the guards above catch everything and teardown never raises, so no
    # rank can skip this final collective and strand the others.
    return comm.bcast(rc, root=0)
