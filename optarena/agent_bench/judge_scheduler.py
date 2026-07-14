# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dynamic device scheduler for the judge, and the two-stage think->grade pipeline.

The judge grades many kernels; each grade runs on ONE device -- a GPU or a CPU
slot, whichever the run is configured for -- and only one kernel occupies a slot
at a time. This module turns a configurable pool of device slots (across a
configurable number of nodes) into a work-stealing dispatcher: every slot gets a
worker that pulls the next pending kernel as soon as it frees, so faster kernels
do not idle a GPU waiting on a slow one. It is the scheduling layer the
native/container judge drives; it holds no grading logic -- the caller passes a
``run_item(item, slot)`` closure that does the actual verify/score.

For the 3-tier ``agent / reference / inference`` topology (agent nodes "think" by
querying the inference server, judge nodes "measure" the reference + candidate),
:class:`TwoStageScheduler` routes each item through a THINK stage on a separate
AGENT pool and then a GRADE stage on the judge pool, the two pools running
concurrently. An item being graded never occupies an agent slot and vice-versa, so
an agent blocked on the inference endpoint never idles a precious timing GPU. Both
schedulers share one work-stealing primitive (:func:`_work_pool`).

Local GPU slots are pinned WITHOUT a ``CUDA_VISIBLE_DEVICES`` env race: the
worker thread records its slot's GPU index in :mod:`native_call`'s thread-local
(`set_assigned_device`), and the spawned device child selects that physical GPU
with ``cp.cuda.Device(index)``. Remote slots (a hostname, from a multi-node
allocation) are addressed by the caller's stage closure via :func:`srun_wrap`.

Config (all optional, read via :func:`optarena.config.get`, so an env override
``OPTARENA_JUDGE_<KEY>`` / ``OPTARENA_AGENT_<KEY>`` works and no ``config.yaml``
entry is required):

* ``judge.gpus_per_node`` -- GPU slots per node (default: detected local GPUs).
* ``judge.cpu_slots_per_node`` -- CPU slots per node (default: 1 when no GPU,
  else 0). A CPU slot runs a host-residency kernel.
* ``judge.nodelist`` -- comma-separated hostnames for a multi-node judge
  (default: empty = one local node). The node count is ``len(nodelist)`` when
  set, else 1.
* ``agent.workers_per_node`` -- concurrent agent (think) workers per agent node
  (default: 1).
* ``agent.nodelist`` -- comma-separated hostnames for the agent pool (default:
  empty = one local node = co-located with the judge, for a single-box run).
"""
import os
import queue as _queue
import shlex
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from optarena import config, containers
from optarena.agent_bench import native_call


@dataclass(frozen=True)
class DeviceSlot:
    """One schedulable slot: a GPU ordinal, a CPU slot, or an AGENT (think) worker,
    on a local or named (remote) node."""

    kind: str  # "gpu" | "cpu" | "agent"
    index: int  # GPU ordinal (kind == "gpu"), else a slot/worker ordinal on the node
    node: Optional[str] = None  # None = local host; else a hostname for srun dispatch

    @property
    def is_local(self) -> bool:
        return self.node is None

    @property
    def label(self) -> str:
        return f"{self.node or 'local'}:{self.kind}{self.index}"


#: Env vars the sbatch template exports with the SLURM-expanded hostnames (via
#: ``scontrol show hostnames``); the single source for each pool's nodelist name.
JUDGE_NODES_ENV = "OPTARENA_JUDGE_NODES_EXPANDED"
AGENT_NODES_ENV = "OPTARENA_AGENT_NODES_EXPANDED"

#: Default srun template for a REMOTE judge slot (``{node}`` substituted per slot). Built by
#: the one CE-launcher factory so the srun-prefix spelling lives in a single place; on Alps the
#: campaign overrides ``judge.launcher`` to add ``--environment <edf>`` so the remote grade runs
#: INSIDE the Container-Engine image (else it runs outside the toolchain).
DEFAULT_JUDGE_LAUNCHER = containers.ce_launcher()

#: Shared frozen fallback slots (a pool with no configured slot runs one local slot).
LOCAL_CPU_SLOT = DeviceSlot("cpu", 0, None)
LOCAL_AGENT_SLOT = DeviceSlot("agent", 0, None)


class NodePool:
    """Mixin for a pool config carrying a ``nodelist`` -- supplies the node count so
    :class:`JudgeConfig` and :class:`AgentPoolConfig` do not each restate it."""

    nodelist: Tuple[str, ...]

    @property
    def nodes(self) -> int:
        return len(self.nodelist) or 1


def local_gpu_count() -> int:
    """Visible GPUs on this host (0 when cupy or a driver is absent -> a host-only
    judge, which is correct on a CPU box)."""
    try:
        import cupy as cp
        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:  # noqa: BLE001 -- no cupy / no driver -> zero GPUs
        return 0


def _expand_nodelist(config_key: str, env_key: str) -> Tuple[str, ...]:
    """Hostnames for a pool: the config key (comma-separated), else the SLURM
    allocation's already-expanded env var (the sbatch template fills it via
    ``scontrol show hostnames``), else empty (one local node)."""
    raw = config.get(config_key, "") or os.environ.get(env_key, "")
    return tuple(h for h in str(raw).replace("\n", ",").split(",") if h.strip())


def _nodelist_from_config() -> Tuple[str, ...]:
    """Hostnames for a multi-node judge (``judge.nodelist`` /
    ``OPTARENA_JUDGE_NODES_EXPANDED``)."""
    return _expand_nodelist("judge.nodelist", JUDGE_NODES_ENV)


def resolve_launcher() -> Tuple[str, ...]:
    """The srun template for a REMOTE judge slot: ``judge.launcher`` from config (a yaml
    list), or ``OPTARENA_JUDGE_LAUNCHER`` (a shell-split string), else
    :data:`DEFAULT_JUDGE_LAUNCHER`. Alps sets it to carry ``--environment <edf>`` so the
    dispatched grade runs inside the Container-Engine image, not the bare host."""
    raw = config.get("judge.launcher", None)
    if isinstance(raw, str):
        return tuple(shlex.split(raw)) or DEFAULT_JUDGE_LAUNCHER
    if raw:
        return tuple(raw)
    return DEFAULT_JUDGE_LAUNCHER


@dataclass(frozen=True)
class JudgeConfig(NodePool):
    """Resolved device-pool shape for the judge."""

    gpus_per_node: int
    cpu_slots_per_node: int
    nodelist: Tuple[str, ...] = field(default_factory=tuple)
    #: srun template for a remote slot; ``{node}`` is substituted per slot (see
    #: :func:`resolve_launcher` -- config-driven so Alps can add ``--environment``).
    launcher: Tuple[str, ...] = DEFAULT_JUDGE_LAUNCHER

    @classmethod
    def from_config(cls) -> "JudgeConfig":
        gpus = config.get("judge.gpus_per_node", None)
        gpus = int(gpus) if gpus is not None else local_gpu_count()
        cpu_slots = config.get("judge.cpu_slots_per_node", None)
        cpu_slots = int(cpu_slots) if cpu_slots is not None else (0 if gpus else 1)
        return cls(gpus_per_node=gpus,
                   cpu_slots_per_node=cpu_slots,
                   nodelist=_nodelist_from_config(),
                   launcher=resolve_launcher())

    def slots(self) -> List[DeviceSlot]:
        """Expand the pool into concrete slots (GPU slots first per node, then CPU
        slots). A pool with neither a GPU nor a CPU slot falls back to one local
        CPU slot so the judge always has somewhere to run."""
        nodes = list(self.nodelist) or [None]
        out: List[DeviceSlot] = []
        for node in nodes:
            for g in range(self.gpus_per_node):
                out.append(DeviceSlot("gpu", g, node))
            for c in range(self.cpu_slots_per_node):
                out.append(DeviceSlot("cpu", c, node))
        return out or [LOCAL_CPU_SLOT]


def srun_wrap(slot: DeviceSlot, argv: List[str], launcher: Tuple[str, ...]) -> List[str]:
    """Wrap ``argv`` in the srun launcher targeting ``slot``'s node (for a remote
    slot). ``{node}`` in the launcher template is replaced with the hostname; a
    local slot is returned unwrapped."""
    if slot.is_local:
        return list(argv)
    prefix = [tok.replace("{node}", slot.node) for tok in launcher]
    return prefix + list(argv)


#: Sentinel a ``pull`` returns when its pool is drained (distinct from a legitimate
#: ``None`` / falsy payload, so an item whose value is ``0``/``None`` still runs).
_POOL_DONE = object()


def _work_pool(slots: List[DeviceSlot], pull: Callable[[], Any], handle: Callable[[Any, DeviceSlot], None]) -> None:
    """Work-stealing over ``slots``: one worker thread per slot repeatedly pulls the
    next unit of work via ``pull()`` -- a payload, or :data:`_POOL_DONE` when the
    pool is drained (worker exits) -- and runs ``handle(payload, slot)`` with local
    GPU pinning applied for that unit. Blocks until every worker drains. ``pull`` /
    ``handle`` own all queue + result bookkeeping (including capturing per-item
    errors); this helper owns only the threads + the pinning, so it is shared
    verbatim by the single-stage judge run and each stage of the think->grade
    pipeline."""

    def worker(slot: DeviceSlot) -> None:
        while True:
            payload = pull()
            if payload is _POOL_DONE:
                return
            native_call.set_assigned_device(slot.index if (slot.kind == "gpu" and slot.is_local) else None)
            try:
                handle(payload, slot)
            finally:
                native_call.set_assigned_device(None)

    threads = [threading.Thread(target=worker, args=(s, ), daemon=True, name=f"pool-{s.label}") for s in slots]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def drain_pull(q: "_queue.Queue") -> Any:
    """A non-blocking pull for a drain-to-empty pool: the next queued item, or
    :data:`_POOL_DONE` once the queue is empty (the worker then exits)."""
    try:
        return q.get_nowait()
    except _queue.Empty:
        return _POOL_DONE


def finalize_results(results: List[Optional[Tuple[str, Any]]]) -> List[Tuple[str, Any]]:
    """Every item gets a ``(status, value)``: an item never scheduled (still ``None``)
    becomes a scored error -- the belt-and-braces guarantee shared by both schedulers."""
    return [r if r is not None else ("err", RuntimeError("item not scheduled")) for r in results]


class JudgeScheduler:
    """Work-stealing dispatcher over a fixed pool of :class:`DeviceSlot`.

    One worker thread per slot pulls the next pending item as soon as it is free,
    so the pool stays busy (dynamic, not a static round-robin). A local GPU slot
    pins its index in :mod:`native_call` for the duration of the item, so a
    concurrent device score lands on THAT GPU. Each item's result is captured
    independently -- one item raising is a scored failure, never a scheduler
    death."""

    def __init__(self, slots: List[DeviceSlot], log: Optional[Callable[[str], None]] = None):
        self.slots = list(slots) or [LOCAL_CPU_SLOT]
        self._log = log or (lambda _m: None)

    @classmethod
    def from_config(cls, log: Optional[Callable[[str], None]] = None) -> "JudgeScheduler":
        return cls(JudgeConfig.from_config().slots(), log=log)

    def run(self, items: List[Any], run_item: Callable[[Any, DeviceSlot], Any]) -> List[Tuple[str, Any]]:
        """Schedule every item across the slot pool and return results in INPUT
        order as ``(status, value)`` -- ``("ok", <run_item result>)`` or
        ``("err", <exception>)``. ``run_item(item, slot)`` does the grade; when it
        drives a device score on a local GPU slot the pinning is already in effect
        (via the thread-local), and it may call :func:`srun_wrap` for a remote
        slot."""
        results: List[Optional[Tuple[str, Any]]] = [None] * len(items)
        work: "_queue.Queue[Tuple[int, Any]]" = _queue.Queue()
        for i, it in enumerate(items):
            work.put((i, it))

        def handle(payload: Tuple[int, Any], slot: DeviceSlot) -> None:
            i, it = payload
            try:
                results[i] = ("ok", run_item(it, slot))
            except BaseException as exc:  # noqa: BLE001 -- scored failure, not fatal
                results[i] = ("err", exc)

        self._log(f"judge: {len(items)} items over {len(self.slots)} slots ({[s.label for s in self.slots]})")
        _work_pool(self.slots, lambda: drain_pull(work), handle)
        return finalize_results(results)


@dataclass(frozen=True)
class AgentPoolConfig(NodePool):
    """Resolved shape of the AGENT (think) pool -- the nodes that run the LLM
    optimizer workers, separate from the judge's timing slots. Mirrors
    :class:`JudgeConfig` so the two pools are configured the same way."""

    workers_per_node: int
    nodelist: Tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_config(cls) -> "AgentPoolConfig":
        workers = int(config.get("agent.workers_per_node", 1))
        return cls(workers_per_node=max(1, workers), nodelist=_expand_nodelist("agent.nodelist", AGENT_NODES_ENV))

    def slots(self) -> List[DeviceSlot]:
        """Expand into concrete ``agent`` slots (``workers_per_node`` per node).
        An empty nodelist = one local node (the single-box run where the agent
        workers co-locate with the judge)."""
        nodes = list(self.nodelist) or [None]
        out = [DeviceSlot("agent", w, node) for node in nodes for w in range(self.workers_per_node)]
        return out or [LOCAL_AGENT_SLOT]


class TwoStageScheduler:
    """Route each item through a THINK stage on the agent pool then a GRADE stage on
    the judge pool, the two pools running CONCURRENTLY (no barrier): an item begins
    grading as soon as its think finishes, while later items are still thinking. An
    item being graded never occupies an agent slot and an agent blocked on the
    inference endpoint never idles a timing GPU -- the 3-tier separation.

    ``think(item, agent_slot) -> candidate`` runs the LLM optimizer (the agent slot
    names its node so the closure can :func:`srun_wrap` it onto the agent pool);
    ``grade(candidate, item, grade_slot) -> value`` does the reference verify + timed
    score on a judge slot (GPU pinning already in effect for a local GPU slot). A
    think error is recorded and its grade skipped; a grade error is recorded.
    Results come back in INPUT order as ``(status, value)`` -- the same contract as
    :meth:`JudgeScheduler.run`."""

    def __init__(self,
                 agent_slots: List[DeviceSlot],
                 grade_slots: List[DeviceSlot],
                 log: Optional[Callable[[str], None]] = None):
        self.agent_slots = list(agent_slots) or [LOCAL_AGENT_SLOT]
        self.grade_slots = list(grade_slots) or [LOCAL_CPU_SLOT]
        self._log = log or (lambda _m: None)

    @classmethod
    def from_config(cls, log: Optional[Callable[[str], None]] = None) -> "TwoStageScheduler":
        return cls(AgentPoolConfig.from_config().slots(), JudgeConfig.from_config().slots(), log=log)

    def run(self, items: List[Any], think: Callable[[Any, DeviceSlot], Any],
            grade: Callable[[Any, Any, DeviceSlot], Any]) -> List[Tuple[str, Any]]:
        n = len(items)
        results: List[Optional[Tuple[str, Any]]] = [None] * n
        think_q: "_queue.Queue[Tuple[int, Any]]" = _queue.Queue()
        grade_q: "_queue.Queue[Tuple[int, Any, Any]]" = _queue.Queue()
        for i, it in enumerate(items):
            think_q.put((i, it))

        lock = threading.Lock()
        remaining = [n]  # items not yet FINALIZED (graded, or think-failed); list = mutable cell

        def finish_one() -> None:
            with lock:
                remaining[0] -= 1

        # THINK pool (agent slots): pull an item, produce a candidate, hand it to the
        # grade queue. A think error finalizes the item as err and skips its grade.
        def think_handle(payload: Tuple[int, Any], slot: DeviceSlot) -> None:
            i, it = payload
            try:
                candidate = think(it, slot)
            except BaseException as exc:  # noqa: BLE001 -- think failure is a scored failure
                results[i] = ("err", exc)
                finish_one()
                return
            grade_q.put((i, it, candidate))

        # GRADE pool (judge slots): consume candidates as they arrive; keep waiting
        # while any item is still outstanding (in think or in flight), exit once all
        # are finalized -- so grade workers do not quit before think produces.
        def grade_pull() -> Any:
            while True:
                try:
                    return grade_q.get(timeout=0.02)
                except _queue.Empty:
                    with lock:
                        if remaining[0] == 0:
                            return _POOL_DONE

        def grade_handle(payload: Tuple[int, Any, Any], slot: DeviceSlot) -> None:
            i, it, candidate = payload
            try:
                results[i] = ("ok", grade(candidate, it, slot))
            except BaseException as exc:  # noqa: BLE001 -- grade failure is a scored failure
                results[i] = ("err", exc)
            finish_one()

        self._log(f"pipeline: {n} items, {len(self.agent_slots)} agent slots -> "
                  f"{len(self.grade_slots)} grade slots")
        # Both pools run concurrently: each _work_pool call blocks on its own worker
        # threads, so drive them from two driver threads and join both.
        drivers = [
            threading.Thread(target=_work_pool,
                             args=(self.agent_slots, lambda: drain_pull(think_q), think_handle),
                             name="agent-pool"),
            threading.Thread(target=_work_pool, args=(self.grade_slots, grade_pull, grade_handle), name="grade-pool"),
        ]
        for d in drivers:
            d.start()
        for d in drivers:
            d.join()
        return finalize_results(results)
