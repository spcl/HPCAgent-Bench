# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dynamic device scheduler for the judge.

The judge grades many kernels; each grade runs on ONE device -- a GPU or a CPU
slot, whichever the run is configured for -- and only one kernel occupies a slot
at a time. This module turns a configurable pool of device slots (across a
configurable number of nodes) into a work-stealing dispatcher: every slot gets a
worker that pulls the next pending kernel as soon as it frees, so faster kernels
do not idle a GPU waiting on a slow one. It is the scheduling layer the
native/container judge drives; it holds no grading logic -- the caller passes a
``run_item(item, slot)`` closure that does the actual verify/score.

Local GPU slots are pinned WITHOUT a ``CUDA_VISIBLE_DEVICES`` env race: the
worker thread records its slot's GPU index in :mod:`native_call`'s thread-local
(`set_assigned_device`), and the spawned device child selects that physical GPU
with ``cp.cuda.Device(index)``. Remote slots (a hostname, from a multi-node
allocation) are addressed by the caller's ``run_item`` via :func:`srun_wrap`.

Config (all optional, read via :func:`optarena.config.get`, so an env override
``OPTARENA_JUDGE_<KEY>`` works and no ``config.yaml`` entry is required):

* ``judge.gpus_per_node`` -- GPU slots per node (default: detected local GPUs).
* ``judge.cpu_slots_per_node`` -- CPU slots per node (default: 1 when no GPU,
  else 0). A CPU slot runs a host-residency kernel.
* ``judge.nodelist`` -- comma-separated hostnames for a multi-node judge
  (default: empty = one local node). The node count is ``len(nodelist)`` when
  set, else 1.
"""
import os
import queue as _queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from optarena import config
from optarena.agent_bench import native_call


@dataclass(frozen=True)
class DeviceSlot:
    """One schedulable device: a GPU ordinal or a CPU slot, on a local or named
    (remote) node."""

    kind: str  # "gpu" | "cpu"
    index: int  # GPU ordinal (kind == "gpu"), else a CPU-slot ordinal on the node
    node: Optional[str] = None  # None = local host; else a hostname for srun dispatch

    @property
    def is_local(self) -> bool:
        return self.node is None

    @property
    def label(self) -> str:
        return f"{self.node or 'local'}:{self.kind}{self.index}"


def local_gpu_count() -> int:
    """Visible GPUs on this host (0 when cupy or a driver is absent -> a host-only
    judge, which is correct on a CPU box)."""
    try:
        import cupy as cp
        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:  # noqa: BLE001 -- no cupy / no driver -> zero GPUs
        return 0


def _nodelist_from_config() -> Tuple[str, ...]:
    """Hostnames for a multi-node judge: ``judge.nodelist`` (comma-separated), else
    the SLURM allocation's already-expanded ``OPTARENA_JUDGE_NODES_EXPANDED`` (the
    sbatch template fills it via ``scontrol show hostnames``), else empty (local)."""
    raw = config.get("judge.nodelist", "") or os.environ.get("OPTARENA_JUDGE_NODES_EXPANDED", "")
    return tuple(h for h in str(raw).replace("\n", ",").split(",") if h.strip())


@dataclass(frozen=True)
class JudgeConfig:
    """Resolved device-pool shape for the judge."""

    gpus_per_node: int
    cpu_slots_per_node: int
    nodelist: Tuple[str, ...] = field(default_factory=tuple)
    #: srun template for a remote slot; ``{node}`` is substituted per slot.
    launcher: Tuple[str, ...] = ("srun", "--nodelist", "{node}", "--gpus", "1", "-n", "1")

    @classmethod
    def from_config(cls) -> "JudgeConfig":
        gpus = config.get("judge.gpus_per_node", None)
        gpus = int(gpus) if gpus is not None else local_gpu_count()
        cpu_slots = config.get("judge.cpu_slots_per_node", None)
        cpu_slots = int(cpu_slots) if cpu_slots is not None else (0 if gpus else 1)
        return cls(gpus_per_node=gpus, cpu_slots_per_node=cpu_slots, nodelist=_nodelist_from_config())

    @property
    def nodes(self) -> int:
        return len(self.nodelist) or 1

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
        return out or [DeviceSlot("cpu", 0, None)]


def srun_wrap(slot: DeviceSlot, argv: List[str], launcher: Tuple[str, ...]) -> List[str]:
    """Wrap ``argv`` in the srun launcher targeting ``slot``'s node (for a remote
    slot). ``{node}`` in the launcher template is replaced with the hostname; a
    local slot is returned unwrapped."""
    if slot.is_local:
        return list(argv)
    prefix = [tok.replace("{node}", slot.node) for tok in launcher]
    return prefix + list(argv)


class JudgeScheduler:
    """Work-stealing dispatcher over a fixed pool of :class:`DeviceSlot`.

    One worker thread per slot pulls the next pending item as soon as it is free,
    so the pool stays busy (dynamic, not a static round-robin). A local GPU slot
    pins its index in :mod:`native_call` for the duration of the item, so a
    concurrent device score lands on THAT GPU. Each item's result is captured
    independently -- one item raising is a scored failure, never a scheduler
    death."""

    def __init__(self, slots: List[DeviceSlot], log: Optional[Callable[[str], None]] = None):
        self.slots = list(slots) or [DeviceSlot("cpu", 0, None)]
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

        def worker(slot: DeviceSlot) -> None:
            while True:
                try:
                    i, it = work.get_nowait()
                except _queue.Empty:
                    return
                native_call.set_assigned_device(slot.index if (slot.kind == "gpu" and slot.is_local) else None)
                try:
                    results[i] = ("ok", run_item(it, slot))
                except BaseException as exc:  # noqa: BLE001 -- scored failure, not fatal
                    results[i] = ("err", exc)
                finally:
                    native_call.set_assigned_device(None)

        threads = [
            threading.Thread(target=worker, args=(s, ), daemon=True, name=f"judge-{s.label}") for s in self.slots
        ]
        self._log(f"judge: {len(items)} items over {len(self.slots)} slots ({[s.label for s in self.slots]})")
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return [r if r is not None else ("err", RuntimeError("item not scheduled")) for r in results]
