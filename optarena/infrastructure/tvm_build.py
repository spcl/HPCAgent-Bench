# Copyright 2025 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared TVM build/tune plumbing for the per-kernel OptArena TVM impls.

Every ``<kernel>_tvm_cpu.py`` (llvm target) and ``<kernel>_tvm.py`` (cuda
target) constructs a TIR :class:`PrimFunc` describing the kernel and hands
it here. This module owns the parts that are identical across all of them:

* target construction (CPU llvm / GPU cuda with the attrs meta_schedule
  needs),
* the ``tune_tir`` → ``compile_tir`` → ``tvm.compile`` autotuning pipeline
  (the "auto-opt track"), gated by ``metaschedule_trials()``,
* a tiny shape-keyed compile cache (:class:`TvmKernel`) so the harness's
  repeat loop and multi-preset sweeps reuse one compiled Executable,
* output-buffer allocation.

Keeping it here means a per-kernel file is just its TIR plus a thin
entry point, mirroring the one-file-per-kernel shape of the pluto track.
"""
import os
import tempfile

import numpy as np
import tvm
from tvm.s_tir.meta_schedule import tune_tir
from tvm.s_tir.meta_schedule.tir_integration import compile_tir

from optarena.infrastructure.tvm_cpu_framework import metaschedule_trials

# Active TVM backend ("cpu" / "gpu"), set by the running framework (mirrors
# ``tvm_cpu_framework.tvm_dtype``). A *unified* ``<kernel>_tvm.py`` builds both
# a CPU and a GPU :class:`TvmKernel` and calls :func:`active_kernel` to pick the
# one matching the framework driving the run, so one file serves both backends.
tvm_backend: str = "cpu"


def active_kernel(cpu_kernel: "TvmKernel", gpu_kernel: "TvmKernel") -> "TvmKernel":
    """Return the :class:`TvmKernel` matching the active :data:`tvm_backend`.

    Building the GPU kernel is lazy (the device callable fires only on use), so
    a unified file can construct both on a CPU-only box; only the selected one
    is ever compiled / run."""
    return gpu_kernel if tvm_backend == "gpu" else cpu_kernel


def active_target_device():
    """Return ``(target_fn, device)`` for the active :data:`tvm_backend`.

    For kernels that pass a target builder + device into a host driver instead
    of holding a module-level :class:`TvmKernel` (the sparse solvers' ``_solve``
    loop). Only the active backend's device is constructed, so a CPU run never
    touches ``tvm.cuda``."""
    if tvm_backend == "gpu":
        return gpu_target, tvm.cuda(0)
    return cpu_target, tvm.cpu(0)


def cpu_target() -> "tvm.target.Target":
    """llvm target sized to the physical core count."""
    import psutil
    return tvm.target.Target({"kind": "llvm", "num-cores": psutil.cpu_count(logical=False) or 1})


def gpu_target() -> "tvm.target.Target":
    """cuda target with the device attrs meta_schedule's rules require.

    The bare ``{"kind": "cuda"}`` dict only fills ``max_num_threads``;
    the tuning rules also read warp size / shared-mem / registers, so we
    query the live device. Raises if no GPU is present (expected on
    CPU-only boxes — the CPU impl is what runs there)."""
    dev = tvm.cuda(0)
    return tvm.target.Target({
        "kind": "cuda",
        "max_threads_per_block": dev.max_threads_per_block,
        "thread_warp_size": dev.warp_size,
        "max_shared_memory_per_block": dev.max_shared_memory_per_block,
        "registers_per_block": 65536,
    })


def tune_compile(prim_func, target, name: str, key: str):
    """Autotune ``prim_func`` under meta_schedule and return an Executable.

    :param name: stable kernel name (used for the work-dir + log lines).
    :param key: shape/dtype discriminator so different presets tune into
        separate work dirs.

    Fast path: when ``OPTARENA_TVM_NOTUNE`` is set, skip meta_schedule and
    compile the PrimFunc directly with TVM's default schedule. The numerics
    are identical (a schedule only changes *how* the loops run, not the
    result), so this is the right mode for correctness verification — it
    turns a multi-second tune into a sub-second build. The autotuning path
    is the default (unset) and is what real benchmark runs use.
    """
    if os.environ.get("OPTARENA_TVM_NOTUNE"):
        return default_compile(prim_func, target)
    work_root = os.environ.get("OPTARENA_TVM_WORK_DIR", os.path.join(tempfile.gettempdir(), "optarena_tvm_ms"))
    work_dir = os.path.join(work_root, f"{name}_{key}")
    os.makedirs(work_dir, exist_ok=True)
    db = tune_tir(prim_func, target=target, work_dir=work_dir, max_trials_global=metaschedule_trials())
    sch = compile_tir(db, prim_func, target)
    # meta_schedule is best-effort: if it found no schedule (compile_tir is
    # None — common for small/awkward kernels, esp. on GPU), fall back to the
    # plain default-schedule compile, which must always work + verify.
    if sch is None:
        return default_compile(prim_func, target)
    return tvm.compile(sch.mod, target=target)


def default_gpu_schedule(prim_func, max_threads: int = 256):
    """A minimal generic GPU schedule: for every compute block, fuse its
    data-parallel (spatial) loops, split off ``max_threads`` and bind to
    ``blockIdx.x`` / ``threadIdx.x``. Reduction loops stay sequential (each
    thread does its own reduction — correct, if not peak-performance). Enough
    to give a cuda PrimFunc the thread environment ``tvm.compile`` requires
    when meta_schedule is unavailable or declined to schedule it."""
    from tvm.s_tir import Schedule
    sch = Schedule(prim_func)
    try:
        blocks = sch.get_child_blocks(sch.get_sblock("root"))
    except Exception as e:  # noqa: BLE001
        # No standard "root" block (e.g. a hand-written TVMScript serial
        # kernel like crc16): nothing to auto-bind. Such inherently
        # sequential kernels don't map to a default GPU thread layout.
        raise NotImplementedError("default_gpu_schedule: kernel has no auto-bindable spatial "
                                  "structure (sequential/non-te kernel); unsupported on GPU") from e
    for blk in blocks:
        ivs = sch.get(blk).iter_vars
        loops = sch.get_loops(blk)
        spatial = [lp for lp, iv in zip(loops, ivs) if int(iv.iter_type) == 0]
        if not spatial:
            continue
        fused = sch.fuse(*spatial) if len(spatial) > 1 else spatial[0]
        outer, inner = sch.split(fused, factors=[None, max_threads])
        sch.bind(outer, "blockIdx.x")
        sch.bind(inner, "threadIdx.x")
    return sch


def default_compile(prim_func, target):
    """Plain default-schedule compile, no meta_schedule. ``llvm`` needs no
    schedule; ``cuda`` gets the minimal :func:`default_gpu_schedule` thread
    binding. Numerics are schedule-independent, so this always verifies."""
    if "cuda" in str(target.kind):
        try:
            return tvm.compile(default_gpu_schedule(prim_func).mod, target=target)
        except Exception:
            # default_gpu_schedule can't auto-bind this kernel (no spatial
            # structure / non-te TVMScript). It may already carry its own
            # thread binding (e.g. a sequential kernel that bakes a 1-thread
            # launch) — compile as-is. If it truly lacks a thread environment
            # this still raises, which is the honest failure.
            return tvm.compile(prim_func, target=target)
    return tvm.compile(prim_func, target=target)


def empty(shape, dtype, device):
    """Allocate an uninitialised output ``tvm.runtime.Tensor`` on ``device``."""
    return tvm.runtime.tensor(np.empty(shape, dtype=str(dtype)), device=device)


class TvmKernel:
    """Shape-keyed compile cache around one TIR builder.

    ``build`` is a callable returning a TIR PrimFunc; it is invoked with
    the cache key tuple (e.g. ``(N, "float64")``) whenever the shape
    changes, then the result is tuned + compiled once and reused. This
    is the per-kernel object every ``*_tvm*.py`` file instantiates at
    module scope.

    The CPU file builds it with ``cpu_target`` / ``tvm.cpu``; the GPU file
    reuses the *same* ``build`` (imported from the CPU module, so the TIR —
    hence the numerics — is identical) with ``gpu_target`` / ``tvm.cuda``.
    """

    def __init__(self, name: str, build, target_fn, device_fn):
        self.name = name
        self.build = build
        self.target_fn = target_fn
        self.device_fn = device_fn
        self._exe = None
        self._key = None

    def get(self, key):
        """Return the compiled Executable for cache key ``key`` (a tuple)."""
        if self._key == key and self._exe is not None:
            return self._exe
        prim_func = self.build(*key)
        key_str = "_".join(str(k) for k in key)
        self._exe = tune_compile(prim_func, self.target_fn(), self.name, key_str)
        self._key = key
        return self._exe

    def out(self, shape, dtype):
        """Allocate a fresh output tensor on this kernel's device."""
        return empty(shape, dtype, self.device_fn())

    @property
    def device(self):
        return self.device_fn()
