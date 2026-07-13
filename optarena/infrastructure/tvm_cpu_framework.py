# Copyright 2025 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""TVM CPU framework — uses ``tvm.s_tir.meta_schedule.tune_tir`` for
autotuning and ``tvm.compile`` for code generation. Built against
Apache TVM 0.20+; the old ``tvm.auto_scheduler`` path is unsupported.
"""

from optarena.infrastructure import Benchmark, Framework
from typing import Any, Callable, Dict

# Datatype string picked by the harness's set_datatype(). Kernel files
# read this when constructing their te.placeholder shapes.
tvm_dtype: str = "float64"


def tvm_dtype_str(datatype) -> str:
    """The TVM dtype string for a datatype request (numpy or enum spelling).

    fp64/fp32/fp16/bf16 map to their TVM names; anything else (fp8 -- TVM's
    support is partial and the siblings pin float32 there) falls back to float64.
    """
    from optarena.precision import Precision, precision_from_datatype
    return {
        Precision.FP64: "float64",
        Precision.FP32: "float32",
        Precision.FP16: "float16",
        Precision.BF16: "bfloat16",
    }.get(precision_from_datatype(datatype), "float64")


# Per-process trial cap. Smaller = faster sanity smoke; full = paper.
METASCHEDULE_TRIALS_DEFAULT = 64
METASCHEDULE_TRIALS_FULL = 1024


def metaschedule_trials() -> int:
    """How many tuning trials to give ``tune_tir`` per task.

    Delegates to the unified optimize budget (:class:`optarena.optimize.\
OptimizeBudget`) so TVM and Triton share ONE knob (``$OPTARENA_OPTIMIZE_BUDGET``),
    read on every call so a test can change it mid-process."""
    from optarena.optimize import OptimizeBudget
    return OptimizeBudget.from_env().tvm_trials()


class TVMCPUFramework(Framework):
    """Framework binding for Apache TVM running on the CPU (``llvm``
    target). Mirrors :class:`TritonFramework` in shape: a thin set of
    overrides on top of the base :class:`Framework`.

    An :class:`optarena.optimize.Optimizer`: ``tune_tir`` (MetaSchedule) searches
    a schedule within :meth:`optimize_budget`'s ``trials`` (see
    :func:`metaschedule_trials`).
    """

    is_optimizer = True

    def __init__(self, fname: str):
        super().__init__(fname)

    def version(self) -> str:
        """ Return the TVM version. """
        import tvm
        return tvm.__version__

    def imports(self) -> Dict[str, Any]:
        import tvm
        from tvm import te
        return {"tvm": tvm, "te": te}

    def copy_func(self) -> Callable:
        """ Convert numpy array → ``tvm.runtime.Tensor`` on CPU.

        Two pass-throughs: TVM has no complex dtype, so complex array_args
        (FFT / contour / self-energy) stay numpy (kernels split real/imag);
        and a scipy.sparse ``A`` stays a scipy matrix so the sparse kernel
        can pull out its CSR buffers for a gather-reduction SpMV. """
        import numpy as np
        import scipy.sparse as sp
        import tvm

        def inner(arr):
            if sp.issparse(arr):
                return arr.copy()
            if np.iscomplexobj(arr):
                return np.array(arr)
            return tvm.runtime.tensor(arr, device=tvm.cpu(0))

        return inner

    def copy_back_func(self) -> Callable:
        """ Convert ``tvm.runtime.Tensor`` (or list of) → numpy. """
        import tvm

        def inner(x):
            if isinstance(x, tvm.runtime.Tensor):
                return x.numpy()
            return x

        return inner

    def set_datatype(self, datatype):
        super().set_datatype(datatype)
        global tvm_dtype
        tvm_dtype = tvm_dtype_str(datatype)
        # Mark the active backend so a unified <kernel>_tvm.py picks the CPU
        # TvmKernel (see tvm_build.active_kernel).
        from optarena.infrastructure import tvm_build
        tvm_build.tvm_backend = "cpu"

    def implementations(self, bench: "Benchmark"):
        """Load the per-kernel TVM impl, preferring the unified
        ``<kernel>_tvm.py`` (which carries both the CPU and GPU kernels) and
        falling back to the legacy split ``<kernel>_tvm_cpu.py`` while a kernel
        has not been unified yet. ``set_datatype`` has already flagged the CPU
        backend, so the unified file's entry selects the CPU kernel."""
        import importlib
        import pathlib
        rel = bench.info["relative_path"]
        mod = bench.info["module_name"]
        bench_dir = pathlib.Path(__file__).parent.joinpath("..", "..", "optarena", "benchmarks", rel)
        # A still-split kernel keeps its CPU impl in <mod>_tvm_cpu.py; a unified
        # one has only <mod>_tvm.py (carrying both backends).
        postfix = "tvm_cpu" if bench_dir.joinpath(f"{mod}_tvm_cpu.py").exists() else "tvm"
        module = importlib.import_module(f"optarena.benchmarks.{rel.replace('/', '.')}.{mod}_{postfix}")
        return [(vars(module)[bench.info["func_name"]], "default")]

    # No post_call override: the base direct-call path (positional args, no
    # device sync) is exactly the old exec_str behaviour for CPU TVM.
