"""Plain-C17 framework (``cc``).

Dispatches to the ``<short>_d_c`` / ``<short>_f_c`` symbols compiled
from the per-kernel ``cpp_backend/<short>_{d,f}.c`` sources. Same
shared library as :class:`LlvmFramework`; only the symbol suffix
differs (the wrapper picks the right entry via the ``backend`` arg
to :func:`wrap_kernel`).

The C17 sources are generated on demand from the numpy reference by
:mod:`optarena.autogen`; the C / C++ / Fortran sources are
precision-monomorphic (``<short>_<fptype>.c``), built per-framework into
``lib<short>_cc.so``.
"""
from optarena.flags import CPU_BASELINE_GCC, Mode
from optarena.framework import Framework, register_framework
from optarena.precision import Precision


@register_framework("cc")
class CcFramework(Framework):
    full_name = "C17 baseline"
    postfix = "cpp"  # reuse the existing per-kernel <short>_cpp.py wrapper
    arch = "cpu"
    SUPPORTED_PRECISIONS = frozenset({Precision.FP32, Precision.FP64})

    def compile_args(self, mode: Mode) -> str:
        return CPU_BASELINE_GCC
