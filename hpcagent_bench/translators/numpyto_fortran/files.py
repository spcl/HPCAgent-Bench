"""Write one kernel's Fortran file (target ``fortran``; ``fortran_omp`` with ``parallel``)."""

from hpcagent_bench.translators.numpyto_common.emit_helpers.request import EmitRequest, native_names, with_precision
from hpcagent_bench.translators.numpyto_common.emit_io import write_generated
from hpcagent_bench.translators.numpyto_common.frontend import parse_kernel
from hpcagent_bench.translators.numpyto_common.lowering import lower
from hpcagent_bench.translators.numpyto_fortran.emit import emit_fortran, emit_fortran_omp
from hpcagent_bench.translators.numpyto_fortran.intrinsics import renders_natively

__all__ = ["write_fortran"]


def write_fortran(request: EmitRequest) -> str:
    """``<base>.f90``, or ``<base>_omp.f90`` (same ``bind(C)`` symbol) with ``parallel``; the write status."""
    if request.bench_info is None:
        raise ValueError("the Fortran emit needs the kernel's bench_info")
    kir = parse_kernel(request.kernel, request.bench_info, precision=request.precision)
    # Whole-array reductions stay intrinsics and a whole-array 1-D np.fft.* becomes an FFTW3 call (see emit_fftw);
    # everything else lowers to loops.
    kir = with_precision(lower(kir, native_call=renders_natively, fft_library=True), request.precision)
    request.out.mkdir(parents=True, exist_ok=True)
    short, base, sym = native_names(request)
    emit, name = (emit_fortran_omp, f"{base}_omp.f90") if request.parallel else (emit_fortran, f"{base}.f90")
    return write_generated(request.out / name, emit(kir, fn_name=sym), line_comment="! ", source=f"{short}_numpy.py")
