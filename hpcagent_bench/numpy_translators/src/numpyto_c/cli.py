"""CLI entry point for emitting one kernel's C / C++ / Pluto files; backend for ``numpyto --target {c,polly,pluto}``."""

import argparse
import sys

from numpyto_common.emit_helpers.cli import (
    add_precision,
    emit_parser,
    native_names,
    run,
    with_inline_fallback,
    with_precision,
)
from numpyto_common.emit_io import write_generated
from numpyto_common.frontend import parse_kernel
from numpyto_common.lowering import lower

from numpyto_c.bindings import emit_binding, emit_pluto_binding
from numpyto_c.emit import emit_c, emit_c_omp, emit_cpp, emit_cpp_isopar, emit_cpp_omp, emit_pluto

#: Precisions real BLAS has a gemm for; any other keeps the loop nest.
BLAS_PRECISIONS = ("", "float32", "float64")


def emit_once(args: argparse.Namespace) -> int:
    kir = parse_kernel(args.kernel, args.bench_info, config=args.config, precision=args.precision)
    # C and C++ hand a dense 2-D float GEMM to BLAS and a whole-array np.fft.* to FFTW3; Pluto does
    # not (see below). BLAS is gated on the REQUESTED precision: apply_precision runs after lowering.
    kir = with_precision(
        lower(kir, blas=args.precision in BLAS_PRECISIONS, fft_library=True, fft_library_nd=True), args.precision
    )
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    short, base, sym = native_names(args)
    src = f"{short}_numpy.py"
    if args.isopar:
        # C++ only (C has no <algorithm>), same symbol as sequential.
        write_generated(out / f"{base}_isopar.cpp", emit_cpp_isopar(kir, fn_name=sym), line_comment="// ", source=src)
        emit_binding(kir, out / f"{base}_isopar_binding.json", base_name=base, symbol=sym)
        print(f"numpyto_c: emitted {base}_isopar.cpp (ISO algorithms) + {base}_isopar_binding.json")
        return 0
    if args.parallel:
        # Same symbol as sequential; no Pluto (sequential-only track).
        write_generated(out / f"{base}_omp.c", emit_c_omp(kir, fn_name=sym), line_comment="// ", source=src)
        write_generated(out / f"{base}_omp.cpp", emit_cpp_omp(kir, fn_name=sym), line_comment="// ", source=src)
        emit_binding(kir, out / f"{base}_omp_binding.json", base_name=base, symbol=sym)
        print(f"numpyto_c: emitted {base}_omp.{{c,cpp}} (OpenMP) + {base}_omp_binding.json")
        return 0
    write_generated(out / f"{base}.c", emit_c(kir, fn_name=sym), line_comment="// ", source=src)
    write_generated(out / f"{base}.cpp", emit_cpp(kir, fn_name=sym), line_comment="// ", source=src)
    # Pluto optimises the contraction itself, so it gets the loop-lowered matmul: a library call is
    # opaque to the scop. Re-parsed rather than shared, so neither lowering sees the other's rewrites.
    pluto_kir = with_precision(
        lower(parse_kernel(args.kernel, args.bench_info, config=args.config, precision=args.precision)), args.precision
    )
    write_generated(out / f"{base}_pluto_input.c", emit_pluto(pluto_kir, fn_name=sym), line_comment="// ", source=src)
    emit_binding(kir, out / f"{base}_binding.json", base_name=base, symbol=sym)
    # Pluto's VLA-param signature reorders args (symbols first), so it needs its own binding.
    emit_pluto_binding(pluto_kir, out / f"{base}_pluto_binding.json", base_name=base, symbol=sym)
    print(f"numpyto_c: emitted {base}.{{c,cpp}} + {base}_pluto_input.c + {base}_binding.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser, emit = emit_parser("numpyto_c", __doc__, bench_info_required=True)
    # One variant per emit: each writes its own source set, so asking for two is a mistake, not a mix.
    variant = emit.add_mutually_exclusive_group()
    variant.add_argument(
        "--parallel",
        action="store_true",
        help="emit the OpenMP variant (<base>_omp.{c,cpp}); compile with -fopenmp. Refuses (nonzero exit) a "
        "kernel with no sound parallel form (colliding scatter).",
    )
    variant.add_argument(
        "--isopar",
        action="store_true",
        help="emit the ISO standard-algorithm C++ variant (<base>_isopar.cpp): a loop with a faithful "
        "<algorithm>/<numeric> spelling becomes that call (transform / reduce / inclusive_scan), the rest "
        "stay loops. Never refuses a kernel.",
    )
    add_precision(emit)
    emit.set_defaults(func=with_inline_fallback(emit_once))
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser(), argv)


if __name__ == "__main__":
    sys.exit(main())
