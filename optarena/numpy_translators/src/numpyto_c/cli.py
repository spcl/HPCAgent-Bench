"""Single-command entry point for emitting one kernel's C / C++ / Pluto files.

Canonical front door is ``numpyto --target {c,polly,pluto}`` (numpyto_common.cli);
this per-package CLI is the backend that driver dispatches to.

Usage::

    numpyto_c emit \\
        --kernel optarena/benchmarks/foundation/s111/s111_numpy.py \\
        --bench-info bench_info/s111.json \\
        --out optarena/benchmarks/foundation/s111/cpp_backend

Idempotent; overwrites previously-emitted files.
"""

import argparse
import pathlib
import sys

from numpyto_c.bindings import emit_binding, emit_pluto_binding
from numpyto_c.emit import emit_c, emit_c_omp, emit_cpp, emit_cpp_omp, emit_pluto
from numpyto_common.frontend import parse_kernel
from numpyto_common.ir import apply_precision
from numpyto_common.lowering import lower
from numpyto_common.emit_io import write_generated
from numpyto_common.naming import native_base


def cmd_emit(args: argparse.Namespace) -> int:
    kir = parse_kernel(args.kernel, args.bench_info, config=args.config, precision=args.precision)
    kir = lower(kir)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    # Derive the kernel name from the input stem (independent of bench_info's
    # ``short_name`` abbreviation or a legacy ``func_name = "kernel"``).
    short = args.kernel.stem.removesuffix("_numpy")
    # Precision is applied ON THE IR (float/complex dtypes only); every emitter
    # then reads each array's dtype -- so the emitted source is precision-
    # MONOMORPHIC and the file/symbol carries its fp tag.
    if args.precision:
        kir = apply_precision(kir, args.precision)
    # Canonical native name: <short>[_<sparse>]_<fptype> for BOTH the file and
    # the exported symbol (no ``_auto``, no per-compiler suffix -- each compiler
    # variant builds its own lib<short>_<framework>.so from this one source).
    base = native_base(short, precision=args.precision, sparse=args.config)
    src = f"{short}_numpy.py"
    if args.parallel:
        # OpenMP variant: a drop-in ``<base>_omp.{c,cpp}`` with the SAME symbol as the
        # sequential emit (compile with ``-fopenmp``). No Pluto here (Pluto is the
        # sequential polyhedral track). ``emit_c_omp`` raises UnsupportedParallelError
        # for a kernel with no sound parallel form -- propagated as a nonzero exit.
        write_generated(out / f"{base}_omp.c", emit_c_omp(kir, fn_name=base), line_comment="// ", source=src)
        write_generated(out / f"{base}_omp.cpp", emit_cpp_omp(kir, fn_name=base), line_comment="// ", source=src)
        emit_binding(kir, out / f"{base}_omp_binding.json", base_name=base)
        print(f"numpyto_c: emitted {base}_omp.{{c,cpp}} (OpenMP) + {base}_omp_binding.json")
        return 0
    write_generated(out / f"{base}.c", emit_c(kir, fn_name=base), line_comment="// ", source=src)
    write_generated(out / f"{base}.cpp", emit_cpp(kir, fn_name=base), line_comment="// ", source=src)
    write_generated(out / f"{base}_pluto_input.c", emit_pluto(kir, fn_name=base),
                    line_comment="// ", source=src)
    emit_binding(kir, out / f"{base}_binding.json", base_name=base)
    # Pluto's VLA-param signature reorders args (size symbols first), so it needs its
    # own binding for the harness to marshal correctly (see emit_pluto_binding).
    emit_pluto_binding(kir, out / f"{base}_pluto_binding.json", base_name=base)
    print(f"numpyto_c: emitted {base}.{{c,cpp}} + {base}_pluto_input.c + {base}_binding.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="numpyto_c", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit", help="emit one kernel")
    e.add_argument("--kernel", type=pathlib.Path, required=True,
                   help="path to <short>_numpy.py")
    e.add_argument("--bench-info", type=pathlib.Path, required=True,
                   help="path to bench_info/<short>.json")
    e.add_argument("--out", type=pathlib.Path, required=True,
                   help="output cpp_backend/ directory")
    e.add_argument("--parallel", action="store_true",
                   help="emit the OpenMP variant (<base>_omp.{c,cpp}, "
                        "``#pragma omp parallel for``) instead of the sequential "
                        "source; compile with -fopenmp. Refuses (nonzero exit) a "
                        "kernel with no sound parallel form (colliding scatter).")
    e.add_argument("--precision", default="",
                   help="floating precision override (e.g. ``float32`` / "
                        "``float16``). Remaps ONLY float/complex arrays, "
                        "scalars and locals; int index arrays are unchanged. "
                        "Empty = use each array's declared dtype (fp64).")
    e.add_argument("--config", default=None,
                   help="sparse configuration key to emit (one of the "
                        "kernel's ``configurations``, i.e. a "
                        "``ResolvedBench.config_key``). Deterministic "
                        "per-sub-benchmark emission; overrides the "
                        "$OPTARENA_SPARSE_CONFIG fallback. Ignored for "
                        "dense kernels.")
    e.set_defaults(func=cmd_emit)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
