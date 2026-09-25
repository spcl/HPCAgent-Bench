"""The ``<backend> emit`` command line every per-backend CLI shares.

parser, emit = emit_parser("numpyto_fortran", __doc__, bench_info_required=True)
add_precision(emit)
emit.set_defaults(func=with_inline_fallback(emit_once))
...
def main(argv=None) -> int:
    return run(build_parser(), argv)
"""

import argparse
import pathlib
from collections.abc import Callable, Sequence

from hpcagent_bench.translators.numpyto_common.ir import KernelIR, apply_precision
from hpcagent_bench.translators.numpyto_common.naming import entry_symbol, native_base, short_for

type EmitFn = Callable[[argparse.Namespace], int]


def emit_parser(
    prog: str, description: str | None = None, *, bench_info_required: bool
) -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    """The top-level parser and its ``emit`` sub-parser, carrying ``--kernel``, ``--bench-info``,
    ``--out`` and ``--config``."""
    parser = argparse.ArgumentParser(prog=prog, description=description)
    emit = parser.add_subparsers(dest="cmd", required=True).add_parser("emit", help="emit one kernel")
    emit.add_argument("--kernel", type=pathlib.Path, required=True, help="path to <short>_numpy.py")
    emit.add_argument(
        "--bench-info", type=pathlib.Path, required=bench_info_required, help="bench_info JSON of the kernel"
    )
    emit.add_argument("--out", type=pathlib.Path, required=True, help="output directory")
    emit.add_argument(
        "--config",
        default=None,
        help="sparse configuration key (one of the kernel's ``configurations``); tags the emitted name. "
        "Dense kernels omit it.",
    )
    return parser, emit


def add_precision(emit: argparse.ArgumentParser) -> None:
    """``--precision``: remap float/complex arrays, scalars and locals; integers are unchanged.
    Empty keeps each array's declared dtype."""
    emit.add_argument(
        "--precision",
        default="",
        help="floating precision override (e.g. ``float32``); remaps float/complex only, ints unchanged.",
    )


def add_sanitize(emit: argparse.ArgumentParser) -> None:
    """``--sanitize``: strip comments and docstrings before the file crosses into a container."""
    emit.add_argument("--sanitize", action="store_true", help="strip comments/docstrings (container handoff)")


def with_inline_fallback(emit_once: EmitFn) -> EmitFn:
    """``emit_once``, repeated once with helper inlining forced on when the kept-helper form fails."""
    from hpcagent_bench.translators.numpyto_common.frontend import (
        emit_with_inline_fallback,
    )  # heavy; cupy's CLI never needs it

    return lambda args: emit_with_inline_fallback(lambda: emit_once(args))


def run(parser: argparse.ArgumentParser, argv: Sequence[str] | None) -> int:
    """Parse ``argv`` and run the selected sub-command."""
    args = parser.parse_args(argv)
    return args.func(args)


def with_precision(kir: KernelIR, precision: str) -> KernelIR:
    """``kir`` at ``precision`` (applied on the IR, so the emitted source is precision-monomorphic)."""
    return apply_precision(kir, precision) if precision else kir


def native_names(args: argparse.Namespace) -> tuple[str, str, str]:
    """``(short, base, symbol)`` of a native emit: ``<short>[_<sparse>]_<fptype>`` names the file
    and its lowercased form is the exported symbol, since Fortran folds case."""
    short = short_for(args.kernel)
    base = native_base(short, precision=args.precision, sparse=args.config)
    return short, base, entry_symbol(base)
