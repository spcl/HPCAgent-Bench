"""Unified ``numpyto --target <lang> ...`` driver (directive #1).

A thin dispatcher: ``--target`` picks the backend, and every other argument is
passed straight through to that backend's ``emit`` sub-command. So

    numpyto --target c        --kernel k_numpy.py --bench-info k.json --out DIR
    numpyto --target polly    --kernel ... --bench-info ... --out ...
    numpyto --target pluto    --kernel ... --bench-info ... --out ...
    numpyto --target fortran  --kernel ... --bench-info ... --out ...
    numpyto --target cupy     --kernel ... --out ... [--sanitize]
    numpyto --target numba    --kernel ... --out ... --suffix n [--fastmath] [--sanitize]
    numpyto --target pythran  --kernel ... --bench-info ... --out ... [--precision ...]
    numpyto --target cpp_isopar --kernel ... --bench-info ... --out ...

are equivalent to invoking each per-package CLI directly. The per-package CLIs
remain (the regen scripts call them); this is the single front door over them.

``cpp_isopar`` is the C++ backend spelled over ``<algorithm>``/``<numeric>``: a
map is a ``std::transform``, a reduction a ``std::reduce``, a prefix recurrence
a ``std::inclusive_scan``, and anything with no faithful algorithm stays the
loop it already was. Same symbol and same ABI as ``c``/``cpp`` -- only the body
differs, so the source states the STRUCTURE and the toolchain picks the
schedule.

``polly`` and ``pluto`` are the C-family polyhedral targets: a single
``numpyto_c`` emit already writes the C source, the C++ source, AND the
``#pragma scop``-wrapped Pluto input for the *whole* kernel, so all three share
one backend. ``polly`` reuses the same C/C++ source (Polly is a compile-flag
variant -- nothing changes in emission); ``pluto`` is the polycc input that
emit produces. They are distinct front-door names because the runtime exposes
them (Polly/Pluto were separate file tracks; now flag presets).

Importing a backend requires its ``src`` on ``PYTHONPATH`` (the same wiring the
per-package CLIs already need); the driver itself only resolves the module.
"""
import argparse
import importlib
import sys

#: target name -> backend CLI module exposing ``main(argv)``. The C backend
#: backs three targets -- ``c`` / ``polly`` / ``pluto`` -- because one emit
#: produces the whole C-family (C, C++, and the Pluto ``#pragma scop`` input).
_TARGETS = {
    "c": "numpyto_c.cli",
    "polly": "numpyto_c.cli",
    "pluto": "numpyto_c.cli",
    "fortran": "numpyto_fortran.cli",
    "cupy": "numpyto_cupy.cli",
    "numba": "numpyto_numba.cli",
    "pythran": "numpyto_pythran.cli",
    # OpenMP parallel-scope variants: same backend, ``--parallel`` injected. One C
    # emit produces both the C and C++ OpenMP sources, so ``c_omp`` and ``cpp_omp``
    # share the C backend (like ``c``/``polly``/``pluto``).
    "c_omp": "numpyto_c.cli",
    "cpp_omp": "numpyto_c.cli",
    "fortran_omp": "numpyto_fortran.cli",
    # ISO standard-algorithm C++: same backend, ``--isopar`` injected.
    "cpp_isopar": "numpyto_c.cli",
}

#: Targets that inject ``--parallel`` into the backend emit (OpenMP variants).
_PARALLEL_TARGETS = {"c_omp", "cpp_omp", "fortran_omp"}

#: Targets that inject ``--isopar``: C++ over <algorithm>/<numeric> instead of hand-written loops,
#: so the SOURCE states the map / reduce / scan and the toolchain picks the schedule. No execution
#: policy is emitted, so this is a statement of structure, not a request for threads.
_ISOPAR_TARGETS = {"cpp_isopar"}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="numpyto",
                                 description="Unified NumpyToX emitter: emit a numpy kernel to a target language.")
    ap.add_argument("--target", "-t", required=True, choices=sorted(_TARGETS), help="output language / framework")
    args, rest = ap.parse_known_args(argv)
    mod = importlib.import_module(_TARGETS[args.target])
    if args.target in _PARALLEL_TARGETS and "--parallel" not in rest:
        rest = ["--parallel", *rest]
    if args.target in _ISOPAR_TARGETS and "--isopar" not in rest:
        rest = ["--isopar", *rest]
    return mod.main(["emit", *rest])


if __name__ == "__main__":
    sys.exit(main())
