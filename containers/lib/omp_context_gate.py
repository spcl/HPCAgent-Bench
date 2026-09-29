# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One OpenMP context, one process: every OpenMP user of that family runs multi-threaded on ONE runtime.

    python3 omp_context_gate.py --context {gnu,llvm,nvhpc} [--root /opt/omp] [--require PROBE ...]
                                [--torch] [--wheels] [--blas-in-context] [--expect-runtime PATH | --any-runtime]

Run by containers/lib/omp_contexts.sh at image build and by containers/images/verify_image.py in the
finished image, once per context the image carries. The script re-executes itself under the context's
environment (``LD_LIBRARY_PATH`` led by ``<root>/<context>/lib``, the way the judge starts a grading child of
that family), then, in ONE process:

* builds and RUNS an OpenMP probe (``openmp_probe.c`` / ``openmp_probe.f90``: every schedule, tasks, taskloop,
  atomic, critical, simd, locks, threadprivate) with each compiler of the family found: gcc and gfortran in
  ``gnu``; clang, and flang / hipcc / amdclang where present, in ``llvm``; nvc and nvfortran in ``nvhpc``.
  Each probe reports the team size it saw, and every one must exceed 1: a pragma compiled to serial code (the
  silent failure: ``clang -fopenmp=libgomp``) reports a team of 1;
* runs numpy and scipy BLAS (a large matrix product must keep more than one core busy);
* runs a numba ``prange`` whose iterations call BLAS (``gnu`` and ``llvm``): the threading layer must be
  ``omp``, the iterations must land on more than one thread;
* runs a torch op (``--torch``, the gnu context) and imports every wheel that may bundle a runtime
  (``--wheels``, after ``one_openmp.sh`` linked their bundled copies to the image's);

then asserts exactly ONE OpenMP runtime realpath is mapped, and that it is the context's
(``<root>/<context>/lib``'s ``libgomp.so.1`` in gnu and llvm, a ``libnvomp`` in nvhpc; ``--expect-runtime`` names
another). ``--blas-in-context`` also asserts the one mapped ``libopenblas`` is the file the context's
``lib/libopenblas.so.0`` names, which is what makes numpy and scipy run on that family's variant. Exit status 3 is a
second runtime, 4 a serial team or a BLAS that used one core, anything else a crash.

Standard library only apart from the numerical stack it exercises; ``openmp_runtimes.py`` beside it counts.
"""

import argparse
import ctypes
import dataclasses
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import openmp_runtimes

#: Exit status for "two runtimes mapped" and for "not multi-threaded", apart from every other failure.
CONFLICT_EXIT = 3
SERIAL_EXIT = 4

#: The gate counts runtimes and threads, it does not load-test BLAS (numpy_on_openblas.sh does, at 2 x nproc
#: callers): a few threads keep it safe beside a 512-thread OpenBLAS on a large host.
GATE_THREADS = 4

#: A parallel BLAS or torch op keeps more than one core busy: process CPU time over wall time, at least this.
MIN_PARALLELISM = 1.5

#: Result slots and check names, in the order ``openmp_probe.c`` / ``.f90`` write them.
TEAM_SLOTS = ("static", "dynamic", "guided", "task", "barrier")
CHECKS = (
    "static",
    "dynamic",
    "guided",
    "runtime",
    "collapse",
    "ordered",
    "sections",
    "min/max reduction",
    "tasks",
    "taskloop",
    "task depend",
    "atomic",
    "critical",
    "simd",
    "parallel for simd",
    "lock",
    "nest lock",
    "threadprivate",
    "max threads",
)

#: The optional wheels imported (torch is run when asked for): each may bundle or load a runtime of its own.
OPTIONAL_USERS = ("sklearn", "xgboost", "lightgbm", "jax", "cupy", "tvm", "pythran")

HERE = pathlib.Path(__file__).resolve().parent

#: The variable a process of a context carries; the gate re-executes itself until it is set.
CONTEXT_ENV = "HPCAGENT_BENCH_OMP_CONTEXT"


class SerialTeam(RuntimeError):
    """OpenMP code that ran on one thread: a pragma compiled to serial code, or a runtime that never forked."""


@dataclasses.dataclass(frozen=True)
class Probe:
    """One compiler of a family and how it builds ``openmp_probe.<ext>`` into a shared library."""

    name: str
    driver: tuple[str, ...]
    source: str
    flags: tuple[str, ...]


#: family -> (probes that must run, probes that run when their compiler is present). ``CC``/``FC`` name the
#: image's gcc and gfortran (a stale PATH must not pick another).
def probes_for(context: str) -> tuple[list[Probe], list[Probe]]:
    c, f90 = "openmp_probe.c", "openmp_probe.f90"
    gcc = Probe("gcc", (os.environ.get("CC", "gcc"),), c, ("-fopenmp",))
    gfortran = Probe("gfortran", (os.environ.get("FC", "gfortran"),), f90, ("-fopenmp",))
    clang = Probe("clang", ("clang",), c, ("-fopenmp",))
    flang = Probe("flang", ("flang",), f90, ("-fopenmp",))
    hipcc = Probe("hipcc", ("hipcc",), c, ("-fopenmp", "-x", "c++"))
    amdclang = Probe("amdclang", ("amdclang",), c, ("-fopenmp",))
    nvc = Probe("nvc", ("nvc",), c, ("-mp",))
    nvfortran = Probe("nvfortran", ("nvfortran",), f90, ("-mp",))
    return {
        "gnu": ([gcc, gfortran], []),
        "llvm": ([clang], [flang, hipcc, amdclang]),
        "nvhpc": ([nvc], [nvfortran]),
    }[context]


def context_environment(root: pathlib.Path, context: str, base: dict[str, str] | None = None) -> dict[str, str]:
    """The environment ENTRIES of a child in ``context``. The same function as
    ``hpcagent_bench.omp_context.context_env`` (this file runs before that package exists);
    tests/test_omp_context_gate.py pins the two equal."""
    env = dict(os.environ if base is None else base)
    lib = root / context / "lib"
    entries = [str(lib)] if lib.is_dir() else []
    seen: list[str] = []
    for item in [*entries, *env.get("LD_LIBRARY_PATH", "").split(os.pathsep)]:
        if item and item not in seen:
            seen.append(item)
    layer = {"gnu": "omp", "llvm": "omp", "nvhpc": "workqueue"}[context]
    return {
        "LD_LIBRARY_PATH": os.pathsep.join(seen),
        "NUMBA_THREADING_LAYER": layer,
        CONTEXT_ENV: context,
    }


def build_probe(probe: Probe, tmp: pathlib.Path) -> ctypes.CDLL | None:
    """Compile ``probe`` and load it; ``None`` when the compiler is absent."""
    exe = shutil.which(probe.driver[0])
    if exe is None:
        return None
    lib = tmp / f"lib{probe.name}_probe.so"
    command = [
        exe,
        *probe.driver[1:],
        "-O2",
        "-fPIC",
        "-shared",
        *probe.flags,
        str(HERE / probe.source),
        "-o",
        str(lib),
    ]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"{probe.name}: {' '.join(command)} failed:\n{done.stdout}{done.stderr}")
    return ctypes.CDLL(str(lib))


def step(message: str) -> None:
    """Progress on stderr, flushed: a hung step must name itself in the build log."""
    print(f"[omp_context_gate] {message}", file=sys.stderr, flush=True)


def run_probe(probe: Probe, lib: ctypes.CDLL) -> str:
    """Run the probe library with ``GATE_THREADS`` threads; the summary line, or raises."""
    result = (ctypes.c_int * len(TEAM_SLOTS))()
    lib.omp_probe.argtypes = [ctypes.c_int, ctypes.c_void_p]
    lib.omp_probe.restype = ctypes.c_int
    failed = lib.omp_probe(GATE_THREADS, ctypes.addressof(result))
    if failed:
        names = [name for bit, name in enumerate(CHECKS) if failed >> bit & 1]
        raise RuntimeError(f"{probe.name}: OpenMP probe computed a wrong answer for {names}")
    teams = dict(zip(TEAM_SLOTS, result, strict=True))
    serial = [name for name, size in teams.items() if size < 2]
    if serial:
        raise SerialTeam(f"{probe.name}: a team of one thread in {serial}: {teams}")
    return f"{probe.name}: teams {teams}"


def cpu_over_wall(work) -> float:
    """Process CPU seconds per wall second while ``work()`` runs: above 1 only if threads really ran."""
    cpu0, wall0 = time.process_time(), time.perf_counter()
    work()
    return (time.process_time() - cpu0) / max(time.perf_counter() - wall0, 1e-9)


def require_parallel(what: str, ratio: float) -> str:
    if len(os.sched_getaffinity(0)) < 2:
        raise SerialTeam(f"{what}: this process may use one CPU, so no threading can be shown")
    if ratio < MIN_PARALLELISM:
        raise SerialTeam(f"{what}: CPU time was {ratio:.2f} x wall time, expected at least {MIN_PARALLELISM}")
    return f"{what}: CPU/wall {ratio:.1f}"


def blas_multithreaded() -> list[str]:
    """numpy and scipy on the context's BLAS: a big product keeps more than one core busy."""
    import numpy as np
    import scipy.linalg

    rng = np.random.default_rng(0)
    a = rng.random((2048, 2048))
    a @ a  # first call: thread pool and kernels
    ratio = cpu_over_wall(lambda: [a @ a for _ in range(3)])
    small = rng.random((256, 256))
    scipy.linalg.lu_factor(small)
    assert np.allclose((small @ small)[0, 0], small[0] @ small[:, 0])
    return [require_parallel("numpy dgemm 2048", ratio)]


def numba_prange_calling_blas() -> list[str]:
    """numba's OpenMP pool with BLAS inside each iteration: the pair whose second runtime cost nproc^2 threads."""
    import numba
    import numpy as np

    threads = min(GATE_THREADS, numba.config.NUMBA_NUM_THREADS)
    numba.set_num_threads(threads)

    @numba.njit(parallel=True)
    def prange_blas(a: np.ndarray, out: np.ndarray, ids: np.ndarray) -> None:
        for i in numba.prange(out.shape[0]):
            out[i] = np.dot(a, a)[0, 0]
            ids[i] = numba.get_thread_id()

    a = np.random.default_rng(1).random((64, 64))
    out, ids = np.empty(4 * threads), np.empty(4 * threads, dtype=np.int64)
    prange_blas(a, out, ids)
    assert np.allclose(out, (a @ a)[0, 0])
    layer = numba.threading_layer()
    if layer != "omp":
        raise RuntimeError(f"numba's threading layer is {layer!r}, not omp")
    if len(set(ids.tolist())) < 2:
        raise SerialTeam(f"numba prange ran on thread ids {sorted(set(ids.tolist()))}")
    return [f"numba prange + BLAS: layer {layer}, {len(set(ids.tolist()))} threads"]


def torch_parallel_op() -> list[str]:
    """A torch elementwise op over a large tensor keeps more than one core busy (ATen's OpenMP pool)."""
    import torch

    torch.set_num_threads(GATE_THREADS)
    if not torch.backends.openmp.is_available():
        raise RuntimeError("torch was built without OpenMP")
    x = torch.rand(1 << 24)
    torch.exp(x)
    ratio = cpu_over_wall(lambda: [torch.exp(x) for _ in range(8)])
    return [require_parallel(f"torch exp, {torch.get_num_threads()} threads", ratio)]


def blas_in_context(root: pathlib.Path, context: str) -> list[str]:
    """Every mapped ``libopenblas`` is the file the context's ``lib/libopenblas.so.0`` names (its own variant),
    not the image default's: that is what makes numpy and scipy run on that family's BLAS."""
    link = root / context / "lib" / "libopenblas.so.0"
    if not link.exists():
        raise RuntimeError(f"the {context} context has no libopenblas.so.0 for numpy and scipy to resolve")
    ours = os.path.realpath(link)
    blas = sorted({os.path.realpath(p) for p in mapped_files() if "openblas" in os.path.basename(p)})
    if not blas:
        raise RuntimeError("no libopenblas is mapped: numpy and scipy are not on the image's OpenBLAS")
    if blas != [ours]:
        raise RuntimeError(f"{context} context: numpy and scipy mapped OpenBLAS {blas}, the context's is {ours}")
    return [f"OpenBLAS of the {context} context: {ours}"]


def mapped_files() -> list[str]:
    """The files this process has mapped, by path."""
    paths: set[str] = set()
    with open(openmp_runtimes.MAPS_PATH, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.split(maxsplit=5)
            if len(fields) == openmp_runtimes.MAPS_FIELDS and fields[5].startswith("/"):
                paths.add(fields[5].strip().removesuffix(openmp_runtimes.DELETED))
    return sorted(paths)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True, choices=("gnu", "llvm", "nvhpc"))
    parser.add_argument("--root", default="/opt/omp", type=pathlib.Path)
    parser.add_argument("--require", nargs="*", default=None, metavar="PROBE", help="probes that must be present")
    parser.add_argument("--torch", action="store_true", help="run a torch op (the gnu context)")
    parser.add_argument(
        "--wheels", action="store_true", help="import every wheel that may bundle a runtime (after one_openmp.sh)"
    )
    parser.add_argument("--blas-in-context", action="store_true", help="numpy's OpenBLAS must be the context's own")
    parser.add_argument(
        "--expect-runtime", default=None, help="the one runtime that may be mapped (default: the context's)"
    )
    parser.add_argument(
        "--any-runtime", action="store_true", help="assert one runtime, whichever file (a host that is no image)"
    )
    parser.add_argument("--no-numpy", action="store_true", help="skip numpy, scipy and numba (an image without them)")
    return parser.parse_args(argv)


def expected_runtime(root: pathlib.Path, context: str, override: str | None) -> str | None:
    if override:
        return os.path.realpath(override)
    if context == "nvhpc":
        return None  # any single libnvomp
    lib = root / context / "lib" / "libgomp.so.1"
    return os.path.realpath(lib) if lib.exists() else None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if os.environ.get(CONTEXT_ENV) != args.context:
        # The loader reads LD_LIBRARY_PATH at exec: become a process of the context.
        env = {**os.environ, **context_environment(args.root, args.context)}
        os.execve(sys.executable, [sys.executable, str(pathlib.Path(__file__).resolve()), *sys.argv[1:]], env)
    os.environ["OMP_NUM_THREADS"] = str(GATE_THREADS)
    os.environ["NUMBA_NUM_THREADS"] = str(GATE_THREADS)
    must, maybe = probes_for(args.context)
    required = set(args.require) if args.require is not None else {probe.name for probe in must}
    lines: list[str] = []
    try:
        with tempfile.TemporaryDirectory() as raw:
            tmp = pathlib.Path(raw)
            for probe in [*must, *maybe]:
                step(f"probe {probe.name}")
                lib = build_probe(probe, tmp)
                if lib is None:
                    if probe.name in required:
                        raise RuntimeError(f"{probe.name} is required in the {args.context} context and is not on PATH")
                    lines.append(f"{probe.name}: absent, skipped")
                    continue
                lines.append(run_probe(probe, lib))
        if not args.no_numpy:
            step("numpy and scipy BLAS")
            lines += blas_multithreaded()
            if args.context != "nvhpc":
                step("numba prange calling BLAS")
                lines += numba_prange_calling_blas()
        if args.torch:
            step("torch op")
            lines += torch_parallel_op()
        step("optional wheels")
        present = openmp_runtimes.import_all([], OPTIONAL_USERS if args.wheels else ())
        if args.blas_in_context:
            lines += blas_in_context(args.root, args.context)
    except SerialTeam as serial:
        print(serial, file=sys.stderr)
        return SERIAL_EXIT
    runtimes = openmp_runtimes.mapped_runtimes()
    expected = None if args.any_runtime else expected_runtime(args.root, args.context, args.expect_runtime)
    try:
        openmp_runtimes.assert_single_runtime(runtimes, f"{args.context} context")
        if not runtimes:
            raise openmp_runtimes.OpenMPRuntimeConflict(f"{args.context} context: no OpenMP runtime is mapped at all")
        if expected is not None and runtimes != (expected,):
            raise openmp_runtimes.OpenMPRuntimeConflict(
                f"{args.context} context: mapped {runtimes}, expected {expected}"
            )
    except openmp_runtimes.OpenMPRuntimeConflict as conflict:
        print(conflict, file=sys.stderr)
        return CONFLICT_EXIT
    print(*lines, sep="\n")
    print(f"{args.context} context: one OpenMP runtime mapped: {runtimes} (also imported: {present})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
