"""Guards for glibc's vector libm (libmvec) across the compiler matrix: EVERY CPU baseline must reach
libmvec (a ~3x gap), by whatever knob its compiler family offers, or the compiler axis silently
measures a library difference instead of a compiler one. clang gets ``-fveclib=libmvec`` built in;
gcc/g++ need ``-include vecmath.h`` (glibc hides its decls behind __FAST_MATH__, which we don't set);
gfortran gets it for free from the driver spec, a host property asserted here rather than assumed."""
import ctypes.util
import pathlib
import re
import shutil
import subprocess

import pytest

from optarena import flags, languages, osinfo
from optarena.languages import _load_compilers

#: Any libmvec entry point: _ZGV <isa> N <width> <v...> _ <fn>, e.g. _ZGVeN8v_exp.
LIBMVEC_SYMBOL = re.compile(r"_ZGV[a-z]N\d+v+_\w+")

#: ``#pragma omp declare simd ...`` followed by ``<ret> <name>(...)`` in vecmath.h.
DECLARED_IN_HEADER = re.compile(r"^\s*(?:double|float)\s+(\w+)\s*\(", re.MULTILINE)

C_LIBM_LOOP = """
#include <math.h>
void kern(double *restrict a, double *restrict b, long n) {
  for (long i = 0; i < n; i++) b[i] = exp(a[i]) + log(a[i] + 2.0);
}
"""

CXX_LIBM_LOOP = """
#include <cmath>
void kern(double *__restrict a, double *__restrict b, long n) {
  for (long i = 0; i < n; i++) b[i] = std::exp(a[i]) + std::log(a[i] + 2.0);
}
"""

FORTRAN_LIBM_LOOP = """
subroutine kern(a, b, n)
  integer, intent(in) :: n
  real(8), intent(in) :: a(n)
  real(8), intent(out) :: b(n)
  integer :: i
  do i = 1, n
     b(i) = exp(a(i)) + log(a(i) + 2.0d0)
  end do
end subroutine kern
"""


def compile_object(tmp_path, source: str, suffix: str, exe: str, baseline: str, *extra: str) -> pathlib.Path:
    """Compile ``source`` with ``baseline`` and return the object path; ``baseline`` is passed exactly
    as the build path expands it, so this exercises the same string a kernel build gets."""
    src = tmp_path / f"probe{suffix}"
    src.write_text(source)
    obj = tmp_path / f"probe{suffix}.o"
    cmd = [exe, *baseline.split(), *extra, "-c", str(src), "-o", str(obj)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"compile failed:\n$ {' '.join(cmd)}\n{proc.stderr}"
    return obj


def libmvec_calls(obj: pathlib.Path) -> set:
    """The libmvec symbols ``obj`` calls, read via ``nm --undefined-only`` (no disassembling needed)."""
    out = subprocess.run(["nm", "--undefined-only", str(obj)], capture_output=True, text=True, check=True).stdout
    return set(LIBMVEC_SYMBOL.findall(out))


def declared_functions() -> list:
    return DECLARED_IN_HEADER.findall(flags.VECMATH_H.read_text())


# --- Config guards: no toolchain needed, so they run in every job -----------------------


def test_the_vecmath_header_ships_with_the_package():
    """flags.py -include's this path on every gcc/g++ compile; setup.py + MANIFEST.in must list it."""
    assert flags.VECMATH_H.is_file(), f"{flags.VECMATH_H} is missing"
    root = pathlib.Path(flags.__file__).resolve().parents[1]
    manifest = (root / "MANIFEST.in").read_text()
    assert "envs/vecmath.h" in manifest, "vecmath.h is not listed in MANIFEST.in; wheels will drop it"
    setup_py = (root / "setup.py").read_text()
    assert "envs/vecmath.h" in setup_py, "vecmath.h is not in setup.py package_data; wheels will drop it"


def test_every_cpu_baseline_reaches_libmvec():
    """The point of the whole file: no CPU baseline may silently lack the vector libm while another
    has it, or the compiler axis measures libmvec instead of the compiler."""
    if not osinfo.IS_LINUX:
        # Not a skip: on macOS there is no libmvec, so the correct state is that neither knob is present.
        assert "-fveclib" not in flags.CPU_BASELINE_CLANG
        assert "-include" not in flags.CPU_BASELINE_GCC
        return
    assert "-fveclib=libmvec" in flags.CPU_BASELINE_CLANG
    assert "-include" in flags.CPU_BASELINE_GCC and "vecmath.h" in flags.CPU_BASELINE_GCC


def test_the_fortran_baseline_does_not_carry_the_c_header():
    """gfortran rejects a C header (a warning on every compile, fatal under -Werror); it gets libmvec
    from glibc's Fortran directives instead (asserted for real below)."""
    assert "-include" not in flags.CPU_BASELINE_GFORTRAN
    assert "vecmath.h" not in flags.CPU_BASELINE_GFORTRAN


@pytest.mark.parametrize("block", ["gfortran", "mpifort"])
def test_fortran_compilers_use_the_fortran_baseline(block):
    """Both gfortran blocks must name CPU_BASELINE_GFORTRAN; mpifort wraps gfortran and is easy to forget."""
    compilers = _load_compilers()
    assert compilers[block]["baseline_ref"] == "CPU_BASELINE_GFORTRAN"


def test_the_header_declares_nothing_libmvec_does_not_export():
    """Declaring a function libmvec does not export makes every kernel using it fail to link; assert it
    here, where the message says which function, instead of mid-build."""
    if not osinfo.IS_LINUX:
        return
    libmvec = ctypes.util.find_library("mvec")
    assert libmvec, "glibc libmvec not found on this host; CPU_BASELINE_GCC would emit unresolvable calls"
    path = subprocess.run(["gcc", "-print-file-name=libmvec.so"], capture_output=True, text=True,
                          check=True).stdout.strip()
    out = subprocess.run(["nm", "-D", "--defined-only", path], capture_output=True, text=True, check=True).stdout
    exported = {s.split("@")[0] for s in re.findall(r"_ZGV\w+", out)}
    declared = declared_functions()
    assert declared, "parsed no declarations out of vecmath.h -- the regex has drifted from the file"
    for fn in declared:
        # The 128-bit SSE2 form (_ZGVbN2v_/_ZGVbN4v_) is the one every x86-64 host has.
        assert any(s.endswith(f"_{fn}") and s.startswith("_ZGVb") for s in exported), \
            f"vecmath.h declares {fn}(), but this host's libmvec exports no vector {fn} -- kernels using it will not link"


# --- Behavioural guards: gcc/g++/gfortran are present in every job that runs these -------


def test_gcc_vectorizes_libm_at_the_baseline(tmp_path):
    """The regression this file exists for: before the header, gcc called scalar libm in a loop while
    clang vectorized the same source."""
    if not osinfo.IS_LINUX:
        return
    assert shutil.which("gcc"), "gcc is required to build native C kernels"
    obj = compile_object(tmp_path, C_LIBM_LOOP, ".c", "gcc", flags.CPU_BASELINE_GCC, languages.std_flag("c"))
    calls = libmvec_calls(obj)
    assert calls, "gcc emitted NO libmvec calls at CPU_BASELINE_GCC -- the vecmath.h -include is not reaching it"
    assert any("_exp" in s for s in calls) and any("_log" in s for s in calls), f"got {sorted(calls)}"


def test_gxx_vectorizes_libm_at_the_baseline(tmp_path):
    """C++ is a separate risk from C: the decls must survive <cmath>'s extern "C" + noexcept."""
    if not osinfo.IS_LINUX:
        return
    assert shutil.which("g++"), "g++ is required to build native C++ kernels"
    obj = compile_object(tmp_path, CXX_LIBM_LOOP, ".cpp", "g++", flags.CPU_BASELINE_GCC, languages.std_flag("cpp"))
    assert libmvec_calls(obj), "g++ emitted NO libmvec calls at CPU_BASELINE_GCC"


def test_gfortran_vectorizes_libm_at_the_baseline(tmp_path):
    """gfortran gets libmvec for free via the driver spec's pre-include; a host whose spec omits it
    silently loses libmvec while C keeps it, so the fortran column stops being comparable."""
    if not osinfo.IS_LINUX:
        return
    assert shutil.which("gfortran"), "gfortran is required to build native Fortran kernels"
    obj = compile_object(tmp_path, FORTRAN_LIBM_LOOP, ".f90", "gfortran", flags.CPU_BASELINE_GFORTRAN, "-ffree-form",
                         languages.std_flag("fortran"))
    assert libmvec_calls(obj), ("gfortran emitted NO libmvec calls at CPU_BASELINE_GFORTRAN -- this host's gcc spec "
                                "does not pre-include glibc's math-vector-fortran.h, so Fortran needs an explicit "
                                "-fpre-include that C does not")


def test_the_fortran_baseline_compiles_without_warnings(tmp_path):
    """The concrete reason CPU_BASELINE_GFORTRAN exists: the C/C++ baseline made gfortran warn on every compile."""
    if not osinfo.IS_LINUX:
        return
    assert shutil.which("gfortran"), "gfortran is required to build native Fortran kernels"
    src = tmp_path / "warn.f90"
    src.write_text(FORTRAN_LIBM_LOOP)
    cmd = [
        "gfortran", *flags.CPU_BASELINE_GFORTRAN.split(), "-ffree-form",
        languages.std_flag("fortran"), "-c",
        str(src), "-o",
        str(tmp_path / "warn.o")
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "not for Fortran" not in proc.stderr, f"gfortran was handed a C-only flag:\n{proc.stderr}"


# --- Why the header, and not -D__FAST_MATH__ --------------------------------------------


def test_the_header_does_not_leak_fast_math_into_libstdcxx(tmp_path):
    """-D__FAST_MATH__ was rejected because <bits/c++config.h> turns it into _GLIBCXX_FAST_MATH=1,
    changing libstdc++'s complex infinity handling; our header must not do that."""
    if not osinfo.IS_LINUX:
        return
    assert shutil.which("g++"), "g++ is required to build native C++ kernels"
    probe = tmp_path / "leak.cpp"
    probe.write_text(f'#include "{flags.VECMATH_H}"\n'
                     "#include <cmath>\n"
                     "#if _GLIBCXX_FAST_MATH\n"
                     "#error fast_math_leaked\n"
                     "#endif\n"
                     "int main() { return 0; }\n")
    base = ["g++", languages.std_flag("cpp"), "-fopenmp", "-fsyntax-only", str(probe)]
    clean = subprocess.run(base, capture_output=True, text=True)
    assert clean.returncode == 0, f"vecmath.h leaked __FAST_MATH__ into libstdc++:\n{clean.stderr}"
    poisoned = subprocess.run([*base, "-ffast-math"], capture_output=True, text=True)
    assert poisoned.returncode != 0, "probe is vacuous: it does not even detect a real -ffast-math"


def test_the_header_does_not_change_math_errhandling(tmp_path):
    """The C half of the same lie: -D__FAST_MATH__ flips math_errhandling from MATH_ERREXCEPT to 0."""
    if not osinfo.IS_LINUX:
        return
    assert shutil.which("gcc"), "gcc is required to build native C kernels"
    values = {}
    for label, extra in (("with", ["-include", str(flags.VECMATH_H)]), ("without", [])):
        src = tmp_path / f"meh_{label}.c"
        src.write_text("#include <math.h>\n"
                       "#include <stdio.h>\n"
                       "int main(void) { printf(\"%d\\n\", math_errhandling); return 0; }\n")
        exe = tmp_path / f"meh_{label}"
        subprocess.run(
            ["gcc", "-O2", "-fopenmp", "-fno-math-errno", *extra,
             str(src), "-o", str(exe), "-lm"],
            capture_output=True,
            text=True,
            check=True)
        values[label] = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.strip()
    assert values["with"] == values["without"], \
        f"vecmath.h changed math_errhandling ({values['without']} -> {values['with']}); it must be transparent"
