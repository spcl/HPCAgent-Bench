# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Three Fortran emissions that gfortran accepts while meaning something else, or not at all.

Fortran has no ``implicit none`` in the emitted subroutine, so an identifier that does not exist
is not an error -- it is a fresh variable of whatever type its first letter implies. Every case
here is that hazard or its mirror image:

* **A temp sized by a loop iterator.** Each ``For`` target is uniquified (``k`` -> ``k_l0``) so
  nested loops cannot share a DO variable, but the harvested shape TOKENS of a local live in a
  side-table no tree rewrite reaches. Left at ``k`` the ``ALLOCATE`` extent named a variable
  nobody declared, gfortran typed it INTEGER by implicit rule, and the allocation took whatever
  was on the stack -- durbin and stockham_fft died on SIG11, vector_stencil_4d_vc landed on the
  ``f_``-prefixed spelling instead and was rejected as a REAL array index.

* **A helper's scalar result dummy.** It was pinned to ``real(c_double)`` while the caller's
  hoisted temp follows the KERNEL's float precision, so at fp32 the two disagreed and the call
  did not typecheck (nussinov: ``passed REAL(4) to REAL(8)``). Invisible at fp64, where both
  spellings are the same type.

* **LOGICAL where a number belongs, and the reverse.** numpy adds a mask straight into an array
  (``bin_id + (edges <= radius)``) and stores an int into a bool (``uspp = 1 if uspp else 0``);
  Fortran has neither implicit conversion and rejects both outright (azimint_naive, cegterg).

Every case is asserted on the emitted TEXT and then compiled with ``-fimplicit-none``, which is
the gate the emitted source does not carry: it turns "a name that does not exist" from a silent
retype into a diagnostic, so a regression cannot pass by being merely well-formed.
"""
import json
import pathlib
import re
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

import _op_oracle as oo
from _op_oracle import run_op

from numpyto_common import dtypes
from numpyto_common.frontend import parse_kernel
from numpyto_common.ir import apply_precision
from numpyto_common.lowering import lower
from numpyto_fortran.emit import emit_fortran
from numpyto_fortran.intrinsics import renders_natively

#: ``-fimplicit-none`` is the point of this compile, not a style flag -- see the module docstring.
_GFORTRAN_IMPLICIT_NONE = [
    "gfortran", "-fsyntax-only", "-ffree-form", "-ffree-line-length-none", "-std=f2018", "-fimplicit-none"
]

#: An ALLOCATE statement and everything inside its parentheses.
_ALLOCATE = re.compile(r"allocate\(([A-Za-z_][A-Za-z0-9_]*)\((.*?)\)\)$", re.MULTILINE)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _build(src: str, func: str, inputs, outputs, shapes, syms, level=None):
    d = pathlib.Path(tempfile.mkdtemp())
    npy = d / f"{func}_numpy.py"
    npy.write_text(src)
    info = oo._bench_info(func, inputs, outputs, shapes, syms)
    if level is not None:
        info["benchmark"]["level"] = level
    bi = d / "bi.json"
    bi.write_text(json.dumps(info))
    return npy, bi


def fortran(src, func, inputs, outputs, shapes, syms, precision=None, level=None) -> str:
    npy, bi = _build(src, func, inputs, outputs, shapes, syms, level=level)
    kir = lower(parse_kernel(npy, bi), native_call=renders_natively)
    if precision:
        kir = apply_precision(kir, precision)
    return emit_fortran(kir, fn_name=func)


def compiles_with_implicit_none(text: str) -> str:
    """``""`` when gfortran accepts ``text`` under ``-fimplicit-none``, else its diagnostics."""
    if shutil.which("gfortran") is None:  # pragma: no cover -- toolchain gate
        pytest.skip("gfortran not installed")
    d = pathlib.Path(tempfile.mkdtemp())
    f = d / "k.f90"
    f.write_text(text)
    r = subprocess.run(_GFORTRAN_IMPLICIT_NONE + [str(f)], capture_output=True, text=True, cwd=str(d))
    return "" if r.returncode == 0 else (r.stderr or r.stdout)


# --------------------------------------------------------------------------- #
# A. a local sized by a loop iterator                                          #
# --------------------------------------------------------------------------- #

#: ``r[:k]`` inside the ``k`` loop spills into a temp whose ONLY extent is the loop variable --
#: durbin's shape, reduced to the one statement that carries it.
_ITER_SIZED = ("import numpy as np\n"
               "def f(r, y, out):\n"
               " n = r.shape[0]\n"
               " for k in range(1, n):\n"
               "  out[k] = np.dot(np.flip(r[:k]), y[:k])\n")

_ITER_SHAPES = {"r": "(N,)", "y": "(N,)", "out": "(N,)"}
_ITER_SYMS = {"N": 8}


def _iter_sized_fortran() -> str:
    return fortran(_ITER_SIZED, "f", ["r", "y"], ["out"], _ITER_SHAPES, _ITER_SYMS)


def test_an_iterator_sized_temp_allocates_against_the_renamed_do_variable():
    text = _iter_sized_fortran()
    allocs = _ALLOCATE.findall(text)
    assert allocs, f"the iterator-sized temp no longer allocates at all:\n{text}"
    # The rename pass uniquifies the target, so the extent must name THAT spelling, never the
    # Python-level ``k`` -- which is exactly the name no declaration in the subroutine carries.
    assert any(re.fullmatch(r"k_l\d+", extent) for _name, extent in allocs), \
        f"no ALLOCATE is sized by the renamed loop iterator; extents were {[e for _, e in allocs]}:\n{text}"
    assert not any(extent.strip() == "k" for _name, extent in allocs), \
        f"an ALLOCATE is still sized by the pre-rename ``k``, which no declaration binds:\n{text}"


def test_an_iterator_sized_temp_allocates_inside_the_loop_not_at_the_top():
    """The extent only EXISTS inside the loop, so a function-top ALLOCATE cannot be right even if
    it somehow named a declared variable."""
    text = _iter_sized_fortran()
    body = text.split("do k_l", 1)
    assert len(body) == 2, f"the k loop is gone from the emitted body:\n{text}"
    assert "allocate(" not in body[0], \
        f"a temp sized by the loop iterator is allocated BEFORE the loop it is sized by:\n{text}"
    assert "allocate(" in body[1], f"the temp never allocates inside the loop:\n{text}"


def test_every_allocate_extent_names_something_the_subroutine_declares():
    """The general form of the bug: with no ``implicit none``, an undeclared extent is not an
    error, it is a garbage-valued INTEGER. ``-fimplicit-none`` is what says so."""
    text = _iter_sized_fortran()
    diag = compiles_with_implicit_none(text)
    assert not diag, f"the emitted subroutine names an undeclared identifier:\n{diag}\n{text}"


def test_the_iterator_sized_temp_computes_the_reference_numbers():
    rng = np.random.default_rng(0)
    status = run_op(_ITER_SIZED,
                    "f", {
                        "r": rng.standard_normal(8),
                        "y": rng.standard_normal(8)
                    }, {"out": (8, )},
                    _ITER_SYMS,
                    shapes=_ITER_SHAPES,
                    backends=("c", "fortran"))
    assert status["fortran"] == "ok", status
    assert status["c"] == "ok", status


# --------------------------------------------------------------------------- #
# B. a kept helper's scalar result dummy follows the kernel's precision        #
# --------------------------------------------------------------------------- #

#: nussinov's shape: a kept helper returning a non-literal scalar, so the result dummy takes the
#: float path rather than the all-integer-literals one.
_HELPER_RET = ("import numpy as np\n"
               "def pick(a, b, bonus):\n"
               " if a > b:\n"
               "  return bonus\n"
               " else:\n"
               "  return 0.0\n"
               "def f(x, y, bonus, out):\n"
               " for i in range(x.shape[0]):\n"
               "  out[i] = pick(x[i], y[i], bonus)\n")

_HELPER_SHAPES = {"x": "(N,)", "y": "(N,)", "out": "(N,)"}
_HELPER_SYMS = {"N": 6}

#: name -> the ``real(<kind>)`` the result dummy must carry at that precision.
_RESULT_KIND = {None: "real(c_double)", "float32": "real(c_float)"}


def _helper_ret_fortran(precision) -> str:
    return fortran(_HELPER_RET,
                   "f", ["x", "y", "bonus"], ["out"],
                   _HELPER_SHAPES,
                   _HELPER_SYMS,
                   precision=precision,
                   level=3)


@pytest.mark.parametrize("precision", [None, "float32"])
def test_the_helper_result_dummy_carries_the_kernels_float_kind(precision):
    text = _helper_ret_fortran(precision)
    if "intent(out) :: hret_" not in text:
        pytest.skip("the fixture helper is no longer kept as a contained subroutine")
    want = _RESULT_KIND[precision]
    assert f"{want}, intent(out) :: hret_" in text, \
        f"the result dummy does not follow the kernel's float precision ({precision}):\n{text}"
    # The caller's hoisted temp is what it has to agree WITH, so the other spelling must be absent.
    other = _RESULT_KIND[None if precision else "float32"]
    assert f"{other}, intent(out) :: hret_" not in text, f"both float kinds are declared at once:\n{text}"


@pytest.mark.parametrize("precision", [None, "float32"])
def test_the_helper_call_typechecks_at_both_precisions(precision):
    """The wrong kind is not a warning: gfortran refuses the CALL, so the kernel does not build."""
    diag = compiles_with_implicit_none(_helper_ret_fortran(precision))
    assert not diag, f"precision={precision}:\n{diag}"


def test_the_kernel_float_precision_is_where_the_kind_comes_from():
    """Pins the derivation rather than the literal: the dummy takes the accumulator dtype of the
    kernel's precision, the same rule ``_collect_implicit_locals`` types the caller's temp by."""
    assert dtypes.fortran_kind(dtypes.accumulator_dtype("float32")) == "real(c_float)"
    assert dtypes.fortran_kind(dtypes.accumulator_dtype("float64")) == "real(c_double)"


# --------------------------------------------------------------------------- #
# C. LOGICAL <-> numeric, in both directions                                   #
# --------------------------------------------------------------------------- #

#: ``m[i] = <int>`` stores a number into a bool; ``a[i] + m[i]`` reads a bool as a number.
_LOGICAL_BOTH_WAYS = ("import numpy as np\n"
                      "def f(a, out):\n"
                      " m = np.zeros(a.shape[0], dtype=np.bool_)\n"
                      " for i in range(a.shape[0]):\n"
                      "  m[i] = 1 if a[i] > 0.0 else 0\n"
                      " for i in range(a.shape[0]):\n"
                      "  out[i] = a[i] + m[i]\n")

_LOGICAL_SHAPES = {"a": "(N,)", "out": "(N,)"}
_LOGICAL_SYMS = {"N": 8}


def _logical_fortran() -> str:
    return fortran(_LOGICAL_BOTH_WAYS, "f", ["a"], ["out"], _LOGICAL_SHAPES, _LOGICAL_SYMS)


def test_a_number_stored_into_a_logical_converts_by_truthiness():
    text = _logical_fortran()
    assert "logical(c_bool) :: m(" in text, f"the mask local is no longer LOGICAL:\n{text}"
    assert re.search(r"m\(\(i_l\d+\) \+ 1\) = \(x_ifexp\d+\) /= 0", text), \
        f"the integer store into the LOGICAL mask is not converted:\n{text}"


def test_a_logical_read_in_arithmetic_promotes_to_zero_or_one():
    """numpy's bool -> 0/1 promotion, spelled as the MERGE Fortran needs; ``a + m`` is a type
    error without it, so the alternative is not a wrong number but no kernel at all."""
    text = _logical_fortran()
    assert re.search(r"\+ merge\(1_c_int64_t, 0_c_int64_t, m\(", text), \
        f"the LOGICAL operand is not promoted inside the arithmetic:\n{text}"


def test_the_logical_conversions_compile_and_compute_the_reference_numbers():
    diag = compiles_with_implicit_none(_logical_fortran())
    assert not diag, diag
    rng = np.random.default_rng(1)
    a = rng.standard_normal(8)
    status = run_op(_LOGICAL_BOTH_WAYS,
                    "f", {"a": a}, {"out": (8, )},
                    _LOGICAL_SYMS,
                    shapes=_LOGICAL_SHAPES,
                    backends=("c", "fortran"))
    assert status["fortran"] == "ok", status
    assert status["c"] == "ok", status


def test_a_logical_operand_of_a_comparison_is_left_alone():
    """The promotion is for ARITHMETIC only -- a mask feeding ``.and.`` / a condition must stay
    LOGICAL, or the same fix that lets ``a + m`` compile stops ``if (m(i))`` from compiling."""
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           " m = np.zeros(a.shape[0], dtype=np.bool_)\n"
           " for i in range(a.shape[0]):\n"
           "  m[i] = a[i] > 0.0\n"
           " for i in range(a.shape[0]):\n"
           "  if m[i]:\n"
           "   out[i] = a[i]\n")
    text = fortran(src, "f", ["a"], ["out"], _LOGICAL_SHAPES, _LOGICAL_SYMS)
    assert "merge(1_c_int64_t" not in text, f"a condition was wrapped in the numeric promotion:\n{text}"
    assert not compiles_with_implicit_none(text)


#: A helper that is KEPT (the early return blocks inlining) and builds a boolean mask of its own,
#: so the mask is declared by the helper's own declaration pass and used by the helper's own body
#: emitter -- two tables that have to say the same thing.
_HELPER_MASK = ("import numpy as np\n"
                "def masked(v, lo):\n"
                " if lo < 0.0:\n"
                "  return -v\n"
                " nz = v > lo\n"
                " return np.where(nz, v, 0.0)\n"
                "def f(x, thr, out):\n"
                " out[:] = masked(x, thr)\n")


def test_a_kept_helpers_mask_is_logical_to_its_own_body_too():
    """The logical-ness oracle is per-EMITTER, and a kept helper gets its own.

    Left unfed, the helper's body emitter saw an untyped name where its own declaration pass had
    written ``logical(c_bool)``, so the comparison stored into the mask was converted as if the
    target were a number (``Cannot convert INTEGER(8) to LOGICAL(1)``). Same failure the caller
    side had, one scope in.
    """
    text = fortran(_HELPER_MASK, "f", ["x", "thr"], ["out"], {"x": "(N,)", "out": "(N,)"}, _LOGICAL_SYMS, level=3)
    if "logical(c_bool) :: nz" not in text:
        pytest.skip("the fixture helper is no longer kept with a mask local of its own")
    assert "merge(1_c_int64_t" not in text, f"the helper stores its mask through a numeric promotion:\n{text}"
    assert not compiles_with_implicit_none(text)


# --------------------------------------------------------------------------- #
# D. sibling loops that shared a Python name must not share a DO variable      #
# --------------------------------------------------------------------------- #

#: ``np.linalg.solve`` lowers to a Gauss-Jordan nest that reuses ONE Python loop name (``__sol_c``)
#: across six sibling loops -- legal in Python and in C, where each ``for`` scopes its own
#: declaration, and the reason the Fortran backend uniquifies every DO variable in the first place.
_SOLVE = ("import numpy as np\n"
          "def f(A, B, out):\n"
          " out[:] = np.linalg.solve(A, B)\n")

_SOLVE_SHAPES = {"A": "(N, N)", "B": "(N, M)", "out": "(N, M)"}
_SOLVE_SYMS = {"N": 5, "M": 3}

_DO_HEADER = re.compile(r"^\s*do ([A-Za-z_][A-Za-z0-9_]*) = ", re.MULTILINE)


def _solve_fortran() -> str:
    return fortran(_SOLVE, "f", ["A", "B"], ["out"], _SOLVE_SHAPES, _SOLVE_SYMS)


def test_every_do_variable_is_read_by_the_body_it_controls():
    """A DO variable that appears ONLY in its own header is the signature of this bug.

    The rename pass gives each of the six ``__sol_c`` loops its own DO variable, but the tree it
    rewrites hands out ALIASED nodes, so renaming one occurrence renamed every other occurrence with
    it. The later loops' bodies then kept indexing the FIRST loop's variable: the pivot swap, the
    normalisation and the elimination all ran on one column, and contour_integral aborted rather
    than returning a wrong answer.
    """
    text = _solve_fortran()
    heads = _DO_HEADER.findall(text)
    assert len(heads) >= 6, f"the solve lowering no longer emits its loop nest:\n{text}"
    dead = [v for v in heads if len(re.findall(rf"\b{re.escape(v)}\b", text)) < 3]
    assert not dead, (f"DO variable(s) {dead} are declared and opened but never read -- their loop "
                      f"bodies are indexing a sibling's variable:\n{text}")


def test_no_two_sibling_loops_share_a_do_variable():
    """Fortran has one scope per subroutine, so the uniquification is not cosmetic: two DO
    statements on one variable in the same nest is a different program, not a style question."""
    heads = _DO_HEADER.findall(_solve_fortran())
    assert len(heads) == len(set(heads)), f"a DO variable is opened twice: {sorted(heads)}"


def test_the_solve_lowering_computes_the_reference_numbers():
    rng = np.random.default_rng(7)
    a = rng.standard_normal((5, 5)) + 5.0 * np.eye(5)  # diagonally dominant: no pivot degeneracy
    status = run_op(_SOLVE,
                    "f", {
                        "A": a,
                        "B": rng.standard_normal((5, 3))
                    }, {"out": (5, 3)},
                    _SOLVE_SYMS,
                    shapes=_SOLVE_SHAPES,
                    rtol=1e-9,
                    atol=1e-9,
                    backends=("c", "fortran"))
    assert status["fortran"] == "ok", status
    assert status["c"] == "ok", status
