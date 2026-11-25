# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The scalar reduction accumulator must land in the array cell it feeds.

``hoist_matmul``'s 1-D x 1-D branch (and the ``np.dot`` call-hoist that shares its
shape) lowered a contraction to ``s = 0.0; for k: s += f(k); T[idx] = s`` with ``s`` a
plain ``double``. ``emit_pluto`` hoists every scalar declaration above ``#pragma scop``,
so pet models ``s`` as a one-element array live across the whole iteration domain:

* the false WAW/WAR between successive ``T[idx]`` instances fragments the schedule, and
* Pluto's ``--parallel`` privatises only its own tile counters, so ``s`` stays SHARED in
  the parallel band -- a data race, i.e. wrong numbers.

``syrk`` was never affected because its reduction carrier is ``C[i, si1]``, an affine
array cell. ``symm`` (``temp2[j] = B[:i, j] @ A[i, :i]``) and ``trmm``
(``B[i, j] += np.dot(A[i+1:, i], B[i+1:, j])``, which additionally crashed Pluto's
``pluto_auto_transform`` assertion) both carried the scalar. The peephole retargets the
reduction onto the destination cell, so all three emit the same shape.

The AST tests pin the rewrite and the guard that declines it: the
loop body must not be able to read back the cell being accumulated into. The native-TU
tests are the numerical consumers -- symm and trmm have no other c/cpp coverage.
"""

import ast
import re
import tempfile

import numpy as np
import pytest

from hpcagent_bench.translators.numpyto_c.emit import emit_c
from hpcagent_bench.translators.numpyto_common.lib_nodes import reduction_misses_target, retarget_scalar_accumulator
from tests.fresh_module import module_at
from tests.translators import native_tu as tu
from tests.translators.bench_yaml import kir_for

DLA = tu.REPO / "hpcagent_bench" / "benchmarks" / "scientific_computing" / "dense_linear_algebra"


def pattern(
    target: str,
    body: str,
    *,
    augmented: bool = False,
    iterable: str = "range(i)",
    init: str = "0.0",
    step: str = "+=",
) -> tuple[ast.stmt, list[ast.stmt]]:
    """``__mm1 = <init>; for __mml1 in <iterable>: __mm1 += <body>`` plus the statement
    that consumes ``__mm1`` -- the exact shape ``hoist_value`` leaves behind.

    ``step="="`` spells the accumulation ``__mm1 = __mm1 + <body>`` instead, which is what the
    reduction expanders emit; both are the same reduction and both must retarget."""
    accum = f"__mm1 += {body}" if step == "+=" else f"__mm1 = __mm1 + {body}"
    prelude = ast.parse(f"__mm1 = {init}\nfor __mml1 in {iterable}:\n    {accum}\n").body
    node = ast.parse(f"{target} {'+=' if augmented else '='} __mm1").body[0]
    return node, prelude


def unparse_(stmts: list[ast.stmt]) -> str:
    return ast.unparse(ast.fix_missing_locations(ast.Module(body=list(stmts), type_ignores=[])))


def emit_fortran(short: str) -> str:
    with tempfile.TemporaryDirectory() as d:
        return tu.emit_source(short, DLA / short / f"{short}_numpy.py", "fortran", d)


def joined(src: str) -> str:
    """Fortran free-form continuations (``... &\\n    &...``) split one statement across lines, so a
    statement only matches after they are rejoined. Which statements wrap depends on identifier
    length, which the DO-variable uniquifier below changes."""
    return re.sub(r"&\s*\n\s*&", "", src)


#: A local scalar ``real`` declaration -- ``real(c_double) :: x_mm1``. An array local carries its
#: extent (``:: temp2(N)``) and a dummy argument its attributes (``, intent(in) ::``), so neither
#: matches. One of these surviving IS the un-retargeted accumulator, whatever it happens to be
#: named, which is what the C leg spells ``__mm1``/``__cb1``.
LOCAL_SCALAR_REAL = re.compile(r"^\s*real\(c_\w+\) :: (\w+)\s*$", re.MULTILINE)

# A  the rewrite                                                               #


@pytest.mark.parametrize(
    ("target", "step"),
    [
        # ``s = s * f(k)`` is a product, not the reduction this fold is sound for.
        ("out[i]", "__mm1 = __mm1 * a[__mml1]"),
        # ``s = f(k) + g(k)`` overwrites rather than accumulates -- folding it would sum every term.
        ("out[i]", "__mm1 = a[__mml1] + b[__mml1]"),
        # A bare-Name destination is already the accumulator; there is nothing to retarget.
        ("s", "__mm1 += a[__mml1] * b[__mml1]"),
        ("t[j]", "__mm1 += a[__mml1] * __mm1"),
    ],
    ids=["non_add_assign_step", "assign_step_not_reading_the_scalar", "scalar_target", "self_referential_body"],
)
def test_a_pattern_that_is_not_a_scalar_accumulation_into_a_cell_is_left_alone(target: str, step: str) -> None:
    node, prelude = pattern(target, "a[__mml1]")
    prelude[1].body = ast.parse(step).body
    assert retarget_scalar_accumulator(node, prelude) is None


def test_aug_assign_retargets_without_a_zero_init() -> None:
    # ``T[idx] += s`` already holds the running value -- zeroing it would drop it.
    out = retarget_scalar_accumulator(
        *pattern("B[i, j]", "A[__mml1 + (i + 1), i] * B[__mml1 + (i + 1), j]", augmented=True)
    )
    assert out is not None
    text = unparse_(out)
    assert "= 0.0" not in text
    assert "B[i, j] += A[__mml1 + (i + 1), i] * B[__mml1 + (i + 1), j]" in text
    assert "__mm1" not in text


# B  the aliasing guard                                                        #


def misses(target: str, body: str, iterable: str = "range(i)") -> bool:
    loop = ast.parse(f"for __mml1 in {iterable}:\n    pass\n").body[0]
    return reduction_misses_target(ast.parse(target).body[0].value, loop, ast.parse(body).body[0].value)


@pytest.mark.parametrize(
    ("target", "body"),
    [
        ("temp2[j]", "B[__mml1, j] * A[i, __mml1]"),
        # trmm: k runs from i + 1, so B[i, j] is never read back mid-accumulation.
        ("B[i, j]", "A[__mml1 + (i + 1), i] * B[__mml1 + (i + 1), j]"),
    ],
    ids=["body_never_touches_the_destination_array", "read_offset_past_the_destination_cell"],
)
def test_a_reduction_that_cannot_read_its_destination_cell_is_safe(target: str, body: str) -> None:
    assert misses(target, body)


@pytest.mark.parametrize(
    ("body", "iterable"),
    [
        # B[__mml1, j] covers B[i, j] when __mml1 == i -- retargeting would feed the
        # partial sum back into itself.
        ("A[__mml1, i] * B[__mml1, j]", "range(i)"),
        ("A[__mml1, i] * B", "range(i)"),
        # The miss proof rests on ``__mml1 >= 0``; without a bare ``range(n)`` there is
        # nothing to rest it on, so the same trmm shape must be declined.
        ("A[__mml1 + (i + 1), i] * B[__mml1 + (i + 1), j]", "range(-M, M)"),
        ("A[__mml1 + (i + 1), i] * B[__mml1 + (i + 1), j]", "ks"),
    ],
    ids=["read_that_may_hit_the_cell", "whole_array_read", "negative_based_iterable", "non_range_iterable"],
)
def test_a_reduction_that_may_read_its_destination_cell_is_refused(body: str, iterable: str) -> None:
    assert not misses("B[i, j]", body, iterable=iterable)


def test_guard_declines_the_retarget_end_to_end() -> None:
    assert retarget_scalar_accumulator(*pattern("B[i, j]", "A[__mml1, i] * B[__mml1, j]", augmented=True)) is None


# C  the emitted C for the three PolyBench kernels                             #


def test_trmm_reduces_into_b_with_no_scalar() -> None:
    c = emit_c(kir_for("trmm", do_lower=True))
    assert "B[(i)*(N) + (j)] += (A[((__r0 + (i + 1)))*(M) + (i)] * B[((__r0 + (i + 1)))*(N) + (j)]);" in c
    assert "double __cb" not in c


def test_fortran_carries_the_same_retarget() -> None:
    # The rewrite lands in the shared AST lowering, so every backend must see it --
    # a scalar left behind here would race under the Fortran OpenMP leg just the same.
    # Fortran rejects two DO variables sharing an identifier in one subroutine scope, so the
    # backend uniquifies every one of them (``j`` -> ``j_l3``, ``x_mml1`` -> ``x_mml1_5``) with a
    # counter that shifts whenever loop structure changes anywhere in the kernel. Match the
    # suffix and tie the cells to ONE capture instead of pinning a spelling: that asserts what
    # the retarget actually promises -- init and accumulation name the SAME cell -- and does not
    # go stale the next time the counter moves.
    symm = joined(emit_fortran("symm"))
    init = re.search(r"temp2\(\((j\w*)\) \+ 1\) = 0\.0_c_double", symm)
    assert init, symm
    j = re.escape(init.group(1))
    cell = rf"temp2\(\({j}\) \+ 1\)"
    assert re.search(
        rf"{cell} = {cell} \+ \(\(B\(\({j}\) \+ 1, \((x_mml1\w*)\) \+ 1\) \* "
        rf"A\(\(\1\) \+ 1, \(i\w*\) \+ 1\)\)\)",
        symm,
    ), symm
    assert not LOCAL_SCALAR_REAL.findall(symm), LOCAL_SCALAR_REAL.findall(symm)
    trmm = joined(emit_fortran("trmm"))
    assert re.search(r"(B\(\(j\w*\) \+ 1, \(i\w*\) \+ 1\)) = \1 \+ ", trmm), trmm
    assert not LOCAL_SCALAR_REAL.findall(trmm), LOCAL_SCALAR_REAL.findall(trmm)


# D  numerics: emit + compile + run against the PolyBench recurrence           #

M, N = 9, 7


def emit_(short: str, cpp: bool) -> str:
    with tempfile.TemporaryDirectory() as d:
        numpy_py = DLA / short / f"{short}_numpy.py"
        return tu.emit_cpp_source(short, numpy_py, d) if cpp else tu.emit_source(short, numpy_py, "c", d)


def inputs(short: str) -> dict[str, object]:
    rng = np.random.default_rng(0)
    if short == "symm":
        return dict(A=rng.random((M, M)), B=rng.random((M, N)), C=rng.random((M, N)), alpha=1.5, beta=0.75)
    return dict(A=rng.random((M, M)), B=rng.random((M, N)), alpha=1.5)


def reference(short: str, args: dict[str, object]) -> np.ndarray:
    """Run the kernel's OWN ``*_numpy.py`` (in-place) and return the output buffer --
    the emitted C must reproduce the shipped reference, not a paraphrase of it."""
    path = DLA / short / f"{short}_numpy.py"
    mod = module_at(path, f"{short}_numpy_ref")
    out = "C" if short == "symm" else "B"
    ref = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in args.items()}
    if short == "symm":
        mod.kernel(ref["alpha"], ref["beta"], ref["C"], ref["A"], ref["B"], M, N)
    else:
        mod.kernel(ref["alpha"], ref["A"], ref["B"], M, N)
    return ref[out]


def driver(short: str, args: dict[str, object], want: np.ndarray) -> str:
    if short == "symm":
        call = f"symm_fp64(A, B, C, {M}, {N}, {args['alpha']}, {args['beta']});"
        bufs = (
            f"static const double A[] = {{{tu.c_double_list(args['A'].ravel())}}};\n"
            f"static const double B[] = {{{tu.c_double_list(args['B'].ravel())}}};\n"
            f"static double C[] = {{{tu.c_double_list(args['C'].ravel())}}};\n"
        )
        out, count = "C", M * N
    else:
        call = f"trmm_fp64(A, B, {M}, {N}, {args['alpha']});"
        bufs = (
            f"static const double A[] = {{{tu.c_double_list(args['A'].ravel())}}};\n"
            f"static double B[] = {{{tu.c_double_list(args['B'].ravel())}}};\n"
        )
        out, count = "B", M * N
    return f"""
#include <stdio.h>
#include <math.h>
{bufs}
static const double want[] = {{{tu.c_double_list(want.ravel())}}};
int main(void) {{
    {call}
    for (int i = 0; i < {count}; ++i) {{
        if (fabs({out}[i] - want[i]) > 1e-10 * (1.0 + fabs(want[i]))) {{
            printf("{short}[%d] got %.17g want %.17g\\n", i, {out}[i], want[i]);
            return 1;
        }}
    }}
    return 0;
}}
"""


@pytest.mark.parametrize(
    ("short", "cpp"),
    [
        pytest.param("symm", False, marks=tu.have_gcc, id="symm_c"),
        pytest.param("symm", True, marks=tu.have_gpp, id="symm_cpp"),
        pytest.param("trmm", False, marks=tu.have_gcc, id="trmm_c"),
        pytest.param("trmm", True, marks=tu.have_gpp, id="trmm_cpp"),
    ],
)
def test_the_retargeted_kernel_matches_numpy_natively(short: str, cpp: bool) -> None:
    args = inputs(short)
    want = reference(short, args)
    run = tu.build_run_c(emit_(short, cpp), driver(short, args, want), cpp=cpp)
    assert run.returncode == 0, run.stdout + run.stderr
