# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Parallel-loop analysis + OpenMP parallel-scope emission.

Two layers: pure-AST predicate unit tests (loop classification / reduction /
scatter detection) and end-to-end ``emit_c_omp`` text assertions (the pragma a
parallel loop nest gets).
"""

import ast

import pytest

from hpcagent_bench.translators.numpyto_common.ir import KernelIR
from hpcagent_bench.translators.numpyto_common.parallelism import (
    UnsupportedParallelError,
    any_parallelizable_loop,
    collapsible_depth,
    has_indirect_scatter,
    is_timestep_loop,
    loop_is_parallel_safe,
    loop_reduction,
    subscript_idx_safe,
)
from tests.translators.op_oracle import parse_source


def stmt_(src: str) -> ast.Assign | ast.For:
    return ast.parse(src).body[0]


def sub_(src: str) -> ast.Subscript:
    return ast.parse(src).body[0].value


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        ("for t in range(TSTEPS):\n    step(t)\n", True),
        ("for i in range(N):\n    a[i] = 0\n", False),
        ("for k in range(NITER):\n    pass\n", True),
    ],
    ids=["timestep_for", "plain_range", "niter"],
)
def test_loops_over_a_timestep_bound_are_timestep_loops(src: str, expected: bool) -> None:
    assert bool(is_timestep_loop(stmt_(src))) is expected


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        ("a[i]", True),
        ("a[i - 1]", False),
        ("a[2 * i]", False),
        ("a[p[i]]", False),
        ("a[0]", False),
        ("a[i, j]", True),  # bare in one axis
    ],
    ids=["bare", "shifted", "scaled", "indirect", "absent", "multidim_bare_in_one_axis"],
)
def test_only_a_bare_loop_index_subscript_is_safe(src: str, expected: bool) -> None:
    assert bool(subscript_idx_safe(sub_(src), "i")) is expected


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        ("for i in range(N):\n    c[i] = a[i] + b[i]\n", True),
        # b written idx-safe; a is READ-ONLY (never written), so its shifted read does not race.
        ("for i in range(N):\n    b[i] = a[i - 1] + a[i + 1]\n", True),
        ("for i in range(N):\n    s = s + a[i]\n", False),
        # a is written AND read shifted -> loop-carried dependence.
        ("for i in range(N):\n    a[i] = a[i - 1] + a[i + 1]\n", False),
        ("for i in range(N):\n    a[p[i]] = a[p[i]] + b[i]\n", False),
    ],
    ids=["elementwise_map", "jacobi_read_only_neighbours", "scalar_reduction", "inplace_stencil", "scatter"],
)
def test_loop_is_parallel_safe_only_without_a_carried_dependence(src: str, expected: bool) -> None:
    assert bool(loop_is_parallel_safe(stmt_(src))) is expected


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        ("for i in range(N):\n    s = s + a[i]\n", ("+", "s")),
        ("for i in range(N):\n    s += a[i]\n", ("+", "s")),
        ("for i in range(N):\n    p = p * a[i]\n", ("*", "p")),
        ("for i in range(N):\n    m = np.maximum(m, a[i])\n", ("max", "m")),
        ("for i in range(N):\n    m = min(m, a[i])\n", ("min", "m")),
        ("for i in range(N):\n    c[i] = a[i] + b[i]\n", None),
        ("for i in range(N):\n    s = s + a[i]\n    t = t + b[i]\n", None),
        # ``x`` is recomputed each iteration (not self-referential) -> not an accumulator.
        ("for i in range(N):\n    x = a[i] * 2.0\n    c[i] = x\n", None),
    ],
    ids=[
        "sum_assign_form",
        "sum_augassign_form",
        "product",
        "max_via_np_maximum",
        "min_via_builtin",
        "pure_map_has_none",
        "two_accumulators_are_not_one_reduction",
        "private_temp_is_not_a_reduction",
    ],
)
def test_loop_reduction_names_its_single_accumulator(src: str, expected: tuple[str, str] | None) -> None:
    assert loop_reduction(stmt_(src)) == expected


@pytest.mark.parametrize(
    ("src", "depth"),
    [
        # 3 perfectly-nested, rectangular, independent levels -> the whole nest collapses.
        (
            "for i in range(N):\n"
            "    for j in range(M):\n"
            "        for k in range(K):\n"
            "            out[i, j, k] = a[i, j, k] * 2.0\n",
            3,
        ),
        # gemm-shaped: i, j are independent (subscript accumulator, not a bare-Name reduction);
        # the innermost k is a genuine reduction and must NOT be folded into the collapse.
        (
            "for i in range(N):\n"
            "    for j in range(M):\n"
            "        out[i, j] = 0.0\n"
            "        for k in range(K):\n"
            "            out[i, j] += a[i, k] * b[k, j]\n",
            2,
        ),
        # A triangular inner bound (range(i)) makes the iteration space non-rectangular --
        # collapse must not fold it in, however independent the writes look.
        ("for i in range(N):\n    for j in range(i):\n        out[i, j] = a[i, j] * 2.0\n", 1),
        # A statement alongside the nested loop breaks OpenMP's canonical perfectly-nested form.
        ("for i in range(N):\n    total = 0.0\n    for j in range(N):\n        out[i, j] = a[i, j] * 2.0\n", 1),
        # out[i] does not depend on j -- every j iteration of a fixed i overwrites the same cell,
        # so the j level alone is unsafe to reorder even though the outer i level is fine.
        ("for i in range(N):\n    for j in range(M):\n        out[i] = a[i, j]\n", 1),
        ("for i in range(N):\n    for t in range(TSTEPS):\n        out[i, t] = a[i, t]\n", 1),
    ],
    ids=[
        "perfectly_nested_rectangular_map",
        "stops_before_inner_reduction",
        "stops_at_non_rectangular_bound",
        "stops_at_sibling_statement",
        "stops_when_inner_index_absent_from_write",
        "stops_at_timestep_bound",
    ],
)
def test_collapse_depth_covers_only_independent_rectangular_levels(src: str, depth: int) -> None:
    assert collapsible_depth(stmt_(src)) == depth


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        ("for i in range(N):\n    out[idx[i]] += x[i]\n", True),
        ("for i in range(N):\n    out[i] += x[i]\n", False),
    ],
    ids=["scatter_write", "affine_write"],
)
def test_only_a_data_indexed_write_is_an_indirect_scatter(src: str, expected: bool) -> None:
    assert bool(has_indirect_scatter(ast.parse(src))) is expected


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        ("for i in range(N):\n    c[i] = a[i]\n", True),
        ("for i in range(N):\n    s = s + a[i]\n", True),
        ("for i in range(N):\n    out[idx[i]] += x[i]\n", False),
    ],
    ids=["map", "reduction", "scatter_only"],
)
def test_a_nest_is_parallelizable_unless_it_only_scatters(src: str, expected: bool) -> None:
    assert bool(any_parallelizable_loop(ast.parse(src))) is expected


# emit_c_omp (end to end: parse -> lower -> emit)
def kir_(
    src: str,
    args: list[str],
    shapes: dict[str, str],
    dtypes: dict[str, str] | None = None,
    params: dict[str, int] | None = None,
) -> KernelIR:
    from hpcagent_bench.translators.numpyto_common.lowering import lower

    return lower(parse_source(src, "f", args[:-1], args[-1:], shapes, params or {"N": 16}, dtypes))


def test_emit_c_omp_elementwise_parallel_for() -> None:
    from hpcagent_bench.translators.numpyto_c.emit import emit_c, emit_c_omp

    kir = kir_(
        "def f(x, y, out):\n    for i in range(N):\n        out[i] = y[i] + 2.0 * x[i]\n",
        ["x", "y", "out"],
        {"x": "(N,)", "y": "(N,)", "out": "(N,)"},
    )
    c = emit_c_omp(kir, fn_name="f")
    assert "#pragma omp parallel for\n" in c
    assert "reduction(" not in c
    assert "#pragma omp" not in emit_c(kir, fn_name="f")  # sequential emit is unchanged


def test_emit_c_omp_sum_reduction_clause() -> None:
    from hpcagent_bench.translators.numpyto_c.emit import emit_c_omp

    kir = kir_(
        "def f(x, out):\n    s = 0.0\n    for i in range(N):\n        s = s + x[i]\n    out[0] = s\n",
        ["x", "out"],
        {"x": "(N,)", "out": "(N,)"},
    )
    assert "#pragma omp parallel for reduction(+:s)" in emit_c_omp(kir, fn_name="f")


def test_emit_c_omp_nested_tags_outer_only() -> None:
    from hpcagent_bench.translators.numpyto_c.emit import emit_c_omp

    kir = kir_(
        "def f(a, out):\n    for i in range(N):\n        for j in range(N):\n            out[i, j] = a[i, j] * 2.0\n",
        ["a", "out"],
        {"a": "(N, N)", "out": "(N, N)"},
    )
    c = emit_c_omp(kir, fn_name="f")
    assert c.count("#pragma omp parallel for") == 1  # only the outermost loop, no nested regions


@pytest.mark.parametrize("backend", ["c", "fortran"])
def test_emit_c_omp_scatter_is_refused(backend: str) -> None:
    from hpcagent_bench.translators.numpyto_c.emit import emit_c_omp
    from hpcagent_bench.translators.numpyto_fortran.emit import emit_fortran_omp

    emit = {"c": emit_c_omp, "fortran": emit_fortran_omp}[backend]
    kir = kir_(
        "def f(idx, x, out):\n    for i in range(N):\n        out[idx[i]] = out[idx[i]] + x[i]\n",
        ["idx", "x", "out"],
        {"idx": "(N,)", "x": "(N,)", "out": "(N,)"},
        {"idx": "int64"},
    )
    with pytest.raises(UnsupportedParallelError):
        emit(kir, fn_name="f")


def test_emit_fortran_omp_elementwise_parallel_do() -> None:
    from hpcagent_bench.translators.numpyto_fortran.emit import emit_fortran, emit_fortran_omp

    kir = kir_(
        "def f(x, y, out):\n    for i in range(N):\n        out[i] = y[i] + 2.0 * x[i]\n",
        ["x", "y", "out"],
        {"x": "(N,)", "y": "(N,)", "out": "(N,)"},
    )
    f = emit_fortran_omp(kir, fn_name="f")
    assert "!$omp parallel do" in f
    assert "reduction(" not in f
    assert "!$omp" not in emit_fortran(kir, fn_name="f")  # sequential emit is unchanged


def test_emit_fortran_omp_sum_reduction_clause() -> None:
    from hpcagent_bench.translators.numpyto_fortran.emit import emit_fortran_omp

    kir = kir_(
        "def f(x, out):\n    s = 0.0\n    for i in range(N):\n        s = s + x[i]\n    out[0] = s\n",
        ["x", "out"],
        {"x": "(N,)", "out": "(N,)"},
    )
    assert "!$omp parallel do reduction(+:s)" in emit_fortran_omp(kir, fn_name="f")


def test_reduction_rejects_self_referential_recurrence() -> None:
    # Genuine reductions: the non-accumulator operand is loop data, not the accumulator.
    assert loop_reduction(stmt_("for i in range(N):\n s = s + a[i]\n")) == ("+", "s")
    assert loop_reduction(stmt_("for i in range(N):\n p = p * a[i]\n")) == ("*", "p")
    assert loop_reduction(stmt_("for i in range(N):\n m = max(m, a[i])\n")) == ("max", "m")
    # A self-referential recurrence reads the accumulator on BOTH sides -- reduction(+:ema)
    # would start each thread at the identity and lose the compounding, so it must NOT match.
    assert loop_reduction(stmt_("for i in range(N):\n ema = ema + ema * alpha\n")) is None
    assert loop_reduction(stmt_("for i in range(N):\n m = max(m, m * 2.0)\n")) is None
