# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compile-time tuples must be gone before emit -- C and Fortran have no tuple to lower one to.

Each test asserts on the desugared source, since the failure this pass exists to prevent is an
``ast.Tuple`` surviving into value position, where the emitter refuses it.
"""

import ast
import textwrap

import numpy as np
import pytest

from hpcagent_bench.translators.numpyto_common.numpy_desugar import fold_list_accumulators, rewrite_curve_fit
from hpcagent_bench.translators.numpyto_common.tuple_desugar import desugar_tuples
from tests.translators.source_module import run_source

SCALARS = frozenset({"p", "s", "k"})
ARRAYS = frozenset({"x", "out"})

RANKS = {"x": 4, "out": 4}


def desugared(
    body: str,
    int_scalars: frozenset[str] = SCALARS,
    float_scalars: frozenset[str] = frozenset(),
    arrays: frozenset[str] = ARRAYS,
    ranks: dict[str, int] | None = None,
) -> str:
    fn = ast.parse(textwrap.dedent(body)).body[0]
    desugar_tuples(
        fn, int_scalars=int_scalars, float_scalars=float_scalars, arrays=arrays, ranks=RANKS if ranks is None else ranks
    )
    return ast.unparse(fn)


def test_a_tuple_local_is_scalarized_away() -> None:
    got = desugared("""
        def k(x, p, out):
            t = (p, p + 1)
            out[0] = t[0] + t[1]
        """)
    assert "out[0] = p + (p + 1)" in got
    assert "t = " not in got  # the binding itself is gone, not just its uses


def test_a_self_referential_binding_does_not_re_substitute() -> None:
    """``p = (p,)`` binds p to a tuple whose element IS p; naive substitution yields ``((p,),)``."""
    got = desugared("""
        def k(x, p, out):
            p = (p,)
            out[0] = p[0]
        """)
    assert got.endswith("out[0] = p")


def test_tuple_concatenation_folds_into_one_shape() -> None:
    got = desugared("""
        def k(x, p, out):
            shape = (x.shape[0],) + (p, 2)
            out[:] = np.zeros(shape)
        """)
    assert "np.zeros((x.shape[0], p, 2))" in got


@pytest.mark.parametrize(("rank", "want"), [(4, "(1, x.shape[1], 1, 1)"), (2, "(1, x.shape[1])")])
def test_a_broadcast_shape_padded_to_an_array_rank_folds(rank: int, want: str) -> None:
    """``(1,) * (x.ndim - 2)``. The rank-2 case repeats ZERO times: the empty tuple is falsy but
    correct, so the fold must test for None rather than truthiness."""
    got = desugared(
        """
        def k(x, out):
            shape = (1, x.shape[1]) + (1,) * (x.ndim - 2)
            out[:] = x.reshape(shape)
        """,
        int_scalars=frozenset(),
        arrays=frozenset({"x", "out"}),
        ranks={"x": rank, "out": rank},
    )
    assert f"x.reshape({want})" in got


@pytest.mark.parametrize("spelling", ["int", "np.integer"])
def test_a_one_sided_isinstance_on_a_declared_knob_stays_undecided(spelling: str) -> None:
    """A preset symbol arrives as a Python int, an init.scalars entry as a numpy scalar, and
    ``isinstance(np.int64(3), int)`` is FALSE. Folding either spelling would make the emitted kernel
    take a branch the numpy oracle does not."""
    got = desugared(f"""
        def k(x, p, out):
            if isinstance(p, {spelling}):
                p = (p, p)
            out[0] = p[1]
        """)
    assert f"isinstance(p, {spelling})" in got


def test_an_isinstance_guard_on_a_tuple_takes_the_false_branch() -> None:
    got = desugared("""
        def k(x, p, out):
            p = (p, p)
            if isinstance(p, (int, np.integer)):
                p = (p, p, p)
            out[0] = len(p)
        """)
    assert got.endswith("out[0] = 2")


def test_literal_index_arithmetic_folds_so_a_tuple_element_can_be_selected() -> None:
    """A comprehension unroll leaves ``i + 2`` as ``0 + 2``; without folding it the tuple subscript
    has no literal index and the whole tuple survives into value position."""
    got = desugared("""
        def k(x, p, out):
            t = (1, 2, p)
            out[0] = t[0 + 2]
        """)
    assert got.endswith("out[0] = p")


def test_a_string_comparison_of_two_literals_folds_its_branch() -> None:
    """The ports carry ``-np.inf if 'mean' == 'max' else 0.0`` from a templated generator."""
    got = desugared("""
        def k(x, p, out):
            fill = -np.inf if 'mean' == 'max' else 0.0
            out[0] = fill
        """)
    assert "'mean'" not in got
    assert "0.0" in got


def test_a_bare_shape_expands_wherever_it_stands() -> None:
    """``.reshape(x.shape)`` -- the argument is not a tuple context, so without forcing the
    expansion the reshape reaches lowering with no compile-time rank, and group-norm keeps the
    PREVIOUS statement's rank-5 extent for the target and indexes past the end of it."""
    got = desugared("""
        def k(x, p, out):
            out[:] = x.reshape(x.shape)
        """)
    assert "x.reshape((x.shape[0], x.shape[1], x.shape[2], x.shape[3]))" in got


def test_a_shape_of_unknown_rank_is_left_alone() -> None:
    """Only the LENGTH is compile-time here. With no rank there is no length, and inventing one
    would emit a reshape to a shape the kernel does not have."""
    got = desugared("""
        def k(x, p, out):
            out[:] = q.reshape(q.shape)
        """)
    assert "q.reshape(q.shape)" in got


def folded_lists(body: str) -> str:
    fn = ast.parse(textwrap.dedent(body)).body[0]
    fold_list_accumulators(fn)
    return ast.unparse(fn)


def test_a_list_grown_by_append_becomes_an_array_and_a_fill_loop() -> None:
    # raman_fitting's initial centre guesses. Left as a list, lowering reads ``len(centre)`` as the
    # ARRAY extent it later gives the local: the guard becomes ``npeaks < npeaks`` (never taken) and
    # the truncation a self-copy, so the kernel emits and computes the wrong guesses.
    out = folded_lists("""
        def f(npeaks, out):
            centre = [1580.0, 2670.0]
            while len(centre) < npeaks:
                centre.append(1200.0 + 200.0 * len(centre))
            centre = centre[:npeaks]
            out[:] = centre
        """)
    assert "centre = np.zeros(npeaks, dtype=np.float64)" in out
    assert "for __la1 in range(npeaks):" in out
    # Element i is the i-th literal inside the display and the growth step beyond it, with the
    # loop index standing in for the ``len`` the step counted with.
    assert "centre[__la1] = 1580.0 if __la1 == 0 else 2670.0 if __la1 == 1 else 1200.0 + 200.0 * __la1" in out
    assert "while" not in out
    assert "append" not in out
    # An all-integer display folds to an integer array; the cut pins the length to ``n``, else the build is
    # ``max(2, n)`` long and refused.
    ints = folded_lists("""
        def f(n, out):
            idx = [0, 1]
            while len(idx) < n:
                idx.append(len(idx) * 2)
            idx = idx[:n]
            out[:] = idx
        """)
    assert "np.zeros(n, dtype=np.int64)" in ints


KEPT_CENTRE = ("centre = [1.0]", "while len(centre) < n:")


@pytest.mark.parametrize(
    ("source", "kept"),
    [
        # Grown to ``n`` and cut to ``m`` is not this idiom: the closed form would be the wrong
        # length. Nothing is guessed -- the list stays and the refusal that owns it still fires.
        (
            """
        def f(n, m, out):
            centre = [1.0]
            while len(centre) < n:
                centre.append(2.0)
            centre = centre[:m]
            out[:] = centre
        """,
            KEPT_CENTRE,
        ),
        # A second append outside the growth loop is a mutation the closed form does not account for.
        (
            """
        def f(n, out):
            centre = [1.0]
            while len(centre) < n:
                centre.append(2.0)
            centre.append(3.0)
            out[:] = centre
        """,
            KEPT_CENTRE,
        ),
        # The build itself is complete and cut; the append in a later branch is what the guard refuses.
        (
            """
        def f(n, flag, out):
            centre = [1.0]
            while len(centre) < n:
                centre.append(2.0)
            centre = centre[:n]
            if flag:
                centre.append(3.0)
            out[:] = centre
        """,
            KEPT_CENTRE,
        ),
        # An extended list mutated in a later branch.
        (
            """
        def f(flag, lo):
            guess = [lo]
            guess += [2.0]
            if flag:
                guess.append(3.0)
            return guess
        """,
            ("guess = [lo]", "guess += [2.0]"),
        ),
    ],
    ids=[
        "cut_to_a_different_length",
        "mutated_outside_the_growth_loop",
        "resized_after_its_cut",
        "extended_then_mutated",
    ],
)
def test_a_list_the_closed_form_cannot_account_for_is_left_alone(source: str, kept: tuple[str, ...]) -> None:
    out = folded_lists(source)
    for text in kept:
        assert text in out


def as_written_and_folded(source: str, *args: object) -> tuple[np.ndarray, np.ndarray]:
    """``f(*args)`` from ``source`` as Python runs it, and from its folded form."""
    written: dict[str, object] = {"np": np}
    run_source(textwrap.dedent(source), written)
    folded: dict[str, object] = {"np": np}
    run_source(folded_lists(source), folded)
    return np.asarray(written["f"](*args)), np.asarray(folded["f"](*args))


@pytest.mark.parametrize(
    ("source", "args", "present", "absent"),
    [
        # No truncation: Python keeps max(len(seed), n) elements, so a length-``n`` array drops the tail.
        (
            """
        def f(n):
            centre = [1580.0, 2670.0, 3000.0]
            while len(centre) < n:
                centre.append(1200.0 + 200.0 * len(centre))
            return centre
        """,
            [(1,), (5,)],
            (),
            (),
        ),
        # raman_fitting's spelling: the cut binds a NEW name and the grown list is read nowhere else.
        (
            """
        def f(n):
            centre1 = [1580.0, 2670.0]
            while len(centre1) < n:
                centre1.append(1200.0 + 200.0 * len(centre1))
            centre2 = centre1[:n]
            return centre2
        """,
            [(1,), (5,)],
            (),
            ("append",),
        ),
        # ``centre[-1]`` is the last APPENDED element in Python; in a preallocated array it is the last slot.
        (
            """
        def f(n):
            centre = [1.0]
            while len(centre) < n:
                centre.append(centre[-1] * 2.0)
            centre = centre[:n]
            return centre
        """,
            [(2,), (5,)],
            ("centre.append(centre[-1] * 2.0)",),
            (),
        ),
        # Python keeps 1.5 as 1.5; an int64 buffer would truncate every element the growth rule writes.
        (
            """
        def f(n):
            idx = [0, 1]
            while len(idx) < n:
                idx.append(0.5 * len(idx))
            idx = idx[:n]
            return idx
        """,
            [(2,), (5,)],
            ("np.zeros(n, dtype=np.float64)",),
            (),
        ),
        (
            """
        def f(lo, span):
            out = np.zeros((2, 4))
            for t in range(2):
                guess = [lo + t]
                guess += [span, 10.0]
                guess = guess + [lo * span]
                out[t, :] = guess
            return out
        """,
            [(1.5, 4.0)],
            ("guess = np.zeros(4, dtype=np.float64)", "guess[0] = lo + t", "guess[2] = 10.0", "guess[3] = lo * span"),
            ("+=", "guess + ["),
        ),
        (
            """
        def f(n, width):
            params = [0.5]
            for j in range(n):
                params += [1.0 * j, width]
            params.append(-1.0)
            return params
        """,
            [(0, 7.0), (3, 7.0)],
            (
                "params = np.zeros(1 + 2 * n + 1, dtype=np.float64)",
                "params[1 + 2 * j + 1] = width",
                "params[1 + 2 * n] = -1.0",
            ),
            ("+=", "append"),
        ),
        # Three elements are built before the cut; at n == 2 the third store must not land.
        (
            """
        def f(n, a):
            v = [a]
            for j in range(2):
                v += [a * j]
            while len(v) < n:
                v.append(0.5 * len(v))
            v = v[:n]
            return v
        """,
            [(2, 3.0), (5, 3.0)],
            (),
            ("while",),
        ),
    ],
    ids=[
        "seed_longer_than_the_growth_bound",
        "cut_into_a_fresh_name",
        "growth_step_reading_the_list",
        "integer_seed_float_rule",
        "straight_extends_in_a_loop_body",
        "fixed_stride_loop",
        "cut_shorter_than_the_built_prefix",
    ],
)
def test_straight_extends_inside_a_loop_body_fold_to_indexed_stores(
    source: str, args: list[tuple[object, ...]], present: tuple[str, ...], absent: tuple[str, ...]
) -> None:
    """Each fold keeps the values and the length Python gives: the folded text carries its stores and none of the
    list operations, or (for a growth the closed form cannot account for) stays as written."""
    folded = folded_lists(source)
    for text in present:
        assert text in folded
    for text in absent:
        assert text not in folded
    for call in args:
        expected, got = as_written_and_folded(source, *call)
        assert got.shape == expected.shape
        assert np.array_equal(got, expected)


CURVE_FIT_MODULE = """
    import numpy as np
    from scipy.optimize import curve_fit

    def line(grid, *p):
        return p[0] * grid + p[-1]

    def fit(x, y, lo, out):
        {prelude}
        popt, _ = curve_fit(line, x, y, p0=guess)
        out[0] = popt[-1]
    """


def curve_fit_rewritten(prelude: str) -> str:
    tree = ast.parse(textwrap.dedent(CURVE_FIT_MODULE).replace("{prelude}", prelude))
    kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "fit")
    rewrite_curve_fit(tree, kernel)
    return ast.unparse(kernel)


@pytest.mark.parametrize(
    ("prelude", "lists"),
    [
        ("guess = [lo]\n    guess += [0.0]", ()),
        ("scales = [1.0, 2.0]\n    guess = [lo, 0.0]", ("scales = [1.0, 2.0]",)),  # other displays stay lists
    ],
    ids=["grown", "bare-display"],
)
def test_a_grown_p0_becomes_an_array_before_the_fit_indexes_it(prelude: str, lists: tuple[str, ...]) -> None:
    out = curve_fit_rewritten(prelude)
    assert "guess = np.zeros(2, dtype=np.float64)" in out
    assert "guess[0] = lo" in out
    assert "guess[1] = 0.0" in out
    for text in lists:
        assert text in out
    assert "curve_fit" not in out
