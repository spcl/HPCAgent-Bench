# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
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


def test_a_generator_over_a_literal_range_unrolls() -> None:
    got = desugared("""
        def k(x, p, out):
            t = tuple((x.shape[i + 1] * p for i in range(2)))
            out[0] = t[1]
        """)
    assert "out[0] = x.shape[2] * p" in got


@pytest.mark.parametrize("rank,want", [(4, "(1, x.shape[1], 1, 1)"), (2, "(1, x.shape[1])")])
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


def test_len_of_a_compile_time_tuple_is_a_constant() -> None:
    assert desugared("""
        def k(x, p, out):
            t = (p, p, p)
            out[0] = len(t)
        """).endswith("out[0] = 3")


def test_an_isinstance_guard_on_an_integer_knob_takes_the_true_branch() -> None:
    """``(int, np.integer)`` is true whichever way the harness passed the knob, so it decides."""
    got = desugared("""
        def k(x, p, out):
            if isinstance(p, (int, np.integer)):
                p = (p, p)
            out[0] = p[1]
        """)
    assert got.endswith("out[0] = p")


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


def test_a_cast_pins_the_provenance_and_makes_a_bare_int_test_decide() -> None:
    """The helper inliner materialises an argument as ``__inlN_stride = int(stride)``; that cast is
    what turns an otherwise-undecidable guard into a known Python int."""
    got = desugared("""
        def k(x, p, out):
            q = int(p)
            if isinstance(q, int):
                q = (q, q)
            out[0] = q[1]
        """)
    assert got.endswith("out[0] = q")
    assert "isinstance" not in got


def test_an_isinstance_guard_on_a_tuple_takes_the_false_branch() -> None:
    got = desugared("""
        def k(x, p, out):
            p = (p, p)
            if isinstance(p, (int, np.integer)):
                p = (p, p, p)
            out[0] = len(p)
        """)
    assert got.endswith("out[0] = 2")


def test_an_isinstance_on_an_unknown_name_is_left_alone() -> None:
    """This pass narrows; it never guesses. An undecidable guard must survive verbatim."""
    got = desugared("""
        def k(x, p, out):
            if isinstance(unknown, (int, np.integer)):
                out[0] = 1
        """)
    assert "isinstance(unknown, (int, np.integer))" in got


def test_a_bound_name_is_never_none() -> None:
    got = desugared("""
        def k(x, p, s, out):
            if s is None:
                s = p
            out[0] = s
        """)
    assert got.endswith("out[0] = s")


def test_slice_calls_in_a_concatenated_index_become_real_slices() -> None:
    got = desugared("""
        def k(x, p, out):
            idx = (slice(None), slice(None)) + (slice(p, p + 2),)
            out[idx] = x
        """)
    assert "out[:, :, p:p + 2] = x" in got


def test_literal_index_arithmetic_folds_so_a_tuple_element_can_be_selected() -> None:
    """A comprehension unroll leaves ``i + 2`` as ``0 + 2``; without folding it the tuple subscript
    has no literal index and the whole tuple survives into value position."""
    got = desugared("""
        def k(x, p, out):
            t = (1, 2, p)
            out[0] = t[0 + 2]
        """)
    assert got.endswith("out[0] = p")


def test_a_tuple_bound_inside_a_loop_is_not_folded_across_iterations() -> None:
    """A read can come from the previous iteration, so the binding does not dominate it."""
    got = desugared("""
        def k(x, p, out):
            t = (1, 2)
            for i in range(3):
                out[i] = t[0]
                t = (3, 4)
        """)
    assert "t = (3, 4)" in got  # left standing rather than folded to a stale value


def test_a_string_comparison_of_two_literals_folds_its_branch() -> None:
    """The ports carry ``-np.inf if 'mean' == 'max' else 0.0`` from a templated generator."""
    got = desugared("""
        def k(x, p, out):
            fill = -np.inf if 'mean' == 'max' else 0.0
            out[0] = fill
        """)
    assert "'mean'" not in got and "0.0" in got


def test_a_none_default_left_dead_by_the_fold_is_dropped() -> None:
    """``None`` has no C spelling, so a defaulted argument the fold made unreachable would fail the
    emit over a statement that no longer does anything."""
    got = desugared("""
        def k(x, p, out):
            s = None
            if s is None:
                s = (p, p)
            out[0] = s[0]
        """)
    assert "None" not in got and got.endswith("out[0] = p")


def test_a_none_binding_that_is_still_read_survives() -> None:
    """The drop is dead-code only; a live None must reach the emitter and fail loudly there."""
    got = desugared("""
        def k(x, p, out):
            s = None
            out[0] = f(s)
        """)
    assert "s = None" in got


def test_a_bare_shape_expands_wherever_it_stands() -> None:
    """``.reshape(x.shape)`` -- the argument is not a tuple context, so nothing used to force the
    expansion, and the reshape reached lowering with no compile-time rank. Group-norm then kept the
    PREVIOUS statement's rank-5 extent for the target and indexed past the end of it."""
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


def test_a_simultaneous_bind_is_not_split_sequentially() -> None:
    """``a, b = b, a + b`` binds both from the OLD values. A sequential split reads the new ``a``
    and the kernel computes a different sequence with no diagnostic, so the statement is left for
    the staging rewriter in ``lowering`` rather than split here."""
    got = desugared("""
        def k(x, p, out):
            a = x[0]
            b = x[1]
            for i in range(3):
                a, b = b, a + b
            out[0] = a
        """)
    assert "a, b = (b, a + b)" in got
    assert "a = b" not in got


def test_an_unpack_with_no_hazard_still_splits() -> None:
    """The decline is the hazard case only: a plain unpack must still scalarize away."""
    got = desugared("""
        def k(x, p, out):
            oh, ow = (p, p + 1)
            out[0] = oh + ow
        """)
    assert "oh = p" in got and "ow = p + 1" in got
    assert "oh, ow = " not in got  # the tuple itself is gone, not just its uses


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
    assert "while" not in out and "append" not in out


def test_an_all_integer_display_folds_to_an_integer_array() -> None:
    # The cut pins the length to ``n``; without it the build is ``max(2, n)`` long and refused.
    out = folded_lists("""
        def f(n, out):
            idx = [0, 1]
            while len(idx) < n:
                idx.append(len(idx) * 2)
            idx = idx[:n]
            out[:] = idx
        """)
    assert "np.zeros(n, dtype=np.int64)" in out


def test_a_cut_to_a_different_length_is_left_alone() -> None:
    # Grown to ``n`` and cut to ``m`` is not this idiom: the closed form above would be the wrong
    # length. Nothing is guessed -- the list stays and the refusal that owns it still fires.
    out = folded_lists("""
        def f(n, m, out):
            centre = [1.0]
            while len(centre) < n:
                centre.append(2.0)
            centre = centre[:m]
            out[:] = centre
        """)
    assert "centre = [1.0]" in out and "while len(centre) < n:" in out


def as_written_and_folded(source: str, *args: object) -> tuple[np.ndarray, np.ndarray]:
    """``f(*args)`` from ``source`` as Python runs it, and from its folded form."""
    written: dict[str, object] = {"np": np}
    exec(textwrap.dedent(source), written)
    folded: dict[str, object] = {"np": np}
    exec(folded_lists(source), folded)
    return np.asarray(written["f"](*args)), np.asarray(folded["f"](*args))


@pytest.mark.parametrize("n", [1, 5])
def test_a_seed_longer_than_the_growth_bound_keeps_its_python_length(n: int) -> None:
    # No truncation: Python keeps max(len(seed), n) elements, so a length-``n`` array drops the tail.
    expected, got = as_written_and_folded(
        """
        def f(n):
            centre = [1580.0, 2670.0, 3000.0]
            while len(centre) < n:
                centre.append(1200.0 + 200.0 * len(centre))
            return centre
        """,
        n,
    )
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("n", [1, 5])
def test_a_growth_cut_into_a_fresh_name_folds_and_keeps_python_values(n: int) -> None:
    # raman_fitting's spelling: the cut binds a NEW name and the grown list is read nowhere else.
    source = """
        def f(n):
            centre1 = [1580.0, 2670.0]
            while len(centre1) < n:
                centre1.append(1200.0 + 200.0 * len(centre1))
            centre2 = centre1[:n]
            return centre2
        """
    expected, got = as_written_and_folded(source, n)
    assert "append" not in folded_lists(source)
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)


def test_a_list_mutated_anywhere_else_is_left_alone() -> None:
    # A second append outside the growth loop is a mutation the closed form does not account for.
    out = folded_lists("""
        def f(n, out):
            centre = [1.0]
            while len(centre) < n:
                centre.append(2.0)
            centre.append(3.0)
            out[:] = centre
        """)
    assert "while len(centre) < n:" in out


def test_a_list_resized_after_its_cut_is_left_alone() -> None:
    # The build itself is complete and cut; the append in a later branch is what the guard refuses.
    out = folded_lists("""
        def f(n, flag, out):
            centre = [1.0]
            while len(centre) < n:
                centre.append(2.0)
            centre = centre[:n]
            if flag:
                centre.append(3.0)
            out[:] = centre
        """)
    assert "centre = [1.0]" in out and "while len(centre) < n:" in out


@pytest.mark.parametrize("n", [2, 5])
def test_a_growth_step_reading_the_list_itself_is_left_alone(n: int) -> None:
    # ``centre[-1]`` is the last APPENDED element in Python; in a preallocated array it is the last slot.
    source = """
        def f(n):
            centre = [1.0]
            while len(centre) < n:
                centre.append(centre[-1] * 2.0)
            centre = centre[:n]
            return centre
        """
    expected, got = as_written_and_folded(source, n)
    assert "centre.append(centre[-1] * 2.0)" in folded_lists(source)
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("n", [2, 5])
def test_an_integer_seed_grown_by_a_float_rule_stays_floating(n: int) -> None:
    # Python keeps 1.5 as 1.5; an int64 buffer would truncate every element the growth rule writes.
    source = """
        def f(n):
            idx = [0, 1]
            while len(idx) < n:
                idx.append(0.5 * len(idx))
            idx = idx[:n]
            return idx
        """
    expected, got = as_written_and_folded(source, n)
    assert "np.zeros(n, dtype=np.float64)" in folded_lists(source)
    assert np.array_equal(got, expected)


def test_straight_extends_inside_a_loop_body_fold_to_indexed_stores() -> None:
    source = """
        def f(lo, span):
            out = np.zeros((2, 4))
            for t in range(2):
                guess = [lo + t]
                guess += [span, 10.0]
                guess = guess + [lo * span]
                out[t, :] = guess
            return out
        """
    folded = folded_lists(source)
    assert "guess = np.zeros(4, dtype=np.float64)" in folded
    assert "guess[0] = lo + t" in folded and "guess[2] = 10.0" in folded and "guess[3] = lo * span" in folded
    assert "+=" not in folded and "guess + [" not in folded
    expected, got = as_written_and_folded(source, 1.5, 4.0)
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("n", [0, 3])
def test_a_fixed_stride_loop_folds_to_stores_at_a_symbolic_offset(n: int) -> None:
    source = """
        def f(n, width):
            params = [0.5]
            for j in range(n):
                params += [1.0 * j, width]
            params.append(-1.0)
            return params
        """
    folded = folded_lists(source)
    assert "params = np.zeros(1 + 2 * n + 1, dtype=np.float64)" in folded
    assert "params[1 + 2 * j + 1] = width" in folded and "params[1 + 2 * n] = -1.0" in folded
    assert "+=" not in folded and "append" not in folded
    expected, got = as_written_and_folded(source, n, 7.0)
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("n", [2, 5])
def test_a_cut_shorter_than_the_built_prefix_writes_nothing_past_it(n: int) -> None:
    # Three elements are built before the cut; at n == 2 the third store must not land.
    source = """
        def f(n, a):
            v = [a]
            for j in range(2):
                v += [a * j]
            while len(v) < n:
                v.append(0.5 * len(v))
            v = v[:n]
            return v
        """
    assert "while" not in folded_lists(source)
    expected, got = as_written_and_folded(source, n, 3.0)
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)


def test_an_extended_list_mutated_in_a_later_branch_is_left_alone() -> None:
    out = folded_lists("""
        def f(flag, lo):
            guess = [lo]
            guess += [2.0]
            if flag:
                guess.append(3.0)
            return guess
        """)
    assert "guess = [lo]" in out and "guess += [2.0]" in out


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


def test_a_grown_p0_becomes_an_array_before_the_fit_indexes_it() -> None:
    out = curve_fit_rewritten("guess = [lo]\n    guess += [0.0]")
    assert "guess = np.zeros(2, dtype=np.float64)" in out
    assert "guess[0] = lo" in out and "guess[1] = 0.0" in out
    assert "curve_fit" not in out


def test_a_bare_p0_display_becomes_an_array_and_other_displays_stay_lists() -> None:
    out = curve_fit_rewritten("scales = [1.0, 2.0]\n    guess = [lo, 0.0]")
    assert "guess = np.zeros(2, dtype=np.float64)" in out
    assert "guess[0] = lo" in out and "guess[1] = 0.0" in out
    assert "scales = [1.0, 2.0]" in out
    assert "curve_fit" not in out
