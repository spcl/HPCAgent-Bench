# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compile-time tuples must be gone before emit -- C and Fortran have no tuple to lower one to.

Each test asserts on the desugared source, since the failure this pass exists to prevent is an
``ast.Tuple`` surviving into value position, where the emitter refuses it.
"""
import ast
import textwrap

import pytest

from numpyto_common.tuple_desugar import desugar_tuples

SCALARS = frozenset({"p", "s", "k"})
ARRAYS = frozenset({"x", "out"})

RANKS = {"x": 4, "out": 4}


def desugared(body: str, int_scalars=SCALARS, float_scalars=frozenset(), arrays=ARRAYS, ranks=None) -> str:
    fn = ast.parse(textwrap.dedent(body)).body[0]
    desugar_tuples(fn,
                   int_scalars=int_scalars,
                   float_scalars=float_scalars,
                   arrays=arrays,
                   ranks=RANKS if ranks is None else ranks)
    return ast.unparse(fn)


def test_a_tuple_local_is_scalarized_away():
    got = desugared("""
        def k(x, p, out):
            t = (p, p + 1)
            out[0] = t[0] + t[1]
        """)
    assert "out[0] = p + (p + 1)" in got
    assert "t = " not in got  # the binding itself is gone, not just its uses


def test_a_self_referential_binding_does_not_re_substitute():
    """``p = (p,)`` binds p to a tuple whose element IS p; naive substitution yields ``((p,),)``."""
    got = desugared("""
        def k(x, p, out):
            p = (p,)
            out[0] = p[0]
        """)
    assert got.endswith("out[0] = p")


def test_tuple_concatenation_folds_into_one_shape():
    got = desugared("""
        def k(x, p, out):
            shape = (x.shape[0],) + (p, 2)
            out[:] = np.zeros(shape)
        """)
    assert "np.zeros((x.shape[0], p, 2))" in got


def test_a_generator_over_a_literal_range_unrolls():
    got = desugared("""
        def k(x, p, out):
            t = tuple((x.shape[i + 1] * p for i in range(2)))
            out[0] = t[1]
        """)
    assert "out[0] = x.shape[2] * p" in got


@pytest.mark.parametrize("rank,want", [(4, "(1, x.shape[1], 1, 1)"), (2, "(1, x.shape[1])")])
def test_a_broadcast_shape_padded_to_an_array_rank_folds(rank, want):
    """``(1,) * (x.ndim - 2)``. The rank-2 case repeats ZERO times: the empty tuple is falsy but
    correct, so the fold must test for None rather than truthiness."""
    got = desugared("""
        def k(x, out):
            shape = (1, x.shape[1]) + (1,) * (x.ndim - 2)
            out[:] = x.reshape(shape)
        """,
                    int_scalars=frozenset(),
                    arrays=frozenset({"x", "out"}),
                    ranks={
                        "x": rank,
                        "out": rank
                    })
    assert f"x.reshape({want})" in got


def test_len_of_a_compile_time_tuple_is_a_constant():
    assert desugared("""
        def k(x, p, out):
            t = (p, p, p)
            out[0] = len(t)
        """).endswith("out[0] = 3")


def test_an_isinstance_guard_on_an_integer_knob_takes_the_true_branch():
    """``(int, np.integer)`` is true whichever way the harness passed the knob, so it decides."""
    got = desugared("""
        def k(x, p, out):
            if isinstance(p, (int, np.integer)):
                p = (p, p)
            out[0] = p[1]
        """)
    assert got.endswith("out[0] = p")


@pytest.mark.parametrize("spelling", ["int", "np.integer"])
def test_a_one_sided_isinstance_on_a_declared_knob_stays_undecided(spelling):
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


def test_a_cast_pins_the_provenance_and_makes_a_bare_int_test_decide():
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


def test_an_isinstance_guard_on_a_tuple_takes_the_false_branch():
    got = desugared("""
        def k(x, p, out):
            p = (p, p)
            if isinstance(p, (int, np.integer)):
                p = (p, p, p)
            out[0] = len(p)
        """)
    assert got.endswith("out[0] = 2")


def test_an_isinstance_on_an_unknown_name_is_left_alone():
    """This pass narrows; it never guesses. An undecidable guard must survive verbatim."""
    got = desugared("""
        def k(x, p, out):
            if isinstance(unknown, (int, np.integer)):
                out[0] = 1
        """)
    assert "isinstance(unknown, (int, np.integer))" in got


def test_a_bound_name_is_never_none():
    got = desugared("""
        def k(x, p, s, out):
            if s is None:
                s = p
            out[0] = s
        """)
    assert got.endswith("out[0] = s")


def test_slice_calls_in_a_concatenated_index_become_real_slices():
    got = desugared("""
        def k(x, p, out):
            idx = (slice(None), slice(None)) + (slice(p, p + 2),)
            out[idx] = x
        """)
    assert "out[:, :, p:p + 2] = x" in got


def test_literal_index_arithmetic_folds_so_a_tuple_element_can_be_selected():
    """A comprehension unroll leaves ``i + 2`` as ``0 + 2``; without folding it the tuple subscript
    has no literal index and the whole tuple survives into value position."""
    got = desugared("""
        def k(x, p, out):
            t = (1, 2, p)
            out[0] = t[0 + 2]
        """)
    assert got.endswith("out[0] = p")


def test_a_tuple_bound_inside_a_loop_is_not_folded_across_iterations():
    """A read can come from the previous iteration, so the binding does not dominate it."""
    got = desugared("""
        def k(x, p, out):
            t = (1, 2)
            for i in range(3):
                out[i] = t[0]
                t = (3, 4)
        """)
    assert "t = (3, 4)" in got  # left standing rather than folded to a stale value


def test_a_string_comparison_of_two_literals_folds_its_branch():
    """The ports carry ``-np.inf if 'mean' == 'max' else 0.0`` from a templated generator."""
    got = desugared("""
        def k(x, p, out):
            fill = -np.inf if 'mean' == 'max' else 0.0
            out[0] = fill
        """)
    assert "'mean'" not in got and "0.0" in got


def test_a_none_default_left_dead_by_the_fold_is_dropped():
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


def test_a_none_binding_that_is_still_read_survives():
    """The drop is dead-code only; a live None must reach the emitter and fail loudly there."""
    got = desugared("""
        def k(x, p, out):
            s = None
            out[0] = f(s)
        """)
    assert "s = None" in got
