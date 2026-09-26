# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""The one tuple-unpack split behind dace, native lowering and the C / Fortran emitters.

Python evaluates the whole right side before it binds any target, so a split must read the OLD
values; a right side with no per-target spelling (a starred element, a call returning a tuple) must
stay whole. Each split is executed next to the source it came from and the bound names compared.
"""

import ast
from collections.abc import Callable

import numpy as np
import pytest

from hpcagent_bench.translators.numpyto_c.dace_emit import SplitTupleAssign
from hpcagent_bench.translators.numpyto_common.emitter import TupleTargetSplitter
from hpcagent_bench.translators.numpyto_common.lowering import ShapeTableTupleSplit
from hpcagent_bench.translators.numpyto_common.statement_desugar import SplitTupleUnpack
from tests.translators.source_module import run_source

EVERY_BACKEND = [
    pytest.param(SplitTupleAssign, id="dace"),
    pytest.param(lambda: ShapeTableTupleSplit({}), id="lowering"),
    pytest.param(TupleTargetSplitter, id="c-fortran"),
]
LATCHING_BACKENDS = EVERY_BACKEND[:2]


def split(splitter: SplitTupleUnpack, source: str) -> str:
    tree = splitter.visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def unchanged(source: str) -> str:
    return ast.unparse(ast.parse(source))


def bound(source: str, **values: object) -> dict[str, object]:
    """The names ``source`` leaves bound, starting from ``values``; minted ``__`` temps are dropped."""
    scope: dict[str, object] = {"np": np, **values}
    run_source(source, scope)
    return {name: value for name, value in scope.items() if not name.startswith("__")}


@pytest.mark.parametrize("make", EVERY_BACKEND)
def test_a_starred_element_has_no_per_target_spelling(make: Callable[[], SplitTupleUnpack]) -> None:
    """``a = *p`` is not python, and every split used to emit it."""
    assert split(make(), "a, b = *p, q") == unchanged("a, b = *p, q")


@pytest.mark.parametrize("make", EVERY_BACKEND)
def test_a_call_returning_a_tuple_binds_once(make: Callable[[], SplitTupleUnpack]) -> None:
    assert split(make(), "a, b = f(x)") == unchanged("a, b = f(x)")


@pytest.mark.parametrize("make", LATCHING_BACKENDS)
@pytest.mark.parametrize("source", ["a, b = b, a", "a, b, c = b, c, a", "a, b = b + 1, a * 2", "c, a, b = c, b, a"])
def test_a_racing_unpack_binds_every_target_from_the_old_values(
    make: Callable[[], SplitTupleUnpack], source: str
) -> None:
    out = split(make(), source)
    assert len(out.splitlines()) > len(ast.parse(source).body[0].targets[0].elts), out
    assert bound(out, a=1, b=2, c=3) == bound(source, a=1, b=2, c=3), out


def test_a_self_copy_binds_plainly_after_the_latched_values() -> None:
    """``c = c`` writes back what it reads, so it needs no temp and must stay a bare self-copy."""
    lines = split(ShapeTableTupleSplit({}), "c, a, b = c, b, a").splitlines()
    assert lines == ["__swap1_1 = b", "__swap1_2 = a", "a = __swap1_1", "b = __swap1_2", "c = c"]


def test_the_emitter_split_projects_a_conditional_over_tuples() -> None:
    source = "a, b = (p, q) if c else (r, s)"
    out = split(TupleTargetSplitter(), source)
    assert out.splitlines() == ["a = p if c else r", "b = q if c else s"]
    assert bound(out, p=1, q=2, r=3, s=4, c=False) == bound(source, p=1, q=2, r=3, s=4, c=False)


def test_the_emitter_split_leaves_a_racing_unpack_whole() -> None:
    """A temp minted after the C / Fortran locals are harvested would reach the emitter undeclared."""
    source = "a, b = (b, a) if c else (a, b)"
    assert split(TupleTargetSplitter(), source) == unchanged(source)


def test_a_shape_unpack_onto_the_other_axes_symbols_reads_the_old_symbols() -> None:
    """``ny, nx = u.shape`` over ``u: [nx, ny]`` became ``ny = nx; nx = ny``, reading the NEW ``ny``."""
    splitter = ShapeTableTupleSplit({"u": ["nx", "ny"]})
    out = split(splitter, "ny, nx = u.shape")
    after_split = bound(out, nx=3, ny=5)
    after_source = bound("ny, nx = u.shape", u=np.zeros((3, 5)), nx=3, ny=5)
    assert (after_split["ny"], after_split["nx"]) == (after_source["ny"], after_source["nx"]) == (3, 5), out
    assert splitter.int_locals == ["ny", "nx"]


def test_a_shape_unpack_binds_the_declared_symbols_as_int_locals() -> None:
    splitter = ShapeTableTupleSplit({"x": ["N", "M"]})
    assert split(splitter, "n, M = x.shape").splitlines() == ["n = N", "M = M"]
    assert splitter.int_locals == ["n"]


def test_an_integer_tuple_binds_int_locals() -> None:
    splitter = ShapeTableTupleSplit({})
    assert split(splitter, "n, m = 3, 4").splitlines() == ["n = 3", "m = 4"]
    assert splitter.int_locals == ["n", "m"]


def test_a_subscript_swap_through_the_lowering_split_matches_python() -> None:
    source = "out[i], out[j] = out[j], out[i]"
    out = split(ShapeTableTupleSplit({}), source)
    split_out, source_out = [1, 2, 3], [1, 2, 3]
    bound(out, out=split_out, i=0, j=2)
    bound(source, out=source_out, i=0, j=2)
    assert split_out == source_out == [3, 2, 1], out


@pytest.mark.parametrize("source", ["a, *b = p, q", "(a, b), c = (1, 2), a"])
def test_a_starred_or_nested_target_stays_whole(source: str) -> None:
    """``*b = q`` is not python; ``(a, b) = (1, 2); c = a`` read the NEW ``a``."""
    assert split(ShapeTableTupleSplit({}), source) == unchanged(source)


def test_a_split_inside_a_loop_body_is_spliced_into_that_body() -> None:
    out = split(ShapeTableTupleSplit({}), "for i in range(3):\n    a, b = b, a\n")
    assert bound(out, a=1, b=2) == bound("for i in range(3):\n    a, b = b, a\n", a=1, b=2), out
