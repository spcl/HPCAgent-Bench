# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""A rebound input parameter stays a read in the emitted DaCe program.

numpy rebinding a parameter NAME leaves the caller's array alone; dace's frontend spells the same
assignment as a copy into the argument. chebyshev_filter_subspace's ``X, Y, sigma = Y, Ynew,
sigma_new`` wrote the caller's ``X``, which the C ABI declares const, so the CPF render refused it
and the dace leg mutated an input the numpy reference leaves untouched.
"""

import ast

from hpcagent_bench.translators.numpyto_c.dace_emit import emit_dace, localize_rebound_inputs
from hpcagent_bench.translators.numpyto_common.frontend import emit_with_inline_fallback, parse_kernel
from tests.translators.bench_yaml import bench_info_for


def localized(source: str, inputs: set[str]) -> str:
    fn = ast.parse(source).body[0]
    assert isinstance(fn, ast.FunctionDef)
    localize_rebound_inputs(fn, inputs)
    return ast.unparse(fn)


def stored_names(source: str) -> set[str]:
    return {n.id for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


def test_a_loop_rebind_reads_a_copy_bound_at_entry() -> None:
    """Rebound inside the loop, ``X`` is read before the rebind on the next trip: every use reads
    the entry copy, and the argument itself is only ever the copy's source."""
    out = localized(
        "def k(X, out):\n"
        "    Y = X * 2.0\n"
        "    for i in range(3):\n"
        "        Z = Y + X\n"
        "        X = Y\n"
        "        Y = Z\n"
        "    out[:] = Y\n",
        {"X"},
    )
    lines = out.splitlines()
    assert lines[1].strip() == "X__v1 = np.copy(X)", out
    assert "X" not in stored_names(out), out
    assert "Y = X__v1 * 2.0" in out and "Z = Y + X__v1" in out and "X__v1 = Y" in out, out


def test_a_top_level_rebind_renames_from_the_binding_without_a_copy() -> None:
    """The first store is a top-level binding, which dominates every later use: its own right-hand
    side reads the argument, and everything after reads the new name."""
    out = localized("def k(nl, ngm, out):\n    nl = nl[:ngm]\n    out[:] = nl\n", {"nl"})
    assert out.splitlines()[1:] == ["    nl__v1 = nl[:ngm]", "    out[:] = nl__v1"], out


def test_a_parameter_that_is_not_rebound_is_left_alone() -> None:
    """``out[:] = ..`` writes the caller's buffer in numpy too, and is not a rebind."""
    source = "def k(X, out):\n    out[:] = X + 1.0\n"
    assert localized(source, {"X"}) == ast.unparse(ast.parse(source))


def test_an_input_also_written_through_keeps_its_name() -> None:
    """numpy writes the caller's array through ``X[0] = ..``; a copy would drop that write."""
    source = "def k(X, Y):\n    for i in range(2):\n        X[0] = 1.0\n        X = Y\n"
    assert localized(source, {"X"}) == ast.unparse(ast.parse(source))


def test_chebyshev_never_stores_its_input() -> None:
    """The emitted program binds the local copy and never assigns the argument ``X``."""

    def render() -> tuple[object, str]:
        with bench_info_for("chebyshev_filter_subspace") as (unused, numpy_py, bi):
            kir = parse_kernel(numpy_py, bi, open_mesh_grids=False)
        return kir, emit_dace(kir)

    unused, src = emit_with_inline_fallback(render)
    program = next(
        n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "chebyshev_filter_subspace"
    )
    assert "X" not in stored_names(ast.unparse(program)), src
    assert "X__v1 = np.copy(X)" in src, src
