"""DI-5: the debug-gated between-phase invariant checker in ``lower()``.

The lowering pipeline is a list of phases mutating one shared
:class:`LoweringContext`. When ``OPTARENA_LOWER_INVARIANTS`` is set, ``lower()``
runs :func:`_assert_lowering_invariants` after every phase so a corrupted
side-table or an AST the context stopped tracking is localised to the phase that
broke it. These tests prove the checker (a) passes on real kernels with the flag
on, and (b) actually fires -- naming the phase -- for each corruption mode.
"""
import ast
import copy
import json
import pathlib
import tempfile

import pytest

from numpyto_c.frontend import parse_kernel
from numpyto_common.lowering import (LoweringContext, _assert_lowering_invariants, _INVARIANT_ENV, lower)

_SRC = ("import numpy as np\n"
        "def f(x, out):\n"
        " s = np.zeros((6,))\n"
        " for i in range(6):\n"
        "  s[i] = x[i] * 2.0\n"
        " out[:] = s\n")


def _parsed_kir():
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "k_numpy.py").write_text(_SRC)
    bi = {
        "benchmark": {
            "name": "k",
            "short_name": "k",
            "relative_path": "",
            "module_name": "k",
            "func_name": "f",
            "parameters": {
                "S": {
                    "N": 6
                }
            },
            "input_args": ["x", "out"],
            "array_args": ["x", "out"],
            "output_args": ["out"],
            "init": {
                "shapes": {
                    "x": "(N,)",
                    "out": "(N,)"
                }
            }
        }
    }
    (d / "bi.json").write_text(json.dumps(bi))
    return parse_kernel(d / "k_numpy.py", d / "bi.json")


def _fresh_ctx():
    """A structurally-valid context over a freshly lowered kir (aliasing holds)."""
    lowered = lower(_parsed_kir())
    ctx = LoweringContext(lowered, lowered)
    ctx.tree = lowered.tree
    return ctx


def test_lower_with_flag_on_does_not_false_trip(monkeypatch):
    # The checker must accept every intermediate state of a real lowering.
    monkeypatch.setenv(_INVARIANT_ENV, "1")
    kir = lower(_parsed_kir())
    assert kir.zeros_locals  # np.zeros((6,)) harvested -> proves the pipeline ran


def test_clean_context_passes():
    # A well-formed post-lowering context trips nothing.
    _assert_lowering_invariants("some-phase", _fresh_ctx())


def test_tree_drift_is_caught():
    ctx = _fresh_ctx()
    ctx.tree = ast.parse("x = 1")  # no longer aliases ctx.kir.tree
    with pytest.raises(AssertionError, match=r"drift-phase.*no longer aliases"):
        _assert_lowering_invariants("drift-phase", ctx)


def test_non_functiondef_tree_is_caught():
    ctx = _fresh_ctx()
    ctx.kir.tree = ast.parse("x = 1")  # a Module, not a FunctionDef
    ctx.tree = ctx.kir.tree  # keep aliasing so the FunctionDef check is what fires
    with pytest.raises(AssertionError, match=r"bad-tree-phase.*expected ast.FunctionDef"):
        _assert_lowering_invariants("bad-tree-phase", ctx)


def test_wrong_side_table_container_is_caught():
    ctx = _fresh_ctx()
    ctx.kir.local_dtypes = []  # declared a dict; a list is a corruption
    with pytest.raises(AssertionError, match=r"container-phase.*kir.local_dtypes.*expected dict"):
        _assert_lowering_invariants("container-phase", ctx)


def test_malformed_ast_is_caught():
    ctx = _fresh_ctx()
    # A fresh FunctionDef whose one statement is an Assign with no value -- valid
    # container types, aliasing holds, but it cannot unparse.
    bad = copy.deepcopy(ctx.kir.tree)
    bad.body = [ast.Assign(targets=[ast.Name(id="x", ctx=ast.Store())], value=None)]
    ctx.kir.tree = bad
    ctx.tree = bad
    with pytest.raises(AssertionError, match=r"unparse-phase.*ast.unparse"):
        _assert_lowering_invariants("unparse-phase", ctx)
