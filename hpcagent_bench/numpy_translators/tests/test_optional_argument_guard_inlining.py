"""An OPTIONAL argument is guarded with ``is None``, and the guard has to fuse just the same.

``_fuse_guarded_returns`` turns ``if FLAG: return A`` before a trailing ``return B`` into one
``return A if FLAG else B``, which is what makes an early-returning helper inlinable at all. Two
things kept it from firing on the shape torch's affine-less normalizations ship, and each cost a
kernel its DaCe program outright rather than degrading anything visible:

* the guard is spelled ``if weight is None``, and only ``==``/``!=`` counted as a decidable
  compare, so an identity test against a call-site literal was treated as a runtime condition;
* the affine arm binds its broadcast ``shape`` AFTER the guard, and the fuse only ever looked at
  the last two statements, so the guard and the trailing return were never adjacent to begin with.

conv2d_instance_norm_divide's ``_instance_norm`` has both at once. It matched no inlinable form,
survived as a call a ``@dc.program`` cannot make, and emitted no program at all.

The load-bearing assertion is the last one. Picking the wrong arm still produces a program, and
every value it computes is wrong -- so the arm is checked against the reference, not assumed.
"""
import ast

import numpy as np

import pytest

from _op_oracle import run_op
from numpyto_common.frontend import _collect_inlinable_helpers, _fuse_guarded_returns, _is_static_flag_test

BACKENDS = ("c", "cpp", "fortran", "numba", "pythran", "jax")

#: ``_instance_norm``'s exact shape: an ``is None`` guard with a pure binding BETWEEN it and the
#: trailing return. The affine arm is scaled far from 1.0 so selecting it cannot pass as round-off.
NORM_SRC = ("import numpy as np\n"
            "def _norm(x, weight):\n"
            " mean = np.sum(x) / x.shape[0]\n"
            " y = x - mean\n"
            " if weight is None:\n"
            "  return y\n"
            " shape = (x.shape[0],)\n"
            " return y * np.reshape(weight, shape) * 100.0\n"
            "def f(x, out):\n"
            " out[:] = _norm(x, None)\n")


def parse(src: str) -> ast.Module:
    return ast.parse(src)


def helper_of(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def flags(*names):
    return frozenset(names)


def test_an_identity_test_against_a_literal_is_a_static_flag_test():
    """``weight is None`` decides on the same grounds ``weight == 0`` does: the call site binds the
    parameter to a literal, so substitution leaves two constants."""
    assert _is_static_flag_test(ast.parse("weight is None", mode="eval").body, flags("weight"))
    assert _is_static_flag_test(ast.parse("weight is not None", mode="eval").body, flags("weight"))
    assert _is_static_flag_test(ast.parse("None is weight", mode="eval").body, flags("weight"))


@pytest.mark.parametrize(
    "expr,reason",
    [("weight is other", "neither side is a literal, so nothing is decided"),
     ("thing is None", "the name is not a parameter every call site pins to a literal"),
     ("weight is None is None", "a chained compare has no single decidable pair")],
)
def test_an_undecidable_identity_test_is_declined(expr, reason):
    """An undecidable guard fused into an ``IfExp`` over ARRAY branches has no target form: C's
    ``?:`` rejects the operand types and Fortran's ``merge`` evaluates BOTH arms."""
    assert not _is_static_flag_test(ast.parse(expr, mode="eval").body, flags("weight")), reason


def test_a_binding_between_the_guard_and_the_return_is_lifted_over_it():
    """``shape`` is bound on the affine path only, which left the guard two statements from the
    trailing return -- and the fuse only ever looked at the last two."""
    tree = parse(NORM_SRC)
    _fuse_guarded_returns(tree)
    body = helper_of(tree, "_norm").body
    assert [type(s).__name__ for s in body] == ["Assign", "Assign", "Assign", "Return"]
    assert isinstance(body[-1].value, ast.IfExp)
    assert body[2].targets[0].id == "shape", "the lifted binding runs ahead of the guard now"


def test_the_fused_helper_becomes_inlinable():
    """The whole point of fusing: an early return anywhere but the last statement matches no form,
    and a helper that matches no form survives as a CALL that a ``@dc.program`` cannot make."""
    tree = parse(NORM_SRC)
    assert "_norm" not in _collect_inlinable_helpers(tree, helper_of(tree, "f"))
    _fuse_guarded_returns(tree)
    assert "_norm" in _collect_inlinable_helpers(tree, helper_of(tree, "f"))


def test_a_binding_that_calls_is_not_lifted():
    """Lifting runs the statement on a path that never ran it. A call may write an argument array
    in place, so hoisting one over an early return is a side effect the source never had."""
    impure = NORM_SRC.replace(" shape = (x.shape[0],)\n", " shape = np.zeros(x.shape[0])\n")
    tree = parse(impure)
    _fuse_guarded_returns(tree)
    body = helper_of(tree, "_norm").body
    assert [type(s).__name__ for s in body] == ["Assign", "Assign", "If", "Assign", "Return"]
    assert "_norm" not in _collect_inlinable_helpers(tree, helper_of(tree, "f"))


def test_a_binding_the_guard_reads_is_not_lifted():
    """Moving it above the guard would change which value the guard tests -- the one case where
    the lift is not merely early but wrong."""
    read_by_guard = ("import numpy as np\n"
                     "def _norm(x, weight):\n"
                     " shape = 0\n"
                     " y = x - 1.0\n"
                     " if weight is None:\n"
                     "  return y + shape\n"
                     " shape = x.shape[0]\n"
                     " return y * shape\n"
                     "def f(x, out):\n"
                     " out[:] = _norm(x, None)\n")
    tree = parse(read_by_guard)
    _fuse_guarded_returns(tree)
    body = helper_of(tree, "_norm").body
    assert [type(s).__name__ for s in body] == ["Assign", "Assign", "If", "Assign", "Return"]


def test_the_selected_arm_is_the_one_the_reference_takes():
    """Every backend, against numpy's own answer. ``weight=None`` selects the CENTERED array, and
    the affine arm it must not select is a hundred times larger."""
    x = np.arange(1.0, 9.0)
    verdicts = run_op(NORM_SRC,
                      "f", {"x": x}, {"out": (8, )}, {"N": 8},
                      shapes={
                          "x": "(N,)",
                          "out": "(N,)"
                      },
                      backends=BACKENDS)
    assert all(v == "ok" or v.startswith("skip") for v in verdicts.values()), verdicts
