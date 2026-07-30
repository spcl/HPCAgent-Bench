"""``np.divide`` / ``np.true_divide`` are TRUE division on every spelling.

numpy ``/`` (and its ``divide`` / ``true_divide`` ufunc aliases) promote int / int to
float64; C and Fortran ``/`` truncate. Lowering's ``_TrueDivisionPromoter`` handles that
by casting the left operand -- but it runs at the *promote-true-division* phase, well
BEFORE *libnode-expand*, where ``expand_divide`` synthesized its own ``Div`` node. That
node was never seen by the promoter, so only the value-returning CALL form broke:

    out[:] = np.divide(a, b)   # int64 -> C emitted a / b, truncating: 1/2 gave 0

while the operator form ``a / b`` and the ``out=`` kwarg form (both written by the numpy
source, hence visible to the promoter) were already correct. All three are pinned here so
the phase-order dependence cannot come back.
"""
import numpy as np
from _op_oracle import run_op

_NATIVE = ("c", "cpp", "fortran")
_A = np.array([1, 3, 7, 9, 2, 5], dtype=np.int64)
_B = np.array([2, 2, 2, 4, 4, 4], dtype=np.int64)
_SYMS = {"N": 6}
_SHAPES = {"a": "(N,)", "b": "(N,)", "out": "(N,)"}
_DTYPES = {"a": "int64", "b": "int64", "out": "float64"}


def _assert_ok(res):
    for backend, status in res.items():
        assert status == "ok" or status.startswith("skip"), f"{backend}: {status}"
    assert any(status == "ok" for status in res.values()), f"all skipped (vacuous): {res}"


def _run(body: str):
    src = f"import numpy as np\ndef f(a, b, out):\n    {body}\n"
    return run_op(src, "f", {"a": _A, "b": _B}, {"out": (6, )}, _SYMS, shapes=_SHAPES, backends=_NATIVE, dtypes=_DTYPES)


def test_divide_call_form_is_true_division():
    _assert_ok(_run("out[:] = np.divide(a, b)"))


def test_true_divide_call_form_is_true_division():
    _assert_ok(_run("out[:] = np.true_divide(a, b)"))


def test_divide_by_integer_literal_is_true_division():
    _assert_ok(_run("out[:] = np.divide(a, 4)"))


def test_operator_form_still_true_division():
    # Already correct (the promoter sees it); pinned so the expander fix cannot regress it.
    _assert_ok(_run("out[:] = a / b"))


def test_out_kwarg_form_still_true_division():
    _assert_ok(_run("np.divide(a, b, out=out)"))


def test_float_divide_is_not_cast_to_fp64():
    # A float operand must NOT pick up the int/int cast -- it would pin an fp32 kernel's
    # arithmetic to double. Numerically identical at fp64; asserted on the emitted text.
    import ast

    from numpyto_common.lib_nodes import expand_divide
    args = [ast.Name(id="a", ctx=ast.Load()), ast.Name(id="b", ctx=ast.Load())]
    shapes = {"a": ("N", ), "b": ("N", ), "out": ("N", )}
    target = ast.Name(id="out", ctx=ast.Store())
    float_src = ast.unparse(
        ast.fix_missing_locations(
            ast.Module(body=expand_divide(target, args, shapes, local_dtypes={
                "a": "float64",
                "b": "float64"
            }),
                       type_ignores=[])))
    assert "float64" not in float_src, float_src
    int_src = ast.unparse(
        ast.fix_missing_locations(
            ast.Module(body=expand_divide(target, args, shapes, local_dtypes={
                "a": "int64",
                "b": "int64"
            }),
                       type_ignores=[])))
    assert "np.float64(a[__r0])" in int_src, int_src
