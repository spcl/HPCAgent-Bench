"""Pythran cannot reshape a non-materialized object, index a lazy ``numpy_expr`` passed into a
helper (KernelBench lenet/mlp), nor reduce a lazy broadcast ``numpy_expr`` correctly -- a column
broadcast fed to ``np.sum`` reduces to garbage (nbody KE). ``_PythranMaterialize`` forces evaluation
with ``np.ascontiguousarray``; these AST tests pin the rewrite, and the end-to-end bit-exact
numba/pythran validation lives in the machine_learning + scientific_computing (nbody) oracle.
"""
import ast

from numpyto_pythran.emit import _PythranMaterialize


def _apply(src, local_funcs):
    tree = ast.parse(src)
    tree = _PythranMaterialize(set(local_funcs)).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def test_np_reshape_becomes_ascontiguous_method_reshape():
    # the shape tuple is preserved as the reshape method arg (pythran accepts
    # ``.reshape((N, C))``; validated end-to-end by lenet's pythran gate).
    out = _apply("y = np.reshape(x, (N, C))", local_funcs=[])
    assert out == "y = np.ascontiguousarray(x).reshape((N, C))"


def test_helper_call_compound_arg_is_materialized():
    out = _apply("y = softmax(x @ w + b)", local_funcs=["softmax"])
    assert out == "y = softmax(np.ascontiguousarray(x @ w + b))"


def test_helper_call_plain_name_arg_untouched():
    # a bare Name is already concrete -> no ascontiguousarray wrap.
    out = _apply("y = softmax(t)", local_funcs=["softmax"])
    assert out == "y = softmax(t)"


def test_non_local_call_arg_untouched():
    # np.exp is not a local helper; its lazy arg is fine (elementwise).
    out = _apply("y = np.exp(x - m)", local_funcs=["softmax"])
    assert out == "y = np.exp(x - m)"


def test_method_reshape_not_double_wrapped():
    # an existing ``x.reshape(...)`` method form is left alone (only the
    # ``np.reshape`` function form is the failing shape).
    out = _apply("y = x.reshape(N, C)", local_funcs=[])
    assert out == "y = x.reshape(N, C)"


def test_reduction_over_compound_broadcast_is_materialized():
    # pythran reduces a lazy column-vector broadcast (mass(N,1) * vel(N,3)) to garbage;
    # force the operand concrete first. nbody KE regression.
    out = _apply("ke = np.sum(mass * vel ** 2)", local_funcs=[])
    assert out == "ke = np.sum(np.ascontiguousarray(mass * vel ** 2))"


def test_reduction_axis_kwarg_is_preserved():
    # only the array operand is wrapped; axis/other args pass through untouched.
    out = _apply("m = np.mean(a * b, axis=0)", local_funcs=[])
    assert out == "m = np.mean(np.ascontiguousarray(a * b), axis=0)"


def test_reduction_over_plain_name_untouched():
    # a bare Name is already concrete -> no wrap (ascontiguousarray would be a needless copy).
    out = _apply("s = np.sum(x)", local_funcs=[])
    assert out == "s = np.sum(x)"
