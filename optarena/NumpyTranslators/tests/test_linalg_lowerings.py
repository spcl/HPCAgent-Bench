"""Numerical validation of the np.linalg.{norm,lstsq} lowerings.

Both expanders emit a numpy-style loop nest; exec it against numpy
buffers and compare to numpy's reference (the same single-source-of-
truth approach as ``test_sparse_matvec``).
"""
import ast

import numpy as np

from numpyto_c import lib_nodes as ln


def _is_alloc_marker(s):
    """``X = __optarena_zeros__()`` -- a C deferred-malloc directive, not
    executable Python. The tests pre-allocate these buffers in ``scope``,
    so the marker is stripped before exec."""
    return (isinstance(s, ast.Assign)
            and isinstance(s.value, ast.Call)
            and isinstance(s.value.func, ast.Name)
            and s.value.func.id == "__optarena_zeros__")


def _run(stmts, scope):
    body = [s for s in stmts if not _is_alloc_marker(s)]
    mod = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, "<linalg>", "exec"), {"range": range}, scope)
    return scope


def test_linalg_norm_vector_2norm():
    fn = ln.NP_CALL_EXPANDERS[("np", "linalg.norm")]
    M = 9
    r = np.random.default_rng(1).random(M)
    stmts = fn(ast.Name(id="nrm", ctx=ast.Store()),
               [ast.Name(id="r", ctx=ast.Load())], {"r": ("M",)})
    sc = _run(stmts, {"r": r.copy(), "nrm": 0.0, "M": M, "sqrt": np.sqrt})
    assert np.isclose(sc["nrm"], np.linalg.norm(r))


def _solve(M, b_node_builder, scope_extra):
    """Build + exec ``y = lstsq(A, <b>)`` and return (y, A, b)."""
    rng = np.random.default_rng(0)
    A = rng.random((M, M)) + M * np.eye(M)   # well-conditioned
    b = rng.random(M)
    b_node = b_node_builder()
    fla = {}
    stmts = ln.expand_lstsq(
        ast.Name(id="y", ctx=ast.Store()),
        [ast.Name(id="A", ctx=ast.Load()), b_node],
        {"A": ("M", "M"), "b": ("M",), "y": ("M",)},
        fresh_local_allocs=fla)
    scope = {"A": A.copy(), "b": b.copy(), "y": np.zeros(M), "M": M}
    scope.update(scope_extra)
    _run(stmts, scope)
    return scope["y"], A, b, fla


def test_lstsq_square_bare_name_b():
    y, A, b, _ = _solve(6, lambda: ast.Name(id="b", ctx=ast.Load()), {})
    assert np.allclose(y, np.linalg.solve(A, b))


def test_lstsq_square_binop_b_materialized():
    """gmres passes ``beta * e1[:m]`` -- a BinOp b is materialized to a
    fresh temp vector (registered in fresh_local_allocs) before the solve."""
    beta = 2.5
    bnode = lambda: ast.BinOp(left=ast.Name(id="beta", ctx=ast.Load()),
                              op=ast.Mult(),
                              right=ast.Name(id="b", ctx=ast.Load()))
    y, A, b, fla = _solve(6, bnode, {"beta": beta, "__lq_b": np.zeros(6)})
    assert np.allclose(y, np.linalg.solve(A, beta * b))
    assert "__lq_b" in fla          # temp vector was registered for alloc
