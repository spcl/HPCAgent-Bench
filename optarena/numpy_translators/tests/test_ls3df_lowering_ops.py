"""AST-level unit tests for the general lowering capabilities added to make the
level-3 LS3DF macro-kernel ``ls3df_scf`` emit + validate on native c / fortran.

Each test isolates ONE capability so a regression points straight at the cause:

  * ``.shape`` / ``.shape[k]`` folded on a rank-shifting SUBSCRIPT base (newaxis
    broadcast) via the extent oracle -- the ``X.reshape(-1, X.shape[-1])`` /
    ``.reshape(X.shape)`` in an inlined ``_hpsi(v[..., None])``;
  * ``np.fft.fftfreq(n)`` and method-form ``.reshape`` sized by ``_iter_extent_of``;
  * shape-tuple local forward-substitution (``shp = Y.shape; ... .reshape(shp)``);
  * a mutated scalar (Lanczos ``na += 1``) excluded from inlined-dim substitution;
  * a compound shape token (``na - 1``) surviving ``.size`` as a real BinOp;
  * ``expand_copy`` allocating its (possibly runtime-shaped) target;
  * per-call-unique ``inv`` / ``solve`` / ``det`` working buffers.

These are pure AST transforms, so no compiler is needed.
"""
import ast

from numpyto_common.frontend import _collect_inlined_scalar_defs
from numpyto_common.lib_nodes import (_iter_extent_of, expand_copy,
                                      expand_linalg_inv)
from numpyto_common.lowering import (_ShapeMidExpressionRewriter,
                                     _TupleLocalPropagator)


def _expr(src):
    return ast.parse(src, mode="eval").body


def _unparse(node):
    return ast.unparse(ast.fix_missing_locations(node))


# --------------------------------------------------------------------------- #
# .shape / .shape[k] on a newaxis-subscript base                              #
# --------------------------------------------------------------------------- #


def test_shape_index_on_newaxis_subscript_folds_to_static_dim():
    # ``v[..., None].shape[-1]`` on a 3-D ``v`` -> the inserted size-1 axis (1).
    tree = ast.parse("y = v[:, :, :, None].shape[-1]")
    _ShapeMidExpressionRewriter({"v": ["Lb", "Lb", "Lb"]}).visit(tree)
    assert _unparse(tree.body[0].value) == "1"


def test_shape_index_on_newaxis_subscript_positive_axis():
    tree = ast.parse("y = v[:, :, :, None].shape[0]")
    _ShapeMidExpressionRewriter({"v": ["Lb", "Lb", "Lb"]}).visit(tree)
    assert _unparse(tree.body[0].value) == "Lb"


def test_bare_shape_on_subscript_base_folds_to_tuple():
    # ``psi[f].shape`` -> the residual (Lb, Lb, Lb, nstate) as a literal Tuple.
    tree = ast.parse("s = psi[f].shape")
    _ShapeMidExpressionRewriter({"psi": ["nfrag", "Lb", "Lb", "Lb", "nstate"]}).visit(tree)
    val = tree.body[0].value
    assert isinstance(val, ast.Tuple)
    assert _unparse(val) == "(Lb, Lb, Lb, nstate)"


def test_shape_fold_leaves_name_base_untouched_when_unknown():
    # Never-worse: an unknown base is left intact (not mangled).
    tree = ast.parse("y = w[:, None].shape[-1]")
    _ShapeMidExpressionRewriter({}).visit(tree)
    assert "shape" in _unparse(tree.body[0].value)


# --------------------------------------------------------------------------- #
# _iter_extent_of: fftfreq + method-form reshape                              #
# --------------------------------------------------------------------------- #


def test_iter_extent_of_fftfreq_is_length_n():
    ext = _iter_extent_of(_expr("np.fft.fftfreq(N, d=h)"), {})
    assert ext is not None and len(ext) == 1 and _unparse(ext[0]) == "N"


def test_iter_extent_of_fftn_preserves_shape():
    ext = _iter_extent_of(_expr("np.fft.fftn(rho)"), {"rho": ("N", "N", "N")})
    assert ext is not None and tuple(_unparse(e) for e in ext) == ("N", "N", "N")


def test_iter_extent_of_method_reshape_to_tuple():
    ext = _iter_extent_of(_expr("mm.reshape((Lb, Lb, Lb, nstate))"), {"mm": ("A", "B")})
    assert tuple(_unparse(e) for e in ext) == ("Lb", "Lb", "Lb", "nstate")


def test_iter_extent_of_method_reshape_resolves_neg1():
    ext = _iter_extent_of(_expr("X.reshape(-1, k)"), {"X": ("Lb", "Lb", "Lb", "nstate")})
    assert ext is not None and len(ext) == 2 and _unparse(ext[1]) == "k"
    # -1 dim = total / product(other dims).
    assert "/" in _unparse(ext[0])


# --------------------------------------------------------------------------- #
# shape-tuple local forward-substitution                                       #
# --------------------------------------------------------------------------- #


def test_tuple_local_propagator_inlines_and_drops_assignment():
    tree = ast.parse("shp = (Lb, Lb, Lb, nstate)\n"
                     "k = shp[-1]\n"
                     "y = np.reshape(mm, shp)\n")
    _TupleLocalPropagator().run(tree)
    out = _unparse(tree)
    assert "shp = " not in out                       # dead assignment dropped
    assert "(Lb, Lb, Lb, nstate)[-1]" in out         # inlined into the subscript
    assert "np.reshape(mm, (Lb, Lb, Lb, nstate))" in out


def test_tuple_local_propagator_skips_reassigned_name():
    # A twice-assigned Name is not a fixed shape descriptor -- leave it alone.
    tree = ast.parse("shp = (a, b)\nshp = (c, d)\ny = shp[0]\n")
    _TupleLocalPropagator().run(tree)
    assert "shp = (a, b)" in _unparse(tree)


# --------------------------------------------------------------------------- #
# mutated scalar excluded from inlined-dim substitution                        #
# --------------------------------------------------------------------------- #


def test_collect_inlined_scalar_defs_excludes_augassigned_counter():
    fn = ast.parse("def k():\n"
                   " __inl2_na = 0\n"
                   " __inl2_n = a.shape[0]\n"
                   " for _ in range(6):\n"
                   "  __inl2_na += 1\n").body[0]
    defs = _collect_inlined_scalar_defs(fn)
    assert "__inl2_na" not in defs        # mutated counter -- not a fixed dim
    assert defs.get("__inl2_n") == "a.shape[0]"


def test_collect_inlined_scalar_defs_excludes_multiply_assigned():
    fn = ast.parse("def k():\n"
                   " __inl1_m = 3\n"
                   " __inl1_m = 5\n").body[0]
    assert "__inl1_m" not in _collect_inlined_scalar_defs(fn)


# --------------------------------------------------------------------------- #
# compound shape token survives .size as a real BinOp (not a mangled Name)     #
# --------------------------------------------------------------------------- #


def test_size_of_compound_token_reparses_to_binop():
    tree = ast.parse("y = off.size")
    _ShapeMidExpressionRewriter({"off": ["na - 1"]}).visit(tree)
    val = tree.body[0].value
    assert isinstance(val, ast.BinOp)                 # not ast.Name(id="na - 1")
    assert _unparse(val) == "na - 1"


# --------------------------------------------------------------------------- #
# expand_copy allocates its target                                             #
# --------------------------------------------------------------------------- #


def test_expand_copy_emits_allocation_marker():
    tgt = ast.Name(id="Cm", ctx=ast.Store())
    stmts = expand_copy(tgt, [ast.Name(id="a", ctx=ast.Load())], {"a": ("n", "n")})
    # First emitted statement allocates the fresh target.
    assert isinstance(stmts[0], ast.Assign)
    assert isinstance(stmts[0].value, ast.Call)
    assert stmts[0].value.func.id == "__optarena_zeros__"
    assert stmts[0].targets[0].id == "Cm"


# --------------------------------------------------------------------------- #
# per-call-unique linalg working buffer                                        #
# --------------------------------------------------------------------------- #


def _inv_buffer_names(stmts):
    return {n.id for n in ast.walk(ast.Module(body=stmts, type_ignores=[]))
            if isinstance(n, ast.Name) and n.id.startswith("__inv_aw")}


def test_inv_working_buffer_is_unique_per_call():
    st = {"A": ("__inl3_k", "__inl3_k"), "B": ("__inl5_k", "__inl5_k")}
    s1 = expand_linalg_inv(ast.Name(id="X1", ctx=ast.Store()),
                           [ast.Name(id="A", ctx=ast.Load())], st, local_dtypes={}, fresh_local_allocs={})
    s2 = expand_linalg_inv(ast.Name(id="X2", ctx=ast.Store()),
                           [ast.Name(id="B", ctx=ast.Load())], st, local_dtypes={}, fresh_local_allocs={})
    b1, b2 = _inv_buffer_names(s1), _inv_buffer_names(s2)
    assert b1 and b2 and b1.isdisjoint(b2), f"inv buffers collide: {b1} vs {b2}"
