"""``x.shape[k]`` is rewritten to the extent the manifest declares for ``x``.

The emitted kernel has no descriptor beside its buffers: the extents cross the ABI as named
symbols and nothing on the other side can ask an array how long it is. Every shape read must
therefore be gone before an emitter runs, and it always can be -- the manifest declares a shape
for every array in the corpus, and the resolver carries that shape through allocations, aliases
and slices to whichever local does the reading.

Left in place the read does not fail loudly; it forks the SPELLING of an extent. ls3df_scf's ``v``
is ``(Lb, Lb, Lb)`` going in and ``(vcol.shape[0], nb1, nb2)`` coming back out of ``hpsi``, and
lowering's rebind check then sees one name bound to two shapes and refuses a kernel that has only
ever had one.
"""
import ast

from numpyto_common.frontend import (ArrayDesc, _apply_subscript_axes, _parse_shape_expression, resolve_shape_reads)


def _fn(src):
    return ast.parse(src).body[0]


def _env(**shapes):
    return {n: ArrayDesc(name=n, dtype="float64", shape=_parse_shape_expression(s)) for n, s in shapes.items()}


def _axes(dims, expr):
    """Result dims of subscripting ``dims`` with the subscript written in ``expr``."""
    return _apply_subscript_axes(list(dims), ast.parse(expr, mode="eval").body.slice)


def test_ellipsis_stands_for_the_axes_left_unindexed():
    # Read positionally, ``...`` is just another non-Slice and lands on the wrong end of the
    # array: ls3df_scf's ``psi_frag[f][..., 0]`` selects the first state at every grid point and
    # came back as the first two axes instead. Wrong rank AND wrong extents, reported by nothing.
    assert _axes(("Lb", "Lb", "Lb", "nstate"), "a[..., 0]") == ["Lb", "Lb", "Lb"]
    assert _axes(("Lb", "Lb", "Lb", "nstate"), "a[0, ...]") == ["Lb", "Lb", "nstate"]
    assert _axes(("n", "m"), "a[...]") == ["n", "m"]
    # An explicit index still drops its own axis, and a full slice still keeps it.
    assert _axes(("n", "m", "k"), "a[:, 0, ...]") == ["n", "k"]


def test_a_chain_of_subscripts_resolves_like_one():
    # ``psi_frag[f]`` picks a fragment and ``[..., 0]`` its first state. Stopping the resolver at
    # a Name base left the whole chain unresolved, so every extent read off the local it binds
    # stayed spelled ``local.shape[k]``.
    fn = _fn("def f(psi_frag, out):\n"
             " v = psi_frag[0][..., 0]\n"
             " out[:] = v.shape[0] + v.shape[2]\n")
    assert resolve_shape_reads(fn, _env(psi_frag="(nfrag, Lb, Lb, Lb, nstate)", out="(1,)")) == []
    assert "out[:] = Lb + Lb" in ast.unparse(fn)


def test_a_read_on_a_local_allocation_resolves_through_it():
    fn = _fn("def f(x, out):\n"
             " buf = np.zeros((x.shape[1], 4), dtype=np.float64)\n"
             " out[:] = buf.shape[0]\n")
    assert resolve_shape_reads(fn, _env(x="(n, m)", out="(1,)")) == []
    assert "out[:] = m" in ast.unparse(fn)


def test_a_negative_axis_counts_from_the_end():
    fn = _fn("def f(x, out):\n out[:] = x.shape[-1]\n")
    assert resolve_shape_reads(fn, _env(x="(n, m, k)", out="(1,)")) == []
    assert "out[:] = k" in ast.unparse(fn)


def test_an_unresolvable_read_is_reported_and_left_alone():
    # Nothing is guessed. An extent that does not resolve is handed back to the caller by name,
    # and the read stays exactly as written so the pass that owns its refusal still sees it.
    fn = _fn("def f(x, out):\n out[:] = q.shape[0]\n")
    assert resolve_shape_reads(fn, _env(x="(n,)", out="(1,)")) == ["q.shape[0]"]
    assert "q.shape[0]" in ast.unparse(fn)


def test_an_axis_past_the_rank_is_reported_not_wrapped():
    fn = _fn("def f(x, out):\n out[:] = x.shape[3]\n")
    assert resolve_shape_reads(fn, _env(x="(n, m)", out="(1,)")) == ["x.shape[3]"]


def test_a_scalar_local_bound_once_to_an_extent_is_that_extent():
    # ``nb0, nb1, nb2 = v.shape`` leaves three locals that are second names for extents the ABI
    # already carries. The buffer allocated from them is then described as ``(nb0, nb1, nb2)``
    # while the same buffer coming the other way is ``(Lb, Lb, Lb)`` -- one shape, two spellings,
    # and the rebind check reads that as two shapes.
    fn = _fn("def f(v, out):\n"
             " nb0 = v.shape[0]\n"
             " nb1 = v.shape[1]\n"
             " vcol = np.zeros((nb0, nb1, 1), dtype=np.float64)\n"
             " out[:] = vcol.shape[0]\n")
    assert resolve_shape_reads(fn, _env(v="(Lb, Lb)", out="(1,)")) == []
    src = ast.unparse(fn)
    assert "np.zeros((Lb, Lb, 1)" in src
    assert "out[:] = Lb" in src


def test_a_rebound_scalar_is_not_folded():
    # Only a local bound ONCE to a declared extent is interchangeable with it. A counter that
    # happens to start at an extent is not, and substituting it would move the loop bound.
    fn = _fn("def f(x, out):\n"
             " k = x.shape[0]\n"
             " k = k - 1\n"
             " out[:] = k\n")
    resolve_shape_reads(fn, _env(x="(n,)", out="(1,)"))
    src = ast.unparse(fn)
    assert "k = n" in src and "k = k - 1" in src and "out[:] = k" in src


def test_a_name_read_inside_a_store_target_is_not_a_rebinding():
    # ``row[i % nb0] += w`` writes ``row`` and only READS ``nb0``. Counting every name in the
    # target subtree made the index look rebound, and the extent local stopped folding.
    fn = _fn("def f(x, row, out):\n"
             " nb0 = x.shape[0]\n"
             " row[0, 0 % nb0] += 1.0\n"
             " out[:] = nb0\n")
    resolve_shape_reads(fn, _env(x="(n,)", row="(n, n)", out="(1,)"))
    src = ast.unparse(fn)
    assert "row[0, 0 % n] += 1.0" in src
    assert "out[:] = n" in src


def test_a_bounded_or_strided_slice_is_sized_not_passed_through():
    # A slice axis used to keep the SOURCE extent, which is right only for a whole-axis slice.
    # raman_fitting reads ``centres.shape[0]`` off ``p[0:3 * npeaks:3]``: passed through it came
    # back the full ``3 * npeaks + 1``, so the jacobian was allocated three times over and strided
    # against a count it does not have. Wrong numbers, and nothing downstream reports it.
    assert _axes(("M", ), "a[0:3 * npeaks:3]") == ["((3 * npeaks) + 2) // 3"]
    assert _axes(("M", ), "a[:npeaks]") == ["npeaks"]
    assert _axes(("M", ), "a[2:7]") == ["(7) - (2)"]
    assert _axes(("M", ), "a[::2]") == ["((M) + 1) // 2"]
    assert _axes(("M", ), "a[:-1]") == ["(M) - 1"]
    # A whole-axis slice is the one case where the source extent IS the answer, and it stays the
    # very object it was handed -- callers pass AST exprs through this untouched.
    assert _axes(("M", "N"), "a[:, :]") == ["M", "N"]


def test_a_step_this_cannot_size_refuses_the_whole_shape():
    # A symbolic or reversed step has no ceiling form here, and answering with the source extent
    # would be a wrong number presented as a resolved one. Empty == the callers' "unresolved".
    assert _axes(("M", ), "a[::k]") == []
    assert _axes(("M", ), "a[::-1]") == []
    # One unsizable axis takes the whole shape with it -- a partially-right shape is not a shape.
    assert _axes(("M", "N"), "a[::k, :]") == []


def test_a_rank_zero_shape_read_is_reported_not_folded_to_an_empty_tuple():
    # ``()`` is a resolver out of evidence, not a rank-0 array. Folded, the ``[k]`` beside the read
    # becomes ``()[k]`` -- an index off the end of a tuple that no emitter has a form for and no
    # reader can trace back to the array it came from.
    fn = _fn("def f(x, out):\n"
             " s = x[0].shape\n"
             " out[:] = s[0]\n")
    assert resolve_shape_reads(fn, _env(x="(n,)", out="(1,)")) != []
    assert "()" not in ast.unparse(fn)
