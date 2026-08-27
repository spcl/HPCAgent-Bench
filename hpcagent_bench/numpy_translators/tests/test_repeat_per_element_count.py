"""``np.repeat`` with a PER-ELEMENT count (``np.repeat(np.arange(M), np.diff(p))``) is the
standard CSR row-index idiom (spmv: ``row_index`` repeats each row id by its nnz count).
``expand_repeat`` was written for a SCALAR count -- it used the count both as the ``range(K)``
bound and as ``outer * K`` in the destination index, so handed an array it read the array name
as a scalar multiplier: wrong offsets, and it did not even compile.

The fix is a running prefix sum, not ``outer * K``::

    pos = 0
    for i in range(<source extent>):
        for r in range(counts[i]):
            out[pos] = src[i]
            pos += 1

numpy's result length is ``sum(counts)``, which is data -- except when ``counts`` is
``np.diff(p)``, whose sum TELESCOPES to ``p[-1] - p[0]`` (structural, exact, no algebra over
data). ``np.diff`` is never materialised: ``counts[i]`` is expressed inline as ``p[i+1] - p[i]``.
Any other per-element form has a data-dependent sum with no static extent and is refused --
guessing would under-size the buffer (a heap overflow, not a wrong number).
"""
import ast

import numpy as np
import pytest
from _op_oracle import run_op

from numpyto_common.lib_nodes import expand_repeat

_NATIVE = ("c", "cpp", "fortran")


def _expand(src: str, shapes: dict) -> str:
    """Run ``expand_repeat`` over the single ``out = np.repeat(...)`` statement in ``src``."""
    assign = ast.parse(src).body[0]
    shape_table = dict(shapes)
    stmts = expand_repeat(assign.targets[0],
                          assign.value.args,
                          shape_table,
                          assign.value.keywords,
                          local_dtypes={},
                          fresh_local_allocs={})
    return "\n".join(ast.unparse(ast.fix_missing_locations(s)) for s in stmts)


def _assert_ok(res: dict) -> None:
    for backend, status in res.items():
        assert status == "ok" or status.startswith("skip"), f"{backend}: {status}"
    assert any(status == "ok" for status in res.values()), f"all skipped (vacuous): {res}"


def test_per_element_count_uses_running_offset_not_multiply():
    """The destination index is a running scalar offset (``pos``), and the count is the counts
    array INDEXED at the source position (``p[i + 1] - p[i]``) -- never the count array used as a
    bare scalar (the old ``outer * K`` formula's wrong shortcut)."""
    got = _expand("row_index = np.repeat(a, np.diff(p))", {"a": ("M", ), "p": ("M + 1", )})
    assert "= 0" in got and "+= 1" in got, f"no running offset init/advance in:\n{got}"
    assert "p[__rep_i0 + 1] - p[__rep_i0]" in got, f"count not read as p[i+1] - p[i]:\n{got}"
    assert "* p" not in got and "p *" not in got, f"count array used as a scalar multiplier:\n{got}"
    # The written element comes from the SOURCE at the outer index, not the count.
    assert "row_index[" in got and "] = a[__rep_i0]" in got


def test_extent_telescopes_to_diff_endpoints():
    """The result extent (both shape_table and fresh_local_allocs) is ``p[-1] - p[0]``, derived
    structurally from ``p``'s own shape -- never a guess over the count values."""
    shape_table = {"a": ("M", ), "p": ("M + 1", )}
    fresh_local_allocs = {}
    assign = ast.parse("row_index = np.repeat(a, np.diff(p))").body[0]
    expand_repeat(assign.targets[0],
                  assign.value.args,
                  shape_table,
                  assign.value.keywords,
                  local_dtypes={},
                  fresh_local_allocs=fresh_local_allocs)
    assert shape_table["row_index"] == ("p[M + 1 - 1] - p[0]", )
    assert fresh_local_allocs["row_index"] == ("p[M + 1 - 1] - p[0]", )


def test_non_diff_per_element_count_refused():
    """A per-element count with no derivable sum (a bare counts array, not ``np.diff(p)``) must
    NOT guess an extent -- refuse with a clear message naming the offending form."""
    with pytest.raises(NotImplementedError, match="derivable sum"):
        _expand("row_index = np.repeat(a, counts)", {"a": ("M", ), "counts": ("M", )})


def test_scalar_count_still_uses_the_multiply_form():
    """Regression guard: a SCALAR repeat count must keep using ``outer * K`` (unaffected by the
    per-element branch, since ``_iter_extent_of`` on a bare int constant is None)."""
    got = _expand("out = np.repeat(a, 3)", {"a": ("M", )})
    assert "__rep_pos0" not in got
    assert "* 3" in got or "3 *" in got


def test_numeric_row_lengths_from_csr_row_ptr_zero_and_unequal_counts():
    """The idiom that found the gap, exercised end to end: spmv's ``row_index =
    np.repeat(np.arange(M), np.diff(A_indptr))``. ``p``'s diffs are ``2, 0, 3, 4`` -- a ZERO count
    (row 1 has no nonzeros, the case a naive ``outer * K`` formula silently skips) and unequal
    nonzero counts, in the same fixture."""
    p = np.array([0, 2, 2, 5, 9], dtype=np.int64)
    m = p.shape[0] - 1
    expected = np.repeat(np.arange(m), np.diff(p))
    k = int(expected.shape[0])
    src = ("import numpy as np\n\n"
           "def k(p, out):\n"
           "    m = p.shape[0] - 1\n"
           "    row_index = np.repeat(np.arange(m), np.diff(p))\n"
           "    out[:] = row_index\n")
    _assert_ok(
        run_op(src,
               "k", {"p": p}, {"out": (k, )}, {},
               shapes={
                   "p": "(5,)",
                   "out": f"({k},)"
               },
               backends=_NATIVE,
               dtypes={
                   "p": "int64",
                   "out": "int64"
               }))
