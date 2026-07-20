"""A negative integer literal in a ternary must adopt its partner branch's KIND.

Fortran's ``merge(t, f, cond)`` is strict on TYPE *and* KIND. GROMACS'
``ci_sh = ci if ish == 0 else -1`` lowers to ``merge(ci, -1, ...)`` where ``ci``
is an ``integer(c_int64_t)`` local (assigned ``int(cluster_array[i])``) and the
``-1`` literal defaults to int32 -- gfortran rejected the kind clash.

The merge-branch kind promotion only recognised a bare ``ast.Constant`` literal,
but ``-1`` parses as ``UnaryOp(USub, Constant(1))``, so the negative branch fell
through un-kinded. ``_int_literal_value`` now extracts the value from either
form, so the literal is emitted ``(-1_c_int64_t)`` and matches its int64 partner.
"""
import numpy as np

from _op_oracle import run_op

_ALL = ("c", "cpp", "fortran", "numba", "pythran", "jax")


def _all_ok(res):
    return all(v == "ok" or v.startswith("skip") for v in res.values()), res


# ``c = int(idx[i])`` -> an int64 local (the loop iter makes it int64, ``tab[c]``
# makes it int-used); ``s = c if c > 0 else -1`` is the int64-vs-(-1) ternary.
_SRC = ("import numpy as np\n"
        "def f(idx, tab, out):\n"
        " for i in range(len(idx)):\n"
        "  c = int(idx[i])\n"
        "  s = c if c > 0 else -1\n"
        "  out[i] = tab[c] + float(s)\n")


def test_negative_literal_ternary_matches_int64_partner():
    idx = np.array([0, 2, 1, 3, 2, 0, 3, 1], dtype=np.int64)
    tab = np.linspace(10.0, 20.0, 4, dtype=np.float64)
    out = np.zeros(8, dtype=np.float64)
    ok, res = _all_ok(
        run_op(_SRC,
               "f", {
                   "idx": idx,
                   "tab": tab
               }, {"out": (8, )}, {
                   "N": 8,
                   "T": 4
               },
               shapes={
                   "idx": "(N,)",
                   "tab": "(T,)",
                   "out": "(N,)"
               },
               backends=_ALL))
    assert ok, res
    _ = out


def test_merge_branch_kinds_the_negative_literal():
    # Fortran emit: the ``-1`` branch carries the int64 kind suffix, not a bare
    # (int32) literal, so the merge kinds match.
    import json
    import pathlib
    import tempfile
    from numpyto_common.frontend import parse_kernel
    from numpyto_common.lowering import lower
    from numpyto_fortran.emit import emit_fortran
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
                    "N": 8,
                    "T": 4
                }
            },
            "input_args": ["idx", "tab", "out"],
            "array_args": ["idx", "tab", "out"],
            "output_args": ["out"],
            "init": {
                "shapes": {
                    "idx": "(N,)",
                    "tab": "(T,)",
                    "out": "(N,)"
                },
                "dtypes": {
                    "idx": "int64"
                }
            }
        }
    }
    (d / "bi.json").write_text(json.dumps(bi))
    f90 = emit_fortran(lower(parse_kernel(d / "k_numpy.py", d / "bi.json")), fn_name="f")
    assert "merge(" in f90
    # the negative literal is kind-suffixed (int64), never a bare ``-1`` / ``-(1)``.
    assert "1_c_int64_t" in f90
