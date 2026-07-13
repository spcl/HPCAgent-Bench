"""``int(x)`` is a value-CHANGING truncation, not a droppable no-op.

The GROMACS NBNxM kernel computes a floating-point distance chain and then uses
``ri = int(rs)`` as a table index::

    rsq  = dx*dx + dy*dy + dz*dz     # float
    rinv = 1.0 / sqrt(rsq) * skip    # float
    rs   = rsq * rinv * tab_scale    # float, feeds int()
    ri   = int(rs)                   # TRUNCATION -> table index (int)

Two coupled bugs made every force come out zero:

1. The lowering dropped ``int(x)`` entirely (relying on the target being
   int-declared to truncate implicitly). That erased the barrier the
   used-as-int analysis needs: from the index ``ri`` it walked BACKWARD across
   ``ri = int(rs)`` into ``rs`` and mistyped the whole chain (``rsq`` / ``rinv``
   / ``dx``) as integer, so each sub-1.0 coordinate difference truncated to 0.
2. The backward int-ness closure had no ``pure_int_arith`` guard on that step.

``int(x)`` is now KEPT (rendered ``(int64_t)(x)`` / ``INT(x, kind)``) and the closure
is bounded, so the float chain stays ``double`` and the result is bit-exact. This
pins both the numerical result and the emitted C types.
"""
import ast
import json
import pathlib
import tempfile

import numpy as np

from _op_oracle import run_op

_ALL = ("c", "cpp", "fortran", "numba", "pythran", "jax")

# rs = rsq * rinv * 50 = 50 * sqrt(rsq) = 50 * |x - 0.5|; for x in [0.6, 0.9]
# rs lands in [5, 20], so ri = int(rs) indexes a 24-entry table in range.
_SRC = ("import numpy as np\n"
        "def f(x, table, out):\n"
        " for i in range(len(x)):\n"
        "  d = x[i] - 0.5\n"
        "  rsq = d * d\n"
        "  rinv = 1.0 / np.sqrt(rsq)\n"
        "  rs = rsq * rinv * 50.0\n"
        "  ri = int(rs)\n"
        "  ri = min(max(ri, 0), len(table) - 2)\n"
        "  frac = rs - float(ri)\n"
        "  val = (1.0 - frac) * table[ri] + frac * table[ri + 1]\n"
        "  out[i] = rsq + rinv + val\n")


def _all_ok(res):
    return all(v == "ok" or v.startswith("skip") for v in res.values()), res


def test_int_truncation_keeps_float_chain_bit_exact():
    # Sub-1.0 coordinate differences: if the chain were int-typed they would
    # truncate to 0 (rsq=0 -> rinv=inf), so a wrong result is loud.
    x = np.linspace(0.6, 0.9, 12, dtype=np.float64)
    table = np.linspace(1.0, 2.0, 24, dtype=np.float64)
    out = np.zeros(12, dtype=np.float64)
    ok, res = _all_ok(
        run_op(_SRC,
               "f", {
                   "x": x,
                   "table": table
               }, {"out": (12, )}, {
                   "N": 12,
                   "T": 24
               },
               shapes={
                   "x": "(N,)",
                   "table": "(T,)",
                   "out": "(N,)"
               },
               backends=_ALL))
    assert ok, res
    _ = out


def _emit_c(src):
    from numpyto_common.frontend import parse_kernel
    from numpyto_common.lowering import lower
    from numpyto_c.emit import emit_c
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "k_numpy.py").write_text(src)
    bi = {
        "benchmark": {
            "name": "k",
            "short_name": "k",
            "relative_path": "",
            "module_name": "k",
            "func_name": "f",
            "parameters": {
                "S": {
                    "N": 12,
                    "T": 24
                }
            },
            "input_args": ["x", "table", "out"],
            "array_args": ["x", "table", "out"],
            "output_args": ["out"],
            "init": {
                "shapes": {
                    "x": "(N,)",
                    "table": "(T,)",
                    "out": "(N,)"
                }
            }
        }
    }
    (d / "bi.json").write_text(json.dumps(bi))
    return emit_c(lower(parse_kernel(d / "k_numpy.py", d / "bi.json")), fn_name="f")


def test_float_chain_declared_double_and_int_cast_kept():
    c = _emit_c(_SRC)
    # the float-chain scalars must NOT be int-typed.
    for var in ("d", "rsq", "rinv", "rs"):
        assert f"double {var};" in c, f"{var} should be double, not int:\n{c}"
    # the index truncation is rendered as an explicit cast, not dropped. The
    # canonical integer is int64, so the cast is ``(int64_t)`` (a 32-bit ``int``
    # would truncate a value past 2^31).
    assert "(int64_t)(" in c
    # ``ri`` (the actual index) stays integer-typed -- int64, the ABI integer.
    assert "int64_t ri;" in c


def test_int_call_is_not_dropped_in_lowering():
    # Unit-level: the builtin-cast rewriter keeps ``int(...)`` (drops only
    # ``float(...)``) so the used-as-int barrier survives lowering.
    from numpyto_common.lowering import _BuiltinCastRewriter
    mod = ast.parse("def f(rs):\n ri = int(rs)\n y = float(rs)\n return ri + y\n")
    _BuiltinCastRewriter().visit(mod)
    src = ast.unparse(mod)
    assert "int(rs)" in src  # truncation preserved
    assert "float(rs)" not in src  # no-op cast dropped
