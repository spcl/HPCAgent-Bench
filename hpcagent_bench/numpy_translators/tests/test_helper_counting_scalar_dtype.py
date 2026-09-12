"""A kept helper that counts or indexes with a scalar parameter declares it an integer.

A parameter's dtype is inferred from the CALL-SITE argument, and an argument the resolver cannot
type falls through to ``float64``. An element of a kernel LOCAL array is exactly that: spgemm_hash
passes ``row_bin[row]`` and ``ts`` (itself bound from another helper's call), both ``int64``, and
both arrived as ``double``. Inlined that cost nothing -- the body was spliced into the caller and
the value kept its own type -- but as a kept ``@dc.program`` the body then counts and subscripts
with a double, and g++ refuses the generated code with ``invalid types 'int64_t*[double]' for array
subscript``. C and gfortran say the same thing in their own words.
"""

import ast
import json
import pathlib
import tempfile

from numpyto_c.dace_emit import emit_dace
from numpyto_common.frontend import parse_kernel, widen_counting_scalar_params
from numpyto_common.ir import ScalarDesc

#: ``_table_size`` counts with ``b``; the call site passes an element of a kernel LOCAL, which is
#: the argument shape no resolver types. The reference is spgemm_hash's own bin-size helper.
COUNTING_HELPER = """import numpy as np


def _table_size(b):
    ts = 32
    for _ in range(b):
        ts = ts * 2
    return ts


def k(a, out):
    bins = np.zeros(n, dtype=np.int64)
    for i in range(n):
        bins[i] = a[i]
    for i in range(n):
        out[i] = _table_size(bins[i])
"""

#: ``_pick`` indexes with ``j`` rather than counting with it -- the other half of the same defect.
INDEXING_HELPER = """import numpy as np


def _pick(table, j):
    if j < 0:
        return 0
    return table[j]


def k(a, out):
    bins = np.zeros(n, dtype=np.int64)
    for i in range(n):
        bins[i] = a[i]
    for i in range(n):
        out[i] = _pick(bins, bins[i])
"""


def emitted(source: str) -> str:
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "k_numpy.py").write_text(source)
    bench = {
        "name": "k",
        "short_name": "k",
        "relative_path": ".",
        "module_name": "k",
        "func_name": "k",
        "dwarf": "d",
        "level": 3,
        "parameters": {"S": {"n": 8}},
        "input_args": ["a", "out", "n"],
        "array_args": ["a", "out"],
        "output_args": ["out"],
        "init": {"shapes": {"a": "(n,)", "out": "(n,)"}, "dtypes": {"a": "int64", "out": "int64"}},
    }
    (d / "k.json").write_text(json.dumps({"benchmark": bench}))
    return emit_dace(parse_kernel(d / "k_numpy.py", d / "k.json"))


def annotation_of(src: str, helper: str, param: str) -> str:
    fn = next(f for f in ast.parse(src).body if isinstance(f, ast.FunctionDef) and f.name == helper)
    arg = next(a for a in fn.args.args if a.arg == param)
    return ast.unparse(arg.annotation)


def test_a_range_bound_parameter_is_declared_int() -> None:
    src = emitted(COUNTING_HELPER)
    assert "def _table_size(" in src, f"the helper was inlined, so this proves nothing:\n{src}"
    assert annotation_of(src, "_table_size", "b") == "dc.int64", src


def test_a_subscript_index_parameter_is_declared_int() -> None:
    src = emitted(INDEXING_HELPER)
    assert "def _pick(" in src, f"the helper was inlined, so this proves nothing:\n{src}"
    assert annotation_of(src, "_pick", "j") == "dc.int64", src


def test_a_scalar_the_body_only_compares_keeps_its_inferred_dtype() -> None:
    """The widening reads evidence, not a hunch: a parameter nothing counts with is left alone."""
    hfn = ast.parse("def h(x, y):\n    if x > y:\n        return x\n    return y\n").body[0]
    scalars = [ScalarDesc(name="x", dtype="float64"), ScalarDesc(name="y", dtype="float64")]
    widen_counting_scalar_params(hfn, scalars)
    assert [s.dtype for s in scalars] == ["float64", "float64"]
