"""Integer floor division in a pluto scop is spelled ``floord``, pet's named quasi-affine builtin.

``int_floor`` preprocesses to an opaque ``__npb_floordiv_i`` call, which pet reads in a loop BOUND
as a data-dependent condition and aborts on (POLYCC-008, pagerank). ``floord`` name-matches and
carries the same semantics, so the fix is a spelling; these tests pin both halves of it -- what the
pluto emit writes, and that the prelude still defines the name for the compiler.
"""
import json
import pathlib
import re
import tempfile

import pytest

from _native_tu import build_run_c, have_gcc
from _op_oracle import _bench_info
from numpyto_c.emit import _C_HEADER, emit_c, emit_pluto, pluto_floordiv
from numpyto_common.frontend import parse_kernel
from numpyto_common.lowering import lower

from hpcagent_bench.pluto_affine import KNOWN_POLYCC_ISSUES

#: (a, b) with every sign combination, plus exact division and a zero dividend.
_PAIRS = [(7, 2), (-7, 2), (7, -2), (-7, -2), (8, 4), (-8, 4), (0, 5)]


def _lower_src(src: str, fn: str, shapes, syms, dtypes=None):
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "k_numpy.py").write_text(src)
    bi = _bench_info(fn, ["a"], ["out"], shapes, syms, dtypes)
    (d / "bi.json").write_text(json.dumps(bi))
    return lower(parse_kernel(d / "k_numpy.py", d / "bi.json"))


def _int_bound_kir():
    """``N // 8`` as a loop BOUND: both operands are Python ints (a symbol and a literal)."""
    return _lower_src(
        "import numpy as np\n"
        "def blk_op(a, out):\n"
        "    N, = a.shape\n"
        "    for i in range(N // 8):\n"
        "        out[i] = a[i] * 2.0\n", "blk_op", {
            "a": "(N,)",
            "out": "(N,)"
        }, {"N": 64})


def _float_floordiv_kir():
    """``a[i] // 3.0`` on a float array: the operands are NOT integers."""
    return _lower_src(
        "import numpy as np\n"
        "def flt_op(a, out):\n"
        "    N, = a.shape\n"
        "    for i in range(N):\n"
        "        out[i] = a[i] // 3.0\n", "flt_op", {
            "a": "(N,)",
            "out": "(N,)"
        }, {"N": 64})


def _scop_body(text: str) -> str:
    m = re.search(r"#pragma scop(.*?)#pragma endscop", text, re.S)
    assert m, f"no scop emitted:\n{text}"
    return m.group(1)


def test_integer_floordiv_in_a_scop_is_spelled_floord():
    body = _scop_body(emit_pluto(_int_bound_kir(), fn_name="blk_op"))
    assert "floord(N, 8)" in body, body
    assert "int_floor" not in body, body


def test_the_c_leg_keeps_int_floor():
    """Only the pluto reader cares about the name; the C emit is unchanged."""
    body = emit_c(_int_bound_kir(), fn_name="blk_op")
    assert "int_floor(N, 8)" in body.split("void blk_op", 1)[1], body


def test_float_floordiv_stays_on_the_generic_macro():
    """A float operand must keep ``int_floor``: ``floord`` is the int64 form and would truncate."""
    for text in (emit_pluto(_float_floordiv_kir(), fn_name="flt_op"), emit_c(_float_floordiv_kir(), fn_name="flt_op")):
        body = _scop_body(text) if "#pragma scop" in text else text.split("void flt_op", 1)[1]
        assert "int_floor" in body, body
        assert "floord(" not in body, body


@pytest.mark.parametrize("name,helper", [("floord", "__npb_floordiv_i"), ("ceild", "__npb_ceildiv_i")])
def test_prelude_defines_the_named_builtins_over_the_existing_helpers(name, helper):
    """Guarded, because polycc prepends its own ``#define floord``/``ceild`` (POLYCC-004), and
    delegating rather than restating keeps one definition of the semantics."""
    assert f"#ifndef {name}\nstatic inline int64_t {name}(int64_t a, int64_t b) {{\n    return {helper}(a, b);" \
        in _C_HEADER


@pytest.mark.skipif(not have_gcc(), reason="gcc not installed")
def test_floord_and_ceild_agree_with_the_helpers_they_alias():
    """The spelling claim, executed: same values for both signs, and the guarded block compiles."""
    lines = ["#include <stdio.h>", "int main(void) {"]
    for a, b in _PAIRS:
        lines.append(f'    printf("%lld %lld\\n", (long long)floord((int64_t){a}, (int64_t){b}), '
                     f'(long long)__npb_floordiv_i((int64_t){a}, (int64_t){b}));')
        lines.append(f'    printf("%lld %lld\\n", (long long)ceild((int64_t){a}, (int64_t){b}), '
                     f'(long long)__npb_ceildiv_i((int64_t){a}, (int64_t){b}));')
    lines.append("    return 0;\n}")
    kernel = _C_HEADER + "\nvoid __unused_anchor(void) {}\n"
    result = build_run_c(kernel, "\n".join(lines))
    assert result.returncode == 0, result.stderr
    rows = [r.split() for r in result.stdout.split("\n") if r.strip()]
    assert len(rows) == 2 * len(_PAIRS), result.stdout
    expected = []
    for a, b in _PAIRS:
        expected += [a // b, -((-a) // b)]
    for (got, alias), exp in zip(rows, expected):
        assert int(got) == int(alias) == exp, (rows, expected)


def test_polycc_008_names_the_rule_that_avoids_it():
    entry = KNOWN_POLYCC_ISSUES["POLYCC-008"]
    assert entry.avoided_by == f"numpyto_c.emit.{pluto_floordiv.__name__}"
