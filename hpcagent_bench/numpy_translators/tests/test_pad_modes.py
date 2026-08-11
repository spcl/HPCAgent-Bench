"""``np.pad(a, w, mode=...)`` edge / reflect / wrap / symmetric -> a per-axis
boundary index remap, lowered by ``expand_pad`` (the native c / c++ / fortran path).

Each output cell reads the source cell whose index folds ``out - before`` back
into ``[0, d)``: ``edge`` = clamp, ``wrap`` = periodic (mod d), ``symmetric`` =
mirror INCLUDING the edge (period 2d), ``reflect`` = mirror EXCLUDING the edge
(period 2(d-1)).

Validated bit-exact vs numpy, including a pad width larger than the axis
(multi-period wraparound) and BOTH a symbolic-int64 extent (period bound to an
int local) and a literal extent (period folded to a literal) -- the modulus must
match the int64 index kind under Fortran's kind-strict MODULO.

``edge`` also carries a SHAPE assertion (see the emit test at the bottom): its
clamp must lower to a conditional EXPRESSION, not to guard ``if``s. Guard ifs are
data-dependent control flow inside the loop body, which pet/pluto refuses to
schedule -- measured, the whole scop came back with empty statement bodies.
"""
import json
import pathlib
import re
import tempfile

import numpy as np
import pytest

from _op_oracle import _bench_info, run_op
from numpyto_common.frontend import parse_kernel
from numpyto_common.lowering import lower
from numpyto_c.emit import emit_c

_NATIVE = ("c", "cpp", "fortran")


def _assert_ok(res, label):
    fails = {b: s for b, s in res.items() if not (s == "ok" or s.startswith("skip"))}
    assert not fails, f"{label}: {fails}"


@pytest.mark.parametrize("mode", ["edge", "reflect", "wrap", "symmetric"])
@pytest.mark.parametrize("n,w", [(6, 2), (4, 5)])  # w > n exercises the multi-period remap
@pytest.mark.parametrize("symbolic", [True, False])
def test_pad_boundary_mode(mode, n, w, symbolic):
    src = (f"import numpy as np\n"
           f"def pad_op(a, out):\n"
           f"    out[:] = np.pad(a, {w}, mode='{mode}')\n")
    a = np.random.default_rng(0).random((n, ))
    out_shape = (n + 2 * w, )
    label = f"pad-{mode}-n{n}-w{w}-{'sym' if symbolic else 'lit'}"
    if symbolic:
        res = run_op(src,
                     "pad_op", {"a": a}, {"out": out_shape}, {"N": n},
                     shapes={
                         "a": "(N,)",
                         "out": f"(N + {2 * w},)"
                     },
                     backends=_NATIVE)
    else:
        res = run_op(src, "pad_op", {"a": a}, {"out": out_shape}, {}, backends=_NATIVE)
    _assert_ok(res, label)


def test_pad_reflect_size1_axis_repeats():
    # reflect on a size-1 axis has period 0 in numpy -> it just repeats the one
    # element; the lowering guards this (no modulo-by-zero) and returns index 0.
    src = "import numpy as np\ndef pad_op(a, out):\n    out[:] = np.pad(a, 2, mode='reflect')\n"
    a = np.array([7.0])
    _assert_ok(run_op(src, "pad_op", {"a": a}, {"out": (5, )}, {}, backends=_NATIVE), "pad-reflect-size1")


def _emit_c(mode: str) -> str:
    """The C the minimal 1-D pad fixture emits, from ``void pad_op(`` on (the
    preamble's own helpers are full of ``if``s and are not what is asserted)."""
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "k_numpy.py").write_text("import numpy as np\n"
                                  "def pad_op(a, out):\n"
                                  f"    out[:] = np.pad(a, 2, mode='{mode}')\n")
    bi = _bench_info("pad_op", ["a"], ["out"], {"a": "(N,)", "out": "(N + 4,)"}, {"N": 6})
    (d / "bi.json").write_text(json.dumps(bi))
    text = emit_c(lower(parse_kernel(d / "k_numpy.py", d / "bi.json")), fn_name="pad_op")
    return text[text.index("void pad_op("):]


def test_pad_edge_clamp_is_a_conditional_expression_not_control_flow():
    # The property pluto consumes: the clamp is an EXPRESSION, so the pad loop
    # body stays straight-line. Two guard ifs here made pet drop every statement.
    body = _emit_c("edge")
    clamps = re.findall(r"^\s+__ps\d+ = .+;$", body, re.M)
    assert clamps, f"no __ps clamp assign emitted:\n{body}"
    assert all("?" in c for c in clamps), clamps
    assert "if (" not in body, f"data-dependent control flow in the padded-index region:\n{body}"


def test_pad_edge_clamp_never_self_reads_the_index_scalar():
    # Each arm recomputes the pre-clamp index; reading __ps<k> back would add a
    # RAW dependence on top of the WAW the single assign already carries.
    for clamp in re.findall(r"^\s+(__ps\d+) = (.+);$", _emit_c("edge"), re.M):
        assert clamp[0] not in clamp[1], clamp
