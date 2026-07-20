"""A multi-dimensional array read must FLATTEN to 1-D pointer arithmetic in C, never chain.

C arrays passed across the ABI are flat ``double *`` pointers, so ``w[i][j]`` is a hard compile
error ("subscripted value is neither array nor pointer"), not a slower-but-correct access. The
emitter flattens ``w[i, j]`` to ``w[i*stride + j]`` using the array's declared shape -- but when the
shape was missing or the wrong rank it silently fell back to the chained ``w[i][j]`` and shipped
uncompilable C. conv_2d hit exactly this: its ``w_box`` was inferred 1-D but indexed 2-D.

The fix is twofold, both pinned here: the read flattens when the rank is known (so conv-style
kernels emit and run), and the emitter RAISES rather than emit the chained form when it cannot.
"""
import json
import pathlib
import tempfile

import numpy as np
import pytest
from _op_oracle import _bench_info, run_op

from numpyto_common.frontend import parse_kernel
from numpyto_common.lowering import lower
from numpyto_c.emit import emit_c

_NATIVE = ("c", "cpp", "fortran")

# A conv-style body: a 2-D weight read producing a scalar, times a shifted slice of a 2-D grid --
# the exact shape conv_2d has, minus the np.pad boundary handling.
_WEIGHTED_STENCIL = ("import numpy as np\n"
                     "def f(g, w, out):\n"
                     "    n = g.shape[0]\n"
                     "    for di in range(3):\n"
                     "        for dj in range(3):\n"
                     "            c = w[di, dj]\n"
                     "            for i in range(n):\n"
                     "                for j in range(n):\n"
                     "                    out[i, j] = out[i, j] + c * g[i, j]\n")


def _emit_c_source(body, shapes, syms):
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "k.py").write_text(body)
    (d / "bi.json").write_text(
        json.dumps(_bench_info("f", ["g", "w"], ["out"], shapes, syms, {a: "float64"
                                                                        for a in ("g", "w", "out")})))
    return emit_c(lower(parse_kernel(d / "k.py", d / "bi.json")), fn_name="f")


def test_two_d_weight_read_flattens_and_does_not_chain():
    src = _emit_c_source(_WEIGHTED_STENCIL, {"g": "(n, n)", "w": "(3, 3)", "out": "(n, n)"}, {"n": 4})
    # The load of w must be flattened (w[.. * 3 + ..]) -- never a chained w[..][..].
    w_reads = [ln for ln in src.splitlines() if "w[" in ln and "w_" not in ln.replace("w[", "@")]
    assert w_reads, "expected a read of w in the emitted C"
    assert not any("][" in ln for ln in src.splitlines()), \
        "emitted C multi-subscripts a flat pointer (w[i][j]); it must flatten to w[i*stride + j]"


def test_weighted_stencil_matches_numpy_on_every_native_backend():
    g = np.arange(16, dtype=np.float64).reshape(4, 4)
    w = (np.arange(9, dtype=np.float64) * 0.1).reshape(3, 3)
    res = run_op(_WEIGHTED_STENCIL,
                 "f", {
                     "g": g,
                     "w": w
                 }, {"out": (4, 4)}, {"n": 4},
                 shapes={
                     "g": "(n, n)",
                     "w": "(3, 3)",
                     "out": "(n, n)"
                 },
                 dtypes={
                     "g": "float64",
                     "w": "float64",
                     "out": "float64"
                 },
                 backends=_NATIVE)
    for backend, status in res.items():
        assert status == "ok" or status.startswith("skip"), f"{backend}: {status}"
    assert any(v == "ok" for v in res.values()), res


def test_multi_index_without_a_matching_rank_raises_not_chains():
    """The hardening: a 2-D index of an array declared 1-D must fail loudly, not emit chained C.

    This is the state conv_2d shipped in -- w_box declared/inferred 1-D, read 2-D. The emitter used
    to emit ``w[i][j]`` on a flat pointer; it must raise instead so the gap is caught at emit time.
    """
    body = ("import numpy as np\n"
            "def f(w, out):\n"
            "    for i in range(3):\n"
            "        for j in range(3):\n"
            "            out[i] = w[i, j]\n")
    with pytest.raises(NotImplementedError, match="flatten"):
        _emit_c_source(body, {"w": "(3,)", "out": "(3,)"}, {"n": 3})
