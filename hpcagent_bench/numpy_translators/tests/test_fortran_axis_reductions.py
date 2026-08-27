# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A reduction over ONE axis reaches Fortran as ``SUM(a, dim=k)``, not as a loop nest.

The whole-array case was the easy half: it returns a scalar, so nothing downstream has to shape a
result. Per-axis is where the mapping can go wrong and where the win is -- 115 corpus kernels carry
an axis reduction.

numpy and Fortran count axes in opposite directions. An array whose numpy shape is ``(d0, d1, d2)``
is DECLARED ``(d2, d1, d0)`` by this backend, because Fortran is column-major -- so numpy axis ``k``
of a rank-``n`` array is Fortran ``dim = n - k``. Passing numpy's number straight through reduces a
DIFFERENT axis: it compiles, and it returns a wrong array of the right shape. That is why the tests
below pin the ``dim=`` number for each axis of a rank-3 operand AND run the numbers -- a shape-only
check cannot tell the two apart when the extents happen to match.

The claim is decided in lowering, with the shape table in hand, because past that point the loop
nest is gone and there is nothing to fall back to. So the declines are pinned too: a runtime axis, a
tuple axis, ``keepdims``, an operand of unknown rank.
"""
import ast
import json
import pathlib
import re
import tempfile

import numpy as np

import _op_oracle as oo
from _op_oracle import run_op

from numpyto_common.frontend import parse_kernel
from numpyto_common.lowering import lower
from numpyto_fortran.emit import emit_fortran
from numpyto_fortran.intrinsics import literal_axis, renders_natively

NATIVE = ("c", "cpp", "fortran")

#: Floating: the claim needs positive evidence of a float element type, because numpy and Fortran
#: disagree on integer overflow and an integer ``mean`` would become integer division.
_DTYPES = {"a": "float64"}

_SHAPES3 = {"a": "(N, M, K)", "out": "(N,)"}
_SYMS3 = {"N": 6, "M": 4, "K": 3}


def build(body: str, shapes, syms):
    src = "import numpy as np\ndef f(a, out):\n" + body
    d = pathlib.Path(tempfile.mkdtemp())
    npy = d / "f.py"
    npy.write_text(src)
    bi = d / "bi.json"
    bi.write_text(json.dumps(oo._bench_info("f", ["a"], ["out"], shapes, syms)))
    return src, npy, bi


def fortran(body: str, shapes=None, syms=None) -> str:
    _, npy, bi = build(body, shapes or _SHAPES3, syms or _SYMS3)
    return emit_fortran(lower(parse_kernel(npy, bi), native_call=renders_natively), fn_name="f")


#: ``a`` is rank 3, so numpy axis k must become Fortran ``dim=3-k``. Reducing two axes leaves a
#: rank-1 result the kernel can write out, which keeps every case one shape.
_AXIS_TO_DIM = {0: 3, 1: 2, 2: 1}


def test_each_numpy_axis_maps_to_its_reversed_fortran_dim() -> None:
    for axis, dim in _AXIS_TO_DIM.items():
        text = fortran(f"    c = np.sum(a, axis={axis})\n    out[:] = c[:, 0] * 2.0\n")
        assert f"SUM(a, dim={dim})" in text, (axis, text)


def test_a_negative_axis_resolves_against_the_rank_first() -> None:
    """``axis=-1`` is numpy axis 2 on a rank-3 operand, so ``dim=1`` -- the fastest-varying one."""
    text = fortran("    c = np.sum(a, axis=-1)\n    out[:] = c[:, 0] * 2.0\n")
    assert "SUM(a, dim=1)" in text, text


def test_every_per_axis_reduction_emits_its_intrinsic() -> None:
    want = {
        "np.sum(a, axis=1)": "SUM(a, dim=2)",
        "np.prod(a, axis=1)": "PRODUCT(a, dim=2)",
        "np.mean(a, axis=1)": "SUM(a, dim=2)",
    }
    for call, marker in want.items():
        text = fortran(f"    c = {call}\n    out[:] = c[:, 0] * 2.0\n")
        assert marker in text, (call, text)


def test_the_mean_divides_by_the_reduced_axis_not_the_whole_array() -> None:
    """``SIZE(a)`` is the element count of ALL of it; the per-axis mean divides by ONE extent."""
    text = fortran("    c = np.mean(a, axis=1)\n    out[:] = c[:, 0] * 2.0\n")
    assert "SIZE(a, 2)" in text, text


#: A ``do`` opening a loop, so a declined case can be shown to still lower to one.
_DO_RE = re.compile(r"^\s*do\s", re.IGNORECASE | re.MULTILINE)


def test_a_tuple_axis_still_lowers_to_loops() -> None:
    """Fortran's ``dim=`` reduces exactly one axis; a tuple is not what this mapping expresses."""
    text = fortran("    c = np.sum(a, axis=(1, 2))\n    out[:] = c * 2.0\n")
    assert _DO_RE.search(text), text


def test_keepdims_still_lowers_to_loops() -> None:
    """``keepdims`` decides the result RANK and no ``dim=`` reduction preserves it."""
    for call in ("np.sum(a, axis=1, keepdims=True)", "np.max(a, axis=1, keepdims=True)"):
        node = ast.parse(call, mode="eval").body
        assert literal_axis(node) is None, call


def test_the_claim_needs_a_known_rank() -> None:
    """Decided in lowering because there is no loop left to fall back to at emit.

    An operand the shape table does not know, or an axis out of range for the rank it does know,
    would leave a call the emitter cannot map -- a refused kernel rather than a slower one.
    """
    node = ast.parse("np.sum(a, axis=1)", mode="eval").body
    assert not renders_natively(("np", "sum"), node, {}, _DTYPES)
    assert not renders_natively(("np", "sum"), node, {"a": ("N", )}, _DTYPES)
    assert renders_natively(("np", "sum"), node, {"a": ("N", "M")}, _DTYPES)


def test_a_runtime_axis_is_not_claimed() -> None:
    """One emitted artifact serves every preset, so a runtime axis picks its nest at run time."""
    node = ast.parse("np.sum(a, axis=k)", mode="eval").body
    assert not renders_natively(("np", "sum"), node, {"a": ("N", "M", "K")}, _DTYPES)


def test_per_axis_reductions_match_numpy_on_every_backend() -> None:
    """The mapping is what these catch: a wrong ``dim`` reduces the wrong axis and still compiles.

    ``M`` and ``K`` differ from ``N`` and from each other, so a reduction over the wrong axis
    cannot accidentally produce a conformable result.
    """
    rng = np.random.default_rng(0)
    a = rng.standard_normal((6, 4, 3)) + 2.0
    for axis in (0, 1, 2, -1):
        src, _, _ = build(f"    c = np.sum(a, axis={axis})\n    out[:] = c[:, 0] * 2.0\n", _SHAPES3, _SYMS3)
        out_shape = (4, ) if axis == 0 else (6, )
        status = run_op(src,
                        "f", {"a": a}, {"out": out_shape},
                        _SYMS3,
                        shapes={
                            "a": "(N, M, K)",
                            "out": "(M,)" if axis == 0 else "(N,)"
                        },
                        backends=NATIVE)
        assert status == {"c": "ok", "cpp": "ok", "fortran": "ok"}, (axis, status)


def test_the_mean_matches_numpy_on_every_backend() -> None:
    rng = np.random.default_rng(1)
    a = rng.standard_normal((6, 4, 3)) + 2.0
    src, _, _ = build("    c = np.mean(a, axis=1)\n    out[:] = c[:, 0] * 2.0\n", _SHAPES3, _SYMS3)
    assert run_op(src, "f", {"a": a}, {"out": (6, )}, _SYMS3, shapes=_SHAPES3, backends=NATIVE) == {
        "c": "ok",
        "cpp": "ok",
        "fortran": "ok"
    }
