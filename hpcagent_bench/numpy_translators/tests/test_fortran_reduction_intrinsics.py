# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A whole-array reduction reaches Fortran as an intrinsic, never as a loop nest.

Lowering expands a numpy call to explicit loops for every target. That is the only choice C has,
and in Fortran it throws away the compiler's own ``SUM`` / ``MAXVAL`` / ``PRODUCT`` -- vectorized,
and self-documenting where a loop nest is anonymous. ``_emit_call`` could already render every one
of these; nothing ever reached it, because lowering had consumed the call first.

The Fortran driver now hands ``lower()`` :func:`numpyto_fortran.intrinsics.renders_natively`, so the
claimed calls survive to emit. The test is in two halves and needs both: the emitted TEXT must be
the intrinsic (an equally-correct loop would pass a numbers-only test and defeat the point), and the
numbers must still match numpy (an intrinsic spelled wrong compiles fine).

C is asserted unchanged in the same file. The predicate is off by default, and a C backend that
started skipping expansion would emit a call it has no rendering for.
"""
import ast
import json
import pathlib
import re
import tempfile

import numpy as np

import _op_oracle as oo
from _op_oracle import run_op

from numpyto_c.emit import emit_c
from numpyto_common.frontend import parse_kernel
from numpyto_common.lowering import lower
from numpyto_fortran.emit import emit_fortran
from numpyto_fortran.intrinsics import WHOLE_ARRAY_REDUCTIONS, renders_natively

NATIVE = ("c", "cpp", "fortran")

_SHAPES = {"a": "(N, M)", "out": "(N,)"}
_SYMS = {"N": 8, "M": 4}

#: numpy call -> the Fortran intrinsic its emission must contain. ``mean`` has no single intrinsic
#: and is a composite, but still an intrinsic EXPRESSION. ``norm`` uses ``NORM2``, which scales its
#: operand internally: ``SQRT(SUM(x ** 2))`` returns inf for a vector whose squares overflow.
#: ``max``/``min`` are NOT here: numpy propagates NaN and ``MAXVAL``/``MINVAL`` do not, so the
#: intrinsic would answer with a non-NaN element. ``any``/``all``/``count_nonzero`` are not here
#: either: their result is ``LOGICAL`` (or an integer count) while the hoisted temp is declared
#: ``real``, and ``COUNT(m /= 0)`` on a ``LOGICAL`` operand is not a legal comparison.
_REDUCTIONS = {
    "np.sum(a)": "SUM(",
    "np.prod(a)": "PRODUCT(",
    "np.mean(a)": "SUM(",
    "np.linalg.norm(a)": "NORM2(",
}

#: Reductions Fortran has an intrinsic for and this backend still DECLINES, each for a measured
#: disagreement rather than a missing rendering. Pinned so a future widening has to face them.
_DISAGREEING = {
    "np.max(a)": "MAXVAL",
    "np.min(a)": "MINVAL",
    "np.any(a)": "ANY",
    "np.all(a)": "ALL",
    "np.count_nonzero(a)": "COUNT",
}


def build(call: str):
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           f"    s = {call}\n"
           "    out[:] = a[:, 0] * s\n")
    d = pathlib.Path(tempfile.mkdtemp())
    npy = d / "f.py"
    npy.write_text(src)
    bi = d / "bi.json"
    bi.write_text(json.dumps(oo._bench_info("f", ["a"], ["out"], _SHAPES, _SYMS)))
    return src, npy, bi


def fortran(call: str) -> str:
    _, npy, bi = build(call)
    return emit_fortran(lower(parse_kernel(npy, bi), native_call=renders_natively), fn_name="f")


#: A ``do`` opening a loop, so the reduction's own absence of one can be asserted.
_DO_RE = re.compile(r"^\s*do\s", re.IGNORECASE | re.MULTILINE)


def test_each_whole_array_reduction_emits_its_intrinsic() -> None:
    for call, want in _REDUCTIONS.items():
        text = fortran(call)
        assert want in text, (call, text)


def test_the_reduction_temp_is_assigned_in_one_statement() -> None:
    """The whole point: one line, not a loop nest.

    The reduction lands in a hoisted temp, so the check is that the temp's ASSIGNMENT is a single
    statement naming the intrinsic. Loops elsewhere in the kernel (the ``out[:]`` write) are fine.
    """
    for call in _REDUCTIONS:
        text = fortran(call)
        assigns = [ln.strip() for ln in text.splitlines() if re.match(r"^\s*x_cb\d+ = ", ln)]
        assert len(assigns) == 1, (call, assigns, text)
        assert "(" in assigns[0], (call, assigns[0])


def test_a_reduction_carrying_an_axis_still_lowers_to_loops() -> None:
    """``dim=`` counts in Fortran's axis order, not numpy's, so an axis is not claimed.

    Getting that mapping wrong is a wrong answer that compiles, which is why the predicate declines
    every call with a second argument rather than trying.
    """
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           "    c = np.sum(a, axis=1)\n"
           "    out[:] = c * 2.0\n")
    d = pathlib.Path(tempfile.mkdtemp())
    npy = d / "f.py"
    npy.write_text(src)
    bi = d / "bi.json"
    bi.write_text(json.dumps(oo._bench_info("f", ["a"], ["out"], _SHAPES, _SYMS)))
    text = emit_fortran(lower(parse_kernel(npy, bi), native_call=renders_natively), fn_name="f")
    assert _DO_RE.search(text), text


#: A rank-2 operand, so an axis-carrying call has a rank to resolve against when one is offered.
_SHAPE_TABLE = {"a": ("N", "M")}
#: Floating, because the claim needs POSITIVE evidence of a float element type.
_DTYPE_TABLE = {"a": "float64"}


def test_the_disagreeing_reductions_stay_on_loops() -> None:
    """Each of these has an intrinsic, and each would answer a DIFFERENT question than numpy.

    NaN is the one that matters most: ``np.max`` propagates it, ``MAXVAL`` returns the largest
    non-NaN element instead. That is a wrong number with no diagnostic, so the loop lowering -- which
    emits the NaN-faithful comparison -- stays.
    """
    for call, intrinsic in _DISAGREEING.items():
        text = fortran(call)
        assert intrinsic + "(" not in text, (call, text)


def test_the_whole_array_claim_needs_exactly_one_operand() -> None:
    """These renderings reduce ALL of the array, so a second argument asks a different question.

    ``keepdims`` changes the result rank and ``norm``'s ``ord`` names a different norm entirely.
    (An ``axis`` is not refused outright any more -- it routes to the per-axis ``dim=`` form, pinned
    in ``test_fortran_axis_reductions.py``. ``linalg.norm`` has no per-axis form and stays refused.)
    """
    for call in ("np.sum(a, keepdims=True)", "np.linalg.norm(a, 1)", "np.linalg.norm(a, axis=1)"):
        node = ast.parse(call, mode="eval").body
        key = ("np", "linalg.norm" if "norm" in call else node.func.attr)
        assert not renders_natively(key, node, _SHAPE_TABLE, _DTYPE_TABLE), call


def test_the_predicate_declines_an_op_with_no_intrinsic() -> None:
    """``np.cumsum`` has no Fortran intrinsic; claiming it would leave a call the emitter refuses."""
    node = ast.parse("np.cumsum(a)", mode="eval").body
    assert not renders_natively(("np", "cumsum"), node, _SHAPE_TABLE, _DTYPE_TABLE)
    assert ("np", "cumsum") not in WHOLE_ARRAY_REDUCTIONS


def test_c_still_lowers_every_one_of_them_to_loops() -> None:
    """The predicate is off by default. C has no array intrinsic to fall back on, so a call left
    unexpanded there is a refusal, not a slower kernel."""
    for call in _REDUCTIONS:
        _, npy, bi = build(call)
        text = emit_c(lower(parse_kernel(npy, bi)), fn_name="f")
        assert "for (" in text, (call, text)


def test_an_integer_operand_is_not_claimed() -> None:
    """numpy's int32 ``sum`` WRAPS on overflow and Fortran's does not, and an integer ``mean``
    would become integer division. Only positive evidence of a float claims the call."""
    node = ast.parse("np.sum(a)", mode="eval").body
    assert not renders_natively(("np", "sum"), node, _SHAPE_TABLE, {"a": "int32"})
    assert not renders_natively(("np", "sum"), node, _SHAPE_TABLE, {}), "an untagged name is not a float"
    assert renders_natively(("np", "sum"), node, _SHAPE_TABLE, _DTYPE_TABLE)


def test_the_intrinsics_match_numpy_on_every_backend() -> None:
    """An intrinsic spelled wrong compiles; only the numbers catch it.

    The reduction scales every output element, so a wrong scalar shows up everywhere.
    """
    rng = np.random.default_rng(0)
    a = rng.standard_normal((8, 4)) + 3.0
    for call in _REDUCTIONS:
        src, _, _ = build(call)
        status = run_op(src, "f", {"a": a}, {"out": (8, )}, _SYMS, shapes=_SHAPES, backends=NATIVE)
        assert status == {"c": "ok", "cpp": "ok", "fortran": "ok"}, (call, status)
