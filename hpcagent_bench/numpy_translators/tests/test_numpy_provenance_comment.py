# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A numpy operation lowered to a loop nest carries its numpy spelling as a comment.

Fortran can say ``MATMUL`` or ``SUM`` and the name documents the operation. C has no array
intrinsics at all, so the same operation arrives as an anonymous loop nest: nothing in
``for (i...) for (j...) acc[i] += a[i][j];`` says "this was ``np.sum(a, axis=1)``". The note is what
the generated source has instead of an intrinsic, and it is attached where the numpy call is
replaced by statements -- the one place that still knows both halves.

It is a comment and nothing else: no statement changes, so a backend that ignores comments emits
exactly what it emitted before.
"""
import json
import pathlib
import tempfile

import numpy as np

import _op_oracle as oo
from _op_oracle import run_op

from numpyto_c.emit import emit_c
from numpyto_common.frontend import parse_kernel
from numpyto_common.ir import NUMPY_NOTE_CHARS
from numpyto_common.lowering import lower
from numpyto_fortran.emit import emit_fortran

NATIVE = ("c", "cpp", "fortran")

_SHAPES = {"a": "(N, M)", "out": "(N,)"}
_SYMS = {"N": 8, "M": 4}


def emit(body: str, target: str, shapes=None, syms=None) -> str:
    src = "import numpy as np\ndef f(a, out):\n" + body
    d = pathlib.Path(tempfile.mkdtemp())
    npy = d / "f.py"
    npy.write_text(src)
    bi = d / "bi.json"
    bi.write_text(json.dumps(oo._bench_info("f", ["a"], ["out"], shapes or _SHAPES, syms or _SYMS)))
    kir = lower(parse_kernel(npy, bi))
    return emit_c(kir, fn_name="f") if target == "c" else emit_fortran(kir, fn_name="f")


def notes(text: str):
    return [line.strip() for line in text.splitlines() if "numpy:" in line]


#: One op per lowering route: a scan, an axis reduction, a shift, a shape op, a select, a pad.
#: Each reaches the expander through a different path, and each is anonymous once it is a loop.
_OPS = {
    "np.cumsum(a[:, 0])": "    c = np.cumsum(a[:, 0])\n    out[:] = c * 2.0\n",
    "np.sum(a, axis=1)": "    c = np.sum(a, axis=1)\n    out[:] = c * 2.0\n",
    "np.roll(a, 2, axis=0)": "    c = np.roll(a, 2, axis=0)\n    out[:] = c[:, 0] * 2.0\n",
    "np.transpose(a)": "    c = np.transpose(a)\n    out[:] = c[0, :] * 2.0\n",
    "np.where(a > 0.0, a, -a)": "    c = np.where(a > 0.0, a, -a)\n    out[:] = c[:, 0]\n",
    "np.pad(a, ((1, 1), (0, 0)))": "    c = np.pad(a, ((1, 1), (0, 0)))\n    out[:] = c[1:9, 0]\n",
}


def test_every_lowered_op_names_itself_in_c() -> None:
    for want, body in _OPS.items():
        shapes = {"a": "(N, M)", "out": "(N,)"} if "transpose" not in want else {"a": "(N, N)", "out": "(N,)"}
        syms = _SYMS if "transpose" not in want else {"N": 8}
        got = notes(emit(body, "c", shapes, syms))
        assert got == [f"/* numpy: {want} */"], (want, got)


def test_every_lowered_op_names_itself_in_fortran() -> None:
    for want, body in _OPS.items():
        shapes = {"a": "(N, M)", "out": "(N,)"} if "transpose" not in want else {"a": "(N, N)", "out": "(N,)"}
        syms = _SYMS if "transpose" not in want else {"N": 8}
        got = notes(emit(body, "fortran", shapes, syms))
        assert got == [f"! numpy: {want}"], (want, got)


def test_the_note_names_the_numpy_call_not_the_hoisted_temp() -> None:
    """The call hoister splits ``out[:] = np.sum(a, axis=1) * 2.0`` into a ``__cb`` temp assign.

    The note is captured before that, so it reads as the numpy the kernel was written in rather
    than as the emitter's bookkeeping -- a note saying ``__cb1 = ...`` documents nothing.
    """
    got = notes(emit("    out[:] = np.sum(a, axis=1) * 2.0\n", "c"))
    assert got == ["/* numpy: np.sum(a, axis=1) */"], got


def test_a_long_expression_is_truncated_to_keep_the_column_budget() -> None:
    """The house limit is 120 columns and the note sits at the statement's own indent."""
    long_call = "np.where(a > 0.0, a * 1.0 + 2.0 - 3.0, a * 4.0 + 5.0 - 6.0 + 7.0 * 8.0 - 9.0 + 10.0 * 11.0 - 12.0)"
    got = notes(emit(f"    c = {long_call}\n    out[:] = c[:, 0]\n", "c"))
    assert len(got) == 1, got
    assert got[0].endswith("... */"), got[0]
    assert len(got[0]) <= NUMPY_NOTE_CHARS + len("/* numpy:  */"), got[0]


def test_the_note_leaves_the_numbers_alone() -> None:
    """A comment must not change what the kernel computes -- pinned on real output, not on emit."""
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           "    c = np.sum(a, axis=1)\n"
           "    out[:] = c * 2.0\n")
    rng = np.random.default_rng(0)
    assert run_op(src, "f", {"a": rng.standard_normal((8, 4))}, {"out": (8, )}, _SYMS, shapes=_SHAPES,
                  backends=NATIVE) == {
                      "c": "ok",
                      "cpp": "ok",
                      "fortran": "ok"
                  }


def test_a_dropped_statement_leaves_no_orphan_comment() -> None:
    """Every note in the output sits directly above a statement.

    ``emit_stmt`` returns ``""`` for a statement the backends drop (an input-validation raise, a
    bare return temp). A note attached to one of those would be a comment with nothing under it.
    """
    for target, marker in (("c", "/* numpy:"), ("fortran", "! numpy:")):
        text = emit(_OPS["np.sum(a, axis=1)"], target)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        for i, line in enumerate(lines):
            if marker in line:
                assert i + 1 < len(lines), text
                assert marker not in lines[i + 1], text
