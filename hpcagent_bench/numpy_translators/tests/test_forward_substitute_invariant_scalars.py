"""A loop-invariant scalar is replayed at its deeper use sites, and the assign is gone.

pet drops such an assign and the deeper consumers then read an uninitialised value (POLYCC-001);
substituting it also makes scalar-laundered indirection literal (POLYCC-006). Asserted on the
emitted C, since the emitted text is what polycc reads.
"""
import json
import pathlib
import re
import tempfile

from _op_oracle import _bench_info
from hpcagent_bench.pluto_affine import KNOWN_POLYCC_ISSUES, scop_nonaffine_reason
from numpyto_c.emit import emit_c, emit_pluto
from numpyto_common.frontend import parse_kernel
from numpyto_common.lowering import lower

#: conv_2d's own ``w = w_box[di + R, dj + R]``, with the pad already materialised.
_CONV = ("import numpy as np\n"
         "def conv_op(w_box, padded, out_grid, K, N):\n"
         "    for di in range(K):\n"
         "        for dj in range(K):\n"
         "            w = w_box[di, dj]\n"
         "            for i in range(N):\n"
         "                for j in range(N):\n"
         "                    out_grid[i, j] += w * padded[i + di, j + dj]\n")


def _emit(src: str, func: str, extra_args=(), dtypes=None) -> str:
    """Emit C for a throwaway kernel over ``w_box`` / ``padded`` / ``out_grid``."""
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "k_numpy.py").write_text(src)
    shapes = {"w_box": "(K, K)", "padded": "(N + K, N + K)", "out_grid": "(N, N)"}
    shapes.update({a: "(N,)" for a in extra_args})
    bi = _bench_info(func, ["w_box", "padded", *extra_args], ["out_grid"], shapes, {"K": 3, "N": 8}, dtypes)
    (d / "bi.json").write_text(json.dumps(bi))
    return emit_c(lower(parse_kernel(d / "k_numpy.py", d / "bi.json")), fn_name=func)


def _assigns_to(text: str, name: str):
    """Whole-statement ``<name> = ...;`` lines, the form the substitution deletes."""
    return re.findall(rf"^\s*{name} = .*;$", text, re.M)


def test_invariant_scalar_is_replayed_at_the_deeper_use_site():
    body = _emit(_CONV, "conv_op")
    assert not _assigns_to(body, "w"), f"the invariant scalar assign survived (POLYCC-001):\n{body}"
    inner = [ln for ln in body.splitlines() if "padded[" in ln and "out_grid[" in ln]
    assert inner, f"no accumulate statement in:\n{body}"
    assert all("w_box[" in ln for ln in inner), f"the weight read was not replayed inline:\n{inner}"


def test_declines_when_the_source_array_is_written_in_the_nest():
    # Condition 4: replaying the read would cross a store to its own buffer.
    body = _emit(
        _CONV.replace("conv_op", "alias_op").replace(
            "                    out_grid[i, j] += w * padded[i + di, j + dj]\n",
            "                    out_grid[i, j] += w * padded[i + di, j + dj]\n"
            "                    w_box[di, dj] = out_grid[i, j]\n"), "alias_op")
    assert _assigns_to(body, "w"), f"a written source array must decline the substitution:\n{body}"


def test_declines_when_the_scalar_is_read_after_the_loop():
    # Condition 6: a read past the loop sees the last iteration's value.
    body = _emit(
        _CONV.replace("conv_op(w_box, padded, out_grid, K, N)", "live_op(w_box, padded, tail, out_grid, K, N)").replace(
            "def conv_op", "def live_op") + "    tail[0] = w\n",
        "live_op",
        extra_args=("tail", ))
    assert _assigns_to(body, "w"), f"a scalar live past its loop must decline:\n{body}"


def test_declines_when_an_operand_is_reassigned():
    # Condition 5: ``k`` is written twice, so a deeper replay reads the second binding.
    body = _emit(
        _CONV.replace("conv_op", "unstable_op").replace(
            "            w = w_box[di, dj]\n", "            k = dj\n"
            "            w = w_box[di, k]\n"
            "            k = dj + 1\n"), "unstable_op")
    assert _assigns_to(body, "w"), f"an unstable operand must decline the substitution:\n{body}"


def test_declines_when_no_read_is_deeper():
    # Condition 7: a same-depth read is not the defect, so replaying is pure growth.
    body = _emit(
        _CONV.replace("conv_op", "flat_op").replace(
            "            w = w_box[di, dj]\n"
            "            for i in range(N):\n"
            "                for j in range(N):\n", "            for i in range(N):\n"
            "                for j in range(N):\n"
            "                    w = w_box[di, dj]\n"), "flat_op")
    assert _assigns_to(body, "w"), f"a same-depth read must decline the substitution:\n{body}"


#: lavamd's shape: a box offset laundered through two scalars before a subscript.
_GATHER = ("import numpy as np\n"
           "def gather_op(box_offsets, rv, fv, NB, PB):\n"
           "    for l in range(NB):\n"
           "        first_i = box_offsets[l]\n"
           "        for i in range(PB):\n"
           "            ai = first_i + i\n"
           "            for j in range(PB):\n"
           "                fv[ai] += rv[ai] * rv[j]\n")


def test_scalar_laundered_indirection_becomes_literal():
    """POLYCC-006: the detector reads subscript TEXT, so the gather must reach it."""
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "k_numpy.py").write_text(_GATHER)
    bi = _bench_info("gather_op", ["box_offsets", "rv"], ["fv"], {
        "box_offsets": "(NB,)",
        "rv": "(NB * PB,)",
        "fv": "(NB * PB,)"
    }, {
        "NB": 4,
        "PB": 8
    }, {"box_offsets": "int64"})
    (d / "bi.json").write_text(json.dumps(bi))
    scop = emit_pluto(lower(parse_kernel(d / "k_numpy.py", d / "bi.json")), fn_name="gather_op")
    assert scop_nonaffine_reason(scop) == "indirection", f"the gather stayed laundered:\n{scop}"


def test_registry_credits_the_substitution():
    """The tripwire's other half: both entries name this pass."""
    dotted = "numpyto_common.lowering._ForwardSubstituteInvariantScalars"
    assert KNOWN_POLYCC_ISSUES["POLYCC-001"].avoided_by == dotted
    assert KNOWN_POLYCC_ISSUES["POLYCC-006"].avoided_by == dotted
