"""A slice STEP the ABI supplies keeps its name: it lowers as ``lo + pos * step``, it is never folded.

``_reject_unsupported_slices`` refuses a non-literal step only when the slice has no UPPER BOUND --
there the step's sign is unconstrained and the two signs walk the axis in opposite directions. A
bounded step lowers symbolically on every backend, so the stride reaches the emitted code as the
argument the harness passed.

The KernelBench conv/pool ports (``resnet_basic_block``, ``efficientnet_mb_conv``) declare the
stride as an ``init.scalars`` value that is ALSO an ABI argument, then slice with it inside a helper
that inlines into the body::

    padded[:, :, ky:ky + (oh - 1) * stride + 1:stride, kx:...]

That step was once folded to its manifest value, because ``_slice_step_const`` read a non-literal
step as 1 and the stride was silently lost. The symbolic lowering removed that premise, so the fold
was removed with it: a signature that takes ``stride`` and ignores it is the failure the ABI
ratchets exist to catch, and it is the same reason the AXIS slot was never folded.

A manifest-constant symbol that is NOT an ABI argument still folds -- ``_FoldConstantSymbols`` --
because nothing passes it at run time. Only names the binding passes are excluded.
"""
import ast
import json
import pathlib
import tempfile
from typing import Dict, List, Optional

import numpy as np
import pytest

from _op_oracle import run_op

from numpyto_common.frontend import parse_kernel

NATIVE = ("c", "cpp", "fortran")

#: 1..12, so a stride is visible in the RESULT: ``[::2]`` -> 1 3 5 7 9 11, ``[::3]`` -> 1 4 7 10.
A12 = np.arange(1.0, 13.0)


def assert_ok(res: Dict[str, str]) -> None:
    for backend, status in res.items():
        assert status == "ok" or status.startswith("skip"), f"{backend}: {status}"
    assert any(status == "ok" for status in res.values()), f"all skipped (vacuous): {res}"


def parse(src: str, args: List[str], arrays: List[str], shapes: Dict[str, str], preset: Dict[str, int]):
    """Parse ``src``'s ``f`` against a synthesized manifest (``preset`` is the ``S`` block)."""
    d = pathlib.Path(tempfile.mkdtemp())
    npy = d / "k_numpy.py"
    npy.write_text(src)
    bi = d / "bi.json"
    bi.write_text(
        json.dumps({
            "benchmark": {
                "name": "k",
                "short_name": "k",
                "relative_path": "",
                "module_name": "k",
                "func_name": "f",
                "parameters": {
                    "S": dict(preset)
                },
                "input_args": args,
                "array_args": arrays,
                "output_args": [args[-1]],
                "init": {
                    "shapes": shapes
                },
            }
        }))
    return parse_kernel(npy, bi)


def steps(kir) -> List[Optional[object]]:
    """Every slice step in the parsed body, as a literal value (``None`` when not a literal)."""
    out: List[Optional[object]] = []
    for node in ast.walk(kir.tree):
        if isinstance(node, ast.Slice) and node.step is not None:
            out.append(node.step.value if isinstance(node.step, ast.Constant) else None)
    return out


# ---- structural: an ABI step stays a name, whatever the manifest says it equals ---- #


def test_manifest_scalar_step_stays_symbolic() -> None:
    """The manifest says ``stride`` is 2 in every preset, and the step STILL keeps the name.

    Folding it would compile and run -- and would ignore whatever the harness passed, which is the
    one failure the caller cannot see.
    """
    src = ("import numpy as np\n"
           "def pool(v, k):\n"
           "    return v[:(6 - 1) * k + 1:k] * 1.0\n"
           "def f(x, stride, out):\n"
           "    out[:] = pool(x, stride)\n")
    kir = parse(src, ["x", "stride", "out"], ["x", "out"], {"x": "(N,)", "out": "(6,)"}, {"N": 12, "stride": 2})
    assert steps(kir) == [None], steps(kir)
    assert "stride" in kir.param_order(), kir.param_order()


def test_two_distinct_manifest_steps_do_not_collapse() -> None:
    """Two manifest-constant strides in one kernel each keep their OWN name.

    Collapsing them is the failure mode that produces a wrong answer rather than a refusal: both
    slices would compile, and the second would silently walk the first's stride. The numbers, not
    the exit code, are what decide it -- see the numeric sibling below.
    """
    src = ("import numpy as np\n"
           "def f(x, stride_a, stride_b, out_a, out_b):\n"
           "    out_a[:] = x[:(6 - 1) * stride_a + 1:stride_a] * 1.0\n"
           "    out_b[:] = x[:(4 - 1) * stride_b + 1:stride_b] * 1.0\n")
    kir = parse(src, ["x", "stride_a", "stride_b", "out_a", "out_b"], ["x", "out_a", "out_b"], {
        "x": "(N,)",
        "out_a": "(6,)",
        "out_b": "(4,)"
    }, {
        "N": 12,
        "stride_a": 2,
        "stride_b": 3
    })
    assert steps(kir) == [None, None], steps(kir)
    assert [p for p in kir.param_order() if p.startswith("stride")] == ["stride_a", "stride_b"], kir.param_order()


def test_two_distinct_manifest_steps_walk_their_own_stride() -> None:
    """out_a reads 1 3 5 7 9 11 and out_b reads 1 4 7 10; a collapse fills one of them with the
    other's stride and still compiles."""
    src = ("import numpy as np\n"
           "def f(x, stride_a, stride_b, out_a, out_b):\n"
           "    out_a[:] = x[:(6 - 1) * stride_a + 1:stride_a] * 1.0\n"
           "    out_b[:] = x[:(4 - 1) * stride_b + 1:stride_b] * 1.0\n")
    assert_ok(
        run_op(src,
               "f", {
                   "x": A12,
                   "stride_a": 2,
                   "stride_b": 3
               }, {
                   "out_a": (6, ),
                   "out_b": (4, )
               }, {
                   "N": 12,
                   "stride_a": 2,
                   "stride_b": 3
               },
               shapes={
                   "x": "(N,)",
                   "out_a": "(6,)",
                   "out_b": "(4,)"
               },
               backends=NATIVE))


# ---- the guard still fires on a step that is genuinely not compile-time ---- #


def test_a_bounded_step_absent_from_the_manifest_lowers_symbolically() -> None:
    """No manifest value to fold, but the slice is BOUNDED, so the stride lowers as an expression.

    The fold is what needs a compile-time value; the index ``lo + pos * step`` does not. Nothing
    here may become a literal -- the step is an ABI argument the harness passes.
    """
    src = ("import numpy as np\n"
           "def f(x, step, out):\n"
           "    out[:] = x[:(6 - 1) * step + 1:step] * 1.0\n")
    kir = parse(src, ["x", "step", "out"], ["x", "out"], {"x": "(N,)", "out": "(6,)"}, {"N": 12})
    assert steps(kir) == [None], steps(kir)
    assert "step" in kir.param_order(), kir.param_order()


def test_a_bounded_step_absent_from_the_manifest_matches_numpy() -> None:
    """A lost stride reads 1..6 instead of 1 3 5 7 9 11 -- the discriminating output."""
    src = ("import numpy as np\n"
           "def f(x, step, out):\n"
           "    out[:] = x[:(6 - 1) * step + 1:step] * 1.0\n")
    assert_ok(
        run_op(src,
               "f", {
                   "x": A12,
                   "step": 2
               }, {"out": (6, )}, {"N": 12},
               shapes={
                   "x": "(N,)",
                   "out": "(6,)"
               },
               backends=NATIVE))


def test_a_rebound_step_name_is_not_folded() -> None:
    """Once the body assigns to it, the manifest default is no longer what the slice reads.

    Folding is still refused -- a wrong stride that compiles is the worst outcome of the three, and
    the same rule ``_FoldConstantSymbols`` applies to its own substitution. What changed is the
    fallback: the step is now carried symbolically instead of refusing the kernel outright, so the
    slice reads the REBOUND value.
    """
    src = ("import numpy as np\n"
           "def f(x, stride, out):\n"
           "    stride = stride + 1\n"
           "    out[:] = x[:(4 - 1) * stride + 1:stride] * 1.0\n")
    kir = parse(src, ["x", "stride", "out"], ["x", "out"], {"x": "(N,)", "out": "(4,)"}, {"N": 12, "stride": 2})
    assert steps(kir) == [None], steps(kir)


def test_a_rebound_step_walks_the_rebound_stride_on_every_backend() -> None:
    """``stride`` arrives as 2 and is rebound to 3, so the read is 1 4 7 10.

    Had the manifest's 2 been folded in, the same kernel would fill ``out`` with 1 3 5 7 -- it
    compiles either way, which is why the numbers rather than the exit code decide it.
    """
    src = ("import numpy as np\n"
           "def f(x, stride, out):\n"
           "    stride = stride + 1\n"
           "    out[:] = x[:(4 - 1) * stride + 1:stride] * 1.0\n")
    assert_ok(
        run_op(src,
               "f", {
                   "x": A12,
                   "stride": 2
               }, {"out": (4, )}, {
                   "N": 12,
                   "stride": 2
               },
               shapes={
                   "x": "(N,)",
                   "out": "(4,)"
               },
               backends=NATIVE))


def test_an_unbounded_symbolic_step_is_still_refused() -> None:
    """``x[::s]`` has no upper bound, so the step's SIGN is unconstrained and undecidable here.

    Both signs produce a full-length axis, and they walk it in opposite directions; emitting the
    forward index would silently be the wrong one half the time. The bounded form is what makes the
    positive stride the only reading that has a run to preserve. ``N`` is also an EXTENT, which the
    harness may scale at run time -- the reason the manifest fold declines it too.
    """
    src = ("import numpy as np\n"
           "def f(x, out):\n"
           "    out[:] = x[::N] * 1.0\n")
    with pytest.raises(NotImplementedError, match="needs an upper bound"):
        parse(src, ["x", "out"], ["x", "out"], {"x": "(N,)", "out": "(1,)"}, {"N": 12})


# ---- numerical: every backend walks the declared stride ---- #


def test_manifest_step_matches_numpy_on_every_backend() -> None:
    # The stride arrives across the ABI: a lost one reads 1..6 instead of 1 3 5 7 9 11.
    src = ("import numpy as np\n"
           "def pool(v, k):\n"
           "    return v[:(6 - 1) * k + 1:k] * 1.0\n"
           "def f(x, stride, out):\n"
           "    out[:] = pool(x, stride)\n")
    assert_ok(
        run_op(src,
               "f", {
                   "x": A12,
                   "stride": 2
               }, {"out": (6, )}, {
                   "N": 12,
                   "stride": 2
               },
               shapes={
                   "x": "(N,)",
                   "out": "(6,)"
               },
               backends=NATIVE))


def test_one_helper_two_different_literal_steps_matches_numpy() -> None:
    """The shape every ML port has: ONE ``_conv2d``/``_maxpool2d``, called with DIFFERENT strides.

    Each call site inlines its own copy, so each must keep the literal IT was passed. If the two
    collapsed onto one stride the kernel would still compile and still fill both buffers -- ``out3``
    would just hold ``1 3 5 7`` instead of ``1 4 7 10``.
    """
    src = ("import numpy as np\n"
           "def pool(v, k, m):\n"
           "    return v[:(m - 1) * k + 1:k] * 1.0\n"
           "def f(x, out2, out3):\n"
           "    out2[:] = pool(x, 2, 6)\n"
           "    out3[:] = pool(x, 3, 4)\n")
    assert_ok(
        run_op(src,
               "f", {"x": A12}, {
                   "out2": (6, ),
                   "out3": (4, )
               }, {"N": 12},
               shapes={
                   "x": "(N,)",
                   "out2": "(6,)",
                   "out3": "(4,)"
               },
               backends=NATIVE))
