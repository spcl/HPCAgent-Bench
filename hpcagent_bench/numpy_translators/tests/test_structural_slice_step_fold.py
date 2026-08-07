"""A slice STEP is a structural slot, so a manifest-constant runtime argument folds into it.

``_reject_unsupported_slices`` refuses a non-literal step because ``_slice_step_const`` returns
``None`` for it and every consumer reads that as step 1 -- ``x[::s]`` emitted a contiguous copy and
the stride was silently gone. The guard is right; what was wrong was the INPUT to it.

The KernelBench conv/pool ports (``resnet_basic_block``, ``efficientnet_mb_conv``) declare the
stride as an ``init.scalars`` value that is ALSO an ABI argument, then slice with it inside a helper
that inlines into the body::

    padded[:, :, ky:ky + (oh - 1) * stride + 1:stride, kx:...]

``_FoldStructuralUses`` already folded such an argument in a reduction's AXIS slot -- the same class
of slot, refused by the sibling guard for the same reason -- but not in a slice step, so both ports
were refused outright. It now folds there too, and only there: the BOUNDS keep the name and reach
the ABI, because a bound is an ordinary integer expression a runtime value evaluates fine.

The guard is not weakened. A step whose value is not preset-constant (absent from the manifest, or
present but reachable as an EXTENT, which the harness may scale at run time) is still refused.
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


# ---- structural: the manifest value reaches the step slot, and only that slot ---- #


def test_manifest_scalar_step_folds_to_its_literal() -> None:
    src = ("import numpy as np\n"
           "def pool(v, k):\n"
           "    return v[:(6 - 1) * k + 1:k] * 1.0\n"
           "def f(x, stride, out):\n"
           "    out[:] = pool(x, stride)\n")
    kir = parse(src, ["x", "stride", "out"], ["x", "out"], {"x": "(N,)", "out": "(6,)"}, {"N": 12, "stride": 2})
    assert steps(kir) == [2]
    # The BOUND keeps the name: it is an ordinary integer expression, so the argument still reaches
    # the ABI and the harness may pass a value other than the manifest default.
    assert "stride" in kir.param_order(), kir.param_order()


def test_two_distinct_manifest_steps_do_not_collapse() -> None:
    """Two structural constants in one kernel each fold to their OWN value.

    Collapsing them is the failure mode that produces a wrong answer rather than a refusal: both
    slices would compile, and the second would silently walk the first's stride.
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
    assert steps(kir) == [2, 3]


# ---- the guard still fires on a step that is genuinely not compile-time ---- #


def test_a_step_absent_from_the_manifest_is_still_refused() -> None:
    src = ("import numpy as np\n"
           "def f(x, step, out):\n"
           "    out[:] = x[:(6 - 1) * step + 1:step] * 1.0\n")
    with pytest.raises(NotImplementedError, match="must be a compile-time integer"):
        parse(src, ["x", "step", "out"], ["x", "out"], {"x": "(N,)", "out": "(6,)"}, {"N": 12})


def test_a_rebound_step_name_is_not_folded() -> None:
    """Once the body assigns to it, the manifest default is no longer what the slice reads.

    Folding there is the worse outcome of the two: a wrong stride that compiles, rather than the
    refusal. Same rule ``_FoldConstantSymbols`` already applies to its own substitution.
    """
    src = ("import numpy as np\n"
           "def f(x, stride, out):\n"
           "    stride = stride + 1\n"
           "    out[:] = x[:(6 - 1) * stride + 1:stride] * 1.0\n")
    with pytest.raises(NotImplementedError, match="must be a compile-time integer"):
        parse(src, ["x", "stride", "out"], ["x", "out"], {"x": "(N,)", "out": "(6,)"}, {"N": 12, "stride": 2})


def test_a_step_that_is_also_an_extent_is_still_refused() -> None:
    """An extent may be SCALED at run time, so its manifest value is not the artifact's value."""
    src = ("import numpy as np\n"
           "def f(x, out):\n"
           "    out[:] = x[::N] * 1.0\n")
    with pytest.raises(NotImplementedError, match="must be a compile-time integer"):
        parse(src, ["x", "out"], ["x", "out"], {"x": "(N,)", "out": "(1,)"}, {"N": 12})


# ---- numerical: every backend walks the declared stride ---- #


def test_manifest_step_matches_numpy_on_every_backend() -> None:
    # Step folded to 2, bound left symbolic: a lost stride reads 1..6 instead of 1 3 5 7 9 11.
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
