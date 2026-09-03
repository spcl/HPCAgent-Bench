"""What a kept helper's parameters and return are SIZED from.

Every kernel is built with its helpers kept, so a helper whose parameter or return comes back
unsized is not a corner case -- it is a refusal at emit ("call to np.reshape not supported"), or
worse a by-value ``double`` parameter where the call passes a buffer. Three sources had gaps:

* a caller local bound to an expression over another LOCAL (``h = np.maximum(__hcall1, 0.0)``)
  read no declared array, so it resolved to nothing and the helper it feeds typed its array
  parameter by-value (densenet121_transition_layer, squeezenet);
* a local bound inside an ``if`` was skipped by the forward extent sweep, so a helper returning a
  reduction over it was sized off its INPUT instead (conv2d_gelu_global_avg_pool).
"""

import json
import pathlib
import tempfile

import numpy as np
from _op_oracle import run_op

_POOL_KERNEL = """import numpy as np


def _pool(x, n, c, h, w):
    oh = 1
    ow = 1
    if h % oh == 0 and w % ow == 0:
        y = x.reshape(n, c, oh, h // oh, ow, w // ow)
        return y.mean(axis=(3, 5))
    z = np.zeros((n, c, oh, ow), dtype=x.dtype)
    return z


def pool_demo(x, out, N, C, H, W):
    y = _pool(x, N, C, H, W)
    out[:] = y * 2.0
"""

_CHAIN_KERNEL = """import numpy as np


def _shift(v, n, m):
    return v + 1.0


def chain_demo(x, out, N, M):
    t = np.zeros((N, M))
    t[:] = x * 2.0
    h = np.maximum(t, 0.0)
    out[:] = _shift(h, N, M)
"""

_POOL_BENCH = {
    "benchmark": {
        "func_name": "pool_demo",
        "array_args": ["x", "out"],
        "input_args": ["x", "out"],
        "output_args": ["out"],
        "init": {
            "shapes": {"x": "(N,C,H,W)", "out": "(N,C,1,1)"},
            "dtypes": {"x": "float64", "out": "float64"},
        },
        "parameters": {"S": {"N": 2, "C": 3, "H": 4, "W": 4}},
        "short_name": "pool_demo",
    },
    "track": "loop_level_reasoning",
    "precisions": ["fp64"],
}

_CHAIN_BENCH = {
    "benchmark": {
        "func_name": "chain_demo",
        "array_args": ["x", "out"],
        "input_args": ["x", "out"],
        "output_args": ["out"],
        "init": {"shapes": {"x": "(N,M)", "out": "(N,M)"}, "dtypes": {"x": "float64", "out": "float64"}},
        "parameters": {"S": {"N": 3, "M": 4}},
        "short_name": "chain_demo",
    },
    "track": "loop_level_reasoning",
    "precisions": ["fp64"],
}


def _kir(kernel_src, bench, stem):
    from numpyto_common.frontend import parse_kernel

    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        kp = d / f"{stem}_numpy.py"
        kp.write_text(kernel_src)
        bi = d / "bi.json"
        bi.write_text(json.dumps(bench))
        return parse_kernel(kp, bi)


def _helper(kir, name):
    return next(h for h in kir.helpers if h.kernel_name == name)


def test_return_is_sized_from_the_branch_that_builds_it():
    # ``__hret`` is the (n, c, 1, 1) pooled result, NOT ``x``'s own (N, C, H, W).
    pool = _helper(_kir(_POOL_KERNEL, _POOL_BENCH, "pool_demo"), "_pool")
    hret = next(a for a in pool.arrays if a.name.startswith("__hret"))
    assert tuple(str(s) for s in hret.shape) == ("N", "C", "1", "1"), [(a.name, a.shape) for a in pool.arrays]


def test_pool_helper_emits_and_matches_numpy():
    src = _POOL_KERNEL.replace(
        "def pool_demo(x, out, N, C, H, W):\n    y = _pool(x, N, C, H, W)\n    out[:] = y * 2.0\n",
        "def pool_demo(x, out):\n"
        "    y = _pool(x, x.shape[0], x.shape[1], x.shape[2], x.shape[3])\n"
        "    out[:] = y * 2.0\n",
    )
    N, C, H, W = 2, 3, 4, 4
    x = np.random.default_rng(0).standard_normal((N, C, H, W))
    res = run_op(
        src,
        "pool_demo",
        {"x": x},
        {"out": (N, C, 1, 1)},
        {"N": N, "C": C, "H": H, "W": W},
        shapes={"x": "(N,C,H,W)", "out": "(N,C,1,1)"},
        backends=("c", "cpp", "fortran"),
    )
    assert all(v == "ok" or v.startswith("skip") for v in res.values()), res


def test_parameter_bound_to_a_local_expression_stays_an_array():
    # ``h = np.maximum(t, 0.0)`` reads no declared array; ``_shift``'s ``v`` must still be one.
    kir = _kir(_CHAIN_KERNEL, _CHAIN_BENCH, "chain_demo")
    shift = _helper(kir, "_shift")
    assert "v" in {a.name for a in shift.arrays}, (
        [(a.name, a.shape) for a in shift.arrays],
        [s.name for s in shift.scalars],
    )
    assert "v" not in {s.name for s in shift.scalars}


def test_chained_helper_locals_match_numpy():
    src = _CHAIN_KERNEL.replace(
        "def chain_demo(x, out, N, M):\n", "def chain_demo(x, out):\n    N = x.shape[0]\n    M = x.shape[1]\n"
    )
    N, M = 3, 4
    x = np.random.default_rng(1).standard_normal((N, M))
    res = run_op(
        src,
        "chain_demo",
        {"x": x},
        {"out": (N, M)},
        {"N": N, "M": M},
        shapes={"x": "(N,M)", "out": "(N,M)"},
        backends=("c", "cpp", "fortran"),
    )
    assert all(v == "ok" or v.startswith("skip") for v in res.values()), res
