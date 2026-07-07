"""Array-returning non-inlinable helpers emitted as native out-param functions.

The scalar-return sibling (``test_helper_functions.py``) emits a helper with an
early ``return`` as a native function that returns by value. A helper that
returns a whole ARRAY cannot come back by value in C/Fortran, so it is emitted
with a trailing out-param the body writes into (``return fac`` ->
``__hret[:] = fac``); the call site materialises any slice arguments into
contiguous temps and stores the filled result. Config-flag arguments that are
compile-time constants at the call site are folded into the helper body and the
now-dead branches pruned -- so a QE-``g2_convolution``-style helper (whose vcut /
gamma branches carry un-lowerable tuples) reduces to its live path.
"""
import numpy as np

from _op_oracle import run_op

_ALL = ("c", "cpp", "fortran", "numba", "pythran", "jax")


def _all_ok(res):
    return all(v == "ok" or v.startswith("skip") for v in res.values()), res


def test_array_return_slice_target():
    # Early-return array helper, result stored into a row slice of the output.
    src = ("import numpy as np\n"
           "def clamp_row(v, lo):\n"
           " if lo > 0.0:\n"
           "  return np.maximum(v, lo)\n"
           " return -v\n"
           "def f(x, thr, out):\n"
           " for i in range(x.shape[0]):\n"
           "  out[i, :] = clamp_row(x[i, :], thr)\n")
    x = np.linspace(-2.0, 2.0, 20).reshape(4, 5).astype(np.float64)
    ok, res = _all_ok(
        run_op(src, "f", {"x": x, "thr": 0.5}, {"out": (4, 5)}, {"M": 4, "n": 5},
               shapes={"x": "(M,n)", "out": "(M,n)"}, backends=_ALL))
    assert ok, res


def test_array_return_bare_target():
    # Whole-array target: the out-param is filled in place (no temp copy).
    src = ("import numpy as np\n"
           "def screen(v, s):\n"
           " if s > 0.0:\n"
           "  return v * s\n"
           " return v + 1.0\n"
           "def f(x, s, out):\n"
           " out[:] = screen(x, s)\n")
    x = np.linspace(-3.0, 3.0, 12).astype(np.float64)
    ok, res = _all_ok(
        run_op(src, "f", {"x": x, "s": 2.0}, {"out": (12, )}, {"n": 12},
               shapes={"x": "(n,)", "out": "(n,)"}, backends=_ALL))
    assert ok, res


def test_array_return_specialized_config_flag():
    # A ``g2_convolution``-shaped helper: a config flag (``use_alt``) is a
    # compile-time ``False`` at the call site, so its early-return branch folds
    # away; a strided column arg ``xk[:, k]`` is materialised; the live path runs
    # a reduction + ``np.where`` and stores into a column slice.
    src = ("import numpy as np\n"
           "def gconv(g, xk, scale, use_alt):\n"
           " q = xk[:, None] + g\n"
           " qq = np.sum(q ** 2, axis=0)\n"
           " if use_alt:\n"
           "  return q[0, :] * 0.0 + 7.0\n"
           " nz = qq > 1e-08\n"
           " qn = np.where(nz, qq, 1.0)\n"
           " return np.where(nz, scale / qn, -1.0)\n"
           "def f(g, xk, scale, out):\n"
           " K = out.shape[1]\n"
           " for k in range(K):\n"
           "  out[:, k] = gconv(g, xk[:, k], scale, False)\n")
    rng = np.random.default_rng(0)
    ngm, K = 8, 3
    g = rng.standard_normal((3, ngm))
    xk = rng.standard_normal((3, K))
    ok, res = _all_ok(
        run_op(src, "f", {"g": g, "xk": xk, "scale": 2.0}, {"out": (ngm, K)},
               {"ngm": ngm, "K": K, "three": 3},
               shapes={"g": "(three,ngm)", "xk": "(three,K)", "out": "(ngm,K)"}, backends=_ALL))
    assert ok, res


def test_array_return_helper_native_desugar_bug3():
    # BUG-3: a NON-inlined array-returning helper used to keep native constructs
    # the kernel body had already shed -- the desugars only ran on the kernel, not
    # on ``_build_helper_kirs`` bodies. This helper is non-inlinable (an early
    # ``if s < 0: return`` inside the body) and carries a ``.ndim`` validation
    # guard plus an ``np.newaxis``; both must be desugared away on the HELPER for
    # the native backends to emit. Before DI-2 the ``.ndim`` / ``newaxis`` reached
    # the emitter and it failed.
    src = ("import numpy as np\n"
           "def scale_row(v, s):\n"
           " if s < 0.0:\n"
           "  return -v\n"
           " if v.ndim != 1:\n"
           "  raise ValueError('expected 1-D input')\n"
           " w = v[:, np.newaxis]\n"
           " return w[:, 0] * s\n"
           "def f(x, s, out):\n"
           " for i in range(x.shape[0]):\n"
           "  out[i, :] = scale_row(x[i, :], s)\n")
    x = np.linspace(-2.0, 2.0, 20).reshape(4, 5).astype(np.float64)
    ok, res = _all_ok(
        run_op(src, "f", {"x": x, "s": 1.5}, {"out": (4, 5)}, {"M": 4, "n": 5},
               shapes={"x": "(M,n)", "out": "(M,n)"}, backends=_ALL))
    assert ok, res


def test_array_helper_emitted_as_outparam_c_function():
    # Structural: the helper is a ``void`` C function with a trailing out-param,
    # and the call site is a SINGLE opaque call (not a per-element loop calling
    # the whole-array helper once per element).
    import json
    import pathlib
    import tempfile
    from numpyto_c.frontend import parse_kernel
    from numpyto_c.lowering import lower
    from numpyto_c.emit import emit_c
    src = ("import numpy as np\n"
           "def clamp_row(v, lo):\n"
           " if lo > 0.0:\n"
           "  return np.maximum(v, lo)\n"
           " return -v\n"
           "def f(x, thr, out):\n"
           " for i in range(x.shape[0]):\n"
           "  out[i, :] = clamp_row(x[i, :], thr)\n")
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "k_numpy.py").write_text(src)
    bi = {
        "benchmark": {
            "name": "k", "short_name": "k", "relative_path": "", "module_name": "k", "func_name": "f",
            "parameters": {"S": {"M": 4, "n": 5}}, "input_args": ["x", "thr", "out"],
            "array_args": ["x", "out"], "output_args": ["out"],
            "init": {"shapes": {"x": "(M,n)", "out": "(M,n)"}}, "scalars": {"thr": 0.5}
        }
    }
    (d / "bi.json").write_text(json.dumps(bi))
    kir = lower(parse_kernel(d / "k_numpy.py", d / "bi.json"))
    assert len(kir.helpers) == 1 and kir.helpers[0].return_kind == "__hret_0"
    c = emit_c(kir, fn_name="f")
    assert "static void clamp_row(" in c and "__hret_0" in c
    # a single call statement, not ``__hret_tmp_0[..] = clamp_row(..)`` per element
    assert "clamp_row(__harg_0_0, thr, n, __hret_tmp_0);" in c
