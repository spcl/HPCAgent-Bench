"""``None`` used as a sentinel value -- the single largest C/Fortran-emit refusal cause.

Three related-but-distinct idioms, all unlowerable as written (neither C nor Fortran has a
``None`` value or an ``is`` operator):

* (a) a helper that returns ``None`` OR a tuple (``_tap_range``-style conv/pool tap-range
  helpers): an early ``if <empty range>: return None`` disqualifies it from the ordinary
  single/multi-return inliner (:func:`frontend._collect_inlinable_helpers`), so it used to survive
  as its own un-emittable :class:`KernelIR`. :func:`frontend._collect_none_guarded_helpers` +
  :class:`frontend._SpliceNoneGuardedCalls` splice the helper INTO each call site together with the
  caller's own ``if tap is None: continue`` guard and tuple unpack, so no ``None``-or-tuple value
  ever exists at all.
* (b) a first-iteration accumulator (``acc = None`` before a loop, ``acc = tap if acc is None else
  combiner(acc, tap)`` or the ``if/else`` spelling inside it): :class:`frontend.
  _PeelNoneSeededAccumulators` replaces the ``None`` check with an explicit ``__acc_seen`` flag --
  sound for ANY combiner, since it replays the exact state machine the ``None`` check already was
  rather than guessing a reduction identity.
* (c) a default-argument sentinel resolved once, straight-line (``if stride is None: stride =
  kernel_size``) where the call-site argument is a compile-time ``None`` literal: already partly
  handled by :mod:`tuple_desugar`'s own ``x is None`` kind-tracking, which folds the guard away --
  but the now-dead ``stride = None`` init that DOMINATES it (from :class:`frontend._InlineHelpers`'s
  ``reassigned_params`` handling) survived because it is READ again later, through the SECOND write,
  not the dead one. :func:`tuple_desugar._drop_dead_none_bindings`'s adjacency check drops it.

Each test asserts on the lowered AST (or emitted text) directly, not merely that nothing raised,
plus a numeric check through the real C/C++/Fortran backends via the existing oracle harness
(:func:`_op_oracle.run_op`), plus a NEGATIVE case per idiom that still correctly refuses.
"""
import ast
import json
import pathlib
import tempfile

import numpy as np
import pytest

from _op_oracle import _bench_info, run_op

from numpyto_common.frontend import parse_kernel
from numpyto_common.lowering import lower
from numpyto_c.emit import emit_c
from numpyto_common.tuple_desugar import desugar_tuples

_NATIVE = ("c", "cpp", "fortran")


def _kir_for(src: str, func: str, inputs, outputs, shapes, syms):
    """``parse_kernel`` against a throwaway source + bench_info -- the real file-reading entry
    point, matching the sibling ``test_generator_tuple_fold.py``."""
    d = pathlib.Path(tempfile.mkdtemp())
    npy = d / f"{func}_numpy.py"
    npy.write_text(src)
    bi = d / "bi.json"
    bi.write_text(json.dumps(_bench_info(func, inputs, outputs, shapes, syms)))
    return parse_kernel(npy, bi)


# --------------------------------------------------------------------------------------------- #
# (a) a helper that returns None OR a tuple
# --------------------------------------------------------------------------------------------- #

_TAP_RANGE_SRC = ("import numpy as np\n"
                  "def _tap_range(in_size, out_size, stride, padding, dilation, k):\n"
                  " numer = padding - k * dilation\n"
                  " lo = max(0, -(-numer // stride))\n"
                  " hi = min(in_size, (out_size - 1 + padding - k * dilation) // stride + 1)\n"
                  " if lo >= hi:\n"
                  "  return None\n"
                  " ol_lo = lo * stride - padding + k * dilation\n"
                  " ol_hi = (hi - 1) * stride - padding + k * dilation + 1\n"
                  " return lo, hi, ol_lo, ol_hi\n"
                  "def f(x, weight, stride, padding, dilation, out):\n"
                  " kd = weight.shape[0]\n"
                  " n = x.shape[0]\n"
                  " for k in range(kd):\n"
                  "  tap = _tap_range(n, out.shape[0], stride, padding, dilation, k)\n"
                  "  if tap is None:\n"
                  "   continue\n"
                  "  lo, hi, ol_lo, ol_hi = tap\n"
                  "  for idx in range(lo, hi):\n"
                  "   out[ol_lo + (idx - lo) * stride] += x[idx] * weight[k]\n")

_TAP_RANGE_SHAPES = {"x": "(N,)", "weight": "(K,)", "out": "(M,)"}
_TAP_RANGE_SYMS = {"N": 2, "K": 3, "M": 2}


def test_none_or_tuple_helper_splices_away():
    # Structural: no surviving helper, no None literal, and the caller's own "continue" guard
    # is preserved verbatim (not replaced by a flag -- there is nothing to seed, it just skips).
    kir = _kir_for(_TAP_RANGE_SRC, "f", ["x", "weight", "stride", "padding", "dilation", "out"], ["out"],
                   _TAP_RANGE_SHAPES, _TAP_RANGE_SYMS)
    assert kir.helpers == []
    body = ast.unparse(kir.tree)
    assert "None" not in body
    assert "_tap_range" not in body
    assert "continue" in body


def test_none_or_tuple_helper_numeric_and_skips_the_empty_tap():
    # stride=1, padding=2, dilation=2 over a 2-wide input / 3-tap kernel makes k=0 and k=2 land
    # entirely outside the input (an empty range -> None -> skip) and only k=1 contribute -- this
    # exercises BOTH the skip and the live path, not just one.
    x = np.array([1.0, 2.0])
    weight = np.array([10.0, 20.0, 30.0])
    res = run_op(_TAP_RANGE_SRC,
                 "f", {
                     "x": x,
                     "weight": weight,
                     "stride": 1,
                     "padding": 2,
                     "dilation": 2
                 }, {"out": (2, )},
                 _TAP_RANGE_SYMS,
                 shapes=_TAP_RANGE_SHAPES,
                 backends=_NATIVE)
    assert res == {"c": "ok", "cpp": "ok", "fortran": "ok"}, res


def test_none_or_tuple_helper_wrong_guard_shape_still_refuses():
    # NEGATIVE: the caller wraps the live path in ``if tap is not None:`` with no ``continue`` --
    # not the recognised "guard responds with a plain control statement" shape, so the helper is
    # NOT spliced and keeps its un-emittable early ``return None``.
    src = ("import numpy as np\n"
           "def _tap_range(in_size, out_size, stride, padding, dilation, k):\n"
           " numer = padding - k * dilation\n"
           " lo = max(0, -(-numer // stride))\n"
           " hi = min(in_size, (out_size - 1 + padding - k * dilation) // stride + 1)\n"
           " if lo >= hi:\n"
           "  return None\n"
           " ol_lo = lo * stride - padding + k * dilation\n"
           " ol_hi = (hi - 1) * stride - padding + k * dilation + 1\n"
           " return lo, hi, ol_lo, ol_hi\n"
           "def f(x, weight, stride, padding, dilation, out):\n"
           " kd = weight.shape[0]\n"
           " n = x.shape[0]\n"
           " for k in range(kd):\n"
           "  tap = _tap_range(n, out.shape[0], stride, padding, dilation, k)\n"
           "  if tap is not None:\n"
           "   lo, hi, ol_lo, ol_hi = tap\n"
           "   for idx in range(lo, hi):\n"
           "    out[ol_lo + (idx - lo) * stride] += x[idx] * weight[k]\n")
    kir = _kir_for(src, "f", ["x", "weight", "stride", "padding", "dilation", "out"], ["out"], _TAP_RANGE_SHAPES,
                   _TAP_RANGE_SYMS)
    assert [h.kernel_name for h in kir.helpers] == ["_tap_range"]
    with pytest.raises(NotImplementedError, match="None"):
        emit_c(lower(kir), fn_name="f")


# --------------------------------------------------------------------------------------------- #
# (b) a first-iteration accumulator
# --------------------------------------------------------------------------------------------- #

_ACC_TERNARY_SRC = ("import numpy as np\n"
                    "def f(x, out):\n"
                    " n = x.shape[0]\n"
                    " acc = None\n"
                    " for i in range(n):\n"
                    "  tap = x[i, :]\n"
                    "  acc = tap if acc is None else np.maximum(acc, tap)\n"
                    " out[:] = acc\n")

_ACC_IFELSE_SRC = ("import numpy as np\n"
                   "def f(x, out):\n"
                   " n = x.shape[0]\n"
                   " acc = None\n"
                   " for i in range(n):\n"
                   "  tap = x[i, :]\n"
                   "  if acc is None:\n"
                   "   acc = tap.copy()\n"
                   "  else:\n"
                   "   acc = np.maximum(acc, tap)\n"
                   " out[:] = acc\n")

_ACC_SHAPES = {"x": "(N,M)", "out": "(M,)"}
_ACC_SYMS = {"N": 4, "M": 3}


@pytest.mark.parametrize("src", [_ACC_TERNARY_SRC, _ACC_IFELSE_SRC], ids=["ternary", "if_else"])
def test_accumulator_peels_to_a_flag(src):
    # Structural: no None literal, an explicit __acc_seen-style flag toggling 0 -> 1 once.
    kir = _kir_for(src, "f", ["x", "out"], ["out"], _ACC_SHAPES, _ACC_SYMS)
    body = ast.unparse(kir.tree)
    assert "None" not in body
    assert "_seen" in body
    assert body.count(" = 1") >= 1


@pytest.mark.parametrize("src", [_ACC_TERNARY_SRC, _ACC_IFELSE_SRC], ids=["ternary", "if_else"])
def test_accumulator_all_negative_input_distinguishes_identity_from_zero_seed(src):
    # The bug most likely to slip through a wrong fix: seeding the accumulator with 0.0 (instead of
    # genuinely peeling the first tap) gives the WRONG answer whenever every element is negative,
    # since max(0.0, negative...) never drops below 0. All-negative input makes that divergence
    # observable; the correct running max here equals plain np.max(x, axis=0).
    x = -np.abs(np.random.default_rng(0).standard_normal((4, 3)))
    res = run_op(src, "f", {"x": x}, {"out": (3, )}, _ACC_SYMS, shapes=_ACC_SHAPES, backends=_NATIVE)
    assert res == {"c": "ok", "cpp": "ok", "fortran": "ok"}, res
    assert np.all(x < 0)  # the input premise the test relies on


def test_accumulator_straight_line_no_loop_still_refuses():
    # NEGATIVE: no loop at all -- a genuinely data-dependent None (the runtime ``flag`` decides
    # whether ``acc`` is ever bound), which is a real runtime question, not a first-iteration
    # toggle. Peeling this would be unsound (there is no "later iteration" to distinguish from
    # "first"), so it correctly stays un-emittable.
    src = ("import numpy as np\n"
           "def f(x, flag, out):\n"
           " acc = None\n"
           " if flag > 0:\n"
           "  acc = x\n"
           " if acc is None:\n"
           "  out[:] = 0.0\n"
           " else:\n"
           "  out[:] = acc\n")
    kir = _kir_for(src, "f", ["x", "flag", "out"], ["out"], {"x": "(N,)", "out": "(N,)"}, {"N": 4})
    with pytest.raises(NotImplementedError, match="None"):
        emit_c(lower(kir), fn_name="f")


# --------------------------------------------------------------------------------------------- #
# (c) a default-argument sentinel
# --------------------------------------------------------------------------------------------- #

_DEFAULT_STRIDE_SRC = ("import numpy as np\n"
                       "def _default_stride(stride, k):\n"
                       " if stride is None:\n"
                       "  stride = k\n"
                       " return stride\n"
                       "def f(x, k, out):\n"
                       " s = _default_stride(None, k)\n"
                       " n = x.shape[0]\n"
                       " total = 0.0\n"
                       " for i in range(0, n, s):\n"
                       "  total = total + x[i]\n"
                       " out[0] = total\n")


def test_default_argument_sentinel_folds_away():
    # Structural: no None literal, no surviving helper -- the guard resolves statically since the
    # call site passed the literal None, and the now-dead init doesn't survive just because the
    # SAME renamed local is read again later, through the guard's own (unconditional) rebind.
    kir = _kir_for(_DEFAULT_STRIDE_SRC, "f", ["x", "k", "out"], ["out"], {"x": "(N,)", "out": "(1,)"}, {"N": 8})
    assert kir.helpers == []
    body = ast.unparse(kir.tree)
    assert "None" not in body


def test_default_argument_sentinel_numeric():
    x = np.arange(8, dtype=np.float64)
    res = run_op(_DEFAULT_STRIDE_SRC,
                 "f", {
                     "x": x,
                     "k": 3
                 }, {"out": (1, )}, {"N": 8},
                 shapes={
                     "x": "(N,)",
                     "out": "(1,)"
                 },
                 backends=_NATIVE)
    assert res == {"c": "ok", "cpp": "ok", "fortran": "ok"}, res


def test_default_argument_sentinel_broken_adjacency_still_refuses():
    # NEGATIVE: an unrelated statement (``unrelated = other + 1.0``) sits between the reassigned
    # param's ``None`` init and the guard that resolves it, breaking the STRICT adjacency
    # :func:`tuple_desugar._drop_dead_none_bindings` requires before dropping a dead ``None`` bind
    # -- so the dead init is left standing (it IS read again, through the guard's own rebind, so
    # the "no readers anywhere" fallback does not catch it either) and correctly refuses.
    src = ("import numpy as np\n"
           "def _default_stride(stride, k, other):\n"
           " unrelated = other + 1.0\n"
           " if stride is None:\n"
           "  stride = k\n"
           " return stride + unrelated\n"
           "def f(x, k, other, out):\n"
           " out[0] = _default_stride(None, k, other)\n")
    kir = _kir_for(src, "f", ["x", "k", "other", "out"], ["out"], {"x": "(N,)", "out": "(1,)"}, {"N": 4})
    with pytest.raises(NotImplementedError, match="None"):
        emit_c(lower(kir), fn_name="f")


# --------------------------------------------------------------------------------------------- #
# Unit-level pin: the adjacency fix in isolation, mirroring the sibling
# test_generator_tuple_fold.py's own direct desugar_tuples() unit test.
# --------------------------------------------------------------------------------------------- #


def test_drop_dead_none_bindings_adjacency_unit():
    fn = ast.parse("def h(k):\n"
                   " stride = None\n"
                   " if stride is None:\n"
                   "  stride = k\n"
                   " return stride + 1\n").body[0]
    desugar_tuples(fn, int_scalars=frozenset({"k"}), float_scalars=frozenset(), arrays=frozenset(), ranks={})
    assert ast.unparse(fn) == "def h(k):\n    stride = k\n    return stride + 1"
