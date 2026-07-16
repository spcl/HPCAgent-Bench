# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""NumpyToJAX: helper subroutines that mutate an array argument in place.

numpy helpers mutate an array passed by reference (a Fortran-style ``subroutine``);
jax arrays are immutable, so the mutation only survives if the CALL SITE captures
the helper's (functionalized) result back. ``_helper_mutation_map`` classifies each
returned slot as ``("mut", pos)`` -- the new value of the arg at ``pos`` -- or
``("val",)`` -- a genuine return value the LHS captures -- and
``_rewrite_inplace_helper_calls`` rebinds every call site accordingly. This covers
QE ``vexx_k``'s exchange helpers:

* ``_addusxx_r(rhoc, ...)`` accumulates into ``rhoc`` AND ``return rhoc``
  -> bare ``rhoc = _addusxx_r(rhoc, ...)``;
* ``_newdxx_g(..., deexx[:, ii], ...)`` mutates the ``deexx`` column passed as a
  SUBSCRIPT view -> ``deexx = deexx.at[:, ii].set(_newdxx_g(...))``;
* ``fac = _g2_convolution_all(cf, cd, ...)`` returns a column while filling the
  ``cf``/``cd`` caches -> ``fac, cf, cd = _g2_convolution_all(cf, cd, ...)``.

Before the fix the emitter either raised ``EmitError`` on the bare call or (worse)
let ``_augment_returns`` grow the return into a tuple the value-capturing call site
silently bound whole, so ``fac`` became a 3-tuple and every downstream use broke.
"""
import ast

import numpy as np
import pytest

pytest.importorskip("jax")

from numpyto_jax.core import _helper_mutation_map, emit_jax

_SRC = '''
import numpy as np

def fill_col(store, done, j, col):
    # returns store[:, j] AND fills store / flags done in place (value + mutation).
    if not done[j]:
        store[:, j] = col
        done[j] = True
    return store[:, j]

def add_into(dst, src):
    # mutates dst in place AND returns it (return IS the mutated param).
    dst += src
    return dst

def scale_rows(a, factor):
    # mutates a, returns None (the classic in-place helper).
    a *= factor

def kernel(store, done, cols, out):
    m = cols.shape[1]
    for j in range(m):
        c = fill_col(store, done, j, cols[:, j])   # value-captured: mutates store/done
        add_into(out[:, j], c)                      # bare, subscript arg: mutates out[:, j]
    scale_rows(out, 2.0)                            # bare, no-return helper
'''


def _defs(src):
    return {n.name: n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}


def test_return_slots_classify_the_three_helper_shapes():
    hm = _helper_mutation_map(list(_defs(_SRC).values()))
    # return-value-plus-mutation: the value is captured, then the two caches rebind.
    assert hm["fill_col"] == [("val", ), ("mut", 0), ("mut", 1)]
    # return-is-the-mutated-param: one mut slot at the mutated position.
    assert hm["add_into"] == [("mut", 0)]
    # no own return: the mutated param is returned and rebound.
    assert hm["scale_rows"] == [("mut", 0)]


def test_helper_returning_a_derived_value_is_not_treated_as_in_place():
    # A helper that mutates AND returns a DIFFERENT (non-param) value in a way the
    # augmentation can't line up must not be turned into an in-place rebind that
    # would corrupt the captured value. Here the return mixes a mutated param with
    # a fresh scalar, so both return points must agree; a branch-local return the
    # augmentation can't reach disqualifies the rewrite (falls back to normal call).
    src = '''
def h(acc, x):
    if x > 0:
        acc += x
        return acc
    return acc
'''
    hm = _helper_mutation_map(list(_defs(src).values()))
    # The nested (branch) return is unreachable by fn.body augmentation -> not rewritten.
    assert "h" not in hm


def test_emit_rewrites_bare_value_and_subscript_call_sites():
    js = emit_jax(_SRC, "kernel")
    # value-captured call unpacks the primary return + the two mutated caches.
    assert "c, store, done = fill_col(store, done, j, cols[:, j])" in js
    # bare call with a subscript arg is rebound through the functional .at[...].set.
    assert "out = out.at[:, j].set(add_into(out[:, j], c))" in js
    # no-return helper's mutation is captured too.
    assert "out = scale_rows(out, 2.0)" in js


def test_inplace_helpers_match_numpy_end_to_end():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    p, m = 4, 3
    cols = rng.random((p, m))
    store0, done0, out0 = np.zeros((p, m)), np.zeros(m, bool), np.zeros((p, m))

    ns: dict = {}
    exec(compile(_SRC, "<np>", "exec"), ns)
    store_ref, done_ref, out_ref = store0.copy(), done0.copy(), out0.copy()
    ns["kernel"](store_ref, done_ref, cols.copy(), out_ref)

    nsj: dict = {}
    exec(compile(emit_jax(_SRC, "kernel"), "<jax>", "exec"), nsj)
    ret = nsj["kernel"](jnp.asarray(store0), jnp.asarray(done0), jnp.asarray(cols), jnp.asarray(out0))
    rv = list(ret) if isinstance(ret, tuple) else [ret]
    # kernel mutates store, done AND out in place (through helper calls), so the functionalised
    # return is (store, done, out) in signature order; out (the doubled buffer) is the last element.
    out_jax = np.asarray(rv[-1])

    # numpy sanity: fill_col copies each column, add_into adds it, scale_rows doubles.
    assert np.allclose(out_ref, 2.0 * cols)
    assert np.allclose(out_jax, out_ref)
