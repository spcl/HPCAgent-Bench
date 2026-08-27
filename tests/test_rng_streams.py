# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The per-array RNG stream policy.

The property that matters is INDEPENDENCE: array *k*'s values are a function of the seed and *k*,
and of nothing else. A single shared stream does not have it -- there, array *k* depends on how
many draws arrays *0..k-1* made, so materialising a subset, reordering the declaration, or filling
on threads all change the data.
"""
import numpy as np
import pytest

from hpcagent_bench.initialize import auto_initialize
from hpcagent_bench.precision import Precision, numpy_dtype, safe_max
from hpcagent_bench.spec import KERNELS, BenchSpec
from hpcagent_bench.support import distributions
from hpcagent_bench.support.distributions import streams

#: The six samplers rewritten off scipy, plus the plain uniform.
STANDARD = ("uniform", "normal", "lognormal", "exponential", "gamma", "beta", "laplace")


def auto_init_specs(minimum_arrays: int = 2, limit: int = 4):
    """Specs that opt into :func:`auto_initialize` (declarative ``init.shapes``, no custom func)."""
    found = []
    for key in sorted(KERNELS):
        stem = key.rsplit("/", 1)[-1]
        try:
            spec = BenchSpec.load(stem)
        except Exception:  # noqa: BLE001 -- ambiguous/malformed stem: not this test's business
            continue
        if spec.init is not None and spec.init.shapes and len(spec.init.shapes) >= minimum_arrays:
            found.append((stem, spec))
        if len(found) >= limit:
            break
    return found


def test_spawn_is_reproducible_and_distinct():
    a = [g.random(8) for g in streams.spawn_streams(42, 6)]
    b = [g.random(8) for g in streams.spawn_streams(42, 6)]
    c = [g.random(8) for g in streams.spawn_streams(43, 6)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b)), "same seed must replay"
    assert not any(np.array_equal(x, y) for x, y in zip(a, c)), "different seed must diverge"
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            assert not np.array_equal(a[i], a[j]), f"streams {i} and {j} collided"


def test_stream_k_does_not_depend_on_how_many_were_spawned():
    """The whole point: asking for more arrays must not move the ones already there."""
    short = [g.random(8) for g in streams.spawn_streams(7, 3)]
    long = [g.random(8) for g in streams.spawn_streams(7, 9)]
    assert all(np.array_equal(x, y) for x, y in zip(short, long))


def test_stream_k_does_not_depend_on_draw_order():
    """Drawing array 5 first must give array 5 the same values -- a shared stream cannot do this."""
    forward = [g.random(8) for g in streams.spawn_streams(7, 6)]
    generators = streams.spawn_streams(7, 6)
    backward = [None] * 6
    for k in (5, 0, 3, 1, 4, 2):
        backward[k] = generators[k].random(8)
    assert all(np.array_equal(x, y) for x, y in zip(forward, backward))


def test_bit_generators_are_round_robined():
    got = [type(g.bit_generator) for g in streams.spawn_streams(1, 2 * len(streams.ROUND_ROBIN))]
    assert got[:len(streams.ROUND_ROBIN)] == list(streams.ROUND_ROBIN)
    assert got[len(streams.ROUND_ROBIN):] == list(streams.ROUND_ROBIN), "rotation must repeat"


def test_threaded_fill_matches_sequential(monkeypatch):
    tasks = [(lambda g=g: g.uniform(-1000, 1000, 1 << 16)) for g in streams.spawn_streams(11, 8)]
    monkeypatch.setattr(streams, "THREAD_MIN_ELEMENTS", 1 << 62)
    sequential = streams.fill(tasks, elements=1 << 20)
    tasks = [(lambda g=g: g.uniform(-1000, 1000, 1 << 16)) for g in streams.spawn_streams(11, 8)]
    monkeypatch.setattr(streams, "THREAD_MIN_ELEMENTS", 0)
    threaded = streams.fill(tasks, elements=1 << 20)
    assert all(np.array_equal(x, y) for x, y in zip(sequential, threaded))


#: A spec far outside anything a real manifest asks for. The point of the ceiling is that it holds
#: for the arguments a manifest CAN pass, not only for the defaults -- fp32 and bf16 carried an
#: ``inf`` ceiling (i.e. no clip at all) and went non-finite here, raising numpy's "overflow
#: encountered in cast" on the way.
EXTREME = {"sigma": 90.0, "scale": 1e30, "loc": 1e30}


@pytest.mark.parametrize("name", STANDARD)
@pytest.mark.parametrize("precision", list(Precision))
@pytest.mark.parametrize("spec", [{}, EXTREME], ids=["default", "extreme"])
def test_standard_distributions_stay_in_the_safe_range(name, precision, spec):
    """EVERY precision, not just the two wide ones: the narrow formats are where the clip does the
    work, and the wide ones are where its absence went unnoticed."""
    got = distributions.generate(name, (2048, ), precision, {"rng": np.random.default_rng(5), **spec})
    assert got.dtype == numpy_dtype(precision)
    as_f64 = np.asarray(got, dtype=np.float64)
    assert np.isfinite(as_f64).all(), f"{name} produced a non-finite value at {precision} ({spec})"
    assert np.abs(as_f64).max() <= safe_max(precision)


def test_no_precision_has_an_unbounded_ceiling():
    """``inf`` in the table reads as "wide enough not to worry" and silently disables the clip.
    Every format has a largest finite value, so every entry has to be one."""
    unbounded = [p.value for p in Precision if not np.isfinite(safe_max(p))]
    assert not unbounded, f"safe_max is inf for {unbounded}, which makes clip_to_precision a no-op"


@pytest.mark.parametrize("name", STANDARD)
def test_standard_distributions_follow_their_stream(name):
    kwargs = {"rng": np.random.default_rng(3)}
    a = distributions.generate(name, (64, ), Precision.FP64, kwargs)
    b = distributions.generate(name, (64, ), Precision.FP64, {"rng": np.random.default_rng(3)})
    c = distributions.generate(name, (64, ), Precision.FP64, {"rng": np.random.default_rng(4)})
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("stem_spec", auto_init_specs(), ids=lambda s: s[0])
def test_auto_initialize_is_reproducible(stem_spec):
    _, spec = stem_spec
    a = auto_initialize(spec, "S", Precision.FP64, seed=42)
    b = auto_initialize(spec, "S", Precision.FP64, seed=42)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("stem_spec", auto_init_specs(), ids=lambda s: s[0])
def test_auto_initialize_threading_does_not_change_the_data(stem_spec, monkeypatch):
    _, spec = stem_spec
    monkeypatch.setattr(streams, "THREAD_MIN_ELEMENTS", 1 << 62)
    sequential = auto_initialize(spec, "S", Precision.FP64, seed=42)
    monkeypatch.setattr(streams, "THREAD_MIN_ELEMENTS", 0)
    threaded = auto_initialize(spec, "S", Precision.FP64, seed=42)
    assert all(np.array_equal(x, y) for x, y in zip(sequential, threaded))
