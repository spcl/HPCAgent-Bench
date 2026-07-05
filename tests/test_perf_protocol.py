# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The configs x shapes performance protocol primitives in :mod:`optarena.fuzz`:
config enumeration, correctness edge shapes, and timed large shapes.

See docs/DESIGN_perf_protocol_configs_shapes.md. These are pure resolvers (no
emitter / FFI), so they run everywhere.
"""
import pytest

from optarena import fuzz

# Validates the REAL size ranges/large-shape draws -> opt out of the suite-wide
# small-size cap (the autouse _cap_fuzz_sizes fixture in conftest). Pure resolvers,
# so no speed cost from the full range.
pytestmark = pytest.mark.real_fuzz


# --------------------------------------------------------------------------- #
# enumerate_configs
# --------------------------------------------------------------------------- #
def test_enumerate_configs_none_yields_single_empty():
    assert fuzz.enumerate_configs(None) == [{}]
    assert fuzz.enumerate_configs({}) == [{}]


def test_enumerate_configs_valid_list_verbatim():
    configs = {"valid": [{"mode": "a"}, {"mode": "b"}]}
    assert fuzz.enumerate_configs(configs) == [{"mode": "a"}, {"mode": "b"}]


def test_enumerate_configs_sets_and_rules_cartesian_filtered():
    configs = {"sets": {"x": [1, 2], "y": [10, 20]}, "rules": ["x == 1 or y == 10"]}
    got = fuzz.enumerate_configs(configs, max_configs=10)
    # full product is 4; the rule drops {x:2, y:20}
    assert {"x": 2, "y": 20} not in got
    assert len(got) == 3
    assert all(c["x"] == 1 or c["y"] == 10 for c in got)


def test_enumerate_configs_caps_at_max(caplog):
    # a valid list of 12 configs is capped to a deterministic seeded subset of 5
    configs = {"valid": [{"i": i} for i in range(12)]}
    got = fuzz.enumerate_configs(configs, max_configs=5)
    assert len(got) == 5
    assert all(c in configs["valid"] for c in got)
    # the cap is deterministic (same seed -> same subset)
    assert fuzz.enumerate_configs(configs, max_configs=5) == got


def test_enumerate_configs_no_cap_when_under_limit():
    configs = {"valid": [{"i": 0}, {"i": 1}, {"i": 2}]}
    assert fuzz.enumerate_configs(configs, max_configs=5) == configs["valid"]


# --------------------------------------------------------------------------- #
# edge_shapes
# --------------------------------------------------------------------------- #
def test_edge_shapes_are_small_absolute_independent_of_range():
    # The fuzz range starts LARGE (lo=4096); edge shapes must still be the small
    # structural sizes {1,3,5,6,7}, NOT clamped up to the large lower bound -- this
    # is the central anti-special-casing guarantee.
    params = {"fuzzed": {"N": [4096, 8192]}}
    shapes = dict(fuzz.edge_shapes(params))
    assert {lbl: s["N"] for lbl, s in shapes.items()} == {"one": 1, "odd": 3, "prime": 7, "nonpow2": 6, "nonaligned": 5}


def test_edge_shapes_capped_at_declared_maximum():
    # the only bound that holds: an edge value cannot exceed the declared max.
    params = {"fuzzed": {"N": [1, 4]}}
    vals = {s["N"] for _, s in fuzz.edge_shapes(params)}
    assert all(v <= 4 for v in vals)
    assert 1 in vals  # "one" always probes the degenerate size


def test_edge_shapes_merges_config_and_resolves_derive():
    params = {"fuzzed": {"n": [4, 64], "nn": {"derive": "n*n"}}}
    shapes = fuzz.edge_shapes(params, config={"mode": "x"})
    for _, s in shapes:
        assert s["mode"] == "x"  # config merged in
        assert s["nn"] == s["n"] * s["n"]  # derive resolved off the edge root


def test_edge_shapes_skips_constraint_rejected_category():
    # N must be even -> of {1,3,7,6,5} only nonpow2=6 is even and survives.
    params = {"fuzzed": {"N": [16, 4096]}}
    shapes = fuzz.edge_shapes(params, constraints=["N % 2 == 0"])
    assert all(s["N"] % 2 == 0 for _, s in shapes)
    assert [lbl for lbl, _ in shapes] == ["nonpow2"]  # only the even probe is legal


# --------------------------------------------------------------------------- #
# large_shapes
# --------------------------------------------------------------------------- #
def test_large_shapes_default_mode_n_and_upper_half():
    params = {"fuzzed": {"N": [16, 4096]}}
    shapes = fuzz.large_shapes(params, mode="all_configs_3shapes", n=3)
    assert len(shapes) == 3
    # "large" == upper half of [16, 4096] -> >= midpoint 2056.
    assert all(s["N"] >= 2056 for _, s in shapes)


def test_large_shapes_reproducible_public_seed():
    params = {"fuzzed": {"N": [16, 4096]}}
    a = fuzz.large_shapes(params, mode="all_configs_3shapes", n=3)
    b = fuzz.large_shapes(params, mode="all_configs_3shapes", n=3)
    assert [s for _, s in a] == [s for _, s in b]  # same fixed public seed => identical


def test_large_shapes_secret_mode_n_shapes_and_seed_dependent():
    params = {"fuzzed": {"N": [16, 4096]}}
    # secret mode times the SAME number of shapes as public (n, default 3), just
    # drawn from the hidden seed instead of the public one.
    s1 = fuzz.large_shapes(params, mode="secret_3shapes", n=3, secret_seed=111)
    s2 = fuzz.large_shapes(params, mode="secret_3shapes", n=3, secret_seed=222)
    assert len(s1) == 3 and len(s2) == 3
    assert [lbl for lbl, _ in s1] == ["secret0", "secret1", "secret2"]
    assert all(s["N"] >= 2056 for _, s in s1)  # still "large" (upper half)
    # a different secret seed generally selects different shapes
    assert [s["N"] for _, s in s1] != [s["N"] for _, s in s2]


def test_large_shapes_merges_config():
    params = {"fuzzed": {"N": [16, 4096]}}
    shapes = fuzz.large_shapes(params, config={"layout": "soa"}, mode="all_configs_3shapes", n=2)
    assert all(s["layout"] == "soa" for _, s in shapes)


# --------------------------------------------------------------------------- #
# fuzzed_shape (the per-config crossing of the k-iteration correctness sweep)
# --------------------------------------------------------------------------- #
def test_fuzzed_shape_is_reproducible_and_config_merged():
    params = {"fuzzed": {"N": [16, 4096]}}
    a = fuzz.fuzzed_shape(params, 0, config_ns={"mode": "x"})
    b = fuzz.fuzzed_shape(params, 0, config_ns={"mode": "x"})
    assert a == b  # same iteration seed -> identical draw
    assert a["mode"] == "x"  # config merged in
    assert 16 <= a["N"] <= 4096


def test_fuzzed_shape_iterations_differ():
    params = {"fuzzed": {"N": [16, 4096]}}
    p0 = fuzz.fuzzed_shape(params, 0)
    p1 = fuzz.fuzzed_shape(params, 1)
    assert p0["N"] != p1["N"]  # distinct iterations sample distinct sizes


def test_fuzzed_shape_resolves_derive_against_config():
    params = {"fuzzed": {"n": [4, 64], "nn": {"derive": "n*n"}}}
    s = fuzz.fuzzed_shape(params, 3, config_ns={"flag": 1})
    assert s["nn"] == s["n"] * s["n"]
    assert s["flag"] == 1
