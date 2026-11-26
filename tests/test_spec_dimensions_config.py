# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""SIZE DIMENSIONS vs CONFIG KNOBS manifest schema (see hpcagent_bench/spec.py's module docstring).

Locks the split: a manifest declares its size symbols under ``parameters:`` (+ optional
``config:``/``constraints:``), and ``BenchSpec.parameters`` is the merged
``{preset: {symbol: value}}`` view every consumer reads.
"""

import re
from typing import Any

import pytest

from hpcagent_bench.spec import BenchSpec, ConfigKnob
from tests.bench_specs import hermetic_site


def raw_manifest(**overrides: object) -> dict[str, Any]:
    """A minimal manifest: one graded array of literal extent, so every size symbol is the test's own."""
    base: dict[str, Any] = {"args": {"x": {"shape": "(4,)", "out": True}}}
    base.update(overrides)
    return base


#: Where the manifests load: the reference takes ``x`` alone, read from no file.
SITE = hermetic_site("dimtest", "x")


def _dimensions(spec: BenchSpec) -> dict[str, dict[str, Any]]:
    """The size symbols of ``spec.parameters``: the merged view minus every config knob."""
    return {
        preset: {sym: val for sym, val in row.items() if sym not in spec.config_names}
        for preset, row in spec.parameters.items()
    }


def test_parameters_and_config_stay_apart() -> None:
    """A manifest declaring 'parameters:' + 'config:' keeps them apart as size symbols/.config,
    while .parameters merges each config knob's representative value into every preset -- the
    view frameworks/benchmark.py, support/bindings/contract.py, and initialize.py already read."""
    raw = raw_manifest(
        parameters={"S": {"N": 16}, "M": {"N": 32}},
        config={
            "lvn_only": {"domain": [0, 1], "selects": "branch"},
            "max_iter": {"value": 200, "selects": "iteration"},
        },
    )
    spec = BenchSpec.from_dict(raw, SITE, source="<test>")
    assert _dimensions(spec) == {"S": {"N": 16}, "M": {"N": 32}}
    assert set(spec.config) == {"lvn_only", "max_iter"}
    assert spec.config["lvn_only"] == ConfigKnob(domain=(0, 1), value=None, selects="branch")
    assert spec.config["max_iter"] == ConfigKnob(domain=None, value=200, selects="iteration")
    assert spec.config["lvn_only"].representative == 0
    assert spec.config["max_iter"].representative == 200
    assert spec.parameters == {
        "S": {"N": 16, "lvn_only": 0, "max_iter": 200},
        "M": {"N": 32, "lvn_only": 0, "max_iter": 200},
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        # 'domain' and 'value' together are ambiguous: is the knob an axis or pinned?
        ({"config": {"lvn": {"domain": [32, 64], "value": 32}}}, "both 'domain' and 'value'"),
        # neither 'domain' nor 'value' leaves the knob with no concrete or fuzzable value.
        ({"config": {"lvn": {"selects": "tile"}}}, "neither 'domain'"),
        # one symbol in 'parameters' and 'config' would let the config value silently clobber the dimension.
        ({"config": {"N": {"value": 16}}}, "declared in BOTH"),
        # 'constraints:' is evaluated at LOAD against every preset's merged values; a violated one names itself.
        (
            {
                "parameters": {"S": {"nproma": 16}},
                "config": {"lvn": {"value": 32}},
                "constraints": ["lvn <= nproma"],
            },
            r"constraint 'lvn <= nproma' is violated",
        ),
        # A short curated row would leave that symbol bound to whatever the preset carried.
        ({"config": [{"a": 0, "b": 0}, {"a": 1}]}, "same symbols"),
        # Hand-picked rows are authored, so a violation is a bug: silently dropping it would shrink the graded
        # space without saying so.
        (
            {
                "config": [{"okvan": False, "okpaw": False}, {"okvan": False, "okpaw": True}],
                "constraints": ["okpaw <= okvan"],
            },
            "violates constraint",
        ),
        # ``fuzz.configs`` reads as 'only the fuzzed preset explores configs'; two homes for one space is how a
        # kernel gets graded on a space it did not declare.
        ({"fuzz": {"configs": {"valid": [{"a": 1}]}}}, re.escape("fuzz.configs")),
    ],
    ids=[
        "domain_and_value",
        "neither_domain_nor_value",
        "dimension_and_config_overlap",
        "violated_constraint",
        "curated_row_missing_a_symbol",
        "curated_row_violating_a_constraint",
        "fuzz_configs",
    ],
)
def test_an_ill_formed_config_or_constraint_is_refused_at_load(overrides: dict[str, Any], message: str) -> None:
    raw = raw_manifest(**{"parameters": {"S": {"N": 16}}, **overrides})
    with pytest.raises(ValueError, match=message):
        BenchSpec.from_dict(raw, SITE, source="<test>")


def test_a_knob_in_both_a_preset_and_init_scalars_is_rejected() -> None:
    """The preset copy wins (numerical_oracle's ``syms.setdefault``), so the init.scalars value is
    dead -- and the preset copy is then handed to the e2e size down-scaler as if it were a
    dimension. Both effects are silent, so the loader refuses the manifest instead."""
    raw = raw_manifest(
        parameters={"S": {"N": 16, "max_iter": 50}, "M": {"N": 32, "max_iter": 200}},
        args={"x": {"shape": "(4,)", "out": True}, "max_iter": 100},
    )
    with pytest.raises(ValueError, match="max_iter"):
        BenchSpec.from_dict(raw, hermetic_site("dimtest", "x", "max_iter"), source="<test>")


def test_a_knob_only_in_init_scalars_loads() -> None:
    """The same manifest with the preset copies dropped: 'parameters' is dimensions-only."""
    raw = raw_manifest(
        parameters={"S": {"N": 16}, "M": {"N": 32}},
        args={"x": {"shape": "(4,)", "out": True}, "max_iter": 100},
    )
    spec = BenchSpec.from_dict(raw, hermetic_site("dimtest", "x", "max_iter"), source="<test>")
    assert spec.init.scalars == {"max_iter": 100}
    assert spec.parameters == {"S": {"N": 16}, "M": {"N": 32}}


# The TWO compositions of ``config:``, told apart by YAML shape.
def test_a_mapping_config_crosses_its_domains_into_a_product() -> None:
    spec = BenchSpec.from_dict(
        raw_manifest(
            parameters={"S": {"N": 16}},
            config={
                "lower": {"domain": [False, True]},
                "mode": {"domain": [0, 1]},
                "cap": {"value": 25},
            },
        ),
        SITE,
        source="<test>",
    )
    assert len(spec.config_space) == 4
    assert all(row["cap"] == 25 for row in spec.config_space)
    assert {(r["lower"], r["mode"]) for r in spec.config_space} == {(False, 0), (False, 1), (True, 0), (True, 1)}


def test_constraints_filter_the_product_they_do_not_just_assert_on_it() -> None:
    """The impossible corner is carved out of the space, not merely rejected at load."""
    spec = BenchSpec.from_dict(
        raw_manifest(
            parameters={"S": {"N": 16}},
            config={
                "okvan": {"domain": [False, True]},
                "okpaw": {"domain": [False, True]},
            },
            constraints=["okpaw <= okvan"],
        ),
        SITE,
        source="<test>",
    )
    assert (False, True) not in {(r["okvan"], r["okpaw"]) for r in spec.config_space}
    assert len(spec.config_space) == 3


def test_a_list_config_is_a_curated_space_taken_verbatim() -> None:
    """A curated list is NOT a product: two flags, three rows, and no fourth row invented."""
    rows = [{"a": 0, "b": 0}, {"a": 1, "b": 0}, {"a": 1, "b": 1}]
    spec = BenchSpec.from_dict(raw_manifest(parameters={"S": {"N": 16}}, config=rows), SITE, source="<test>")
    assert list(spec.config_space) == rows
    assert spec.config == {}
    assert spec.config_names == {"a", "b"}


def test_a_curated_config_pins_a_representative_into_every_preset() -> None:
    """A plain ``-p M`` run still has a concrete value for every knob."""
    spec = BenchSpec.from_dict(
        raw_manifest(parameters={"S": {"N": 16}, "M": {"N": 32}}, config=[{"a": 7}, {"a": 9}]), SITE, source="<test>"
    )
    assert spec.parameters == {"S": {"N": 16, "a": 7}, "M": {"N": 32, "a": 7}}
    assert _dimensions(spec) == {"S": {"N": 16}, "M": {"N": 32}}


@pytest.mark.parametrize(
    ("config", "pinned", "merged"),
    [
        ({"max_iter": {"value": 100}, "mode": {"domain": [0, 1]}}, {"max_iter": 100}, {"max_iter": 100, "mode": 0}),
        # A curated list is a SPACE the harness picks a row from, so no member is a constant -- even though each
        # row states one value, the next row states a different one.
        ([{"mode": 0, "tile": 32}, {"mode": 1, "tile": 64}], {}, {}),
    ],
    ids=["value-pins-domain-does-not", "curated-list-pins-nothing"],
)
def test_a_pinned_knob_is_a_compile_time_constant_a_fuzzed_one_is_not(
    config: object, pinned: dict[str, int], merged: dict[str, int]
) -> None:
    """``pinned_config`` is what the native emitters declare as ``constexpr`` / ``parameter``.

    A knob with one ``value:`` has that value for every preset and every fuzz draw, so passing it
    across the ABI spells a compile-time constant as a runtime argument -- and the compiler cannot
    unroll a loop whose trip count it is only handed at run time. A knob with a ``domain:`` is a
    real axis the harness varies, so it stays a parameter."""
    spec = BenchSpec.from_dict(raw_manifest(parameters={"S": {"N": 16}}, config=config), SITE, source="<test>")
    assert spec.pinned_config == pinned
    # Every knob still reaches every existing consumer through the merged parameters view.
    assert merged.items() <= spec.parameters["S"].items()
