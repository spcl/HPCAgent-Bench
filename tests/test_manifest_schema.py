# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The authored manifest schema (:mod:`hpcagent_bench.manifest_schema`) and the strict pydantic
models :class:`~hpcagent_bench.spec.BenchSpec` and :class:`~hpcagent_bench.spec.ConfigKnob`."""

import pathlib

import pytest
from pydantic import ValidationError

from hpcagent_bench import manifest_schema, paths, spec
from hpcagent_bench.spec import BenchSpec, ConfigKnob

#: A corpus manifest with ``args``, a fallback initializer and an ``mpi`` block.
GEMM = paths.BENCHMARKS / "scientific_computing" / "dense_linear_algebra" / "gemm" / "gemm.yaml"


def gemm_raw() -> dict[str, object]:
    """``gemm.yaml`` parsed; each call parses the file again, so callers may mutate the result."""
    return spec.load_yaml(GEMM.read_text(encoding="utf-8"))


def corpus_manifests() -> list[pathlib.Path]:
    """Every manifest the registry scans: ``<stem>.yaml`` under ``benchmarks/``, ``_``-prefixed excluded."""
    return sorted(p for p in paths.BENCHMARKS.rglob("*.yaml") if not p.name.startswith("_"))


def test_every_corpus_manifest_matches_the_schema() -> None:
    """A schema stricter than the corpus would refuse a shipped kernel at load."""
    failures = []
    for path in corpus_manifests():
        try:
            manifest_schema.validate_manifest(spec.load_yaml(path.read_text(encoding="utf-8")), str(path))
        except ValueError as err:
            failures.append(str(err))
    assert len(corpus_manifests()) > 0
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize(
    ("key", "value", "reported"),
    [
        ("constraints", "NI > 0", "constraints: Input should be a valid list"),
        ("languages", "c", "languages: Input should be a valid list"),
        ("allowed_precisions", "fp32", "allowed_precisions: Input should be a valid list"),
        ("scale_axes", "NI", "scale_axes: Input should be a valid list"),
        ("mpi", {"replicateable": ["A"]}, "mpi.replicateable: unknown manifest field"),
        ("mpi", {"decomposition": {"axis": "NI"}}, "mpi.decomposition.axis: Input should be a valid list"),
        ("mpi", {"decomposition": {"axis": ["NI"], "work_exponent": "1"}}, "mpi.decomposition.work_exponent"),
        ("level", True, "level: Input should be a valid integer"),
        ("timeout_s", "600", "timeout_s: expected a number (got str)"),
    ],
)
def test_a_mistyped_block_is_refused_naming_its_key_path(key: str, value: object, reported: str) -> None:
    """A string where a list belongs used to read as an empty list, and an unknown ``mpi`` key was
    kept unread: both dropped a declaration without a word."""
    raw = gemm_raw()
    raw[key] = value
    with pytest.raises(ValueError, match="invalid manifest") as err:
        BenchSpec.from_yaml(raw, source=str(GEMM))
    assert f"{GEMM}: " in str(err.value)
    assert reported in str(err.value), str(err.value)


@pytest.mark.parametrize(
    ("entry", "reported"),
    [
        ({"shape": "(NI, NJ)", "out": True, "shap": "(NI,)"}, "args.C.shap: unknown manifest field"),
        ({"shape": "(NI, NJ)", "out": "yes"}, "args.C.out: Input should be a valid boolean"),
        ({"shape": "(NI, NJ)", "out": True, "domain": [0, None]}, "args.C.domain"),
        (["(NI, NJ)"], "args.C: expected a shape string, a number or a mapping (got list)"),
    ],
)
def test_a_malformed_args_entry_is_refused_naming_its_key_path(entry: object, reported: str) -> None:
    raw = gemm_raw()
    raw["args"] = {**raw["args"], "C": entry}  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="invalid manifest") as err:
        BenchSpec.from_yaml(raw, source=str(GEMM))
    assert reported in str(err.value), str(err.value)


def test_every_violation_is_listed_in_one_error() -> None:
    """Fixing a manifest one error per load costs one round trip per typo."""
    raw = gemm_raw()
    raw["languages"] = "c"
    raw["levle"] = 1
    with pytest.raises(ValueError, match="invalid manifest") as err:
        BenchSpec.from_yaml(raw, source=str(GEMM))
    assert "languages: " in str(err.value)
    assert "levle: " in str(err.value)


def test_a_top_level_typo_names_the_closest_key() -> None:
    raw = gemm_raw()
    raw["levle"] = raw.pop("level")
    with pytest.raises(ValueError, match=r"levle: unknown manifest field \(did you mean 'level'\?\)"):
        BenchSpec.from_yaml(raw, source=str(GEMM))


@pytest.mark.parametrize(
    ("config", "reported"),
    [
        ({"tile": {"domain": [32, None]}}, "config.mapping.tile.domain[1]: expected a bool, a number or a string"),
        ({"tile": {"domian": [32]}}, "config.mapping.tile.domian: unknown manifest field"),
        ([{"tile": [32]}], "config.list[0].tile: expected a bool, a number or a string (got list)"),
        (3, "config: expected a mapping of symbol -> knob or a list of complete configs"),
    ],
)
def test_a_malformed_config_block_is_refused_naming_its_key_path(config: object, reported: str) -> None:
    raw = gemm_raw()
    raw["config"] = config
    with pytest.raises(ValueError, match="invalid manifest") as err:
        BenchSpec.from_yaml(raw, source=str(GEMM))
    assert reported in str(err.value), str(err.value)


def test_a_non_text_parameter_key_is_refused() -> None:
    """YAML 1.1 reads an unquoted ``on`` as ``True``: a size symbol named so matches no shape name."""
    raw = gemm_raw()
    raw["parameters"] = {"S": {True: 1, "NI": 1, "NJ": 1, "NK": 1}}
    with pytest.raises(ValueError, match=r"parameters\.S: key True: Input should be a valid string"):
        BenchSpec.from_yaml(raw, source=str(GEMM))


def test_a_scalar_args_value_keeps_its_int_type() -> None:
    """Strict float validation turns ``2`` into ``2.0``; the ABI passes a scalar in its manifest type."""
    entries = {
        "n": manifest_schema.ArgEntry.model_validate(2),
        "x": manifest_schema.ArgEntry.model_validate({"shape": "(n,)", "out": True}),
    }
    block = spec.parse_args(entries, None, None, "<test>")
    assert block.init.scalars == {"n": 2}
    assert type(block.init.scalars["n"]) is int


@pytest.mark.parametrize(
    ("constant", "keys"),
    [
        (
            manifest_schema.field_keys(manifest_schema.Manifest),
            {
                "name", "short-name", "relative_path", "module_name", "parameters", "config", "constraints",
                "args", "output_extent", "chain_length", "init", "languages", "allowed_precisions", "fuzz",
                "layouts", "mpi", "level", "timeout_s", "memory_cap_gb", "floor_bytes_fraction",
                "conditioning_rtol", "conditioning_atol", "mismatch_fraction", "scale_axes", "public", "array_facts",
            },
        ),
        (spec.ARG_ENTRY_KEYS, spec.ARRAY_ENTRY_KEYS | {"value"} | set(spec.ArgFlag.texts())),
        (manifest_schema.field_keys(manifest_schema.InitBlock), {"scenarios", "revalue", "constants", "workspace"}),
        (
            manifest_schema.field_keys(manifest_schema.LayoutEntry),
            {"logical_shape", "nnz", "offered", "default", "dtype", "pattern"},
        ),
        (manifest_schema.field_keys(manifest_schema.ScenarioEntry), {"description", "layouts"}),
        (manifest_schema.field_keys(manifest_schema.ConfigKnobEntry), {"domain", "value", "selects"}),
    ],
)  # fmt: skip
def test_the_key_sets_are_the_schema_fields(constant: frozenset[str], keys: set[str]) -> None:
    """The schema's key sets are exactly the manifest keys the docs and the loader name."""
    assert constant == keys


def test_a_bench_spec_refuses_a_list_for_a_tuple_field() -> None:
    """A list argument tuple is mutable, so two specs could share and edit one order."""
    with pytest.raises(ValidationError, match="output_args"):
        BenchSpec(
            short_name="k",
            name="k",
            relative_path="t/k",
            module_name="k",
            func_name="k",
            parameters={},
            input_args=(),
            array_args=(),
            output_args=["x"],  # type: ignore[arg-type]
        )


def test_a_bench_spec_refuses_an_unknown_field() -> None:
    with pytest.raises(ValidationError, match="outputs"):
        BenchSpec(
            short_name="k",
            name="k",
            relative_path="t/k",
            module_name="k",
            func_name="k",
            parameters={},
            input_args=(),
            array_args=(),
            output_args=("x",),
            outputs=("x",),  # type: ignore[call-arg]
        )


def test_replace_validates_the_changed_fields() -> None:
    """``model_copy(update=...)`` skips validation; ``replace`` must not."""
    gemm = BenchSpec.from_yaml(gemm_raw(), source=str(GEMM))
    with pytest.raises(ValidationError, match="level"):
        gemm.replace(level="1")
    assert gemm.replace(level=2).level == 2
    assert gemm.level == 1


def test_a_config_knob_refuses_an_unknown_field() -> None:
    with pytest.raises(ValidationError, match="domian"):
        ConfigKnob(domian=(1, 2))  # type: ignore[call-arg]
