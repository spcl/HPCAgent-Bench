# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The figure labels, checked against what the campaigns actually ran.

A display name is the one piece of a figure nothing else validates: a wrong number fails a test
somewhere, a wrong LABEL just quietly mislabels a plot in a paper. These checks are what make the
registry a system rather than a second place to be wrong.
"""

import pathlib
import re

import pytest

from hpcagent_bench import experiment_tags, paths
from hpcagent_bench.stats import palette

ENVS = paths.ROOT / "experiments"

#: `OPTARENA_OPTIMIZER=<checkpoint>` in a generated arm .env -- the ground truth for which model an
#: arm served, because the runner passes exactly this string to the inference endpoint.
OPTIMIZER = re.compile(r"^OPTARENA_OPTIMIZER=(.+)$", re.MULTILINE)

#: `HPCAGENT_BENCH_RECORD_MODEL=<tag>` in a generated arm .env -- the model tag the launcher
#: recorded for this arm. The arm string itself is provenance only and nothing may parse it.
RECORD_MODEL = re.compile(r"^HPCAGENT_BENCH_RECORD_MODEL=(.+)$", re.MULTILINE)


#: Generator SEEDS, not arms. `.env.base-<model>` and `.env.llrbase-<model>-<lang>` are the
#: templates a launcher copies and then stamps; they describe no run and record no identity.
SEED = re.compile(r"^\.env\.(base|llrbase)-")


def arm_envs() -> list[pathlib.Path]:
    return sorted(
        p for p in ENVS.glob(".env.*") if p.is_file() and not p.name.endswith(".example") and not SEED.match(p.name)
    )


def test_the_registry_parses_and_every_section_a_figure_reads_is_populated() -> None:
    """A renamed or emptied YAML key leaves the section EMPTY rather than missing, because the
    registry is a record with declared fields. Empty is what would silently put raw tags on an
    axis, so that -- not the key's presence -- is the property."""
    registry = experiment_tags.registry()
    populated = {
        "experiments": registry.experiments,
        "models": registry.models,
        "languages": registry.languages,
        "packets": registry.packets,
        "hues": registry.hues,
        "markers": registry.markers,
    }
    assert [name for name, block in populated.items() if not block] == []


@pytest.mark.parametrize("model", sorted(experiment_tags.registry().models))
def test_every_registered_model_has_a_name_and_a_checkpoint(model: str) -> None:
    """A model entry with no checkpoint cannot be checked against reality, which is the point."""
    assert experiment_tags.model_name(model) != model, f"{model} maps to itself"
    assert "/" in experiment_tags.model_checkpoint(model), f"{model} names no checkpoint"


def test_the_registered_checkpoint_is_what_the_arms_served() -> None:
    """Each model tag must serve ONE checkpoint, and it must be the one the registry names.

    This is the check that stops a swapped checkpoint from keeping its old axis label. It reads the
    generated .env files, which is what the runner actually hands the endpoint.
    """
    served: dict[str, set[str]] = {}
    for env in arm_envs():
        text = env.read_text(encoding="utf-8", errors="replace")
        optimizer = OPTIMIZER.search(text)
        model = RECORD_MODEL.search(text)
        if optimizer and model:
            served.setdefault(model.group(1).strip(), set()).add(optimizer.group(1).strip())

    assert served, f"no arm .env under {ENVS} carried both OPTARENA_OPTIMIZER and HPCAGENT_BENCH_RECORD_MODEL"
    problems = []
    for model, checkpoints in sorted(served.items()):
        expected = experiment_tags.model_checkpoint(model)
        if not expected:
            problems.append(f"{model}: served {sorted(checkpoints)} but the registry does not list it")
        elif checkpoints != {expected}:
            problems.append(f"{model}: registry says {expected!r}, arms served {sorted(checkpoints)}")
    assert not problems, "registry.yaml disagrees with the arms:\n  " + "\n  ".join(problems)


def test_every_model_the_palette_colours_also_has_a_name() -> None:
    """A model with a hue but no name gets its raw tag on the axis beside properly named ones."""
    unnamed = [m for m in palette.order("models") if experiment_tags.model_name(m) == m]
    assert not unnamed, f"models with a colour but no display name: {unnamed}"


def test_an_unknown_tag_falls_back_instead_of_raising() -> None:
    """A new campaign must not break a figure -- it gets a plain label until someone names it."""
    assert experiment_tags.display_name("brand-new-campaign") == "brand-new-campaign"
    assert experiment_tags.model_name("brand-new-model") == "brand-new-model"
    assert experiment_tags.language_name("zig") == "zig"
    assert experiment_tags.display_name("") == ""


#: The identity an arm .env stamps, and the registry block each value has to be found in. A value
#: the registry does not name still DRAWS -- in a hash colour, under a raw-string label -- so the
#: only thing standing between an unregistered packet and a mislabelled figure is this test.
RECORDED = {"EXPERIMENT": "experiments", "MODEL": "models", "LANGUAGE": "languages", "DEVICE": "devices"}


def recorded_identity(text: str) -> dict[str, str]:
    """``{key: value}`` for the RECORD_ variables one arm .env sets."""
    found = re.findall(r"^HPCAGENT_BENCH_RECORD_([A-Z]+)=(.*)$", text, re.MULTILINE)
    # ENABLED is a runner switch that happens to share the prefix, not a column.
    return {key: value.strip() for key, value in found if key != "ENABLED"}


@pytest.mark.parametrize("env", arm_envs(), ids=lambda p: p.name)
def test_every_value_an_arm_records_is_registered(env: pathlib.Path) -> None:
    """Every identity value that reaches the database can be coloured and labelled.

    Checked per ARM rather than over the union, so the failure names the file to fix. An env that
    stamps nothing is skipped: it predates the identity columns and its rows carry NULL, which is a
    fact about that campaign and not an unregistered value."""
    identity = recorded_identity(env.read_text(encoding="utf-8", errors="replace"))
    if not identity:
        pytest.skip("no recorded identity; predates the identity columns")

    unknown = []
    for key, kind in RECORDED.items():
        value = identity.get(key.removeprefix("HPCAGENT_BENCH_RECORD_"), "")
        if not value:
            continue
        if experiment_tags.canonical(kind, value.lower()) not in experiment_tags.names(kind):
            unknown.append(f"{key}={value!r} is in no `{kind}` key of registry.yaml")

    # The packet is a '+'-joined SET, and '' is the control rather than a missing value, so it is
    # checked part by part instead of as one string.
    for part in experiment_tags.packet_parts(identity.get("PACKET", "")):
        if part not in experiment_tags.names("packets"):
            unknown.append(f"packet {part!r} is in no `packets` key of registry.yaml")

    assert not unknown, f"{env.name}:\n  " + "\n  ".join(unknown)


def test_an_arm_that_records_an_experiment_records_the_whole_tuple() -> None:
    """A half-stamped arm is worse than an unstamped one: its rows join on experiment and then
    group into a NULL model, which reads as a fifth model in every per-model figure.

    This is what the `cpf-llr-focus40` and `gpu-llr-focus40` envs did -- an experiment stamp and
    nothing else, under an experiment name that was really a device and a packet."""
    partial = []
    for env in arm_envs():
        identity = recorded_identity(env.read_text(encoding="utf-8", errors="replace"))
        if not identity:
            continue
        missing = [k for k in ("EXPERIMENT", "MODEL", "LANGUAGE", "DEVICE", "ARM") if not identity.get(k)]
        if missing:
            partial.append(f"{env.name}: stamps {sorted(identity)} but not {missing}")
    assert not partial, "half-stamped arm envs:\n  " + "\n  ".join(partial)
