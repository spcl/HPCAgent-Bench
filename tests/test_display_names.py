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


def arm_envs() -> list[pathlib.Path]:
    return sorted(p for p in ENVS.glob(".env.*") if p.is_file() and not p.name.endswith(".example"))


def test_the_registry_parses_and_has_every_section() -> None:
    registry = experiment_tags.registry()
    assert set(registry) >= {"experiments", "models", "languages"}, sorted(registry)


@pytest.mark.parametrize("model", sorted(experiment_tags.registry()["models"]))
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
    assert not problems, "display_names.yaml disagrees with the arms:\n  " + "\n  ".join(problems)


def test_every_model_the_palette_colours_also_has_a_name() -> None:
    """A model with a hue but no name gets its raw tag on the axis beside properly named ones."""
    unnamed = [m for m in palette.MODEL_ORDER if experiment_tags.model_name(m) == m]
    assert not unnamed, f"models with a colour but no display name: {unnamed}"


def test_an_unknown_tag_falls_back_instead_of_raising() -> None:
    """A new campaign must not break a figure -- it gets a plain label until someone names it."""
    assert experiment_tags.display_name("brand-new-campaign") == "brand-new-campaign"
    assert experiment_tags.model_name("brand-new-model") == "brand-new-model"
    assert experiment_tags.language_name("zig") == "zig"
    assert experiment_tags.display_name("") == ""
