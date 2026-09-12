# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``model_of`` is a CONSUMED function, and deleting it breaks four figures at import time.

THE FAILURE THIS PREVENTS. A consolidation commit deleted this function while
``scripts/plot_tokens.py``, ``plot_score_change.py``, ``plot_arm_summary.py`` and
``plot_single_shot_score.py`` still called it, so all four died with ``AttributeError`` the next
time anyone drew a figure -- and nothing in the suite noticed, because no test called it and the
scripts have no import-time consumer. These are that consumer: the parametrised cases below fail if
the function disappears again, and the scripts are named so the next person to move it knows who
pays.
"""

import importlib
import pathlib

import pytest

from hpcagent_bench import experiment_tags

#: The scripts that call it. A figure that cannot resolve its model draws every arm as one series.
CALLERS = (
    "scripts/plot_tokens.py",
    "scripts/plot_score_change.py",
    "scripts/plot_arm_summary.py",
    "scripts/plot_single_shot_score.py",
)


def test_model_of_exists_and_is_importable() -> None:
    """The bare existence check, because the regression was an AttributeError and nothing else."""
    module = importlib.import_module("hpcagent_bench.experiment_tags")
    assert callable(module.model_of)


@pytest.mark.parametrize("script", CALLERS)
def test_every_caller_still_reaches_it(script: str) -> None:
    """Whoever moves this next finds out here, with the caller named, instead of in a figure run."""
    source = (pathlib.Path(__file__).resolve().parents[1] / script).read_text(encoding="utf-8")
    assert "model_of" in source, f"{script} no longer calls model_of; drop it from CALLERS"
    assert "experiment_tags.model_of" in source, (
        f"{script} calls model_of through something other than experiment_tags, which is how the "
        "last copy drifted out of the registry"
    )


@pytest.mark.parametrize(
    ("arm", "expected"),
    [
        ("llr40v10-oss120b-c-skills", "oss120b"),
        ("llr40v9-qwen38-fortran", "qwen38"),
        ("cpf-llr-focus40-kimi27sglang-c-cpfsrc", "kimi27sglang"),
        ("glm53llr20-glm53-c", "glm53"),
        ("llr4-qwen30b-c", "other"),
        ("", "other"),
    ],
)
def test_an_arm_resolves_to_the_model_that_ran_it(arm: str, expected: str) -> None:
    assert experiment_tags.model_of(arm) == expected


def test_a_language_token_cannot_match_inside_a_model_name() -> None:
    """THE dash-bounded case. A substring search for the language ``c`` finds one inside
    ``kimi27sglang``, and the same search for a MODEL finds ``glm53`` inside a longer word. The
    match is on whole dash-delimited tokens, so neither can happen."""
    assert experiment_tags.model_of("llr40-kimi27sglang-c") == "kimi27sglang"
    assert experiment_tags.model_of("prefix-notglm53here-c") == "other"
    assert experiment_tags.model_of("llr40-oss120bx-c") == "other"
    assert experiment_tags.model_of("xoss120b-c") == "other"


@pytest.mark.parametrize(
    ("spelling", "canonical"),
    [("llr40-gpt-oss-120b-c", "oss120b"), ("llr40-qwen3.8-c", "qwen38"), ("a-kimi-k2.7-c", "kimi27sglang")],
)
def test_a_registered_alias_resolves_to_the_entity_it_names(spelling: str, canonical: str) -> None:
    """Two spellings of one model must not split a figure into two series with two colours."""
    assert experiment_tags.model_of(spelling) == canonical


def test_the_fallback_is_a_registered_word_and_not_a_fragment() -> None:
    """An unrecognised arm gets the explicit ``other``, never a half-parsed token that the palette
    would then colour and label as though it were a model."""
    assert experiment_tags.model_of("something-entirely-new") == "other"
    assert experiment_tags.model_of("something-entirely-new", unknown="") == ""


def test_registry_order_decides_when_an_arm_names_two_models() -> None:
    """A malformed arm carrying two model tokens must resolve the same way in every process, so the
    winner is registry order rather than whichever token came first in the string."""
    order = experiment_tags.order("models")
    first, second = order[0], order[1]
    assert experiment_tags.model_of(f"x-{second}-{first}-c") == first
    assert experiment_tags.model_of(f"x-{first}-{second}-c") == first
