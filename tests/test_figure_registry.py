# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A figure may only colour an entity the registry can also NAME.

Sibling of ``tests/test_display_names.py``, which checks the identity an arm RECORDS. This checks
the identity a figure DRAWS, which is the other half and fails differently: an unregistered value
still draws, in a stable hash colour, under a raw-string label, so the plot looks finished and the
legend quietly says ``cpfsrc`` where every other entry says a sentence.

The fallback is deliberate -- a new campaign must never break a plot -- so it is loud rather than
absent, and these are what stop the loudness from being the only thing between a hash colour and a
paper.
"""

import logging

import pytest

from hpcagent_bench import experiment_tags
from hpcagent_bench.stats import palette
from hpcagent_bench.stats.figures import results, signed

#: The tags the figure modules hard-code and then colour. A name typed into a builder is the one
#: kind of unregistered value no data-driven check can see, because no row has to exist for it to
#: reach a legend.
FIGURE_FRAMEWORKS: dict[str, tuple[str, ...]] = {
    "figures.signed.ARMS": tuple(signed.ARMS),
    "figures.signed.COMPARISONS": tuple(signed.COMPARISONS),
    "figures.signed.BASELINE": (signed.BASELINE,),
    "figures.signed.REFERENCE": (signed.REFERENCE,),
    "figures.results.DEFAULT_BASELINE": (results.DEFAULT_BASELINE,),
}


@pytest.mark.parametrize("source", sorted(FIGURE_FRAMEWORKS))
def test_every_framework_a_figure_names_is_registered(source: str) -> None:
    """A builder that colours a framework the registry does not carry puts a raw tag on a legend."""
    named = experiment_tags.names("frameworks")
    unknown = [tag for tag in FIGURE_FRAMEWORKS[source] if experiment_tags.canonical("frameworks", tag) not in named]
    assert not unknown, f"{source} colours {unknown}, which no `frameworks` key of registry.yaml names"


def test_every_packet_the_palette_hues_is_named() -> None:
    """The hue ramp and the name block are two halves of one identity and must hold one vocabulary."""
    named = set(experiment_tags.names("packets"))
    assert set(palette.hue_order("packets")) <= named


def test_every_framework_the_palette_hues_is_named() -> None:
    named = set(experiment_tags.names("frameworks"))
    assert set(palette.hue_order("frameworks")) <= named


def test_colouring_an_unregistered_packet_is_never_silent(caplog: pytest.LogCaptureFixture) -> None:
    """The fallback keeps the plot drawing. The warning is what stops it reaching a paper unseen."""
    with caplog.at_level(logging.WARNING, logger=palette.LOG.name):
        colour = palette.color("a-packet-nobody-registered")
    assert colour
    assert "registry.yaml" in caplog.text and "a-packet-nobody-registered" in caplog.text


def test_colouring_an_unregistered_framework_is_never_silent(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=palette.LOG.name):
        palette.framework_color("a-framework-nobody-registered")
    assert "registry.yaml" in caplog.text


def test_shaping_an_unregistered_model_is_never_silent(caplog: pytest.LogCaptureFixture) -> None:
    """Shape is the model in every figure, so an unregistered model loses its identity twice."""
    with caplog.at_level(logging.WARNING, logger=palette.LOG.name):
        palette.marker("a-model-nobody-registered")
    assert "registry.yaml" in caplog.text


def test_a_figure_that_colours_a_whole_unregistered_set_warns_once_per_entity(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A builder hands the palette a whole column at once, so the warning has to survive the batch
    call and not only the scalar one."""
    with caplog.at_level(logging.WARNING, logger=palette.LOG.name):
        palette.colors(["cpfsrc", "mystery-one", "mystery-two"])
    assert caplog.text.count("registry.yaml") == 2


def test_an_arm_name_resolves_to_a_registered_model_or_to_nothing() -> None:
    """`model_of` is the last resort for a CSV that predates the identity columns. It must return a
    tag the palette can shape, or the explicit `other` -- never a half-parsed fragment."""
    registered = set(experiment_tags.order("models"))
    for arm in ("llr40-oss120b-c-skills", "cpf-kimi27sglang-fortran", "llr40-gpt-oss-120b-c", "llr4-qwen30b-c"):
        resolved = experiment_tags.model_of(arm)
        assert resolved in registered or resolved == "other", f"{arm} -> {resolved!r}"
