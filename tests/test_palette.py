# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Identity in a figure: colour is the skill packet, shape is the model.

The properties that matter are the ones a reader relies on without being told: the same packet is
the same colour in every figure, dropping a series does not repaint the survivors, a combination
reads as its lead packet's family, and identity is never carried by colour alone.
"""

import logging
import pathlib

import pytest
import yaml

from hpcagent_bench import packets
from hpcagent_bench.stats import palette

REGISTRY = yaml.safe_load((pathlib.Path(palette.__file__).parents[1] / "envs" / "registry.yaml").read_text())


def test_every_registered_packet_has_a_name():
    """A packet a figure can colour must also be a packet it can label; an unnamed one would put a
    raw tag like `cpfsrc` on a legend."""
    named = set(REGISTRY["packets"])
    assert set(palette.hue_order("packets")) <= named, sorted(set(palette.hue_order("packets")) - named)


def test_every_registered_model_has_a_name():
    named = set(REGISTRY["models"])
    assert set(palette.order("models")) <= named, sorted(set(palette.order("models")) - named)


def test_the_control_has_a_name_and_a_neutral_colour():
    """No packet is a condition, so it is drawn and labelled like one -- neutral, because it is the
    reference every treatment is read against rather than one more colour among them."""
    assert REGISTRY["packets"][""] == "No Skill Packet"
    assert palette.color("") == palette.control_color()
    assert palette.control_color() not in palette.hues()


def test_colour_does_not_depend_on_the_other_series():
    """Dropping a series must not repaint the survivors."""
    many = palette.colors(["", "cpfsrc", "cpf", "lang-skills"])
    few = palette.colors(["", "cpf"])
    assert few[""] == many[""] and few["cpf"] == many["cpf"]


def test_a_combination_keeps_its_lead_packets_family():
    """`cpfsrc+lang-skills` is a CPF arm carrying a second packet, and reads as one."""
    assert packets.lead(packets.spec_parts("cpfsrc+lang-skills")) == "cpfsrc"
    assert palette.color("cpfsrc+lang-skills") != palette.color("cpfsrc")
    assert palette.color("cpfsrc+lang-skills") != palette.color("lang-skills")


def test_order_within_a_combination_does_not_change_the_colour():
    assert palette.color("cpfsrc+lang-skills") == palette.color("lang-skills+cpfsrc")


def test_each_extra_packet_is_one_step_lighter():
    base = palette.color("cpfsrc")
    one = palette.color("cpfsrc+lang-skills")
    two = palette.color("cpfsrc+lang-skills+profiling")
    assert base != one != two
    assert packets.lighten(base, 1) == one and packets.lighten(base, 2) == two


def test_the_six_hues_are_distinct():
    assert len(set(palette.hues())) == len(palette.hues())


def test_the_first_six_packets_do_not_share_a_hue():
    """Past the sixth the ramp wraps, which the order comments justify per packet; up to it a
    collision would be an accident."""
    leads = palette.hue_order("packets")[: len(palette.hues())]
    assert len({palette.color(p) for p in leads}) == len(leads)


def test_models_do_not_share_a_shape():
    assert len(set(palette.model_markers(palette.order("models")).values())) == len(palette.order("models"))


@pytest.mark.parametrize("unknown", ["mystery", "not-a-packet"])
def test_an_unregistered_packet_is_stable_and_warns(unknown, caplog):
    """A new packet must not break a plot, and must not get a different colour in two runs of the
    same script -- which `hash()` would, being salted by PYTHONHASHSEED."""
    with caplog.at_level("WARNING"):
        first = palette.color(unknown)
    assert first == palette.color(unknown)
    assert "not in registry.yaml" in caplog.text


def test_an_unregistered_model_is_stable_and_warns(caplog):
    with caplog.at_level("WARNING"):
        first = palette.marker("nomodel")
    assert first == palette.marker("nomodel")
    assert "not in registry.yaml" in caplog.text


def test_two_packets_sharing_a_colour_in_one_figure_warn(caplog):
    """Wrapping is only safe while the wrapped pair never share a figure; this is the guard."""
    with caplog.at_level("WARNING"):
        palette.colors(["cpfsrc", "no-score-tool"])
    assert "both draw" in caplog.text


#: THE PUBLISHED COLOURS. Every entity that has appeared in a figure, pinned by VALUE.
#:
#: The registry decides a colour from its KEY ORDER, which makes the file append-only: inserting a
#: key in the middle shifts every entry after it and repaints figures that are already in a paper.
#: An order test cannot catch that -- the reordered file is still internally consistent. This can,
#: and it fails naming the exact entity whose colour moved.
#:
#: A NEW entity is added here in the same commit that registers it. A CHANGED value is a decision
#: to repaint, so it is made deliberately, with the figures regenerated.
PUBLISHED_PACKET_COLORS = {
    "": "#4d4d4d",
    "cpfsrc": "#0072b2",
    "cpf": "#e69f00",
    "lang-skills": "#009e73",
    "divide-and-conquer": "#cc79a7",
    "profiling": "#d55e00",
    "repo": "#56b4e9",
    "no-score-tool": "#0072b2",
    "cpfsrc+lang-skills": "#009cf4",
    "divide-and-conquer+profiling": "#dea9c7",
}

PUBLISHED_MODEL_MARKERS = {"qwen38": "o", "oss120b": "s", "kimi27sglang": "^", "glm53": "D"}

PUBLISHED_FRAMEWORK_COLORS = {
    "numpy": "#0072b2",
    "numba": "#e69f00",
    "cc": "#009e73",
    "dace_cpu": "#cc79a7",
    "fortran": "#d55e00",
    "cpp": "#56b4e9",
}


@pytest.mark.parametrize("packet,expected", sorted(PUBLISHED_PACKET_COLORS.items()))
def test_a_published_packet_colour_did_not_move(packet, expected):
    assert palette.color(packet) == expected, (
        f"{packet!r} was {expected} and is now {palette.color(packet)}. A key was inserted into "
        "registry.yaml rather than appended, which repaints every figure already drawn with it. "
        "Append instead, or change this table deliberately and regenerate the figures."
    )


@pytest.mark.parametrize("model,expected", sorted(PUBLISHED_MODEL_MARKERS.items()))
def test_a_published_model_shape_did_not_move(model, expected):
    assert palette.marker(model) == expected


@pytest.mark.parametrize("framework,expected", sorted(PUBLISHED_FRAMEWORK_COLORS.items()))
def test_a_published_framework_colour_did_not_move(framework, expected):
    assert palette.framework_color(framework) == expected


def test_an_alias_wears_what_it_aliases():
    """A spelling is not an entity. `gpt-oss-120b` and `oss120b` are one model, so they must take
    one shape and one colour -- otherwise a legend that happens to hold both draws it twice."""
    assert palette.marker("gpt-oss-120b") == palette.marker("oss120b")
    assert palette.model_color("qwen3.8") == palette.model_color("qwen38")


def test_offload_is_a_device_and_a_language_not_a_packet():
    """`device=gpu` with `language=c` IS OpenMP offload, so the old `openmp-offload` packet names
    nothing the other columns do not already say. It resolves to the control, which keeps every row
    recorded under it comparable to a CPU arm on the same packet axis."""
    assert palette.color("openmp-offload") == palette.control_color()
    assert palette.color("openmp-offload+lang-skills") == palette.color("lang-skills")


def test_a_language_wears_the_same_registered_colour_whatever_else_the_figure_holds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=palette.LOG.name):
        alone = palette.language_colors(["c"])
        together = palette.language_colors(["fortran", "cpp", "c"])
    assert alone["c"] == together["c"]
    assert len(set(together.values())) == 3, together
    assert "registry.yaml" not in caplog.text


def test_an_unregistered_language_draws_but_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=palette.LOG.name):
        palette.language_colors(["a-language-nobody-registered"])
    assert "a-language-nobody-registered" in caplog.text
