# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Identity in a figure: colour is the skill packet, shape is the model.

The properties that matter are the ones a reader relies on without being told: the same packet is
the same colour in every figure, dropping a series does not repaint the survivors, a combination
reads as its lead packet's family, and identity is never carried by colour alone.
"""

import pathlib

import pytest
import yaml

from hpcagent_bench.stats import palette

REGISTRY = yaml.safe_load((pathlib.Path(palette.__file__).parents[1] / "envs" / "display_names.yaml").read_text())


def test_every_registered_packet_has_a_name():
    """A packet a figure can colour must also be a packet it can label; an unnamed one would put a
    raw tag like `cpfsrc` on a legend."""
    named = set(REGISTRY["packets"])
    assert set(palette.PACKET_ORDER) <= named, sorted(set(palette.PACKET_ORDER) - named)


def test_every_registered_model_has_a_name():
    named = set(REGISTRY["models"])
    assert set(palette.MODEL_ORDER) <= named, sorted(set(palette.MODEL_ORDER) - named)


def test_the_control_has_a_name_and_a_neutral_colour():
    """No packet is a condition, so it is drawn and labelled like one -- neutral, because it is the
    reference every treatment is read against rather than one more colour among them."""
    assert REGISTRY["packets"][""] == "No Skill Packet"
    assert palette.color("") == palette.CONTROL_COLOR
    assert palette.CONTROL_COLOR not in palette.HUES


def test_colour_does_not_depend_on_the_other_series():
    """Dropping a series must not repaint the survivors."""
    many = palette.colors(["", "cpfsrc", "cpf", "lang-skills"])
    few = palette.colors(["", "cpf"])
    assert few[""] == many[""] and few["cpf"] == many["cpf"]


def test_a_combination_keeps_its_lead_packets_family():
    """`cpfsrc+lang-skills` is a CPF arm carrying a second packet, and reads as one."""
    assert palette.lead("cpfsrc+lang-skills") == "cpfsrc"
    assert palette.color("cpfsrc+lang-skills") != palette.color("cpfsrc")
    assert palette.color("cpfsrc+lang-skills") != palette.color("lang-skills")


def test_order_within_a_combination_does_not_change_the_colour():
    assert palette.color("cpfsrc+lang-skills") == palette.color("lang-skills+cpfsrc")


def test_each_extra_packet_is_one_step_lighter():
    base = palette.color("cpfsrc")
    one = palette.color("cpfsrc+lang-skills")
    two = palette.color("cpfsrc+lang-skills+profiling")
    assert base != one != two
    assert palette.lighten(base, 1) == one and palette.lighten(base, 2) == two


def test_the_six_hues_are_distinct():
    assert len(set(palette.HUES)) == len(palette.HUES)


def test_the_first_six_packets_do_not_share_a_hue():
    """Past the sixth the ramp wraps, which the order comments justify per packet; up to it a
    collision would be an accident."""
    leads = palette.PACKET_ORDER[: len(palette.HUES)]
    assert len({palette.color(p) for p in leads}) == len(leads)


def test_models_do_not_share_a_shape():
    assert len(set(palette.markers(palette.MODEL_ORDER).values())) == len(palette.MODEL_ORDER)


@pytest.mark.parametrize("unknown", ["mystery", "not-a-packet"])
def test_an_unregistered_packet_is_stable_and_warns(unknown, caplog):
    """A new packet must not break a plot, and must not get a different colour in two runs of the
    same script -- which `hash()` would, being salted by PYTHONHASHSEED."""
    with caplog.at_level("WARNING"):
        first = palette.color(unknown)
    assert first == palette.color(unknown)
    assert "not registered" in caplog.text


def test_an_unregistered_model_is_stable_and_warns(caplog):
    with caplog.at_level("WARNING"):
        first = palette.marker("nomodel")
    assert first == palette.marker("nomodel")
    assert "not registered" in caplog.text


def test_two_packets_sharing_a_colour_in_one_figure_warn(caplog):
    """Wrapping is only safe while the wrapped pair never share a figure; this is the guard."""
    with caplog.at_level("WARNING"):
        palette.colors(["cpfsrc", "no-score-tool"])
    assert "both draw" in caplog.text
