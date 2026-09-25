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
    assert palette.lighten(base, 1) == one and palette.lighten(base, 2) == two


def test_the_ramp_is_tab20_dark_first_and_every_slot_is_distinct():
    """One global palette, matplotlib's tab20, taken dark half first: reading tab20 straight
    through spends the second colour of a figure on a pale wash of its first."""
    ramp = palette.hues()
    assert len(set(ramp)) == len(ramp) == 40
    assert ramp[:3] == (palette.tab20_slot(0), palette.tab20_slot(2), palette.tab20_slot(4))
    assert sorted(palette.TAB20_ORDER) == list(range(20))
    # slot 21 onward is tab20b, darkest shade of each of its five hues first
    assert ramp[20:22] == (palette.colormap_slot(palette.TAB20B, 0), palette.colormap_slot(palette.TAB20B, 4))
    assert sorted(palette.TAB20B_ORDER) == list(range(20))


def test_no_registered_packet_shares_a_slot_with_another():
    """Forty slots (tab20, then tab20b) for every packet, so nothing wraps and no packet needs an override. This is
    what the three hand-picked hex `color:` keys in registry.yaml used to buy one packet at a time."""
    leads = palette.hue_order("packets")
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


def test_two_entities_sharing_a_colour_in_one_figure_warn(caplog):
    """The ramp still wraps for a kind with more entities than its forty slots, and
    wrapping is only safe while the wrapped pair never share a figure. Driven through the guard
    itself rather than through a pair that happens to wrap today, so the guard keeps being tested
    on the day the registry grows."""
    with caplog.at_level("WARNING"):
        palette.warn_on_collision({"cpfsrc": palette.color("cpfsrc"), "twin": palette.color("cpfsrc")}, "packet")
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
#:
#: Repainted wholesale once: one global palette, matplotlib's tab20, is the user's decision.
PUBLISHED_PACKET_COLORS = {
    "": "#4d4d4d",
    "cpfsrc": "#1f77b4",
    "cpf": "#ff7f0e",
    "lang-skills": "#2ca02c",
    "divide-and-conquer": "#d62728",
    "profiling": "#9467bd",
    "repo": "#8c564b",
    "no-score-tool": "#e377c2",
    "perf-playbook-cpu": "#ff9896",
    "kernel": "#9edae5",
    "cpfsrc+lang-skills": "#389add",
    "divide-and-conquer+profiling": "#e25e5e",
    "caveman": "#393b79",
}

#: Models are assigned from the front of `markers`, standalone optimizers from the back, so a new
#: model shifts neither. dace and cpf moved off "v"/"P" once, when that rule replaced one shared
#: front-to-back sequence; figures drawn before that carry the old two shapes.
PUBLISHED_MODEL_MARKERS = {
    "qwen38": "o",
    "oss120b": "s",
    "kimi27sglang": "^",
    "glm53": "D",
    "dace": "*",
    "cpf": "X",
}  # fmt: skip

PUBLISHED_FRAMEWORK_COLORS = {
    "numpy": "#1f77b4",
    "numba": "#ff7f0e",
    "cc": "#2ca02c",
    "dace_cpu": "#d62728",
    "fortran": "#9467bd",
    "cpp": "#8c564b",
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


#: The interventions the PAPER's figures draw, each named with the figure it appears in. A reader
#: carries a colour from one figure to the next, so these must be pairwise distinct GLOBALLY -- the
#: six-hue ramp wraps at seven packets and `warn_on_collision` only ever sees one figure at a time.
PAPER_FIGURE_PACKETS: tuple[str, ...] = (
    "",  # the no-packet control, on every paired figure
    "cpf",  # llr40_paired_cpf
    "cpfsrc",  # llr40_paired_cpfsrc
    "lang-skills",  # llr40_paired_skills, gpu_paired_skills
    "divide-and-conquer",  # scicomp perf-playbook panels
    "profiling",  # scicomp perf-playbook panels
    "perf-playbook-cpu",  # playbook_forest
    "repo",  # git_scicomp
    "kernel",  # git_scicomp's control scope
    "no-score-tool",  # llr40_blind_vs_scored
)


def test_the_packets_the_paper_draws_have_pairwise_distinct_hues() -> None:
    """Colour is the entity's, and a reader carries it ACROSS figures: two interventions the paper
    draws in the same colour read as one treatment however far apart their pages are. The committed
    PDFs once had `cpfsrc` and `no-score-tool` in one blue and `cpf`, `perf-playbook-cpu` and
    `kernel` in one orange, because the six-hue ramp wrapped. tab20 has a slot for every packet."""
    chosen = {name: palette.color(name) for name in PAPER_FIGURE_PACKETS}
    clashes = {
        hue: sorted(name for name, own in chosen.items() if own == hue)
        for hue in set(chosen.values())
        if sum(1 for own in chosen.values() if own == hue) > 1
    }
    assert not clashes, clashes


def test_the_paper_packets_never_take_the_control_colour() -> None:
    """The control is the reference every treatment is read against, so no treatment may wear it."""
    treatments = [name for name in PAPER_FIGURE_PACKETS if name]
    assert palette.control_color() not in {palette.color(name) for name in treatments}
    assert palette.color("") == palette.control_color()


def test_every_intervention_git_scicomp_and_llrblind_name_has_a_registered_hue(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A scope is an intervention: `kernel` (the bare kernel), `repo` (the whole repository) and
    `no-score` (the blind condition) are compared exactly like a skill packet, so each has to come
    out of the same table with its own global hue instead of falling through to a hash colour."""
    with caplog.at_level(logging.WARNING, logger=palette.LOG.name):
        chosen = {name: palette.color(name) for name in ("kernel", "repo", "no-score")}
    assert "registry.yaml" not in caplog.text
    assert palette.control_color() not in chosen.values()
    assert len(set(chosen.values())) == len(chosen), chosen


def test_no_score_is_an_alias_of_the_registered_key_and_takes_no_hue_slot_of_its_own() -> None:
    """One intervention, one colour. A SECOND key would hand the blind condition two hues and shift
    every packet registered after it, repainting figures already drawn."""
    assert palette.color("no-score") == palette.color("no-score-tool")
    assert "no-score" not in palette.hue_order("packets")


def test_every_registered_treatment_wears_its_own_shape_and_none_wears_the_control_circle() -> None:
    """USER 2026-09-25: "repository" and "perf playbook" drew the same plus. The shape of a packet or
    a harness comes from one registry pool, one per treatment, so two treatments can never be told
    apart by colour alone -- colour is the model's."""
    table = palette.shape_table()
    shapes = list(table.values())
    assert len(shapes) == len(set(map(repr, shapes))), table
    assert palette.CONTROL_MARKER not in shapes
    assert palette.packet_marker("repo") != palette.packet_marker("perf-playbook-cpu")
    assert palette.harness_marker("openhands") not in [palette.packet_marker(p) for p in palette.hue_order("packets")]


def test_a_packets_shape_does_not_move_when_a_later_treatment_is_registered() -> None:
    """Append-only: the pool is handed out in file order, so the first packet keeps the pool's first
    free shape whatever is registered after it."""
    first = palette.hue_order("packets")[0]
    pool = [shape for shape in palette.registry().shapes if shape != palette.CONTROL_MARKER]
    assert palette.packet_marker(first) == pool[0]


def test_an_unregistered_packet_marker_is_stable_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        first = palette.packet_marker("mystery")
    assert first == palette.packet_marker("mystery")
    assert "not in registry.yaml" in caplog.text


def test_packet_markers_is_the_per_figure_dict_form() -> None:
    chosen = palette.packet_markers(["cpf", "cpfsrc", "cpf"])
    assert set(chosen) == {"cpf", "cpfsrc"}
    assert chosen["cpf"] == palette.packet_marker("cpf")


def test_model_colour_is_reused_by_the_packet_efficacy_panels() -> None:
    """``model_color`` used to serve only a figure whose sole axis was the model; the packet
    efficacy panels now read colour off it too, so a model's hue is the same one everywhere."""
    assert palette.model_color("qwen38") == palette.ordered_color("models", "qwen38")
    assert len({palette.model_color(m) for m in ("qwen38", "oss120b", "kimi27sglang")}) == 3


def test_a_harness_wears_a_registered_colour_of_its_own(caplog: pytest.LogCaptureFixture) -> None:
    """The harness comparison varies the harness and the model, so the harness takes the colour
    channel a packet figure spends on the packet, out of the same registry."""
    with caplog.at_level(logging.WARNING, logger=palette.LOG.name):
        chosen = palette.harness_colors(["claude", "miniswe", "openhands", "optimas"])
    assert "registry.yaml" not in caplog.text
    assert len(set(chosen.values())) == 4, chosen
    assert chosen["claude"] == palette.harness_color("claude")


def test_a_control_is_the_hollow_circle_in_a_lighter_shade_of_its_models_colour_in_every_figure() -> None:
    """USER 2026-09-25: one shape and one shade rule for "no packet" in every figure, so a reader
    learns it once. Each figure module reads it from the palette instead of keeping its own copy."""
    from hpcagent_bench.stats.figures import efficacy, scaling

    assert efficacy.CONTROL_MARKER == palette.CONTROL_MARKER
    style = scaling.series_style("", "qwen38")
    assert style["marker"] == palette.CONTROL_MARKER and style["markerfacecolor"] == "none"
    assert style["color"] == palette.model_shade("qwen38", palette.CONTROL_SHADE)
    assert palette.packet_marker("") == palette.CONTROL_MARKER


@pytest.mark.parametrize("step", [1, 2])
def test_repeated_series_of_one_model_are_close_shades_of_its_colour_not_other_hues(step: int) -> None:
    """A model drawn several times stays recognisably one model: its shades keep the hue and only
    get lighter."""
    import colorsys

    import matplotlib.colors

    base, shade = (matplotlib.colors.to_rgb(palette.model_shade("oss120b", s)) for s in (0, step))
    (h0, l0, _), (h1, l1, _) = (colorsys.rgb_to_hls(*rgb) for rgb in (base, shade))
    assert abs(h0 - h1) < 0.02 and l1 > l0
