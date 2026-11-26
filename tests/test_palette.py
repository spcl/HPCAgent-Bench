# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Identity in a figure: colour is the skill packet, shape is the model.

The properties that matter are the ones a reader relies on without being told: the same packet is
the same colour in every figure, dropping a series does not repaint the survivors, a combination
reads as its lead packet's family, and identity is never carried by colour alone.
"""

from collections.abc import Callable

import pytest

from hpcagent_bench import packets, vocabulary
from hpcagent_bench.stats import palette


def test_every_registered_packet_has_a_name() -> None:
    """A packet a figure can colour must also be a packet it can label; an unnamed one would put a
    raw tag like `cpf-src` on a legend."""
    named = {key for key, entry in vocabulary.PACKETS.entries.items() if entry.name}
    assert set(palette.hue_order("packets")) <= named, sorted(set(palette.hue_order("packets")) - named)
    models = {key for key, entry in vocabulary.MODELS.entries.items() if entry.name}
    assert set(palette.order("models")) <= models, sorted(set(palette.order("models")) - models)


def test_the_control_has_a_name_and_a_neutral_colour() -> None:
    """No packet is a condition, so it is drawn and labelled like one -- neutral, because it is the
    reference every treatment is read against rather than one more colour among them."""
    assert vocabulary.PACKETS.entries[""].name == "No Skill Packet"
    assert palette.color("") == palette.control_color()
    assert palette.control_color() not in palette.hues()


def test_colour_does_not_depend_on_the_other_series() -> None:
    """Dropping a series must not repaint the survivors."""
    many = palette.colors(["", "cpf-src", "cpf-tool", "lang-skills"])
    few = palette.colors(["", "cpf-tool"])
    assert few[""] == many[""]
    assert few["cpf-tool"] == many["cpf-tool"]
    # nor does the order within a combination
    assert palette.color("cpf-src+lang-skills") == palette.color("lang-skills+cpf-src")


def test_a_combination_keeps_its_lead_packets_family() -> None:
    """`cpf-src+lang-skills` is a CPF setup carrying a second packet, and reads as one."""
    assert packets.lead(packets.spec_parts("cpf-src+lang-skills")) == "cpf-cpu-src"
    assert palette.color("cpf-src+lang-skills") != palette.color("cpf-src")
    assert palette.color("cpf-src+lang-skills") != palette.color("lang-skills")


def test_each_extra_packet_is_one_step_lighter() -> None:
    base = palette.color("cpf-src")
    one = palette.color("cpf-src+lang-skills")
    two = palette.color("cpf-src+lang-skills+profiling")
    assert base != one != two
    assert palette.lighten(base, 1) == one
    assert palette.lighten(base, 2) == two


def test_the_ramp_is_tab20_dark_first_and_every_slot_is_distinct() -> None:
    """One global palette, matplotlib's tab20, taken dark half first: reading tab20 straight
    through spends the second colour of a figure on a pale wash of its first."""
    ramp = palette.hues()
    assert len(set(ramp)) == len(ramp) == 40
    assert ramp[:3] == (palette.tab20_slot(0), palette.tab20_slot(2), palette.tab20_slot(4))
    assert sorted(palette.TAB20_ORDER) == list(range(20))
    # slot 21 onward is tab20b, darkest shade of each of its five hues first
    assert ramp[20:22] == (palette.colormap_slot(palette.TAB20B, 0), palette.colormap_slot(palette.TAB20B, 4))
    assert sorted(palette.TAB20B_ORDER) == list(range(20))


def test_no_registered_packet_shares_a_slot_with_another() -> None:
    """Forty slots (tab20, then tab20b) for every packet, so nothing wraps and no packet needs an override."""
    leads = palette.hue_order("packets")
    assert len({palette.color(p) for p in leads}) == len(leads)


@pytest.mark.parametrize(
    ("lookup", "unknown"),
    [
        pytest.param(palette.color, "mystery", id="packet-color"),
        pytest.param(palette.color, "not-a-packet", id="packet-color-other-name"),
        pytest.param(palette.marker, "nomodel", id="model-marker"),
        pytest.param(palette.packet_marker, "mystery", id="packet-marker"),
    ],
)
def test_an_unregistered_entity_is_stable_and_warns(
    lookup: Callable[[str], object], unknown: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A new entity must not break a plot, and must not get a different colour or shape in two runs of
    the same script -- which `hash()` would, being salted by PYTHONHASHSEED."""
    with caplog.at_level("WARNING"):
        first = lookup(unknown)
    assert first == lookup(unknown)
    assert "is not registered" in caplog.text


def test_two_entities_sharing_a_colour_in_one_figure_warn(caplog: pytest.LogCaptureFixture) -> None:
    """The ramp still wraps for a kind with more entities than its forty slots, and
    wrapping is only safe while the wrapped pair never share a figure. Driven through the guard
    itself rather than through a pair that happens to wrap today, so the guard keeps being tested
    on the day the registry grows."""
    with caplog.at_level("WARNING"):
        palette.warn_on_collision({"cpf-src": palette.color("cpf-src"), "twin": palette.color("cpf-src")}, "packet")
    assert "both draw" in caplog.text


#: The published colours, pinned by value. A colour is the entry's registration slot, so registering a new entry
#: anywhere but last repaints figures already drawn; this table fails naming the entity whose colour moved. A new
#: entity is added here in the commit that registers it; a changed value is a deliberate repaint (regenerate the
#: figures). Repainted in v0.2.1 when slots became registration order (pages first).
PUBLISHED_PACKET_COLORS = {
    "": "#4d4d4d",
    "cpf-src": "#ff7f0e",
    "cpf-tool": "#d62728",
    "cpf-cpu-src": "#ff7f0e",
    "cpf-tool-cpu": "#d62728",
    "cpf-gpu-src": "#2ca02c",
    "cpf-tool-gpu": "#9467bd",
    "lang-skills": "#aec7e8",
    "divide-and-conquer": "#8c564b",
    "profiling": "#bcbd22",
    "repo": "#ffbb78",
    "no-score-tool": "#98df8a",
    "perf-playbook-cpu": "#c49c94",
    "kernel": "#637939",
    "cpf-src+lang-skills": "#ffa250",
    "divide-and-conquer+profiling": "#af766a",
    "caveman": "#1f77b4",
    "cpf-src-bare": "#e7ba52",
    "cpf-cpu-src-bare": "#e7ba52",
    "cpf-gpu-src-bare": "#d6616b",
}

#: Models are assigned from the front of `markers`, standalone optimizers from the back, so a new
#: model shifts neither.
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


@pytest.mark.parametrize(("packet", "expected"), sorted(PUBLISHED_PACKET_COLORS.items()))
def test_a_published_packet_colour_did_not_move(packet: str, expected: str) -> None:
    assert palette.color(packet) == expected, (
        f"{packet!r} was {expected} and is now {palette.color(packet)}: the registration order changed, "
        "which repaints every figure already drawn. Register the entry last, or change this table "
        "deliberately and regenerate the figures."
    )


@pytest.mark.parametrize(
    ("paint", "entity", "expected"),
    [(palette.marker, model, shape) for model, shape in sorted(PUBLISHED_MODEL_MARKERS.items())]
    + [(palette.framework_color, name, color) for name, color in sorted(PUBLISHED_FRAMEWORK_COLORS.items())],
)
def test_a_published_framework_colour_did_not_move(paint: Callable[[str], str], entity: str, expected: str) -> None:
    assert paint(entity) == expected


def test_an_alias_wears_what_it_aliases() -> None:
    """A spelling is not an entity. `jax_cpu` and `jax` are one framework, so they must take one colour."""
    assert palette.framework_color("jax_cpu") == palette.framework_color("jax")


def test_every_registered_treatment_wears_its_own_shape_and_none_wears_the_control_circle() -> None:
    """ "repository" and "perf playbook" drew the same plus. The shape of a packet or
    a harness comes from one registry pool, one per treatment, so two treatments can never be told
    apart by colour alone -- colour is the model's."""
    table = palette.shape_table()
    shapes = list(table.values())
    assert len(shapes) == len(set(map(repr, shapes))), table
    assert palette.CONTROL_MARKER not in shapes
    assert palette.packet_marker("repo") != palette.packet_marker("perf-playbook-cpu")
    assert palette.harness_marker("optimas") not in [palette.packet_marker(p) for p in palette.hue_order("packets")]


def test_a_packets_shape_does_not_move_when_a_later_treatment_is_registered() -> None:
    """Append-only: the pool is handed out in file order (harnesses, then packets), so the first
    harness keeps the pool's first shape no packet pins with ``marker:``, whatever is registered after it."""
    first = palette.hue_order("harnesses")[0]
    pinned = {d.marker for d in palette.registry().packet_defs.values() if d.marker}
    pool = [shape for shape in palette.registry().shapes if shape != palette.CONTROL_MARKER and shape not in pinned]
    assert palette.harness_marker(first) == pool[0]


def test_a_control_is_the_hollow_circle_in_a_lighter_shade_of_its_models_colour_in_every_figure() -> None:
    """one shape and one shade rule for "no packet" in every figure, so a reader
    learns it once. Each figure module reads it from the palette instead of keeping its own copy."""
    from hpcagent_bench.stats.figures import scaling

    style = scaling.series_style("", "qwen38")
    assert style["marker"].get_marker() == palette.CONTROL_MARKER
    assert style["markerfacecolor"] == "none"
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
    assert abs(h0 - h1) < 0.02
    assert l1 > l0
