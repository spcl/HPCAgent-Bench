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

import importlib.util
import logging
import sys
from types import ModuleType

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from hpcagent_bench import experiment_tags, paths
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


def load_script(name: str) -> ModuleType:
    """A figure script under ``scripts/``, which is not a package."""
    spec = importlib.util.spec_from_file_location(name, paths.ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def arm_frame(conditions: list[str]) -> pd.DataFrame:
    """``plot_arm_summary.arm_points`` rows, one per (model, language, condition)."""
    rows = []
    for model in ("qwen38", "oss120b"):
        for condition in conditions:
            rows.append(
                {
                    "model": model,
                    "language": "c",
                    "condition": condition,
                    "log2_speedup": 1.0,
                    "log2_speedup_low": 0.5,
                    "log2_speedup_high": 1.5,
                    "tokens": 1e5,
                    "tokens_low": 8e4,
                    "tokens_high": 1.2e5,
                    "baseline_ns": 1e6,
                    "native_ns": 5e5,
                    "kernels": 12,
                }
            )
    return pd.DataFrame(rows)


def identity_warnings(summary: ModuleType, conditions: list[str], caplog: pytest.LogCaptureFixture) -> list[str]:
    """The palette warnings the arm-summary figure raises while drawing ``conditions``."""
    fig, ax = plt.subplots()
    with caplog.at_level(logging.WARNING, logger=palette.LOG.name):
        summary.draw_metric(ax, arm_frame(conditions), "log2_speedup", "Speedup", log=False)
    plt.close(fig)
    return [r.getMessage() for r in caplog.records if r.name == palette.LOG.name and "registry.yaml" in r.getMessage()]


def test_every_packet_the_arm_summary_can_colour_is_registered() -> None:
    """The condition vocabulary is typed into the script, so no row has to exist for it to reach a legend."""
    summary = load_script("plot_arm_summary")
    named = set(experiment_tags.names("packets"))
    packets = [key for _, key in summary.CONDITIONS]
    unknown = [p for p in packets if any(part not in named for part in experiment_tags.packet_parts(p))]
    assert not unknown, f"plot_arm_summary.CONDITIONS colours {unknown}, which no `packets` key of registry.yaml names"


@pytest.mark.parametrize("width", [2, 4], ids=["joined pair", "unjoined conditions"])
def test_the_arm_summary_figure_colours_only_registered_packets(width: int, caplog: pytest.LogCaptureFixture) -> None:
    summary = load_script("plot_arm_summary")
    conditions = [key for _, key in summary.CONDITIONS][:width]
    assert identity_warnings(summary, conditions, caplog) == []


def test_the_registry_check_catches_a_figure_colouring_an_unregistered_packet(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A check that cannot fail guards nothing. The condition vocabulary is the only route a packet has into
    this figure, so an unregistered one is added there and must come back as a warning."""
    summary = load_script("plot_arm_summary")
    monkeypatch.setattr(summary, "CONDITIONS", (*summary.CONDITIONS, ("-mystery", "a-packet-nobody-registered")))
    warnings = identity_warnings(summary, ["", "a-packet-nobody-registered"], caplog)
    assert any("a-packet-nobody-registered" in message for message in warnings), warnings
