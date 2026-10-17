# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A figure may only colour an entity the registry can also NAME.

Sibling of ``tests/test_display_names.py``, which checks the identity a setup RECORDS. This checks
the identity a figure DRAWS, which is the other half and fails differently: an unregistered value
still draws, in a stable hash colour, under a raw-string label, so the plot looks finished and the
legend quietly says ``cpf-src`` where every other entry says a sentence.

The fallback is deliberate -- a new experiment must never break a plot -- so it is loud rather than
absent, and these are what stop the loudness from being the only thing between a hash colour and a
paper.
"""

import logging
import pathlib
from types import ModuleType

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from hpcagent_bench import paths, studies, study_tags
from hpcagent_bench.stats import palette
from tests.fresh_module import module_at


def test_every_packet_the_palette_hues_is_named() -> None:
    """The hue ramp and the name block are two halves of one identity and must hold one vocabulary."""
    named = set(study_tags.names("packets"))
    assert set(palette.hue_order("packets")) <= named


def test_every_framework_the_palette_hues_is_named() -> None:
    named = set(study_tags.names("frameworks"))
    assert set(palette.hue_order("frameworks")) <= named


def test_colouring_an_unregistered_packet_is_never_silent(caplog: pytest.LogCaptureFixture) -> None:
    """The fallback keeps the plot drawing. The warning is what stops it reaching a paper unseen."""
    with caplog.at_level(logging.WARNING, logger=palette.LOG.name):
        colour = palette.color("a-packet-nobody-registered")
    assert colour
    assert "is not registered" in caplog.text
    assert "a-packet-nobody-registered" in caplog.text


def test_colouring_an_unregistered_framework_is_never_silent(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=palette.LOG.name):
        palette.framework_color("a-framework-nobody-registered")
    assert "is not registered" in caplog.text


def test_shaping_an_unregistered_model_is_never_silent(caplog: pytest.LogCaptureFixture) -> None:
    """Shape is the model in every figure, so an unregistered model loses its identity twice."""
    with caplog.at_level(logging.WARNING, logger=palette.LOG.name):
        palette.marker("a-model-nobody-registered")
    assert "is not registered" in caplog.text


def test_a_figure_that_colours_a_whole_unregistered_set_warns_once_per_entity(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A builder hands the palette a whole column at once, so the warning has to survive the batch
    call and not only the scalar one."""
    with caplog.at_level(logging.WARNING, logger=palette.LOG.name):
        palette.colors(["cpf-src", "mystery-one", "mystery-two"])
    assert caplog.text.count("is not registered") == 2


def test_a_setup_name_resolves_to_a_registered_model_or_to_nothing() -> None:
    """`model_of` is the last resort for a CSV that predates the identity columns. It must return a
    tag the palette can shape, or the explicit `other` -- never a half-parsed fragment."""
    registered = set(study_tags.order("models"))
    for setup in ("llr40-oss120b-c-skills", "cpf-kimi27sglang-fortran", "llr40-gpt-oss-120b-c", "llr4-qwen30b-c"):
        resolved = study_tags.model_of(setup)
        assert resolved in registered or resolved == "other", f"{setup} -> {resolved!r}"


def test_setup_rows_read_the_condition_off_the_recorded_packet(tmp_path: pathlib.Path) -> None:
    """A setup whose name carries no ``-cpf`` suffix but which records the ``cpf-tool`` packet loads under
    that packet, read as the CPU tool packet it was: the recorded column decides, never the setup name."""
    path = tmp_path / "observations.csv"
    pd.DataFrame([{"setup": "renamed-qwen38-c", "packet": "cpf-tool", "kernel": "k", "row_kind": "submission"}]).to_csv(
        path, index=False
    )

    frame = studies.setup_rows(path, prefix="")

    assert frame.set_index("setup").packet["renamed-qwen38-c"] == "cpf-tool-cpu"
