# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Config single-source + no-drift regression tests.

Guards the config-parameter consolidation: every drift-prone key is read through
ONE resolver whose CODE default matches the shipped ``config.yaml`` value, so the
runtime behaves identically whether the yaml key is present, deleted, or partially
env-overridden. A drift (code default != yaml, the exact hazard the audit found on
``measurement.baseline`` / ``fuzz.correctness_size_cap``) would only surface if the key
were removed -- these tests exercise the CODE default directly by making ``config.get``
return each caller's default.
"""
import optarena.config as config
from optarena import fuzz
from optarena.harness import service, timing


def _defaults_only(monkeypatch):
    """Make ``config.get`` ignore the yaml file and hand back each caller's code
    default, so a test sees the CODE default (the drift surface), not the shipped
    yaml value."""
    monkeypatch.setattr(config, "get", lambda dotted, default=None: default)


def test_measurement_baseline_code_default_is_auto(monkeypatch):
    _defaults_only(monkeypatch)
    assert timing.measurement_baseline() == "auto"


def test_correctness_size_cap_code_default_matches_yaml_1024(monkeypatch):
    _defaults_only(monkeypatch)
    # both keys missing -> the correctness cap alone bounds the draw (size_cap off).
    assert fuzz.correctness_size_cap() == 1024


def test_n_large_shapes_resolver_is_public_and_single_source(monkeypatch):
    _defaults_only(monkeypatch)
    assert fuzz.default_n_large_shapes() == 3


def test_service_from_config_routes_baseline_through_resolver(monkeypatch):
    # A valid but non-default baseline proves from_config reads the shared resolver
    # rather than its own config key (yaml default is "track").
    monkeypatch.setattr(service, "measurement_baseline", lambda: "numpy")
    assert service.from_config().baseline == "numpy"
