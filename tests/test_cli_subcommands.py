# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Smoke tests for the collection/reporting subcommands folded in from ``scripts/``.

The former standalone ``scripts/`` entrypoints (run_benchmark / run_framework /
run_sparse_benchmark / plot_results / quickstart / pluto_affine_survey) are now
``optarena`` CLI subcommands dispatching DIRECTLY to importable package functions.
These tests assert, without any toolchain (no compile, no plot, no Pluto):

* every new subcommand is registered on the top-level parser;
* each parses a trivial invocation and binds the right ``cmd_*`` dispatcher;
* dispatch reaches the target module function DIRECTLY (no subprocess / importlib) with
  the expected, preset-resolved arguments -- the target is stubbed via ``sys.modules``
  so the heavy framework / matplotlib / Pluto stacks are never imported here;
* the preserved argparse contract holds (``preset_arg`` validation, required ``-b``).
"""
import argparse
import sys
import types

import pytest

from optarena import config
from optarena.cli import build_parser, main

NEW_SUBCOMMANDS = ("run-benchmark", "run-framework", "run-sparse", "plot", "quickstart", "pluto-survey")

#: subcommand -> (module dotted path, function name, trivial argv, expected cmd_* name).
DISPATCH = {
    "run-benchmark": ("optarena.support.collect.sweep", "run_benchmark_sweep", ["run-benchmark", "-b",
                                                                        "gemm"], "cmd_run_benchmark"),
    "run-framework": ("optarena.support.collect.sweep", "run_framework_sweep", ["run-framework", "-b",
                                                                        "gemm"], "cmd_run_framework"),
    "run-sparse": ("optarena.support.collect.sweep", "run_sparse_sweep", ["run-sparse"], "cmd_run_sparse"),
    "plot": ("optarena.plotting", "plot_heatmap", ["plot"], "cmd_plot"),
    "quickstart": ("optarena.support.collect.quickstart", "quickstart", ["quickstart"], "cmd_quickstart"),
    "pluto-survey": ("optarena.support.collect.pluto_survey", "survey", ["pluto-survey"], "cmd_pluto_survey"),
}


def _stub_module(monkeypatch, dotted, funcname, recorder):
    """Install a fake ``dotted`` module exposing ``funcname`` -> ``recorder`` so a
    subcommand's ``from dotted import funcname`` binds the stub, never the real (heavy)
    module."""
    fake = types.ModuleType(dotted)
    vars(fake)[funcname] = recorder
    monkeypatch.setitem(sys.modules, dotted, fake)


def _subcommand_choices(parser):
    action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return action.choices


def test_new_subcommands_are_registered():
    choices = _subcommand_choices(build_parser())
    for name in NEW_SUBCOMMANDS:
        assert name in choices, f"{name} not registered on the top-level parser"


@pytest.mark.parametrize("subcommand", NEW_SUBCOMMANDS)
def test_subcommand_binds_dispatcher(subcommand):
    """The trivial argv parses and binds the expected ``cmd_*`` function."""
    _dotted, _fn, argv, cmd_name = DISPATCH[subcommand]
    ns = build_parser().parse_args(argv)
    assert ns.func.__name__ == cmd_name


@pytest.mark.parametrize("subcommand", NEW_SUBCOMMANDS)
def test_subcommand_dispatches_to_module_function(subcommand, monkeypatch):
    """`main(argv)` reaches the target module function directly and returns cleanly."""
    dotted, funcname, argv, _cmd = DISPATCH[subcommand]
    calls = []

    def recorder(*args, **kwargs):
        calls.append((args, kwargs))
        return 0  # run-sparse / pluto-survey propagate this as the process exit code

    _stub_module(monkeypatch, dotted, funcname, recorder)
    assert main(argv) == 0
    assert len(calls) == 1, f"{subcommand} did not dispatch to {dotted}.{funcname}"


def test_run_benchmark_resolves_preset_and_forwards_flags(monkeypatch):
    """`-p fuzzed:7` is resolved to base `fuzzed` and the selectors are forwarded."""
    calls = []
    _stub_module(monkeypatch, "optarena.support.collect.sweep", "run_benchmark_sweep", lambda *a, **k: calls.append((a, k)))
    try:
        assert main(["run-benchmark", "-b", "atax", "-f", "numba", "-p", "fuzzed:7"]) == 0
    finally:
        config.clear_override("seeds.fuzz")  # resolve_preset('fuzzed:7') sets a process-global override
    (benchmark, framework, preset, *_rest), _kwargs = calls[0]
    assert benchmark == "atax"
    assert framework == "numba"
    assert preset == "fuzzed"  # base preset, seed stripped by resolve_preset


def test_plot_forwards_db_and_output_defaults(monkeypatch):
    calls = []
    _stub_module(monkeypatch, "optarena.plotting", "plot_heatmap", lambda **k: calls.append(k))
    assert main(["plot"]) == 0
    kwargs = calls[0]
    assert kwargs["db"] == "optarena.db"
    assert kwargs["output"] == "heatmap.pdf"
    assert kwargs["preset"] == "S"  # plot's default preset (matches the legacy plot_results.py)


def test_bad_preset_is_rejected():
    """`preset_arg` validation is preserved: a bogus preset is a clean CLI error."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run-benchmark", "-b", "gemm", "-p", "not-a-preset"])


def test_run_benchmark_requires_benchmark():
    """`-b/--benchmark` stays required on run-benchmark (as in the legacy script)."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run-benchmark"])
