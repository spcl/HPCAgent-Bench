# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A sweep metric is one module in ``hpcagent_bench/metrics``: the package scan finds it, the sweep
measures it when its ``metrics.<name>`` switch is on, and a failing metric never sinks the run."""

import pathlib
import sys
import types
from collections.abc import Iterator

import pytest

from hpcagent_bench import metrics
from hpcagent_bench.frameworks import test as sweep

PROBE = """
calls = []


def enabled():
    return True


def measure_sweep(frmwrk, impl, bench, reports, datatype):
    calls.append((impl, datatype))
    return {"probe": 1.0}


def rows(measured, **stamp):
    return [("probe", measured, stamp)]
"""


@pytest.fixture
def probe_dir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[pathlib.Path]:
    """``hpcagent_bench/metrics`` extended by a directory holding one dropped-in metric module."""
    (tmp_path / "zz_probe.py").write_text(PROBE)
    monkeypatch.setattr(metrics, "__path__", [*metrics.__path__, str(tmp_path)])
    yield tmp_path
    sys.modules.pop("hpcagent_bench.metrics.zz_probe", None)


def test_the_shipped_metrics_are_found_and_a_library_module_is_not() -> None:
    names = [name for name, _ in metrics.sweep_metrics()]
    assert names == ["autovec", "parallelism"]


def test_a_module_dropped_into_the_package_is_a_sweep_metric(probe_dir: pathlib.Path) -> None:
    found = dict(metrics.sweep_metrics())
    assert "zz_probe" in found
    stamp = {
        "timestamp": 1,
        "benchmark": "k",
        "framework": "cc",
        "flavor": None,
        "impl": "default",
        "datatype": "float64",
    }
    assert found["zz_probe"].rows({"probe": 1.0}, **stamp) == [("probe", {"probe": 1.0}, stamp)]


def test_the_sweep_measures_every_enabled_metric_and_warns_on_a_failing_one(
    probe_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    found = dict(metrics.sweep_metrics())
    monkeypatch.setattr(found["autovec"], "enabled", lambda: False)

    def broken(*_: object) -> None:
        raise RuntimeError("no sdfg today")

    monkeypatch.setattr(found["parallelism"], "enabled", lambda: True)
    monkeypatch.setattr(found["parallelism"], "measure_sweep", broken)
    owner = types.SimpleNamespace(bench=object())
    frmwrk = types.SimpleNamespace(fname="probe_fw")
    measured = sweep.Test._measure_metrics(owner, frmwrk, "impl", {}, "float32")  # type: ignore[arg-type]
    assert [(metric, value) for metric, value in measured] == [(found["zz_probe"], {"probe": 1.0})]
    assert sys.modules["hpcagent_bench.metrics.zz_probe"].calls == [("impl", "float32")]
    assert "WARNING: parallelism for probe_fw failed: no sdfg today" in capsys.readouterr().out
    assert sweep.Test._measure_metrics(owner, frmwrk, None, {}, "float32") == []  # type: ignore[arg-type]
