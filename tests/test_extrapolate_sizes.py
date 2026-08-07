# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for scripts/extrapolate_sizes.py -- the measured-growth XL sizer.

Two ends are proved here without ever touching the machine (no subprocess, no compile, no
timing): feed :func:`fit_exponent` known ``(n, t)`` pairs on an exact power law and check the
recovered exponent, then check every refusal path (too fast to time, no slope, below
:data:`MIN_EXPONENT`) rejects instead of fitting the noise. :func:`extrapolate`'s XL sizing
(time-bound vs. memory-bound, the per-symbol scale, config knobs excluded) is proved the same
way, against a small stand-in spec. :func:`measured_points` is proved to never anchor a fit on
one preset's native series and the other's python series (they are different clocks -- see the
script's own :func:`read_wall_times` docstring), with ``measure``/``materialised_bytes`` mocked
out so only the series-selection logic is under test.

:func:`measure` is proved to pin ONE precision (:data:`MEASURE_PRECISION`) rather than leave the
CLI's own ``--precision all`` default in place -- 571 of 578 corpus kernels declare both fp64 and
fp32, so an unpinned sweep pools two different clocks (fp32 is often faster) into the same "best"
time the footprint (always fp64) is fit against.

A materialised-bytes regression closes the loop against the real corpus: :func:`materialised_bytes`
must resolve a hand-initialized kernel by its canonical path-key, not by ``spec.short_name`` --
the two diverge for a couple dozen real kernels, and the old code silently reported "unknown"
for every one of them (script is loaded from its file path -- ``scripts/`` is not a package)."""
import dataclasses
import importlib.util
import pathlib
import types
from typing import Dict, Optional

import pytest

from hpcagent_bench.sizing import XL_BYTE_CEILING
from hpcagent_bench.spec import KERNELS

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "extrapolate_sizes.py"


def load_extrapolate():
    """Load scripts/extrapolate_sizes.py as a module (it is a script, not a package member)."""
    spec = importlib.util.spec_from_file_location("extrapolate_sizes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ex = load_extrapolate()


def measured(preset: str, wall_ms: Optional[float], nbytes: Optional[int]) -> "ex.Measured":
    return ex.Measured(preset=preset, wall_ms=wall_ms, nbytes=nbytes)


# --------------------------------------------------------------------------------------------
# fit_exponent: the core power-law recovery, on exact synthetic data.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1.0, 2.0, 3.0])
def test_fit_exponent_recovers_exact_power_law(k: float):
    """t = C * n**k at two footprints 1024x apart must recover k exactly (up to float error)."""
    n_lo, n_hi = 2**20, 2**30  # 1 MiB -> 1 GiB, ratio 2**10
    t_lo = 2.0  # ms; comfortably above MIN_MEASURED_MS
    t_hi = t_lo * (n_hi / n_lo)**k
    points = [measured("S", t_lo, n_lo), measured("M", t_hi, n_hi)]
    fitted, why = ex.fit_exponent(points)
    assert why == ""
    assert fitted == pytest.approx(k, rel=1e-9)


def test_fit_exponent_refuses_measurement_below_floor():
    """A point under MIN_MEASURED_MS must not anchor a fit, even with a clean power law."""
    n_lo, n_hi = 2**20, 2**30
    t_lo = ex.MIN_MEASURED_MS / 2  # too fast to time honestly
    t_hi = 50.0
    fitted, why = ex.fit_exponent([measured("S", t_lo, n_lo), measured("M", t_hi, n_hi)])
    assert fitted is None
    assert "floor" in why


def test_fit_exponent_refuses_below_min_exponent():
    """A fitted k under MIN_EXPONENT (cost not tracking footprint) is refused, not returned."""
    n_lo, n_hi = 2**20, 2**30
    k = ex.MIN_EXPONENT / 2  # deliberately below the floor
    t_lo = 2.0
    t_hi = t_lo * (n_hi / n_lo)**k
    fitted, why = ex.fit_exponent([measured("S", t_lo, n_lo), measured("M", t_hi, n_hi)])
    assert fitted is None
    assert "below" in why


def test_fit_exponent_refuses_equal_footprint():
    """Two presets with the same declared footprint have no slope to fit."""
    fitted, why = ex.fit_exponent([measured("S", 2.0, 2**20), measured("M", 20.0, 2**20)])
    assert fitted is None
    assert "no slope" in why


def test_fit_exponent_refuses_when_time_does_not_grow():
    """Footprint grows but time does not (cache-resident regime): refuse, don't fit a flat line."""
    fitted, why = ex.fit_exponent([measured("S", 5.0, 2**20), measured("M", 5.0, 2**30)])
    assert fitted is None
    assert "cache level" in why


def test_fit_exponent_needs_two_usable_points():
    fitted, why = ex.fit_exponent([measured("S", None, 2**20), measured("M", 20.0, 2**30)])
    assert fitted is None
    assert "fewer than two" in why


# --------------------------------------------------------------------------------------------
# extrapolate: the XL projection built on top of the fit -- time-bound vs. memory-bound, the
# per-symbol scale, and config knobs excluded from the proposal. A minimal stand-in spec exposes
# only the two attributes extrapolate() reads (parameters, config_names).
# --------------------------------------------------------------------------------------------


class FakeSpec:

    def __init__(self,
                 parameters: Dict[str, Dict[str, object]],
                 config_names: frozenset = frozenset(),
                 track: str = "scientific_computing"):
        self.parameters = parameters
        self.config_names = config_names
        # The projection caps on the TRACK's own XL ceiling, so a spec without a track is not a
        # spec this code can be handed.
        self.track = track


def test_extrapolate_is_time_bound_when_the_projection_fits_memory():
    n_lo, n_hi = 1_000_000, 10_000_000  # k=1 (linear) fit
    points = [measured("S", 10.0, n_lo), measured("M", 100.0, n_hi)]
    spec = FakeSpec(parameters={"M": {"N": 1000}})
    out = ex.extrapolate(spec, "fake/kernel", points, target_ms=1000.0)
    assert out.ok
    assert out.exponent == pytest.approx(1.0, rel=1e-9)
    assert out.bound_by == "time"
    assert out.xl_bytes == pytest.approx(n_hi * 10, rel=1e-9)  # 100ms -> 1000ms at k=1 is 10x
    assert out.xl_ms == pytest.approx(1000.0, rel=1e-6)
    assert out.XL["N"] == pytest.approx(1000 * 10, rel=1e-9)
    assert out.S == {"N": 1000}  # the anchor's OWN values, unchanged -- apply_sizes' "S" partner


def test_extrapolate_is_memory_bound_when_the_ceiling_binds_first():
    n_lo, n_hi = 1_000_000, 10_000_000  # k=1 again
    points = [measured("S", 10.0, n_lo), measured("M", 100.0, n_hi)]
    spec = FakeSpec(parameters={"M": {"N": 1000}})
    # An outrageous time target forces the byte ceiling (or the extrapolation cap) to bind.
    out = ex.extrapolate(spec, "fake/kernel", points, target_ms=1e12)
    assert out.ok
    assert out.bound_by == "memory"
    assert out.xl_bytes == min(XL_BYTE_CEILING, n_hi * ex.MAX_EXTRAPOLATION)


def test_extrapolate_excludes_config_knobs_from_xl_and_from_the_scale():
    """A config: knob at the anchor preset is dropped from XL and does not count toward the
    scalable-symbol count the per-symbol root uses."""
    n_lo, n_hi = 1_000_000, 10_000_000
    points = [measured("S", 10.0, n_lo), measured("M", 100.0, n_hi)]
    spec = FakeSpec(parameters={"M": {"N": 1000, "ALGO": 2}}, config_names=frozenset({"ALGO"}))
    out = ex.extrapolate(spec, "fake/kernel", points, target_ms=1000.0)
    assert out.ok
    assert "ALGO" not in out.XL
    assert "ALGO" not in out.S  # dropped from BOTH ends: derive_ladder forbids a config knob at either
    # One scalable symbol (N) only, so the whole xl_bytes growth lands on it -- same answer as
    # the single-symbol case above, proving ALGO was excluded from the root, not just the output.
    assert out.XL["N"] == pytest.approx(1000 * 10, rel=1e-9)


def test_extrapolate_reports_the_fit_refusal_as_its_own_problem():
    spec = FakeSpec(parameters={"M": {"N": 1000}})
    points = [measured("S", 5.0, 2**20), measured("M", 5.0, 2**30)]  # flat: no slope
    out = ex.extrapolate(spec, "fake/kernel", points, target_ms=1000.0)
    assert not out.ok
    assert "cache level" in out.problem


# --------------------------------------------------------------------------------------------
# measure: pins ONE precision rather than the CLI's own "all" default -- most kernels declare
# fp64 AND fp32, and an unpinned sweep would pool both clocks into read_wall_times' "best".
# --------------------------------------------------------------------------------------------


def test_measure_pins_one_precision(monkeypatch, tmp_path):
    captured: Dict[str, list] = {}

    def fake_run(argv, **kw):
        captured["argv"] = argv
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(ex.subprocess, "run", fake_run)
    monkeypatch.setattr(ex, "read_wall_times", lambda path: (1.0, None))
    ex.measure("gemm", "S", framework="numpy", repeat=3, timeout=60, workdir=tmp_path)
    argv = captured["argv"]
    assert "--precision" in argv
    assert argv[argv.index("--precision") + 1] == ex.MEASURE_PRECISION
    assert ex.MEASURE_PRECISION == "fp64"  # matches hpcagent_bench.sizing.DEFAULT_DTYPE


# --------------------------------------------------------------------------------------------
# measured_points: never anchor a fit on two different clocks. measure()/materialised_bytes()
# are mocked so only the series-selection logic (the actual bug this closes) is exercised.
# --------------------------------------------------------------------------------------------


def test_measured_points_never_mixes_native_and_python_across_presets(monkeypatch):
    """S falls back to python (no native at that size); M has native. The fit must not use
    M's tighter native clock against S's looser python one -- both points must end up on the
    SAME series (python, since that is the one common to every point that ran)."""

    def fake_measure(kernel, preset, **kw):
        if preset == "S":
            return ex.Measured(preset="S", wall_ms=None, nbytes=None, python_ms=8.0, native_ms=None)
        return ex.Measured(preset="M", wall_ms=None, nbytes=None, python_ms=50.0, native_ms=20.0)

    monkeypatch.setattr(ex, "measure", fake_measure)
    monkeypatch.setattr(ex, "materialised_bytes", lambda spec, key, preset: {"S": 2**20, "M": 2**30}[preset])

    spec = types.SimpleNamespace(parameters={"S": {}, "M": {}})
    points = ex.measured_points(spec, "fake/kernel", ["S", "M"])
    by_preset = {p.preset: p for p in points}
    assert by_preset["S"].wall_ms == 8.0
    assert by_preset["M"].wall_ms == 50.0  # python, NOT native_ms=20.0 -- never mixed in


def test_measured_points_uses_native_when_every_point_has_it(monkeypatch):
    """When native is available at every measured preset, it is used at every preset (the
    tighter clock, consistently) -- this is the case the mixing guard must not disable."""

    def fake_measure(kernel, preset, **kw):
        table = {"S": (8.0, 3.0), "M": (50.0, 20.0)}
        python_ms, native_ms = table[preset]
        return ex.Measured(preset=preset, wall_ms=None, nbytes=None, python_ms=python_ms, native_ms=native_ms)

    monkeypatch.setattr(ex, "measure", fake_measure)
    monkeypatch.setattr(ex, "materialised_bytes", lambda spec, key, preset: {"S": 2**20, "M": 2**30}[preset])

    spec = types.SimpleNamespace(parameters={"S": {}, "M": {}})
    points = ex.measured_points(spec, "fake/kernel", ["S", "M"])
    by_preset = {p.preset: p for p in points}
    assert by_preset["S"].wall_ms == 3.0
    assert by_preset["M"].wall_ms == 20.0


# --------------------------------------------------------------------------------------------
# materialised_bytes: the Benchmark(...).get_data(preset) fallback for hand-initialized kernels
# must resolve by canonical path-key, not by spec.short_name (they diverge for ~26 real kernels).
# --------------------------------------------------------------------------------------------


def test_materialised_bytes_resolves_hand_initialized_kernel_by_path_key():
    """A kernel the declared shapes cannot size must be materialised through its PATH-KEY -- exactly
    the case that used to crash inside Benchmark(spec.short_name) and get swallowed into "unknown".

    Every hand-written initializer in the corpus has since had its shapes measured and declared
    (``scripts/declare_init_shapes.py``), so ``working_bytes`` answers for all of them and the
    fallback is no longer reachable from a shipped manifest. It is still the answer for the kernels
    whose declared shapes do not EVALUATE, and for the next manifest someone writes, so the shapes
    are cleared here on a real spec whose short_name diverges from its stem."""
    specs = KERNELS.specs()
    candidates = [(key, spec) for key, spec in specs.items()
                  if spec.init.func_name and key.rsplit("/", 1)[-1] != spec.short_name]
    assert candidates, "expected at least one hand-initialized kernel with short_name != stem in the corpus"
    key, real = candidates[0]
    spec = dataclasses.replace(real, init=dataclasses.replace(real.init, shapes={}))
    nbytes = ex.materialised_bytes(spec, key, "S")
    assert nbytes is not None and nbytes > 0
