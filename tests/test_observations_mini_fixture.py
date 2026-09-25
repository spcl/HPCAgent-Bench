# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``tests/data/observations-mini.db`` read through the real loader -- the small, committed,
multi-treatment fixture the other test files' synthetic frames stand in for
(:mod:`tests.data.make_observations_mini` documents how it was built). Exercised here end to end
through :func:`hpcagent_bench.experiments.read_observations` and the two scripts whose behaviour
this session changed, so at least one test in the suite reads a REAL ``.db`` rather than a frame
built in the test body.
"""

import importlib.util
import pathlib
import sys
import types

from tests.conftest import script_path

from hpcagent_bench import experiments
from hpcagent_bench.stats.figures import per_kernel

FIXTURE = pathlib.Path(__file__).with_name("data") / "observations-mini.db"


def load_script(name: str) -> types.ModuleType:
    """Import a standalone script as a module (neither directory is a package)."""
    spec = importlib.util.spec_from_file_location(name, script_path(name))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


score_change = load_script("plot_score_change")


def test_read_observations_reads_the_extracted_db_the_same_shape_as_a_csv() -> None:
    frame = experiments.read_observations(FIXTURE)
    assert not frame.empty
    assert {"arm", "packet", "benchmark", "row_kind", "speedup", "tokens"} <= set(frame.columns)
    assert set(frame.packet.unique()) == {"", "skills", "cpfsrc", "perf-playbook-cpu"}


def test_a_perf_playbook_arm_from_the_real_shaped_fixture_never_enters_the_control_side() -> None:
    """The regression this session's fix guards, read off a fixture shaped like a real extraction
    rather than a hand-built frame in the test body."""
    frame_all = score_change.load(FIXTURE, prefix="")
    control = score_change.control_rows(frame_all)
    assert not any("perf-playbook-cpu" in arm for arm in control.arm.unique())
    assert set(control.packet.unique()) == {""}


def test_three_treatments_against_the_fixtures_control_all_produce_a_panel() -> None:
    """cpf-llr-focus40's real treatments -- skills, cpfsrc, perf-playbook-cpu -- each read against
    the SAME no-packet control and each yield a comparison, which is what lets them join as square
    panels side by side. ``roster`` is every kernel ANY arm of the campaign touched, built the same
    way :func:`plot_score_change.main` builds it, since :func:`one_treatment_panel` gates arm
    coverage against exactly this list (:func:`hpcagent_bench.stats.population.complete_arms`)."""
    frame_all = score_change.load(FIXTURE, prefix="")
    control = score_change.control_rows(frame_all)
    roster = sorted(frame_all["benchmark"].dropna().astype(str).unique())
    built = {
        treatment: score_change.one_treatment_panel(frame_all, control, treatment, roster)
        for treatment in ("skills", "cpfsrc", "perf-playbook-cpu")
    }
    assert all(panel is not None for panel in built.values())
    # The roster argument must actually gate coverage, not merely be accepted: a roster kernel no
    # arm ran drops every arm from the coverage check, so the same call now yields nothing.
    unreachable_roster = [*roster, "kernel-no-arm-ever-ran"]
    assert score_change.one_treatment_panel(frame_all, control, "skills", unreachable_roster) is None, (
        "a roster kernel with zero coverage must fail complete_side_arms for every arm"
    )


def test_per_kernel_cells_read_off_the_fixture_cover_every_kernel_an_arm_ran() -> None:
    frame = experiments.read_observations(FIXTURE)
    one_arm = frame[frame.arm == "cpf-llr-focus40-qwen38-c"]
    cells = {cell.kernel for cell in per_kernel.speedup_cells(one_arm)}
    assert cells == {"argmax_with_index", "tsvc_2_s116", "tsvc_2_s119", "jacobi_1d", "gemver"}
