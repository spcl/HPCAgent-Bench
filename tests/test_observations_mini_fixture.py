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

from hpcagent_bench import experiments
from hpcagent_bench.stats.figures import per_kernel

REPO = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = pathlib.Path(__file__).with_name("data") / "observations-mini.db"


def load_script(name: str) -> types.ModuleType:
    """Import ``scripts/<name>.py`` as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


score_change = load_script("plot_score_change")


def test_read_observations_reads_the_extracted_db_the_same_shape_as_a_csv() -> None:
    frame = experiments.read_observations(FIXTURE)
    assert not frame.empty
    assert {"arm", "packet", "benchmark", "record", "speedup", "tokens"} <= set(frame.columns)
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
    panels side by side."""
    frame_all = score_change.load(FIXTURE, prefix="")
    control = score_change.control_rows(frame_all)
    built = {
        treatment: score_change.one_treatment_panel(frame_all, control, treatment)
        for treatment in ("skills", "cpfsrc", "perf-playbook-cpu")
    }
    assert all(panel is not None for panel in built.values())


def test_per_kernel_cells_read_off_the_fixture_cover_every_kernel_an_arm_ran() -> None:
    frame = experiments.read_observations(FIXTURE)
    one_arm = frame[frame.arm == "cpf-llr-focus40-qwen38-c"]
    cells = {cell.kernel for cell in per_kernel.speedup_cells(one_arm)}
    assert cells == {"argmax_with_index", "tsvc_2_s116", "tsvc_2_s119", "jacobi_1d", "gemver"}
