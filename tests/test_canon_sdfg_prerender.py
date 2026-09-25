# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure-logic tests for scripts/canon_sdfg_prerender.py: roster parsing, the JSON line round-trip,
and the coverage aggregation the `sweep` driver's report is built from. None of this touches dace
or a real kernel -- that is exercised end to end by running the tool itself (see the module
docstring's Usage), not by a unit test that would need a working DaCe install to even collect.
"""

import json
import pathlib

import pytest


import canon_sdfg_prerender as csp  # noqa: E402


def test_parse_roster_reads_a_comma_list_sorted_and_deduplicated() -> None:
    assert csp.parse_roster("b,a,c,a") == ["a", "b", "c"]


def test_parse_roster_reads_a_file_one_kernel_per_line_with_trailing_notes(tmp_path: pathlib.Path) -> None:
    roster_file = tmp_path / "kernels.txt"
    roster_file.write_text("zeta\nalpha  # a note\n\nbeta\n")
    assert csp.parse_roster(str(roster_file)) == ["alpha", "beta", "zeta"]


def test_parse_roster_also_comma_splits_within_a_file_line(tmp_path: pathlib.Path) -> None:
    # experiments/roster.sh's roster_for emits one comma-joined line, not one name per line --
    # dumping that straight to a file and reading it back must not treat the whole line as one
    # "kernel" (an earlier bug here did exactly that and crashed writing its result file).
    roster_file = tmp_path / "kernels.txt"
    roster_file.write_text("alpha,beta,gamma\n")
    assert csp.parse_roster(str(roster_file)) == ["alpha", "beta", "gamma"]


def test_kernel_result_json_round_trips() -> None:
    result = csp.KernelResult(kernel="tsvc_2_s315", cpu="warmed", gpu="fresh", datatype="float64")
    again = csp.parse_result_line(result.to_json())
    assert again == result


def test_parse_result_line_defaults_missing_optional_fields() -> None:
    # A child that never reached the error path emits no "error" key at all (dataclasses.asdict
    # still includes it as "") -- this covers a hand-written or older line missing it outright.
    line = json.dumps({"kernel": "k", "cpu": "fresh", "gpu": "missing"})
    result = csp.parse_result_line(line)
    assert result.datatype == "" and result.error == ""


def test_coverage_table_counts_each_tag_independently() -> None:
    results = [
        csp.KernelResult(kernel="a", cpu="fresh", gpu="fresh"),
        csp.KernelResult(kernel="b", cpu="warmed", gpu="missing"),
        csp.KernelResult(kernel="c", cpu="failed", gpu="failed", error="boom"),
    ]
    table = csp.build_coverage_table(results)
    assert table["cpu"] == {"fresh": 1, "warmed": 1, "failed": 1}
    assert table["gpu"] == {"fresh": 1, "missing": 1, "failed": 1}


def test_format_coverage_table_reports_a_total_per_tag() -> None:
    table = {"cpu": {"fresh": 2, "warmed": 1}, "gpu": {"fresh": 3}}
    text = csp.format_coverage_table(table)
    assert "cpu: 3 kernel(s) -- fresh=2, warmed=1" in text
    assert "gpu: 3 kernel(s) -- fresh=3" in text


def test_default_opt_derives_from_scratch_env_var_not_a_hardcoded_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRATCH", "/some/scratch/root")
    assert csp.default_opt() == "/some/scratch/root/hpcagent-bench"
