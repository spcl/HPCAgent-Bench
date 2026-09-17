# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for scripts/smoke_level3.py's pure result-merging logic.

The three ``run_*_smoke`` phases need dace/numba and a live benchmark corpus, so they are exercised
by actually running the driver on a compute node (see the smoke report), not here. What IS unit
tested is the part a bug would corrupt silently: turning a child's JSON (or its absence) into the
CSV row a human reads.
"""

import csv
import importlib.util
import json
import pathlib
import sys

import pytest

MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "smoke_level3.py"
SPEC = importlib.util.spec_from_file_location("smoke_level3", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
smoke_level3 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke_level3
SPEC.loader.exec_module(smoke_level3)


def test_truncated_bounds_the_message_length_and_strips_newlines() -> None:
    """A long/multiline exception must not blow up a CSV cell or break the row on a newline."""
    exc = ValueError("line one\nline two\n" + "x" * 500)
    text = smoke_level3.truncated(exc, limit=50)
    assert len(text) == 50, text
    assert "\n" not in text


def test_load_result_reports_timeout_when_no_json_was_written(tmp_path: pathlib.Path) -> None:
    """A kernel killed by the per-kernel deadline never gets to write its JSON; the row must still
    say why the kernel is missing, distinct from a kernel that quietly produced nothing."""
    row = smoke_level3.load_result(tmp_path, "ghost_kernel", timed_out=True)
    assert row == {"kernel": "ghost_kernel", "outcome": "timeout"}


def test_load_result_reports_no_result_when_the_child_never_ran(tmp_path: pathlib.Path) -> None:
    """Distinguish 'never scheduled / crashed before the deadline' from an enforced timeout, since
    the two point at different bugs (a scheduler defect vs. a genuinely slow kernel)."""
    row = smoke_level3.load_result(tmp_path, "ghost_kernel", timed_out=False)
    assert row["outcome"] == "no_result"


def test_load_result_flattens_the_three_phase_payloads(tmp_path: pathlib.Path) -> None:
    """The row a human reads must carry every phase's fields under the header's exact names, not
    just 'ok'/'not ok' -- a missing wall_s or node count silently degrades the report to noise."""
    payload = {
        "kernel": "srad",
        "parse": {"ok": True, "wall_s": 0.316, "nodes": 627, "error": ""},
        "numba": {"ok": True, "matched": True, "wall_s": 0.130, "error": ""},
        "fuzz": {"ok": True, "wall_s": 304.26, "k": 1, "solved": False, "error": ""},
    }
    (tmp_path / "srad.json").write_text(json.dumps(payload))
    row = smoke_level3.load_result(tmp_path, "srad", timed_out=False)
    assert row["outcome"] == "ok"
    assert row["parse_nodes"] == "627"
    assert row["numba_matched"] == "True"
    assert row["fuzz_wall_s"] == "304.26"
    assert row["fuzz_solved"] == "False"


def test_load_result_flags_a_worker_crash_that_escaped_all_three_phases(tmp_path: pathlib.Path) -> None:
    """Each phase already catches its own exceptions; a 'worker_error' key means something escaped
    all three guards, and that must stay visible as its own outcome rather than reading as 'ok'."""
    payload = {"kernel": "cloudsc", "worker_error": "Traceback ...\nSystemError: boom"}
    (tmp_path / "cloudsc.json").write_text(json.dumps(payload))
    row = smoke_level3.load_result(tmp_path, "cloudsc", timed_out=False)
    assert row["outcome"] == "worker_error"


def test_write_csv_escapes_commas_so_a_truncated_error_never_shifts_columns(tmp_path: pathlib.Path) -> None:
    """An exception message with a comma in it (common in Python reprs, e.g. tuple args) must not
    silently misalign the CSV -- every row must parse back to exactly len(CSV_HEADER) fields."""
    rows = [{"kernel": "amg_setup", "parse_error": "ValueError: (1, 2) mismatch, shape wrong", "outcome": "ok"}]
    out_path = tmp_path / "out.csv"
    smoke_level3.write_csv(rows, out_path)
    with out_path.open(newline="") as handle:
        parsed = list(csv.reader(handle))
    assert len(parsed[1]) == len(smoke_level3.CSV_HEADER)


def test_write_csv_header_matches_every_row_key_written_by_load_result() -> None:
    """load_result's dict keys are the contract write_csv reads by name; a renamed field on one side
    that is not renamed on the other would silently write empty cells instead of failing loudly."""
    expected_keys = {
        "kernel",
        "parse_ok",
        "parse_wall_s",
        "parse_nodes",
        "parse_error",
        "numba_ok",
        "numba_matched",
        "numba_wall_s",
        "numba_error",
        "fuzz_ok",
        "fuzz_wall_s",
        "fuzz_k",
        "fuzz_solved",
        "fuzz_cache_hit",
        "fuzz_error",
        "outcome",
    }
    assert set(smoke_level3.CSV_HEADER) == expected_keys


def test_fuzz_cache_key_changes_when_the_kernel_source_changes() -> None:
    """The cache is content-addressed by the kernel's own reference bytes: an edited reference
    must be a cache MISS, or a stale result would silently describe code that no longer exists."""
    key_a = smoke_level3.fuzz_cache_key("srad", 1, "def srad(x):\n    return x\n")
    key_b = smoke_level3.fuzz_cache_key("srad", 1, "def srad(x):\n    return x + 1\n")
    assert key_a != key_b


def test_fuzz_cache_key_changes_when_k_changes() -> None:
    """A different fuzz draw count is a different gate, not the same result run twice."""
    source = "def srad(x):\n    return x\n"
    key_a = smoke_level3.fuzz_cache_key("srad", 1, source)
    key_b = smoke_level3.fuzz_cache_key("srad", 2, source)
    assert key_a != key_b


def test_fuzz_cache_roundtrips_an_identical_result(tmp_path: pathlib.Path) -> None:
    """A cache hit must hand back the SAME result a fresh compute would have produced -- not just
    'something', since a smoke report that silently drifted from a rerun would be worse than no
    cache at all."""
    written = smoke_level3.FuzzResult(ok=True, wall_s=304.26, k=1, solved=False, error="")
    smoke_level3.save_cached_fuzz(tmp_path, "srad", "key123", written)
    loaded = smoke_level3.load_cached_fuzz(tmp_path, "srad", "key123")
    assert loaded is not None
    assert loaded.ok == written.ok
    assert loaded.wall_s == written.wall_s
    assert loaded.k == written.k
    assert loaded.solved == written.solved
    assert loaded.cache_hit is True  # the loaded copy says HOW it was satisfied; the saved one does not


def test_fuzz_cache_misses_cleanly_on_an_absent_key(tmp_path: pathlib.Path) -> None:
    """Never having cached this key is a MISS, not an exception -- the caller always has a
    well-defined 'compute it' fallback."""
    assert smoke_level3.load_cached_fuzz(tmp_path, "srad", "never-written") is None


def test_fuzz_cache_root_is_none_without_a_configured_env_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caching is opt-in by environment: a bare invocation with neither cache root set must always
    recompute, never silently write into some made-up default location."""
    monkeypatch.delenv("JIT_CACHE_ROOT", raising=False)
    monkeypatch.delenv("HPCAGENT_BENCH_CACHE", raising=False)
    assert smoke_level3.fuzz_cache_root() is None


def test_reap_kills_a_process_still_alive_past_its_deadline() -> None:
    """The scheduler's whole point is that one hung kernel cannot block the others; this proves a
    process past its deadline is actually terminated rather than left running forever."""
    import multiprocessing
    import time

    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(target=time.sleep, args=(60,))
    process.start()
    running = {"hung_kernel": smoke_level3.RunningKernel(process=process, deadline=time.monotonic() - 1.0)}
    timed_out: list[str] = []
    smoke_level3.reap(running, timed_out)
    assert timed_out == ["hung_kernel"]
    assert "hung_kernel" not in running
    assert not process.is_alive()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", *sys.argv[1:]]))
