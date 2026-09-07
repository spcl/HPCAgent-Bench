# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""promote_unsubmitted.py: recovering a verified kernel an agent was killed before submitting.

Every judge request must name the rank it is addressed to, and an absent rank is refused with a
400 before anything is graded (``service.rank_error``). This script never sent one, so every
promotion it ever attempted was refused: 626521 reported "refused 400" on all three of its lines
and 626523 on its one, while 18 verified-correct-and-faster kernels across the two arms -- one of
them 76.6x -- reached no submissions table at all. Nothing downstream noticed, because a refused
promotion prints a word and exits 0.

The device half matters for the same reason: a hip submission is two translation units, so a GPU
promotion carrying only the host unit would be refused for what looks like the agent's mistake.
"""

import importlib.util
import io
import json
import pathlib
import sqlite3
import sys
from types import ModuleType

import pytest

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "containers/cluster/example-script"


def load_example_module(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="promoter")
def promoter_fixture() -> ModuleType:
    return load_example_module("promote_unsubmitted")


def fake_urlopen(captured: list, health: dict | None = None, body: dict | None = None):
    """Stand-in for urllib.request.urlopen recording every request it is handed."""

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

    def opener(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        captured.append(req)
        if url.endswith("/health"):
            return Response(json.dumps(health if health is not None else {"judge_rank": 3}).encode())
        return Response(json.dumps(body if body is not None else {"correct": True, "build_ok": True}).encode())

    return opener


def test_a_promotion_names_the_judge_rank(promoter, monkeypatch):
    """The whole bug: without this the judge refuses with a 400 and nothing is ever recovered."""
    captured: list = []
    monkeypatch.setattr(promoter.urllib.request, "urlopen", fake_urlopen(captured))
    item = {"kernel": "gemm", "run_id": "arm.n0.p1.w1", "language": "c", "source": "void gemm(void){}"}
    outcome = promoter.promote("http://judge:8800", item, dry_run=False, rank=3)
    body = json.loads(captured[-1].data)
    assert body["rank"] == 3
    assert body["optimizer"] == "promoted-unsubmitted"
    assert outcome.startswith("SUBMITTED")


def test_the_rank_is_asked_of_the_judge_itself(promoter, monkeypatch):
    """Configured ranks go stale; /health is the one route that reports its own."""
    monkeypatch.setattr(promoter.urllib.request, "urlopen", fake_urlopen([], health={"judge_rank": 2}))
    assert promoter.judge_rank("http://judge:8800") == 2


def test_the_upstream_judges_spelling_is_accepted_too(promoter, monkeypatch):
    """The router answers `judge_rank`, the judge behind it `rank`; either is authoritative."""
    monkeypatch.setattr(promoter.urllib.request, "urlopen", fake_urlopen([], health={"rank": 1}))
    assert promoter.judge_rank("http://judge:8800") == 1


def test_an_unreachable_judge_falls_back_rather_than_skipping(promoter, monkeypatch):
    """A promotion attempted with the default rank still beats one never attempted."""

    def boom(req, timeout=None):
        raise OSError("no route to host")

    monkeypatch.setattr(promoter.urllib.request, "urlopen", boom)
    monkeypatch.delenv("JUDGE_RANK", raising=False)
    assert promoter.judge_rank("http://judge:8800") == promoter.DEFAULT_RANK
    monkeypatch.setenv("JUDGE_RANK", "5")
    assert promoter.judge_rank("http://judge:8800") == 5


def test_a_gpu_promotion_carries_both_translation_units(promoter, monkeypatch):
    captured: list = []
    monkeypatch.setattr(promoter.urllib.request, "urlopen", fake_urlopen(captured))
    item = {
        "kernel": "gemm",
        "run_id": "arm.n0.p1.w1",
        "language": "hip",
        "source": "/* host */",
        "device_source": "/* __global__ */",
    }
    promoter.promote("http://judge:8800", item, dry_run=False, rank=0)
    body = json.loads(captured[-1].data)
    assert body["source"] == "/* host */"
    assert body["device_source"] == "/* __global__ */"


def make_run_dir(tmp_path: pathlib.Path, rows: list[tuple[str, str, str]]) -> pathlib.Path:
    """A run dir whose judge shard holds one verified call and ``rows`` of (language, path, text)."""
    rank = tmp_path / "judge" / "rank-0"
    rank.mkdir(parents=True)
    con = sqlite3.connect(rank / "hpcagent_bench0.db")
    con.execute("create table submissions (benchmark text)")
    con.execute("create table calls (benchmark text, run_id text, correct int, speedup real)")
    con.execute("create table sources (benchmark text, run_id text, ts int, path text, language text)")
    con.execute("insert into calls values ('gemm', 'arm.n0.p1.w1', 1, 4.0)")
    for index, (language, name, text) in enumerate(rows):
        (rank / name).write_text(text, encoding="utf-8")
        con.execute("insert into sources values ('gemm', 'arm.n0.p1.w1', ?, ?, ?)", (index + 1, name, language))
    con.commit()
    con.close()
    return tmp_path


def test_the_device_row_is_never_submitted_as_the_host_source(promoter, tmp_path):
    """Both halves sort together by ts, so an unfiltered 'newest wins' picks the device unit."""
    run_dir = make_run_dir(tmp_path, [("hip", "host.hip", "/* host */"), ("hip:device", "dev.hip", "/* __global__ */")])
    (item,) = promoter.candidates(run_dir)
    assert item["language"] == "hip"
    assert item["source"] == "/* host */"
    assert item["device_source"] == "/* __global__ */"


def test_a_host_only_arm_promotes_without_a_device_unit(promoter, tmp_path):
    run_dir = make_run_dir(tmp_path, [("c", "gemm.c", "void gemm(void){}")])
    (item,) = promoter.candidates(run_dir)
    assert item["source"] == "void gemm(void){}"
    assert "device_source" not in item


def make_run_dir_many(tmp_path: pathlib.Path, kernels: list[tuple[str, float]]) -> pathlib.Path:
    """A run dir holding several verified-but-unsubmitted kernels of differing worth."""
    rank = tmp_path / "judge" / "rank-0"
    rank.mkdir(parents=True)
    con = sqlite3.connect(rank / "hpcagent_bench0.db")
    con.execute("create table submissions (benchmark text)")
    con.execute("create table calls (benchmark text, run_id text, correct int, speedup real)")
    con.execute("create table sources (benchmark text, run_id text, ts int, path text, language text)")
    for name, speedup in kernels:
        (rank / f"{name}.c").write_text(f"/* {name} */", encoding="utf-8")
        con.execute("insert into calls values (?, ?, 1, ?)", (name, f"arm.{name}", speedup))
        con.execute("insert into sources values (?, ?, 1, ?, 'c')", (name, f"arm.{name}", f"{name}.c"))
    con.commit()
    con.close()
    return tmp_path


def test_the_biggest_win_is_promoted_first(promoter, tmp_path):
    """The budget can cut this list short, so order has to follow WORTH. Alphabetically, `alpha`
    at 1.1x would outrank `zeta` at 76.6x and be the one that survived a truncation."""
    run_dir = make_run_dir_many(tmp_path, [("alpha", 1.1), ("zeta", 76.6), ("mid", 4.0)])
    assert [item["kernel"] for item in promoter.candidates(run_dir)] == ["zeta", "mid", "alpha"]


def test_the_budget_stops_the_pass_and_names_what_it_cut(promoter, tmp_path, monkeypatch, capsys):
    """Teardown runs inside the job's remaining wall clock: a pass that outlives it is killed with
    the allocation, losing even the promotions it already landed."""
    run_dir = make_run_dir_many(tmp_path, [("alpha", 1.1), ("zeta", 76.6), ("mid", 4.0)])
    attempted: list[str] = []

    def slow(judge, item, dry_run, rank):
        attempted.append(item["kernel"])
        return "SUBMITTED speedup=1.00x"

    monkeypatch.setattr(promoter, "promote", slow)
    monkeypatch.setattr(promoter, "judge_rank", lambda judge: 0)
    # A clock that is inside the budget for the first item and past it for every one after.
    ticks = iter([0.0, 0.0, 10_000.0, 10_000.0, 10_000.0, 10_000.0])
    monkeypatch.setattr(promoter.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(sys, "argv", ["promote_unsubmitted.py", str(run_dir), "--judge", "http://judge:8800"])
    assert promoter.main() == 0

    out = capsys.readouterr().out
    assert attempted == ["zeta"], "the most valuable kernel must be the one that fits"
    assert "budget exhausted; 2 not attempted" in out
    # Named rather than counted: they still exist in the run dir and can be collected later.
    assert "mid" in out and "alpha" in out
