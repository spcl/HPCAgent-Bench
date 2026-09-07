# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""promote_unsubmitted.py against a judge that enforces the judge's OWN request contract.

The bug this covers was not reachable by any mock. `promote()` built a body, posted it, and the
judge refused it 400 for a field the body never carried -- so a test that stubs `urlopen` and
asserts on what was sent passes just as happily against the broken version. What made the failure
invisible in production is the same shape: the script prints "refused 400" per line and exits 0,
and it ran at teardown where nobody reads the log. It refused every promotion it ever attempted.

So the server here validates with :func:`hpcagent_bench.harness.service.rank_error` -- the judge's
own function, not a re-statement of it -- and the control case asserts that a body without the
field really is refused, which is what pins this test to the contract rather than to today's code.
"""

import http.server
import importlib.util
import json
import pathlib
import sqlite3
import sys
import threading
from types import ModuleType
from typing import ClassVar

import pytest

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "containers/cluster/example-script"

#: What the judge answers a promotion it accepted.
GRADE = {"correct": True, "build_ok": True, "speedup": 7.5}

#: The rank this judge answers to -- deliberately NOT the single-judge default, so a promoter that
#: merely omits the field, or hardcodes 0, fails instead of passing by luck.
JUDGE_RANK = 3


@pytest.fixture(name="promoter")
def promoter_fixture() -> ModuleType:
    spec = importlib.util.spec_from_file_location("promote_unsubmitted_live", EXAMPLE / "promote_unsubmitted.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Judge(http.server.BaseHTTPRequestHandler):
    """A judge that answers /health and grades /submit through the real rank guard."""

    posted: ClassVar[list[dict]] = []

    def log_message(self, *args):  # keep pytest output readable
        pass

    def reply(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply(200, {"status": "ok", "judge_rank": JUDGE_RANK})

    def do_POST(self):
        from hpcagent_bench.harness.service import rank_error

        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        # THE judge's own guard, imported rather than restated: a request naming no rank is refused
        # before anything is graded, and that is the whole failure being reproduced here.
        refusal = rank_error(JUDGE_RANK, body.get("rank"))
        if refusal is not None:
            return self.reply(*refusal)
        Judge.posted.append(body)
        self.reply(200, dict(GRADE))


@pytest.fixture(name="judge")
def judge_fixture():
    Judge.posted.clear()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Judge)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def run_dir_with_one_verified_kernel(tmp_path: pathlib.Path) -> pathlib.Path:
    """A run whose judge shard holds one correct-and-faster call and the source behind it."""
    rank_dir = tmp_path / "judge" / "rank-0"
    rank_dir.mkdir(parents=True)
    (rank_dir / "gemm.c").write_text("void gemm(void){}", encoding="utf-8")
    con = sqlite3.connect(rank_dir / "hpcagent_bench0.db")
    con.execute("create table submissions (benchmark text)")
    con.execute("create table calls (benchmark text, run_id text, correct int, speedup real)")
    con.execute("create table sources (benchmark text, run_id text, ts int, path text, language text)")
    con.execute("insert into calls values ('gemm', 'arm.n0.p1.w1', 1, 7.5)")
    con.execute("insert into sources values ('gemm', 'arm.n0.p1.w1', 1, 'gemm.c', 'c')")
    con.commit()
    con.close()
    return tmp_path


def test_a_promotion_is_accepted_by_a_real_judge(promoter, judge, tmp_path, capsys, monkeypatch):
    """End to end through main(): discover the kernel, discover the rank, land the submission."""
    run_dir = run_dir_with_one_verified_kernel(tmp_path)
    monkeypatch.setattr(sys, "argv", ["promote_unsubmitted.py", str(run_dir), "--judge", judge])
    assert promoter.main() == 0
    out = capsys.readouterr().out
    assert "SUBMITTED speedup=7.50x" in out, out
    assert "refused" not in out, out
    (posted,) = Judge.posted
    assert posted["rank"] == JUDGE_RANK, "the rank must be the judge's own, not a guess"
    assert posted["kernel"] == "gemm"
    assert posted["source"] == "void gemm(void){}"
    assert posted["optimizer"] == "promoted-unsubmitted"


def test_the_same_judge_refuses_a_body_that_names_no_rank(promoter, judge, tmp_path, capsys):
    """The control. Without this the test above would pass against a judge that checks nothing,
    which is exactly how a promoter that never sent a rank looked green for its whole life."""
    run_dir = run_dir_with_one_verified_kernel(tmp_path)
    (item,) = promoter.candidates(run_dir)
    original = promoter.promote

    def rankless(judge_url, entry, dry_run, rank):
        return original(judge_url, entry, dry_run, rank=None)

    promoter.promote = rankless
    try:
        outcome = promoter.promote(judge, item, False, JUDGE_RANK)
    finally:
        promoter.promote = original
    assert outcome == "refused 400"
    assert Judge.posted == [], "a refused promotion must not be recorded as graded"
