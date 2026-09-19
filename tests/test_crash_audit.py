# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""What ``experiments/crash_audit.py`` keeps, drops and reports never-ran, per the 2026-09-17
owed-cancel-rule crash audit: a kernel is only carried over if every episode that ever touched it,
in every job of its arm, ended cleanly. A ``submissions`` row alone is not proof of that -- it is
proof one episode finished; a second, crashed episode on the same kernel must still force a rerun.
"""

import json
import pathlib
import sqlite3
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))

import crash_audit  # noqa: E402  -- path insert above must run first

ROSTER = ["a", "b", "c"]

#: A row's ``ts``, arbitrary-but-after-any-real-commit: these tests use fake kernel names ("a", "b",
#: "c") that resolve no real manifest, so comparable_since_ms always returns 0 for them and this
#: value never actually gets compared -- it exists only because the real schema requires the column.
FAR_FUTURE_TS_MS = 10**13


def make_shard(root: pathlib.Path, job_id: str, rank: int = 0) -> sqlite3.Connection:
    """An empty judge shard for ``job_id``, with the three tables the audit reads."""
    shard = root / job_id / "judge" / f"rank-{rank}"
    shard.mkdir(parents=True)
    conn = sqlite3.connect(shard / f"hpcagent_bench{rank}.db")
    with conn:
        conn.execute("create table runs (run_id text, arm text)")
        conn.execute("create table submissions (run_id text, benchmark text, optimizer text, ts integer)")
        conn.execute("create table attempts (run_id text, benchmark text, reason text)")
    return conn


def add_submission(conn: sqlite3.Connection, run_id: str, benchmark: str) -> None:
    with conn:
        conn.execute("insert into submissions values (?, ?, ?, ?)", (run_id, benchmark, "qwen38", FAR_FUTURE_TS_MS))


def add_attempt(conn: sqlite3.Connection, run_id: str, benchmark: str) -> None:
    with conn:
        conn.execute("insert into attempts values (?, ?, ?)", (run_id, benchmark, "score_error"))


def write_problems(root: pathlib.Path, job_id: str, entries: list, name: str = "test-arm") -> None:
    """This job's ``problems-<name>.jsonl``: ``entries`` is ``[(problem id, kernel path), ...]``."""
    launch = root / ".agent-launch" / job_id
    launch.mkdir(parents=True)
    lines = [json.dumps({"id": problem_id, "kernel": kernel}) for problem_id, kernel in entries]
    (launch / f"problems-{name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_log(log_dir: pathlib.Path, job_id: str, lines: list) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"beverin-services-{job_id}.out").write_text("\n".join(lines) + "\n", encoding="utf-8")


def exit_line(problem: int, rc: int, tail: str = "") -> str:
    """One synthetic ``agent_driver.py`` exit line for ``problem``, ending in ``rc`` and ``tail``."""
    return f"problem={problem} worker={problem} judge=0 rc={rc} log=/ritom/x/problem-{problem}/claude.log{tail}"


def one_job_setup(
    tmp_path: pathlib.Path, job_id: str, benchmark: str, submit: bool, attempt: bool, log_lines: list | None
) -> tuple:
    """One job dir carrying ``benchmark``'s judge rows, its own problems file, and its stdout log."""
    root, log_dir = tmp_path / "runs", tmp_path / "logs"
    conn = make_shard(root, job_id)
    run_id = "arm.n0.p0.w0"
    with conn:
        conn.execute("insert into runs values (?, ?)", (run_id, "arm"))
    if submit:
        add_submission(conn, run_id, benchmark)
    if attempt:
        add_attempt(conn, run_id, benchmark)
    conn.close()
    write_problems(root, job_id, [(0, benchmark)])
    if log_lines is not None:
        write_log(log_dir, job_id, log_lines)
    return root, log_dir


def test_a_clean_exit_with_a_submission_is_kept(tmp_path: pathlib.Path) -> None:
    """The ordinary case: one episode, it finished on its own, the submission is real evidence."""
    root, log_dir = one_job_setup(tmp_path, "100", "a", submit=True, attempt=False, log_lines=[exit_line(0, 0)])
    audit = crash_audit.audit_arm("arm", [("100", str(root / "100"))], ROSTER, log_dir, str(tmp_path))
    assert audit.keep == ["a"]
    assert audit.drop == []


def test_a_crashed_exit_with_crash_attempts_is_dropped_even_with_a_submission(tmp_path: pathlib.Path) -> None:
    """rc=127 after the crash budget ran out is the exact evidence the user rule names; a submission
    must not paper over it -- an earlier episode may have submitted before a later one crashed."""
    root, log_dir = one_job_setup(
        tmp_path,
        "100",
        "a",
        submit=True,
        attempt=False,
        log_lines=[exit_line(0, 127, " died=api_timeout crash_attempts=3")],
    )
    audit = crash_audit.audit_arm("arm", [("100", str(root / "100"))], ROSTER, log_dir, str(tmp_path))
    assert audit.keep == []
    assert audit.drop == ["a"]
    assert "rc=127" in audit.drop_evidence["a"][0]


def test_any_nonzero_rc_is_dropped_even_with_no_crash_attempts_field(tmp_path: pathlib.Path) -> None:
    """The user rule is "any nonzero rc", not just the named 127/crash_attempts case."""
    root, log_dir = one_job_setup(tmp_path, "100", "b", submit=True, attempt=False, log_lines=[exit_line(0, 1)])
    audit = crash_audit.audit_arm("arm", [("100", str(root / "100"))], ROSTER, log_dir, str(tmp_path))
    assert audit.drop == ["b"]


def test_a_promoted_self_exit_with_rc_zero_is_kept(tmp_path: pathlib.Path) -> None:
    """promote_at_agent_exit's ``promoted=`` tag rides on an ordinary clean rc=0 line and must not be
    mistaken for crash evidence."""
    root, log_dir = one_job_setup(
        tmp_path,
        "100",
        "c",
        submit=True,
        attempt=False,
        log_lines=[exit_line(0, 0, " promoted=SUBMITTED speedup=3.75x")],
    )
    audit = crash_audit.audit_arm("arm", [("100", str(root / "100"))], ROSTER, log_dir, str(tmp_path))
    assert audit.keep == ["c"]


def test_the_cancelled_by_job_marker_drops_the_kernel_even_at_rc_zero(tmp_path: pathlib.Path) -> None:
    """ "not killed by the job cancel" is its own condition, independent of rc: a line carrying
    ``cancelled=job`` must never count as clean."""
    root, log_dir = one_job_setup(
        tmp_path, "100", "a", submit=True, attempt=False, log_lines=[exit_line(0, 0, " cancelled=job")]
    )
    audit = crash_audit.audit_arm("arm", [("100", str(root / "100"))], ROSTER, log_dir, str(tmp_path))
    assert audit.drop == ["a"]


def test_a_kernel_with_two_episodes_one_crashed_is_dropped(tmp_path: pathlib.Path) -> None:
    """AGENT_SINGLE_SUBMISSION=0 lets two episodes work the same kernel; a clean submission from one
    must not hide a crash the other episode had on the very same kernel."""
    root, log_dir = tmp_path / "runs", tmp_path / "logs"
    conn = make_shard(root, "100")
    with conn:
        conn.execute("insert into runs values (?, ?)", ("arm.n0.p0.w0", "arm"))
        conn.execute("insert into submissions values (?, ?, ?, ?)", ("arm.n0.p0.w0", "a", "qwen38", FAR_FUTURE_TS_MS))
    conn.close()
    write_problems(root, "100", [(0, "a")])
    write_log(log_dir, "100", [exit_line(0, 0)])

    conn = make_shard(root, "101")
    with conn:
        conn.execute("insert into runs values (?, ?)", ("arm.n0.p0.w7", "arm"))
    conn.close()
    write_problems(root, "101", [(0, "a")])
    write_log(log_dir, "101", [exit_line(0, 127, " crash_attempts=2")])

    jobs = [("100", str(root / "100")), ("101", str(root / "101"))]
    audit = crash_audit.audit_arm("arm", jobs, ROSTER, log_dir, str(tmp_path))
    assert audit.keep == []
    assert audit.drop == ["a"]


def test_an_episode_that_cannot_be_mapped_is_reported_not_silently_kept(tmp_path: pathlib.Path) -> None:
    """A problem index outside this job's own roster file must never be guessed at: it is reported
    under ``unmapped``, and it must not be able to make some OTHER kernel look clean by accident."""
    root, log_dir = tmp_path / "runs", tmp_path / "logs"
    conn = make_shard(root, "100")
    with conn:
        conn.execute("insert into runs values (?, ?)", ("arm.n0.p0.w0", "arm"))
        conn.execute("insert into submissions values (?, ?, ?, ?)", ("arm.n0.p0.w0", "a", "qwen38", FAR_FUTURE_TS_MS))
    conn.close()
    write_problems(root, "100", [(0, "a")])  # only problem 0 is on this job's roster
    write_log(log_dir, "100", [exit_line(0, 0), exit_line(5, 0)])  # problem 5 is not

    audit = crash_audit.audit_arm("arm", [("100", str(root / "100"))], ROSTER, log_dir, str(tmp_path))
    assert audit.keep == ["a"]
    assert len(audit.unmapped) == 1
    assert "problem=5" in audit.unmapped[0]


def test_a_job_with_no_problems_file_leaves_every_one_of_its_episodes_unmapped(tmp_path: pathlib.Path) -> None:
    """Zero (or more than one) ``problems-*.jsonl`` in a job's launch dir means this job cannot be
    read at all -- every exit line it printed is unmapped, not silently dropped from the count."""
    root, log_dir = tmp_path / "runs", tmp_path / "logs"
    conn = make_shard(root, "100")
    with conn:
        conn.execute("insert into runs values (?, ?)", ("arm.n0.p0.w0", "arm"))
    conn.close()
    write_log(log_dir, "100", [exit_line(0, 127, " crash_attempts=3")])  # no problems file written

    audit = crash_audit.audit_arm("arm", [("100", str(root / "100"))], ROSTER, log_dir, str(tmp_path))
    assert audit.unmapped == ["job=100 problem=0 rc=127 crash_attempts=3"]


def test_a_job_with_a_missing_stdout_log_is_reported_and_not_treated_as_clean(tmp_path: pathlib.Path) -> None:
    """No log at all means no proof of a clean exit; a submission alone must not default to keep."""
    root, log_dir = one_job_setup(tmp_path, "100", "b", submit=True, attempt=False, log_lines=None)
    audit = crash_audit.audit_arm("arm", [("100", str(root / "100"))], ROSTER, log_dir, str(tmp_path))
    assert audit.keep == []
    assert audit.drop == ["b"]
    assert audit.missing_logs == ["100"]
    assert "log missing" in audit.drop_evidence["b"][0]


def test_a_kernel_with_no_judge_row_at_all_is_never_ran(tmp_path: pathlib.Path) -> None:
    """A roster kernel no job ever touched -- no submission, no attempt -- is NEVER RAN: rerun it,
    but there is nothing of its to delete."""
    root, log_dir = one_job_setup(tmp_path, "100", "a", submit=True, attempt=False, log_lines=[exit_line(0, 0)])
    audit = crash_audit.audit_arm("arm", [("100", str(root / "100"))], ROSTER, log_dir, str(tmp_path))
    assert audit.never_ran == ["b", "c"]


def test_an_attempts_only_kernel_with_no_crash_is_dropped_not_kept(tmp_path: pathlib.Path) -> None:
    """An agent that ran clean but never got a correct/fast-enough answer leaves attempts with no
    submission: it was never done, so it belongs in drop-and-rerun, not in never-ran or keep."""
    root, log_dir = one_job_setup(tmp_path, "100", "a", submit=False, attempt=True, log_lines=[exit_line(0, 0)])
    audit = crash_audit.audit_arm("arm", [("100", str(root / "100"))], ROSTER, log_dir, str(tmp_path))
    assert audit.keep == []
    assert audit.drop == ["a"]
    assert audit.never_ran == ["b", "c"]


def test_keep_drop_never_ran_partition_the_roster_with_no_overlap(tmp_path: pathlib.Path) -> None:
    """The three lists are a partition of the roster: every kernel lands in exactly one."""
    root, log_dir = tmp_path / "runs", tmp_path / "logs"
    conn = make_shard(root, "100")
    with conn:
        conn.execute("insert into runs values (?, ?)", ("arm.n0.p0.w0", "arm"))
        conn.execute("insert into runs values (?, ?)", ("arm.n0.p1.w1", "arm"))
        conn.execute("insert into submissions values (?, ?, ?, ?)", ("arm.n0.p0.w0", "a", "qwen38", FAR_FUTURE_TS_MS))
        conn.execute("insert into submissions values (?, ?, ?, ?)", ("arm.n0.p1.w1", "b", "qwen38", FAR_FUTURE_TS_MS))
    conn.close()
    write_problems(root, "100", [(0, "a"), (1, "b")])
    write_log(log_dir, "100", [exit_line(0, 0), exit_line(1, 127, " crash_attempts=2")])

    audit = crash_audit.audit_arm("arm", [("100", str(root / "100"))], ROSTER, log_dir, str(tmp_path))
    partition = sorted(audit.keep + audit.drop + audit.never_ran)
    assert partition == ROSTER
    assert audit.keep == ["a"]
    assert audit.drop == ["b"]
    assert audit.never_ran == ["c"]


def test_delete_rows_names_only_the_dropped_kernels_rows(tmp_path: pathlib.Path) -> None:
    """The delete report must be precise: a KEPT kernel's rows must never appear in it, so a later
    delete step cannot accidentally erase evidence that is still trusted."""
    root, log_dir = tmp_path / "runs", tmp_path / "logs"
    conn = make_shard(root, "100")
    with conn:
        conn.execute("insert into runs values (?, ?)", ("arm.n0.p0.w0", "arm"))
        conn.execute("insert into runs values (?, ?)", ("arm.n0.p1.w1", "arm"))
        conn.execute("insert into submissions values (?, ?, ?, ?)", ("arm.n0.p0.w0", "a", "qwen38", FAR_FUTURE_TS_MS))
        conn.execute("insert into submissions values (?, ?, ?, ?)", ("arm.n0.p1.w1", "b", "qwen38", FAR_FUTURE_TS_MS))
        conn.execute("insert into attempts values (?, ?, ?)", ("arm.n0.p1.w1", "b", "score_error"))
        # A row under an unrelated run_id that happens to name the same benchmark: not this arm's
        # coverage, and must never be swept into the delete report just because the name matches.
        conn.execute("insert into submissions values (?, ?, ?, ?)", ("adhoc", "b", "human", FAR_FUTURE_TS_MS))
    conn.close()
    write_problems(root, "100", [(0, "a"), (1, "b")])
    write_log(log_dir, "100", [exit_line(0, 0), exit_line(1, 1)])

    audit = crash_audit.audit_arm("arm", [("100", str(root / "100"))], ROSTER, log_dir, str(tmp_path))
    assert audit.drop == ["b"]
    tables = {(table, benchmark) for _, table, _, benchmark, _ in audit.delete_rows}
    assert tables == {("submissions", "b"), ("attempts", "b")}
    assert all(benchmark == "b" for _, _, _, benchmark, _ in audit.delete_rows)
    assert all(run_id != "adhoc" for _, _, run_id, _, _ in audit.delete_rows)


def test_delete_rows_matches_each_jobs_own_arm_not_the_callers_folded_identity(tmp_path: pathlib.Path) -> None:
    """A caller that folds a `-clean` re-run into its base identity (remaining_kernels.py's
    ``collect_arms``/``base_arm``, USER RULE 2026-09-18) can pass ``audit_arm`` one identity label
    covering jobs whose OWN run_ids are still prefixed by their real, unfolded arm names --
    `submit_common.sh`'s `clean_suffix` changes the arm name, not the identity columns. Both jobs'
    rows must still be found: a delete report that only matched the caller's label would silently
    keep the `-clean` job's stale rows around after every rerun.
    """
    root, log_dir = tmp_path / "runs", tmp_path / "logs"
    conn = make_shard(root, "100")
    with conn:
        conn.execute("insert into runs values (?, ?)", ("base-arm.n0.p0.w0", "base-arm"))
        conn.execute("insert into submissions values (?, ?, ?, ?)", ("base-arm.n0.p0.w0", "a", "qwen38", FAR_FUTURE_TS_MS))
    conn.close()
    write_problems(root, "100", [(0, "a")])
    write_log(log_dir, "100", [exit_line(0, 1)])  # crashed: "a" is DROP

    conn = make_shard(root, "101")
    with conn:
        conn.execute("insert into runs values (?, ?)", ("base-arm-clean.n0.p0.w7", "base-arm-clean"))
        conn.execute(
            "insert into submissions values (?, ?, ?, ?)", ("base-arm-clean.n0.p0.w7", "a", "qwen38", FAR_FUTURE_TS_MS)
        )
    conn.close()
    write_problems(root, "101", [(0, "a")])
    write_log(log_dir, "101", [exit_line(0, 1)])  # crashed too: still DROP

    jobs = [("100", str(root / "100")), ("101", str(root / "101"))]
    audit = crash_audit.audit_arm("base-arm", jobs, ROSTER, log_dir, str(tmp_path))
    assert audit.drop == ["a"]
    run_ids = {run_id for _, _, run_id, _, _ in audit.delete_rows}
    assert run_ids == {"base-arm.n0.p0.w0", "base-arm-clean.n0.p0.w7"}


def test_roster_tag_reads_the_wave_boards_own_campaign_table() -> None:
    """The tag lookup must be the SAME table the wave board scores arms under, not a second copy
    that can drift from it."""
    assert crash_audit.roster_tag("cpf-llr-focus40-oss120b-c") == "llr-focus40"
    assert crash_audit.roster_tag("scicomp-dc-cpp-oss120b-plain") == "scicomp40"


def test_roster_tag_raises_for_an_arm_no_campaign_owns() -> None:
    """An arm outside every known campaign prefix must fail loudly, not default to some tag."""
    with pytest.raises(SystemExit, match="matches no campaign"):
        crash_audit.roster_tag("some-unregistered-arm-x9")


def test_main_writes_a_json_report_matching_the_direct_audit(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI wiring (argument parsing, per-arm loop, JSON dump) must not silently diverge from
    calling ``audit_arm`` directly -- that would leave the machine-readable report unverified."""
    root, log_dir = one_job_setup(tmp_path, "100", "a", submit=True, attempt=False, log_lines=[exit_line(0, 0)])
    out_json = tmp_path / "report.json"

    def fake_roster(tag: str, opt: str) -> list:
        assert tag == "llr-focus40"
        return list(ROSTER)

    monkeypatch.setattr(crash_audit.rk, "roster", fake_roster)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "crash_audit.py",
            "--run-root",
            str(root),
            "--log-dir",
            str(log_dir),
            "--out-json",
            str(out_json),
        ],
    )
    # "100" carries arm "arm" in its own runs table above, but roster_tag needs a real campaign
    # prefix, so give this job a campaign-recognised arm instead.
    conn = sqlite3.connect(root / "100" / "judge" / "rank-0" / "hpcagent_bench0.db")
    with conn:
        conn.execute("update runs set arm = ?", ("cpf-llr-focus40-oss120b-c",))
    conn.close()

    assert crash_audit.main() == 0
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["cpf-llr-focus40-oss120b-c"]["keep"] == ["a"]
