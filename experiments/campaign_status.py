#!/usr/bin/env python3
"""One JSON snapshot of every experiment arm submitted today, for the completion dashboard.

Reads Slurm state (``sacct``), each arm's ``.env.<arm>`` and problems file, its run dir under
``$SCRATCH/hpcagent-bench-runs`` (found from the "run dir:" line the launcher prints into
``experiments/beverin-services-<jobid>.out``), the judge sqlite DBs and the agent worker
``claude.log``/``tokens.json`` files, and turns all of it into one document plus a liveness
verdict per running arm.

Every field is best-effort: a missing or partially written file becomes ``null`` (or an empty
list/dict) for that field, with a note appended to that arm's ``errors`` list, never an
exception -- this runs against LIVE run directories that agents and judges are still writing.

Usage::

    campaign_status.py [--since ISO] [--jobs ID ...] [--out FILE]

With no ``--out`` the document is printed to stdout. Read-only: this never touches Slurm state
or any run directory.
"""

import argparse
import csv
import json
import os
import pathlib
import re
import sqlite3
import statistics
import subprocess
import sys
import time
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent

#: agent_driver.py is imported for its token_cost_module() alone -- the one implementation of an
#: episode's token fold (fresh_input/cached_input/output), per the "do not reimplement" rule.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import agent_driver  # noqa: E402  -- path insert above must run first
from hpcagent_bench import paths  # noqa: E402  -- path insert above must run first

DEFAULT_SINCE = "2026-09-17T12:55"

#: KEY=value at the start of a line, same convention audit_envs.py uses for these files: anything
#: else is a comment or a continuation of a quoted value, neither of which is a key.
ENV_ASSIGN = re.compile(r"^([A-Z_][A-Z0-9_]*)=(.*)$")

RECORD_KEYS = ("EXPERIMENT", "MODEL", "LANGUAGE", "DEVICE", "PACKET")

#: A run's health window (docstring, "Liveness"): unchanged for 15 min is still "ok"; unchanged
#: for 20+ min after running 20+ min is "stalled".
OK_WINDOW_SECONDS = 15 * 60
STALL_WINDOW_SECONDS = 20 * 60

#: Helper-job name prefixes (task: "smoke-*, probe-*, cpf-pre-*, canon columns"). Checked only
#: after the .env.<jobname> test below, since a real arm can itself start with "smoke-" (e.g.
#: smoke-enroot-qwen38-c) and is an inference-arm, not a helper, whenever its own env file exists.
PRERENDER_PREFIXES = ("cpf-pre-",)
FRAMEWORK_PREFIXES = ("canon-", "canon40-")

#: sglang's periodic decode-throughput line, and vLLM's own equivalent (task: "Liveness").
SGLANG_DECODE_RE = re.compile(r"gen throughput \(token/s\):\s*([0-9.]+)")
VLLM_THROUGHPUT_RE = re.compile(r"Avg generation throughput:\s*([0-9.]+)\s*tokens/s")

#: Any line naming the engine (or a judge) ready, e.g. "vLLM 0 ready: ..." / "judge 0 ready: ...".
READY_RE = re.compile(r"\bready:", re.IGNORECASE)

#: A crashed engine's own traceback, tagged by the EngineCore process itself -- as opposed to an
#: ordinary per-request traceback the API server logs (serving.py), which is not an engine death.
ENGINE_TRACEBACK_RE = re.compile(r"\(EngineCore pid=\d+\) ERROR[^\n]*Traceback \(most recent call last\)")

#: A .out file can grow for hours; capped so one huge log cannot make this "fast, read-only" tool
#: slow. The tail is what liveness needs -- ready/dead-engine markers near a startup that already
#: happened are still inside this window for any run whose whole .out is smaller than the cap.
OUT_FILE_READ_CAP = 8_000_000


def now_epoch() -> float:
    return time.time()


def read_text_capped(path: pathlib.Path, cap: int = OUT_FILE_READ_CAP) -> str | None:
    """Whole file, or its last ``cap`` bytes when larger. ``None`` on any read failure."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > cap:
                handle.seek(size - cap)
            data = handle.read()
    except OSError:
        return None
    return data.decode("utf-8", errors="replace")


def mtime_or_none(path: pathlib.Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


# --------------------------------------------------------------------------------------------
# sacct
# --------------------------------------------------------------------------------------------

SACCT_FIELDS = ("JobID", "JobName", "State", "Elapsed", "Start", "End", "NNodes", "Timelimit", "ExitCode")


def elapsed_to_seconds(text: str) -> int | None:
    """Slurm's ``[D-]HH:MM:SS`` (or ``MM:SS``) elapsed/timelimit text, in seconds."""
    if not text or text in ("Unknown", "UNLIMITED", "INVALID"):
        return None
    days = 0
    rest = text
    if "-" in text:
        day_part, rest = text.split("-", 1)
        try:
            days = int(day_part)
        except ValueError:
            return None
    parts = rest.split(":")
    try:
        parts_i = [int(p) for p in parts]
    except ValueError:
        return None
    while len(parts_i) < 3:
        parts_i.insert(0, 0)
    hours, minutes, seconds = parts_i[-3:]
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def run_sacct(since: str, jobs: list[str] | None) -> list[dict[str, Any]]:
    """Every job sacct reports from ``since`` on, as the fields the task's exact command names."""
    cmd = [
        "sacct",
        "-u",
        os.environ.get("USER", ""),
        "-S",
        since,
        "-X",
        "-P",
        "-o",
        ",".join(SACCT_FIELDS),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    rows: list[dict[str, Any]] = []
    lines = out.stdout.splitlines()
    if not lines:
        return rows
    header = lines[0].split("|")
    for line in lines[1:]:
        cells = line.split("|")
        if len(cells) != len(header):
            continue
        row = dict(zip(header, cells, strict=False))
        rows.append(row)
    if jobs:
        wanted = set(jobs)
        rows = [r for r in rows if r.get("JobID") in wanted]
    return rows


# --------------------------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------------------------


def classify(jobname: str, env_exists: bool) -> str:
    """inference-arm | prerender | framework-column | other, per the task's helper-name list.

    ``.env.<jobname>`` existing is checked FIRST: a real arm can itself start with "smoke-"
    (smoke-enroot-qwen38-c has its own env file) and must not be swept into "other" by a prefix
    test that was only ever meant for the helper jobs that have no env file of their own.
    """
    if env_exists:
        return "inference-arm"
    if jobname.startswith(PRERENDER_PREFIXES):
        return "prerender"
    if jobname.startswith(FRAMEWORK_PREFIXES):
        return "framework-column"
    return "other"


# --------------------------------------------------------------------------------------------
# env + problems file
# --------------------------------------------------------------------------------------------


def read_env(path: pathlib.Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return values
    for line in text.splitlines():
        found = ENV_ASSIGN.match(line)
        if found:
            values[found.group(1)] = found.group(2)
    return values


def count_problems(path: pathlib.Path) -> int | None:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    total = 0
    for line in text.splitlines():
        if line.strip():
            total += 1
    return total


# --------------------------------------------------------------------------------------------
# out/err file discovery + parsing
# --------------------------------------------------------------------------------------------


def find_service_file(jobid: str, suffix: str) -> pathlib.Path | None:
    """The launcher's own log for this job. Tries the documented exact name first
    (``beverin-services-<jobid>.<suffix>``), then falls back to any ``*-<jobid>.<suffix>`` in
    ``experiments/`` -- prerender jobs (``cpf-pre-*``) keep their own job-name prefix (``%x-%j``)
    rather than the fixed ``beverin-services`` one ``beverin.sbatch`` hardcodes.
    """
    exact = HERE / f"beverin-services-{jobid}.{suffix}"
    if exact.exists():
        return exact
    matches = sorted(HERE.glob(f"*-{jobid}.{suffix}"))
    return matches[0] if matches else None


RUN_DIR_RE = re.compile(r"run dir:\s*(\S+)")


def find_run_dir(out_text: str | None) -> pathlib.Path | None:
    if not out_text:
        return None
    found = RUN_DIR_RE.search(out_text)
    if not found:
        return None
    return pathlib.Path(found.group(1))


def engine_liveness(out_text: str | None) -> dict[str, Any]:
    """Last sglang/vLLM decode line's throughput, whichever engine this arm's log carries."""
    result: dict[str, Any] = {"engine": None, "last_line_seconds_ago": None, "gen_throughput_tokens_s": None}
    if not out_text:
        return result
    last_sglang = None
    for m in SGLANG_DECODE_RE.finditer(out_text):
        last_sglang = m
    last_vllm = None
    for m in VLLM_THROUGHPUT_RE.finditer(out_text):
        last_vllm = m
    # Whichever pattern actually matched something is this arm's engine; a .out carries only one.
    if last_sglang is not None:
        result["engine"] = "sglang"
        result["gen_throughput_tokens_s"] = float(last_sglang.group(1))
    elif last_vllm is not None:
        result["engine"] = "vllm"
        result["gen_throughput_tokens_s"] = float(last_vllm.group(1))
    return result


def dead_engine_reason(out_text: str | None, err_text: str | None) -> str | None:
    """The specific marker found, or None -- never a bare boolean, so `reason` can quote it."""
    for label, text in (("out", out_text), ("err", err_text)):
        if not text:
            continue
        if "Stale file handle" in text:
            return f"'Stale file handle' in .{label}"
        if "RuntimeError: cancelled" in text:
            return f"'RuntimeError: cancelled' in .{label}"
        if ENGINE_TRACEBACK_RE.search(text):
            return f"EngineCore crash traceback in .{label}"
    return None


# --------------------------------------------------------------------------------------------
# run dir contents: agents, judge DBs, monitor
# --------------------------------------------------------------------------------------------


def worker_dirs(run_dir: pathlib.Path) -> list[pathlib.Path]:
    try:
        return sorted(run_dir.glob("agents/node-*/problem-*-worker-*"))
    except OSError:
        return []


def judge_db_paths(run_dir: pathlib.Path) -> list[pathlib.Path]:
    try:
        return sorted(run_dir.glob("judge/rank-*/hpcagent_bench*.db"))
    except OSError:
        return []


def read_calls_and_submissions(
    db_paths: list[pathlib.Path], errors: list[str]
) -> tuple[list[tuple[str, str, int, float]], set[str]]:
    """(benchmark, route, correct, speedup) per call across every rank DB, plus every benchmark
    the submissions table names -- opened read-only so a judge mid-write is never locked out."""
    calls: list[tuple[str, str, int, float]] = []
    submitted: set[str] = set()
    for db_path in db_paths:
        try:
            uri = f"file:{db_path}?mode=ro"
            con = sqlite3.connect(uri, uri=True, timeout=5)
        except sqlite3.Error as exc:
            errors.append(f"cannot open {db_path}: {exc}")
            continue
        try:
            try:
                cur = con.execute("select benchmark, route, correct, speedup from calls")
                for benchmark, route, correct, speedup in cur.fetchall():
                    calls.append((str(benchmark), str(route), int(correct or 0), float(speedup or 0.0)))
            except sqlite3.Error as exc:
                errors.append(f"{db_path}: calls table unreadable: {exc}")
            try:
                cur = con.execute("select distinct benchmark from submissions")
                submitted.update(str(row[0]) for row in cur.fetchall())
            except sqlite3.Error as exc:
                errors.append(f"{db_path}: submissions table unreadable: {exc}")
        finally:
            con.close()
    return calls, submitted


def kernel_metrics(calls: list[tuple[str, str, int, float]], submitted: set[str]) -> dict[str, Any]:
    scored: set[str] = set()
    correct: set[str] = set()
    submit_route: set[str] = set()
    best_by_kernel: dict[str, float] = {}
    for benchmark, route, is_correct, speedup in calls:
        if route in ("score", "submit"):
            scored.add(benchmark)
        if route == "submit":
            submit_route.add(benchmark)
        if is_correct:
            correct.add(benchmark)
            if speedup > best_by_kernel.get(benchmark, float("-inf")):
                best_by_kernel[benchmark] = speedup
    submitted_all = submit_route | submitted
    speedups = [best_by_kernel[k] for k in correct if k in best_by_kernel]
    return {
        "kernels_scored": len(scored),
        "kernels_correct": len(correct),
        "kernels_submitted": len(submitted_all),
        "best_speedup_median": statistics.median(speedups) if speedups else None,
    }


def token_totals(claude_logs: list[pathlib.Path], errors: list[str]) -> int | None:
    """Sum of fresh_input + cached_input + output over every worker's transcript so far.

    Delegates the fold itself to agent_driver's token_cost_module().episode_cost -- see the
    module docstring: the one implementation of what an episode cost, never reimplemented here.
    """
    module = agent_driver.token_cost_module()
    total = 0
    any_ok = False
    for log in claude_logs:
        try:
            row = module.episode_cost(log)
        except Exception as exc:  # noqa: BLE001 -- a cost record must never fail the whole report
            errors.append(f"episode_cost failed for {log}: {exc}")
            continue
        total += int(row.get("fresh_input", 0)) + int(row.get("cached_input", 0)) + int(row.get("output", 0))
        any_ok = True
    return total if any_ok else (0 if not claude_logs else None)


# --------------------------------------------------------------------------------------------
# health verdict
# --------------------------------------------------------------------------------------------


def health_verdict(
    *,
    state: str,
    elapsed_seconds: int | None,
    claude_log_age: float | None,
    db_age: float | None,
    has_ready: bool,
    dead_reason: str | None,
) -> tuple[str | None, str]:
    """ok | stalled | starting | dead-engine, plus the reason that explains it.

    Only meaningful for a RUNNING job -- a finished one is not "frozen", it is finished.
    """
    if state != "RUNNING":
        return None, f"job state is {state}, not running"
    if dead_reason is not None:
        return "dead-engine", dead_reason
    if not has_ready:
        return "starting", "no 'ready' line yet in the service .out"
    ages = [a for a in (claude_log_age, db_age) if a is not None]
    last_change_age = min(ages) if ages else None
    if last_change_age is None:
        return "starting", "no claude.log or judge DB written yet"
    if (
        elapsed_seconds is not None
        and elapsed_seconds > STALL_WINDOW_SECONDS
        and last_change_age >= STALL_WINDOW_SECONDS
    ):
        return "stalled", f"no claude.log or DB write for {int(last_change_age)}s (running {elapsed_seconds}s)"
    if last_change_age <= OK_WINDOW_SECONDS:
        return "ok", f"last claude.log/DB write {int(last_change_age)}s ago"
    return (
        "ok",
        f"last claude.log/DB write {int(last_change_age)}s ago (below the {STALL_WINDOW_SECONDS}s stall window)",
    )


# --------------------------------------------------------------------------------------------
# framework columns (canon_column.sh)
# --------------------------------------------------------------------------------------------


def find_canon_out(jobid: str, jobname: str, scratch: pathlib.Path) -> pathlib.Path | None:
    """canon_column.sh's own OUT_ROOT is set by whoever ran submit-canon-llr40.sh and is not
    derivable from the job name alone (it defaults to ``$SCRATCH/canon-llr40-<date>`` but can be
    overridden), so this looks for the sbatch ``%x-%j.out`` file directly under any ``canon-*``
    directory in $SCRATCH instead of trying to reconstruct OUT_ROOT."""
    try:
        matches = sorted(scratch.glob(f"canon-*/{jobname}-{jobid}.out"))
    except OSError:
        return None
    return matches[0] if matches else None


def framework_column_summary(jobname: str, jobid: str, scratch: pathlib.Path, errors: list[str]) -> dict[str, Any]:
    out_path = find_canon_out(jobid, jobname, scratch)
    if out_path is None:
        return {"csv_files": [], "ok": None, "unsupported": None, "crash": None}
    out_root = out_path.parent
    # canon_column.sh: JOB_PREFIX-col is the job name, and col itself never contains a literal
    # "-" (numba, cc, cc_autopar, dace_cpu, dace_cpu_canonicalize, ...) even though JOB_PREFIX
    # (canon-<tag>) can, so the LAST "-" is always the prefix/column boundary.
    col = jobname.rsplit("-", 1)[-1]
    csv_paths = sorted(out_root.glob(f"{col}.rank*.csv"))
    ok = unsupported = crash = 0
    for csv_path in csv_paths:
        try:
            with csv_path.open(newline="", errors="replace") as handle:
                for row in csv.DictReader(handle):
                    status = row.get("status", "")
                    failure = row.get("failure", "")
                    if status == "ok" and not failure:
                        ok += 1
                    elif failure == "unsupported":
                        unsupported += 1
                    elif status != "ok":
                        crash += 1
        except OSError as exc:
            errors.append(f"cannot read {csv_path}: {exc}")
    return {
        "csv_files": [str(p) for p in csv_paths],
        "ok": ok if csv_paths else None,
        "unsupported": unsupported if csv_paths else None,
        "crash": crash if csv_paths else None,
    }


# --------------------------------------------------------------------------------------------
# per-arm assembly
# --------------------------------------------------------------------------------------------


def build_inference_arm(row: dict[str, Any], now: float, scratch: pathlib.Path) -> dict[str, Any]:
    errors: list[str] = []
    jobid = row["JobID"]
    arm = row["JobName"]
    state = row["State"].split()[0] if row["State"] else row["State"]  # "CANCELLED by 29756" -> "CANCELLED"
    elapsed_seconds = elapsed_to_seconds(row.get("Elapsed", ""))

    env = read_env(HERE / f".env.{arm}")
    if not env:
        errors.append(f"could not read .env.{arm}")

    out_path = find_service_file(jobid, "out")
    err_path = find_service_file(jobid, "err")
    out_text = read_text_capped(out_path) if out_path else None
    err_text = read_text_capped(err_path) if err_path else None
    if out_path is None:
        errors.append("no experiments/*-<jobid>.out found")

    run_dir = find_run_dir(out_text)
    if run_dir is None and out_path is not None:
        errors.append("no 'run dir:' line found in the .out yet")

    kernels_total = None
    if env.get("PROBLEMS_FILE"):
        problems_path = HERE / env["PROBLEMS_FILE"]
        kernels_total = count_problems(problems_path)
        if kernels_total is None:
            errors.append(f"cannot read problems file {problems_path}")

    claude_logs: list[pathlib.Path] = []
    tokens_json_count = 0
    db_paths: list[pathlib.Path] = []
    calls: list[tuple[str, str, int, float]] = []
    submitted: set[str] = set()

    if run_dir is not None:
        dirs = worker_dirs(run_dir)
        for wdir in dirs:
            log = wdir / "claude.log"
            if log.exists():
                claude_logs.append(log)
            if (wdir / "tokens.json").exists():
                tokens_json_count += 1
        db_paths = judge_db_paths(run_dir)
        calls, submitted = read_calls_and_submissions(db_paths, errors)

    kmetrics = kernel_metrics(calls, submitted)
    tokens_so_far = token_totals(claude_logs, errors)

    # Clamped at 0: a run dir on a different node can carry a clock a few seconds ahead of this
    # one's, which would otherwise report a write as happening in the future.
    claude_log_mtimes = [m for m in (mtime_or_none(p) for p in claude_logs) if m is not None]
    newest_claude_log = max(claude_log_mtimes) if claude_log_mtimes else None
    claude_log_age = max(0.0, now - newest_claude_log) if newest_claude_log is not None else None

    newest_db = None
    db_mtimes = [m for m in (mtime_or_none(p) for p in db_paths) if m is not None]
    if db_mtimes:
        newest_db = max(db_mtimes)
    db_age = max(0.0, now - newest_db) if newest_db is not None else None

    engine = engine_liveness(out_text)
    has_ready = bool(out_text and READY_RE.search(out_text))
    dead_reason = dead_engine_reason(out_text, err_text)
    verdict, reason = health_verdict(
        state=state,
        elapsed_seconds=elapsed_seconds,
        claude_log_age=claude_log_age,
        db_age=db_age,
        has_ready=has_ready,
        dead_reason=dead_reason,
    )

    identity = {
        "experiment": env.get("HPCAGENT_BENCH_RECORD_EXPERIMENT"),
        "model": env.get("HPCAGENT_BENCH_RECORD_MODEL"),
        "language": env.get("HPCAGENT_BENCH_RECORD_LANGUAGE"),
        "device": env.get("HPCAGENT_BENCH_RECORD_DEVICE"),
        "packet": env.get("HPCAGENT_BENCH_RECORD_PACKET"),
        "arm": arm,
        "jobid": jobid,
        "state": state,
        "nodes": int(row["NNodes"]) if row.get("NNodes", "").isdigit() else None,
        "elapsed": row.get("Elapsed"),
        "timelimit": row.get("Timelimit"),
    }

    return {
        "kind": "inference-arm",
        "identity": identity,
        "run_dir": str(run_dir) if run_dir else None,
        "kernels_total": kernels_total,
        "kernels_started": len(claude_logs),
        "kernels_scored": kmetrics["kernels_scored"],
        "kernels_correct": kmetrics["kernels_correct"],
        "kernels_submitted": kmetrics["kernels_submitted"],
        "best_speedup_median": kmetrics["best_speedup_median"],
        "tokens_so_far": tokens_so_far,
        "agents_finished": tokens_json_count,
        "liveness": {
            "seconds_since_claude_log_write": claude_log_age,
            "seconds_since_judge_db_write": db_age,
            "engine": engine["engine"],
            "engine_gen_throughput_tokens_s": engine["gen_throughput_tokens_s"],
            "health": verdict,
            "reason": reason,
        },
        "errors": errors,
    }


def build_prerender(row: dict[str, Any]) -> dict[str, Any]:
    jobid = row["JobID"]
    jobname = row["JobName"]
    state = row["State"].split()[0] if row["State"] else row["State"]
    out_path = find_service_file(jobid, "out")
    return {
        "kind": "prerender",
        "identity": {
            "arm": jobname,
            "jobid": jobid,
            "state": state,
            "nodes": int(row["NNodes"]) if row.get("NNodes", "").isdigit() else None,
            "elapsed": row.get("Elapsed"),
            "timelimit": row.get("Timelimit"),
        },
        "out_file": str(out_path) if out_path else None,
        "errors": [] if out_path else ["no experiments/*-<jobid>.out found"],
    }


def build_framework_column(row: dict[str, Any], scratch: pathlib.Path) -> dict[str, Any]:
    jobid = row["JobID"]
    jobname = row["JobName"]
    state = row["State"].split()[0] if row["State"] else row["State"]
    errors: list[str] = []
    summary = framework_column_summary(jobname, jobid, scratch, errors)
    return {
        "kind": "framework-column",
        "identity": {
            "arm": jobname,
            "jobid": jobid,
            "state": state,
            "nodes": int(row["NNodes"]) if row.get("NNodes", "").isdigit() else None,
            "elapsed": row.get("Elapsed"),
            "timelimit": row.get("Timelimit"),
        },
        "ok": summary["ok"],
        "unsupported": summary["unsupported"],
        "crash": summary["crash"],
        "csv_files": summary["csv_files"],
        "errors": errors,
    }


def build_other(row: dict[str, Any]) -> dict[str, Any]:
    state = row["State"].split()[0] if row["State"] else row["State"]
    return {
        "kind": "other",
        "identity": {
            "arm": row["JobName"],
            "jobid": row["JobID"],
            "state": state,
            "nodes": int(row["NNodes"]) if row.get("NNodes", "").isdigit() else None,
            "elapsed": row.get("Elapsed"),
            "timelimit": row.get("Timelimit"),
        },
        "errors": [],
    }


def build_arm(row: dict[str, Any], now: float, scratch: pathlib.Path) -> dict[str, Any]:
    jobname = row.get("JobName", "")
    env_exists = (HERE / f".env.{jobname}").exists()
    kind = classify(jobname, env_exists)
    try:
        if kind == "inference-arm":
            return build_inference_arm(row, now, scratch)
        if kind == "prerender":
            return build_prerender(row)
        if kind == "framework-column":
            return build_framework_column(row, scratch)
        return build_other(row)
    except Exception as exc:  # noqa: BLE001 -- one arm's crash must not blank the whole report
        return {
            "kind": kind,
            "identity": {"arm": jobname, "jobid": row.get("JobID"), "state": row.get("State")},
            "errors": [f"unhandled error building this arm's status: {exc!r}"],
        }


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def build_report(since: str, jobs: list[str] | None, scratch: pathlib.Path) -> dict[str, Any]:
    now = now_epoch()
    rows = run_sacct(since, jobs)
    arms = [build_arm(row, now, scratch) for row in rows]
    return {
        "generated_at": now,
        "since": since,
        "job_count": len(arms),
        "arms": arms,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default=DEFAULT_SINCE, help="sacct -S value (default: %(default)s)")
    parser.add_argument("--jobs", nargs="*", default=None, help="restrict to these job IDs")
    parser.add_argument("--out", default=None, help="write JSON here instead of stdout")
    args = parser.parse_args(argv)

    scratch = paths.scratch_or_repo()
    report = build_report(args.since, args.jobs, scratch)
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.out:
        pathlib.Path(args.out).write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
