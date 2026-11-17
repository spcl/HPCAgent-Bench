# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``hpcagent-bench job retune``: change a queued job's env, safely, with the reason on record.

A submitted job reads its env snapshot (``experiments/.rendered/<setup>-<stamp>-<hash>.env``, read-only) when it starts,
so a setting changed in a pending job takes effect: a serving argument found wrong after queueing, a node count.
Editing the file by hand leaves no trace of what changed or why. ``retune`` does it in one step:

    hpcagent-bench job retune <jobid> KEY=VALUE ... --reason "why"   # only a PENDING job
    hpcagent-bench job retune --show <jobid>                         # what was changed in it, from the ledger

It refuses a running job (it has read its env), a key that is part of the run protocol (the limits, the submission mode),
the identity of the setup (``SETUP``, ``RUN_ROOT``, ``PROBLEMS_FILE``, every ``HPCAGENT_BENCH_RECORD_*``) or a secret, a key
the env does not set (``--add`` allows it), and a value that breaks a serving rule (:mod:`serve_args`). The file is
backed up beside it (``<env>.retune-<n>.bak``, read-only), rewritten by an atomic replace, read back to show that
only the requested lines differ, and every change is appended to ``retune.jsonl`` in its folder.
"""

import argparse
import datetime
import getpass
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from hpcagent_bench.cluster import env_spec, serve_args

__all__ = ["LEDGER", "REFUSED_KEYS", "job_env", "main", "retune"]

LEDGER = "retune.jsonl"
#: Keys that name what the job is, not how it runs: changing one would make its rows another setup's.
REFUSED_KEYS = frozenset({"SETUP", "RUN_ROOT", "PROBLEMS_FILE", "SETUPS_FILE", "EXPERIMENT", "STAMP"})
SECRET_KEY = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
COMMAND = re.compile(r"\bCommand=(.*?)\s+WorkDir=")
STATE = re.compile(r"\bJobState=(\w+)")


def job_env(job: str) -> tuple[str, pathlib.Path]:
    """The state of Slurm job ``job`` and the env file it will read (the argument of ``services.sbatch``)."""
    shown = subprocess.run(
        ["scontrol", "--oneliner", "show", "job", job], capture_output=True, text=True, check=False
    ).stdout
    state, command = STATE.search(shown), COMMAND.search(shown)
    if state is None or command is None:
        raise SystemExit(f"retune: no job {job} in Slurm")
    words = command[1].split()
    if len(words) < 2 or not words[0].endswith("services.sbatch"):
        raise SystemExit(f"retune: job {job} is not a setup job (its command is {command[1]!r})")
    return state[1], pathlib.Path(words[1])


def refused(key: str, *, adding: bool, current: Mapping[str, str]) -> str:
    """Why ``key`` may not be retuned, ``""`` when it may."""
    if key in env_spec.PROTOCOL_KEYS | env_spec.LIMIT_OVERRIDE_KEYS:
        return f"{key} is part of the run protocol: limits never change for a launched setup"
    if key in REFUSED_KEYS or key.startswith("HPCAGENT_BENCH_RECORD_"):
        return f"{key} is the identity of the setup"
    if SECRET_KEY.search(key):
        return f"{key} looks like a secret: set it in the environment, never in a rendered env"
    if key not in current and not adding:
        return f"{key} is not set in the job's env (--add sets a new key)"
    return ""


def backup_of(env: pathlib.Path) -> pathlib.Path:
    number = 1
    while (env.with_name(f"{env.name}.retune-{number}.bak")).exists():
        number += 1
    return env.with_name(f"{env.name}.retune-{number}.bak")


def rewritten(env: pathlib.Path, edits: Mapping[str, str]) -> str:
    """The text of ``env`` with each ``edits`` key's last line replaced, or added at the end."""
    lines = env.read_text(encoding="utf-8").splitlines()
    done: set[str] = set()
    for index in range(len(lines) - 1, -1, -1):
        key = lines[index].partition("=")[0]
        if key in edits and key not in done:
            lines[index] = f"{key}={edits[key]}"
            done.add(key)
    lines += [f"{key}={value}" for key, value in edits.items() if key not in done]
    return "".join(f"{line}\n" for line in lines)


def retune(
    job: str, edits: Mapping[str, str], reason: str, *, adding: bool = False, dry_run: bool = False
) -> list[dict[str, Any]]:
    """Apply ``edits`` to pending job ``job``'s env; returns the ledger entries (one per key). Raises SystemExit
    naming the refusal, with nothing changed."""
    state, env = job_env(job)
    if state != "PENDING":
        raise SystemExit(f"retune: job {job} is {state}: it has read its env; only a PENDING job is retuned")
    if not reason.strip():
        raise SystemExit("retune: --reason is required: the ledger records why")
    current = env_spec.assignments(env)
    for key in edits:
        if message := refused(key, adding=adding, current=current):
            raise SystemExit(f"retune: {message}")
    if message := serve_args.refusal(current | dict(edits)):
        raise SystemExit(f"retune: {message}")
    text = rewritten(env, edits)
    changes: list[dict[str, Any]] = [
        {"key": key, "old": current.get(key), "new": value} for key, value in edits.items()
    ]
    if dry_run:
        return [{"job": job, **change} for change in changes]
    backup = backup_of(env)
    shutil.copy2(env, backup)
    backup.chmod(stat.S_IRUSR)
    mode = stat.S_IMODE(env.stat().st_mode)
    partial = env.with_name(f".{env.name}.retune")
    partial.write_text(text, encoding="utf-8")
    partial.chmod(mode)
    os.replace(partial, env)
    after = env_spec.assignments(env)
    differs = {key for key in current.keys() | after.keys() if current.get(key) != after.get(key)}
    if not differs <= set(edits) or any(after.get(key) != value for key, value in edits.items()):
        shutil.copy2(backup, env)  # a rewrite that did more than asked is undone
        raise SystemExit(f"retune: {env} read back with other changes than {sorted(edits)}; restored from {backup}")
    entry: dict[str, Any] = {
        "time": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "job": job,
        "env": str(env),
        "backup": str(backup),
        "user": getpass.getuser(),
        "reason": reason,
    }
    ledger = env.with_name(LEDGER)
    with ledger.open("a", encoding="utf-8") as out:
        out.writelines(json.dumps(entry | change, sort_keys=True) + "\n" for change in changes)
    return [entry | change for change in changes]


def ledger_of(env: pathlib.Path) -> list[dict[str, str]]:
    """The ledger lines of ``env``, oldest first."""
    ledger = env.with_name(LEDGER)
    lines = ledger.read_text(encoding="utf-8").splitlines() if ledger.is_file() else []
    return [entry for entry in map(json.loads, lines) if entry["env"] == str(env)]


def history(job: str) -> list[dict[str, str]]:
    """The ledger lines of job ``job``'s env, oldest first."""
    return ledger_of(job_env(job)[1])


def copy_ledger(env: pathlib.Path, run_dir: pathlib.Path) -> int:
    """Copy ``env``'s ledger lines into ``run_dir`` as :data:`LEDGER`, so the record travels with the run (what
    ``run_cluster.sh`` does at the job's start); returns how many. Nothing is written when none were changed."""
    entries = ledger_of(env)
    if entries:
        (run_dir / LEDGER).write_text(
            "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries), encoding="utf-8"
        )
    return len(entries)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hpcagent-bench job retune", description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("job", nargs="?")
    parser.add_argument("edits", nargs="*", metavar="KEY=VALUE")
    parser.add_argument("--reason", default="", help="why: written to the ledger (required to change anything)")
    parser.add_argument("--add", action="store_true", help="allow a key the env does not set")
    parser.add_argument("--dry-run", action="store_true", help="check and print the changes, write nothing")
    parser.add_argument("--show", action="store_true", help="print the ledger lines of the job's env")
    parser.add_argument(
        "--copy-ledger",
        nargs=2,
        type=pathlib.Path,
        metavar=("ENV", "RUN_DIR"),
        help="run_cluster.sh: copy ENV's ledger lines into RUN_DIR",
    )
    args = parser.parse_args(argv)
    if args.copy_ledger:
        copy_ledger(*args.copy_ledger)
        return 0
    if args.job is None:
        parser.error("a job id is required")
    if args.show:
        for entry in history(args.job):
            print(
                f"{entry['time']} {entry['user']} {entry['key']}: {entry['old']!r} -> {entry['new']!r} ({entry['reason']})"
            )
        return 0
    pairs = [edit.partition("=") for edit in args.edits]
    if not pairs or any(not sep or not key for key, sep, _ in pairs):
        raise SystemExit("retune: give KEY=VALUE words")
    for entry in retune(
        args.job, {key: value for key, _, value in pairs}, args.reason, adding=args.add, dry_run=args.dry_run
    ):
        print(f"{'would change' if args.dry_run else 'changed'} {entry['key']}: {entry['old']!r} -> {entry['new']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
