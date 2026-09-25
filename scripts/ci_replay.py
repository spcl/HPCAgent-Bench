#!/usr/bin/env python
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Replay the test phases of ``.github/workflows/tests.yml`` on this machine.

Every job is expanded over its matrix into legs; a leg runs its test steps (``run:`` blocks that
call ``python -m pytest`` or ``python -c``) in order, with the workflow, job and step ``env`` and
``${{ }}`` expressions rendered the way GitHub would for a push. Setup steps (``uses:``, apt, pip)
are skipped: the caller provides the environment (scripts/run_tests.sh, or the judge image through
scripts/ci_mi200.sbatch). Legs run concurrently, each step under ``timeout``; the verdict is one
line per step plus the failing test ids, in ``<out>/summary.txt``.

    python scripts/ci_replay.py --list
    python scripts/ci_replay.py --out ci-out --parallel 8 --skip lint,coverage,container-image/6a
    python scripts/ci_replay.py --jobs unit --matrix shard=0

A leg with a ``python`` matrix value runs only on that interpreter (the one running this script).
"""

import argparse
import concurrent.futures
import dataclasses
import itertools
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping
from typing import Any

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "tests.yml"
EXPRESSION = re.compile(r"\$\{\{(.*?)\}\}", re.DOTALL)
TEST_STEP = re.compile(r"^\s*python (-m pytest|-c)\b", re.MULTILINE)
FAILED_LINE = re.compile(r"^(FAILED|ERROR) (\S+)", re.MULTILINE)


class Context(dict[str, Any]):
    """A GitHub expression context: attribute access, and a missing key reads as ''."""

    def __getattr__(self, name: str) -> object:
        value = self.get(name, "")
        return Context(value) if isinstance(value, dict) else value


def evaluate(expression: str, contexts: Mapping[str, Any]) -> object:
    """Evaluate one ``${{ }}`` body; supports the operators and functions this workflow uses."""
    python = expression.strip()
    python = re.sub(r"'((?:[^']|'')*)'", lambda m: repr(m.group(1).replace("''", "'")), python)
    python = python.replace("&&", " and ").replace("||", " or ")
    python = re.sub(r"!(?!=)", " not ", python)
    python = re.sub(
        r"\b(true|false|null)\b", lambda m: {"true": "True", "false": "False", "null": "None"}[m[1]], python
    )
    functions = {
        "cancelled": lambda: False,
        "always": lambda: True,
        "success": lambda: True,
        "failure": lambda: False,
        "contains": lambda haystack, needle: needle in (haystack or ""),
    }
    return eval(python, {"__builtins__": {}}, {**functions, **contexts})


def render(text: str, contexts: Mapping[str, Any]) -> str:
    def one(match: re.Match[str]) -> str:
        value = evaluate(match.group(1), contexts)
        if value is True or value is False:
            return str(value).lower()
        return "" if value is None else str(value)

    return EXPRESSION.sub(one, text)


def matrix_combinations(strategy: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The legs a ``strategy.matrix`` expands into (lists crossed, then ``include`` appended)."""
    matrix = dict((strategy or {}).get("matrix") or {})
    include = matrix.pop("include", [])
    matrix.pop("exclude", None)
    keys = list(matrix)
    combos = [dict(zip(keys, values, strict=True)) for values in itertools.product(*(matrix[k] for k in keys))]
    if include:
        combos = [c for c in combos if c] + [dict(entry) for entry in include]
    return combos or [{}]


@dataclasses.dataclass(slots=True)
class Step:
    name: str
    script: str
    env: dict[str, str]
    timeout_s: int


@dataclasses.dataclass(slots=True)
class Leg:
    job: str
    label: str
    steps: list[Step]


def workflow_inputs(workflow: Mapping[Any, Any]) -> dict[str, Any]:
    """``workflow_dispatch`` input defaults (PyYAML reads the ``on:`` key as True)."""
    triggers = workflow.get("on", workflow.get(True)) or {}
    dispatch = (triggers.get("workflow_dispatch") or {}) if isinstance(triggers, dict) else {}
    return {name: spec.get("default", "") for name, spec in (dispatch.get("inputs") or {}).items()}


def legs(workflow: Mapping[Any, Any], scratch: pathlib.Path, timeout_factor: float) -> Iterator[Leg]:
    inputs = workflow_inputs(workflow)
    root_env = {k: str(v) for k, v in (workflow.get("env") or {}).items()}
    for job_name, job in workflow["jobs"].items():
        for matrix in matrix_combinations(job.get("strategy")):
            label = job_name + "".join(f"[{k}={v}]" for k, v in matrix.items() if not isinstance(v, dict))
            label += "".join(f"[{v.get('id', k)}]" for k, v in matrix.items() if isinstance(v, dict))
            temp = scratch / re.sub(r"[^\w.=-]+", "_", label)
            contexts: dict[str, Any] = {
                "matrix": Context(matrix),
                "github": Context(
                    event_name="push", ref="refs/heads/replay", job=job_name, run_id="0", workspace=str(REPO), event={}
                ),
                "runner": Context(temp=str(temp), os="Linux"),
                "inputs": Context(inputs),
                "secrets": Context(),
                "vars": Context(),
                "needs": Context(),
                "steps": Context(),
            }
            env = dict(root_env)
            contexts["env"] = Context(env)
            env.update({k: render(str(v), contexts) for k, v in (job.get("env") or {}).items()})
            job_timeout = int(job.get("timeout-minutes", 45))
            steps = []
            for step in job.get("steps", []):
                script = step.get("run")
                if not script or not TEST_STEP.search(script):
                    continue
                step_env = {**env, "RUNNER_TEMP": str(temp), "GITHUB_WORKSPACE": str(REPO)}
                contexts["env"] = Context(step_env)
                step_env.update({k: render(str(v), contexts) for k, v in (step.get("env") or {}).items()})
                minutes = int(step.get("timeout-minutes", job_timeout))
                steps.append(
                    Step(
                        name=render(str(step.get("name", "run")), contexts),
                        script=render(script, contexts),
                        env=step_env,
                        timeout_s=int(minutes * 60 * timeout_factor),
                    )
                )
            if steps:
                yield Leg(job=job_name, label=label, steps=steps)


def selected(leg: Leg, step: Step | None, patterns: list[str]) -> bool:
    """``job`` or ``job/step-substring`` patterns; a leg label matches its job name."""
    for pattern in patterns:
        job, _, part = pattern.partition("/")
        if job not in (leg.job, leg.label):
            continue
        if not part or (step is not None and part in step.name):
            return True
    return False


@dataclasses.dataclass(slots=True)
class Outcome:
    leg: str
    step: str
    rc: int
    seconds: float
    failed: list[str]
    log: pathlib.Path


def run_leg(leg: Leg, out: pathlib.Path, base_env: Mapping[str, str], skip: list[str], coverage: bool) -> list[Outcome]:
    outcomes = []
    for index, step in enumerate(leg.steps):
        if selected(leg, step, skip):
            continue
        log = out / f"{re.sub(r'[^\w.=-]+', '_', leg.label)}.{index}.log"
        env = {**base_env, **step.env}
        pathlib.Path(env["RUNNER_TEMP"]).mkdir(parents=True, exist_ok=True)
        if not coverage:
            env["PYTEST_ADDOPTS"] = " ".join(
                t for t in env.get("PYTEST_ADDOPTS", "").split() if not t.startswith("--cov")
            )
        started = time.monotonic()
        with log.open("w") as sink:
            sink.write(f"# {leg.label} :: {step.name}\n{step.script}\n")
            sink.flush()
            proc = subprocess.run(
                ["timeout", "--kill-after=60", str(step.timeout_s), "bash", "-eo", "pipefail", "-c", step.script],
                cwd=REPO,
                env=env,
                stdout=sink,
                stderr=subprocess.STDOUT,
                check=False,
            )
        text = log.read_text(errors="replace")
        failed = sorted({m.group(2) for m in FAILED_LINE.finditer(text)})
        outcomes.append(Outcome(leg.label, step.name, proc.returncode, time.monotonic() - started, failed, log))
    return outcomes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workflow", type=pathlib.Path, default=WORKFLOW)
    ap.add_argument("--jobs", default="", help="comma-separated jobs to run (default: all)")
    ap.add_argument("--skip", default="", help="comma-separated job or job/step-name-substring to skip")
    ap.add_argument(
        "--matrix",
        action="append",
        default=[],
        help="KEY=VALUE: keep only legs whose matrix matches (default: python=<this interpreter>)",
    )
    ap.add_argument("--parallel", type=int, default=4, help="legs run at once")
    ap.add_argument("--timeout-factor", type=float, default=1.5, help="scale every step's timeout-minutes")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "ci-replay")
    ap.add_argument("--coverage", action="store_true", help="keep the workflow's --cov options")
    ap.add_argument("--list", action="store_true", help="print the legs and steps, run nothing")
    args = ap.parse_args(argv)

    workflow = yaml.safe_load(args.workflow.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="ci-replay-", dir=os.environ.get("TMPDIR")))
    jobs = {j for j in args.jobs.split(",") if j}
    skip = [s for s in args.skip.split(",") if s]
    wanted = [tuple(m.split("=", 1)) for m in args.matrix]
    if not any(key == "python" for key, _ in wanted):
        wanted.append(("python", f"{sys.version_info.major}.{sys.version_info.minor}"))
    chosen = []
    for leg in legs(workflow, scratch, args.timeout_factor):
        if (jobs and leg.job not in jobs) or selected(leg, None, skip):
            continue
        if any(f"[{key}=" in leg.label and f"[{key}={value}]" not in leg.label for key, value in wanted):
            continue
        chosen.append(leg)
    if args.list:
        for leg in chosen:
            for step in leg.steps:
                mark = " (skipped)" if selected(leg, step, skip) else ""
                print(f"{leg.label} :: {step.name} [{step.timeout_s // 60} min]{mark}")
        return 0

    # Each leg's `-n auto` gets its share of the machine.
    base_env = dict(os.environ)
    base_env.setdefault("PYTEST_XDIST_AUTO_NUM_WORKERS", str(max(1, (os.cpu_count() or 1) // args.parallel)))
    # Longest budget first, so the tail is short legs.
    chosen.sort(key=lambda leg: -sum(step.timeout_s for step in leg.steps))
    outcomes: list[Outcome] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(run_leg, leg, args.out, base_env, skip, args.coverage) for leg in chosen]
        for future in concurrent.futures.as_completed(futures):
            for outcome in future.result():
                outcomes.append(outcome)
                verdict = "ok" if outcome.rc == 0 else ("TIMEOUT" if outcome.rc in (124, 137) else f"rc={outcome.rc}")
                print(f"{verdict:8} {outcome.seconds / 60:6.1f} min  {outcome.leg} :: {outcome.step}", flush=True)

    lines = []
    for outcome in sorted(outcomes, key=lambda o: (o.rc == 0, o.leg, o.step)):
        verdict = "ok" if outcome.rc == 0 else ("TIMEOUT" if outcome.rc in (124, 137) else f"rc={outcome.rc}")
        lines.append(
            f"{verdict:8} {outcome.seconds / 60:6.1f} min  {outcome.leg} :: {outcome.step}  ({outcome.log.name})"
        )
        lines += [f"           {test}" for test in outcome.failed]
    red = sum(o.rc != 0 for o in outcomes)
    lines.append(f"{len(outcomes) - red} of {len(outcomes)} steps green")
    (args.out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
