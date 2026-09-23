# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""experiments/submit-owed-wave.sh passes NICE to sbatch as ``--nice``.

Harness owed waves are queued behind the LLR and scicomp waves by priority alone, never by a
dependency: NICE=<n> is that knob, the one submit-canon-llr40.sh already has. Runs the real launcher
in a temp tree with a planner stub that plans one wave and a recording sbatch stub.
"""

import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"

#: Plans ONE wave, the way owed_wave.py's plan.tsv lists it: name, env, nodes, walltime.
PLANNER = """
import pathlib, sys
if "--per-problem-keys" in sys.argv:
    print("CAMPAIGN_ARM")
    raise SystemExit(0)
out = pathlib.Path(sys.argv[sys.argv.index("--out") + 1])
env = out / ".env.owed-harness20-qwen38-openhands-w1"
env.write_text("CAMPAIGN_ARM=owed\\n")
pathlib.Path(sys.argv[sys.argv.index("--plan") + 1]).write_text(f"owed-harness20-qwen38-openhands-w1\\t{env}\\t3\\t11:00:00\\n")
"""


def stub(directory: pathlib.Path, name: str, body: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(f"#!/usr/bin/env bash\n{body}\n")
    path.chmod(0o755)


def submit(tmp_path: pathlib.Path, **knobs: str) -> list[str]:
    """The sbatch argv one SUBMIT=1 run of the launcher sends, with ``knobs`` in its environment."""
    experiments = tmp_path / "experiments"
    experiments.mkdir(exist_ok=True)
    for name in ("submit-owed-wave.sh", "submit_common.sh", "arm_nodes.sh", "env_layers.sh"):
        shutil.copy2(EXPERIMENTS / name, experiments / name)
    (experiments / "owed_wave.py").write_text(PLANNER)
    (tmp_path / "scripts" / "cscs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "scripts" / "cscs" / "account_env.sh", tmp_path / "scripts" / "cscs" / "account_env.sh")
    stub(tmp_path / "bin", "sacctmgr", "printf 'a-g34\\n'")
    stub(tmp_path / "bin", "sbatch", 'printf \'%s\\n\' "$@" > "${STUB_MARKERS}/sbatch-argv.txt"; echo 999999')
    env = {
        "PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin",
        "USER": "tester",
        "SCRATCH": str(tmp_path / "scratch"),
        "OPT": str(tmp_path),
        "PY": sys.executable,
        "OUT": str(tmp_path / "out"),
        "STUB_MARKERS": str(tmp_path),
        **knobs,
    }
    done = subprocess.run(
        ["bash", str(experiments / "submit-owed-wave.sh"), "MODEL=qwen38", "SUBMIT=1"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return (tmp_path / "sbatch-argv.txt").read_text(encoding="utf-8").splitlines()


def test_nice_reaches_sbatch_when_set(tmp_path: pathlib.Path) -> None:
    argv = submit(tmp_path, NICE="500")
    assert "--nice=500" in argv
    assert "--dependency" not in " ".join(argv), "lower priority, never a dependency"


def test_no_nice_flag_when_unset(tmp_path: pathlib.Path) -> None:
    """Every caller that never sets NICE keeps its ordinary priority."""
    assert not any(flag.startswith("--nice") for flag in submit(tmp_path))
