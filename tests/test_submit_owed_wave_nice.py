# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""experiments/submit-owed-wave.sh hands its knobs to sbatch and to the planner.

Harness owed waves are queued behind the LLR and scicomp waves by priority alone, never by a
dependency: NICE=<n> is that knob, the one submit-canon-llr40.sh already has. KERNELS_FILE narrows
the plan (the scicomp37 reruns), WAVE_INFERENCE_CE_ENV pins one wave's engine image (oss120b mini-SWE
on vLLM 0.27.1), and a SUBMIT=1 plan must have read the queue. Runs the real launcher in a temp tree
with a planner stub that records its argv and plans one wave, and a recording sbatch stub.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"

#: Plans ONE wave, the way owed_wave.py's plan.tsv lists it: name, env, nodes, walltime.
PLANNER = """
import json, os, pathlib, sys
if "--per-problem-keys" in sys.argv:
    print("CAMPAIGN_ARM")
    raise SystemExit(0)
if "--preflight" in sys.argv:
    pathlib.Path(os.environ["STUB_MARKERS"], "preflight-argv.json").write_text(json.dumps(sys.argv[1:]))
    raise SystemExit(int(os.environ.get("STUB_PREFLIGHT_RC", "0")))
out = pathlib.Path(sys.argv[sys.argv.index("--out") + 1])
out.parent.joinpath("planner-call.json").write_text(json.dumps({"argv": sys.argv[1:], "pythonpath": os.environ.get("PYTHONPATH", "")}))
env = out / ".env.owed-harness20-qwen38-openhands-w1"
env.write_text("CAMPAIGN_ARM=owed\\n")
pathlib.Path(sys.argv[sys.argv.index("--plan") + 1]).write_text(f"owed-harness20-qwen38-openhands-w1\\t{env}\\t3\\t11:00:00\\n")
"""


def stub(directory: pathlib.Path, name: str, body: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(f"#!/usr/bin/env bash\n{body}\n")
    path.chmod(0o755)


def submit(
    tmp_path: pathlib.Path,
    submit_flag: str = "1",
    expect_rc: int = 0,
    associations: str = "a-stub",
    **knobs: str,
) -> list[str]:
    """The sbatch argv one SUBMIT=1 run of the launcher sends, with ``knobs`` in its environment and
    ``associations`` (one per line) as sacctmgr's answer."""
    experiments = tmp_path / "experiments"
    experiments.mkdir(exist_ok=True)
    for name in ("submit-owed-wave.sh", "submit_common.sh", "arm_nodes.sh", "env_layers.sh"):
        shutil.copy2(EXPERIMENTS / name, experiments / name)
    (experiments / "owed_wave.py").write_text(PLANNER)
    (tmp_path / "scripts" / "cscs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "scripts" / "cscs" / "account_env.sh", tmp_path / "scripts" / "cscs" / "account_env.sh")
    stub(tmp_path / "bin", "sacctmgr", f"printf '{associations}'")  # a made-up name
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
        ["bash", str(experiments / "submit-owed-wave.sh"), "MODEL=qwen38", f"SUBMIT={submit_flag}"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == expect_rc, done.stderr
    argv = tmp_path / "sbatch-argv.txt"
    return argv.read_text(encoding="utf-8").splitlines() if argv.exists() else []


def planner_call(tmp_path: pathlib.Path) -> dict[str, object]:
    """What the launcher ran the planner with: its argv and its PYTHONPATH."""
    return json.loads((tmp_path / "planner-call.json").read_text(encoding="utf-8"))


def test_nice_reaches_sbatch_when_set(tmp_path: pathlib.Path) -> None:
    argv = submit(tmp_path, NICE="500")
    assert "--nice=500" in argv
    assert "--dependency" not in " ".join(argv), "lower priority, never a dependency"


def test_no_nice_flag_when_unset(tmp_path: pathlib.Path) -> None:
    """Every caller that never sets NICE keeps its ordinary priority."""
    assert not any(flag.startswith("--nice") for flag in submit(tmp_path))


def option(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_a_kernels_file_and_a_pinned_inference_image_reach_the_planner(tmp_path: pathlib.Path) -> None:
    submit(tmp_path, KERNELS_FILE="/scratch/kernels-scicomp37.txt", WAVE_INFERENCE_CE_ENV="vllm0271")
    argv = planner_call(tmp_path)["argv"]
    assert option(argv, "--kernels-file") == "/scratch/kernels-scicomp37.txt"
    assert option(argv, "--inference-ce-env") == "vllm0271"


def test_a_submission_requires_the_queue_and_a_dry_run_does_not(tmp_path: pathlib.Path) -> None:
    submit(tmp_path)
    assert "--require-queue" in planner_call(tmp_path)["argv"]
    dry = tmp_path / "dry"
    dry.mkdir()
    assert submit(dry, submit_flag="0") == [], "a dry run submits nothing"
    assert "--require-queue" not in planner_call(dry)["argv"]


def test_the_planner_imports_the_package_from_the_checkout_it_runs_in(tmp_path: pathlib.Path) -> None:
    """Without it: ModuleNotFoundError: No module named 'hpcagent_bench' unless the caller exported one."""
    submit(tmp_path)
    assert str(planner_call(tmp_path)["pythonpath"]).split(os.pathsep)[0] == str(tmp_path)


def test_a_family_submits_at_its_priority_band(tmp_path: pathlib.Path) -> None:
    """User 2026-09-23: PRIORITY names the family, submit_common.sh's PRIORITY_NICE its --nice."""
    assert "--nice=2000" in submit(tmp_path, PRIORITY="harness20")


def test_the_priority_bands_follow_the_users_submission_order() -> None:
    script = f'. "{EXPERIMENTS / "submit_common.sh"}" 2>/dev/null; for f in regrade llr-gpu-device harness20 scicomp mlscale kimi; do echo "${{PRIORITY_NICE[$f]}}"; done'
    done = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True, env={**os.environ, "OPT": str(REPO)}
    )
    bands = [int(line) for line in done.stdout.split()]
    assert bands == [0, 1000, 2000, 3000, 4000, 10000]


def test_an_unknown_family_or_a_disagreeing_nice_submits_nothing(tmp_path: pathlib.Path) -> None:
    assert submit(tmp_path, expect_rc=2, PRIORITY="harness") == []
    other = tmp_path / "other"
    other.mkdir()
    assert submit(other, expect_rc=2, PRIORITY="kimi", NICE="0") == []


def test_every_planned_wave_passes_the_preflight_before_anything_is_submitted(tmp_path: pathlib.Path) -> None:
    submit(tmp_path)
    assert json.loads((tmp_path / "preflight-argv.json").read_text())[-1] == str(tmp_path / "out")
    failed = tmp_path / "failed"
    failed.mkdir()
    assert submit(failed, expect_rc=2, STUB_PREFLIGHT_RC="1") == [], "a FAIL submits no wave"


def test_a_submission_with_no_account_resolved_submits_nothing(tmp_path: pathlib.Path) -> None:
    """beverin runs an accountless job on root (fairshare ~0) since its cli_filter went away: no
    account, no sbatch -- where the resolver's "no association" used to be caught by the cluster."""
    assert submit(tmp_path, expect_rc=2, associations="") == []
    dry = tmp_path / "dry"
    dry.mkdir()
    submit(dry, submit_flag="0", associations="")
