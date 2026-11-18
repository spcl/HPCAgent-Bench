# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""What an agent worker inherits of its driver's environment (``agent_driver.agent_environment``): an allowlist.
The Container Engine and apptainer hand the agent step the job's whole environment, so a submitter's token or a
module's path would otherwise reach a process the agent controls. These tests pin both directions: nothing secret or
host-specific passes, and nothing a worker, its harness or the image needs is dropped."""

import pathlib
import re

import pytest

from hpcagent_bench import paths
from tests.fresh_module import fresh

#: Variables a submitting shell or a site module carries that no worker reads.
SUBMITTER = {
    "HF_TOKEN": "hf_x",
    "WANDB_API_KEY": "w",
    "GH_TOKEN": "g",
    "GITHUB_TOKEN": "g",
    "AWS_SECRET_ACCESS_KEY": "a",
    "SSH_AUTH_SOCK": "/run/ssh",
    "HPCAGENT_BENCH_REPO": "/users/x/hpcagent-bench",
    "SCRIPT_DIR": "/users/x/hpcagent-bench/hpcagent_bench/cluster",
    "PROBLEMS_FILE": "/users/x/hpcagent-bench/experiments/problems-a.jsonl",
    "CLUSTER_ENV_FILE": "/users/x/hpcagent-bench/experiments/.env.a",
    "SLURM_SUBMIT_DIR": "/users/x/hpcagent-bench/experiments",
    "AGENT_LAUNCH_DIR": "/scratch/runs/.agent-launch/1",
    "OLDPWD": "/users/x/hpcagent-bench",
}
#: What a harness reaches its model with and tunes itself by, and the fused worker's judge signature.
HARNESS = {
    "ANTHROPIC_API_KEY": "sk",
    "ANTHROPIC_AUTH_TOKEN": "t",
    "ANTHROPIC_BASE_URL": "http://proxy:4000",
    "OPENAI_API_KEY": "sk",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32000",
    "MAX_THINKING_TOKENS": "8000",
    "AGENT_MAX_TOKENS": "100",
    "HPCAGENT_BENCH_WORKER_TOKEN": "w",
    "HTTPS_PROXY": "http://proxy:3128",
    "LC_ALL": "C",
    "PATH": "/opt/venv/bin:/usr/bin",
    "ROCR_VISIBLE_DEVICES": "0",
    "CUDA_VISIBLE_DEVICES": "0",
}


def environment_of(monkeypatch: pytest.MonkeyPatch, variables: dict[str, str]) -> dict[str, str]:
    driver = fresh("agent_driver")
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    return driver.agent_environment({"id": 0, "kernel": "gemm", "task": "t"}, "http://judge", 0, 0, 0)


def test_no_submitter_token_or_checkout_path_reaches_a_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = environment_of(monkeypatch, SUBMITTER)
    assert not sorted(set(SUBMITTER) & set(environment))


def test_a_worker_keeps_its_harness_keys_budgets_proxies_and_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = environment_of(monkeypatch, HARNESS)
    assert {name: environment.get(name) for name in HARNESS} == HARNESS


def image_env_names() -> set[str]:
    """Every name a judge-agent image sets with ENV."""
    names: set[str] = set()
    for dockerfile in (paths.ROOT / "containers" / "images").glob("judge-agent-*/Dockerfile*"):
        for line in dockerfile.read_text(encoding="utf-8").replace("\\\n", " ").splitlines():
            if line.lstrip().startswith("ENV "):
                names |= set(re.findall(r"(?:^|\s)([A-Za-z_]\w*)=", line.lstrip()[4:]))
    return names


def worker_read_names() -> set[str]:
    """Every name the worker side of the agent package (its tools and harness runners) reads by literal."""
    agent = paths.ROOT / "agent" / "hpcagent_agent"
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for folder in (agent / "tools", agent / "harness")
        for path in pathlib.Path(folder).rglob("*.py")
    )
    reads = set(re.findall(r"""environ(?:\.get)?[\[(]\s*["']([A-Za-z_]\w*)["']""", source))
    return reads | set(re.findall(r"""^[A-Z_]+_ENV = ["']([A-Z][A-Z0-9_]+)["']""", source, flags=re.MULTILINE))


def test_every_name_the_image_sets_or_a_worker_reads_is_inherited() -> None:
    """A new ENV line in an image or a new variable a tool reads must not be dropped in silence: a compile in the
    worker would then differ from the judge's, or a tool would lose its setting."""
    driver = fresh("agent_driver")
    needed = image_env_names() | worker_read_names()
    assert {"CC", "LD_PRELOAD", "JUDGE_URL"} <= needed, "the scans found nothing"
    episode_token = driver.http_json.EPISODE_TOKEN_ENV  # set per worker after the filter (identity_env)
    assert sorted(name for name in needed - {episode_token} if not driver.agent_env_allowed(name)) == []


def test_the_episode_token_is_the_drivers_own_and_not_inherited() -> None:
    """A worker signs with the token the driver mints for its episode, never one left in the job environment."""
    driver = fresh("agent_driver")
    assert not driver.agent_env_allowed(driver.http_json.EPISODE_TOKEN_ENV)
