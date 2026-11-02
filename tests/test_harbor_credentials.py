# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""What the Harbor launcher hands Harbor, with no credential and no model call: provider secrets stay in the
environment and out of the argv, consent is the caller's, the argv is one the locked Harbor accepts, and Harbor turns a
Bedrock environment into Claude Code's. A fake here proves the assembly only; real authentication and routing need
the live smoke of docs/hf_dataset_and_harbor.md."""

import dataclasses
import pathlib
import shutil
import subprocess
from typing import Any

import pytest

from hpcagent_bench.export import harbor as A

SECRET = "test-credential-0123456789"
MODEL = "us.anthropic.claude-sonnet-4-6"


@dataclasses.dataclass
class Launched:
    returncode: int = 0


def launched_by(monkeypatch: pytest.MonkeyPatch, *argv: str) -> tuple[list[str], dict[str, str]]:
    """What ``hpcagent-bench harbor generate --run`` starts for ``argv``: the command and its environment."""
    captured: list[tuple[list[str], dict[str, str]]] = []
    real_run = subprocess.run

    def run(cmd: list[str], *args: Any, **kwargs: Any) -> object:  # noqa: ANN401
        if cmd[0] != "harbor":  # task generation's own tools
            return real_run(cmd, *args, **kwargs)
        captured.append((cmd, kwargs["env"]))
        return Launched()

    monkeypatch.setenv("HPCAGENT_BENCH_RUNTIME_BACKEND", "podman")
    monkeypatch.setattr(A.shutil, "which", lambda cmd: "/usr/bin/harbor")
    monkeypatch.setattr(A.subprocess, "run", run)
    assert A.main(["generate", "--selector", "gemm", "--run", *argv]) == 0
    (launch,) = captured
    return launch


def test_a_secret_reaches_harbor_through_the_environment_and_never_the_argv(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", SECRET)
    argv = ("--out", str(tmp_path / "tasks"), "--jobs-dir", str(tmp_path / "runs"))
    cmd, env = launched_by(monkeypatch, *argv)
    assert env["AWS_BEARER_TOKEN_BEDROCK"] == SECRET
    assert SECRET not in " ".join(cmd)
    assert not any(
        SECRET in path.read_bytes().decode(errors="replace") for path in tmp_path.rglob("*") if path.is_file()
    )


def test_consent_to_host_access_is_only_ever_the_callers(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = ("--out", str(tmp_path / "tasks"), "--jobs-dir", str(tmp_path / "runs"))
    assert "--yes" not in launched_by(monkeypatch, *argv)[0]
    assert launched_by(monkeypatch, *argv, "--yes")[0][-1] == "--yes"


def test_a_generated_task_makes_harbor_ask_for_nothing_from_the_host(tmp_path: pathlib.Path) -> None:
    """Without ``--yes`` Harbor prompts for each ``${VAR}`` a task's environment or verifier takes from the host, and a
    batch job has no terminal to answer on."""
    pytest.importorskip("harbor.models.task.config")
    from harbor.models.task.config import TaskConfig
    from harbor.utils.env import get_required_host_vars

    (task,) = A.generate(tmp_path, selector="gemm")
    config = TaskConfig.model_validate_toml((task / "task.toml").read_text(encoding="utf-8"))
    assert get_required_host_vars({**config.environment.env, **config.verifier.env}) == []


@pytest.mark.parametrize("agent", ["claude", "openai"])
def test_the_locked_harbor_accepts_the_launch_argv(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, agent: str
) -> None:
    if shutil.which("harbor") is None:
        pytest.skip("the harbor CLI is installed by the harbor dependency group")
    monkeypatch.setenv("HPCAGENT_BENCH_RUNTIME_BACKEND", "podman")
    monkeypatch.setenv("ANTHROPIC_MODEL", MODEL)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://judge.invalid:8000")
    monkeypatch.setenv("OPENAI_MODEL", "stub")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://judge.invalid:8000/v1")
    A.generate(tmp_path / "tasks", selector="gemm")
    argv = [*A.run_argv(tmp_path / "tasks", job_name="t", jobs_dir=tmp_path / "jobs"), *A.agent_args(agent)]
    done = subprocess.run([*argv, "--dry-run"], capture_output=True, text=True, timeout=120, check=False, cwd=tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "Dry run OK" in done.stdout


@pytest.mark.parametrize(
    ("environment", "region"),
    [
        ({"AWS_BEARER_TOKEN_BEDROCK": SECRET, "AWS_REGION": "eu-west-1"}, "eu-west-1"),
        ({"AWS_BEARER_TOKEN_BEDROCK": SECRET, "ANTHROPIC_API_KEY": "the native key loses"}, "us-east-1"),
    ],
)
def test_harbor_turns_a_bedrock_environment_into_claude_codes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, environment: dict[str, str], region: str
) -> None:
    """The launcher's ``--model`` for ``ANTHROPIC_MODEL``, read back by Harbor's own claude-code agent."""
    pytest.importorskip("harbor.agents.installed.claude_code")
    from harbor.agents.installed.claude_code import ClaudeCode

    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_MODEL", MODEL)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    args = A.agent_args("claude")
    agent = ClaudeCode(logs_dir=tmp_path, model_name=args[args.index("--model") + 1])
    env = agent._resolve_auth_env()
    assert agent._resolved_model_name() == MODEL
    assert (env["CLAUDE_CODE_USE_BEDROCK"], env["AWS_REGION"]) == ("1", region)
    assert env["AWS_BEARER_TOKEN_BEDROCK"] == SECRET
    assert "ANTHROPIC_API_KEY" not in env
