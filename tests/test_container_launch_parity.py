# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Golden parity: the pure-bash launcher (scripts/run_agent_in_container.sh --print) and
the Python factory (containers.local_run_command) fold the SAME launch argv, byte for byte,
because both read optarena/container_backends.txt. This is what makes the single source
real rather than a hand-kept mirror -- if the two folds ever drift, this test fails.

Runs `bash --print` in a controlled environment (no ambient OPTARENA_* leakage) and
compares token-for-token against local_run_command over every backend x hardware."""
import os
import pathlib
import shutil
import subprocess

import pytest

from optarena import containers

REPO_ROOT = pathlib.Path(containers.__file__).resolve().parent.parent
LAUNCHER = REPO_ROOT / "scripts" / "run_agent_in_container.sh"
AGENT_ARGS = ["--kernels", "gemm", "--baseline", "c"]

pytestmark = pytest.mark.skipif(shutil.which("bash") is None or not LAUNCHER.exists(),
                                reason="needs bash + the launcher script")


def controlled_env(backend):
    """A reproducible environment: the ambient PATH etc., every OPTARENA_*/passthrough var
    stripped, then a FIXED set both folds will see identically (incl. a passthrough var and
    two out-of-order dynamic OPTARENA_* vars to exercise the pinned env ordering)."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("OPTARENA_") and k not in ("OLLAMA_HOST", "ANTHROPIC_API_KEY", "OPTARENA_OLLAMA_HOST",
                                                       "OPTARENA_LOCAL_MODEL")
    }
    env["OPTARENA_RUNTIME_BACKEND"] = backend
    env["ANTHROPIC_API_KEY"] = "sk-test"  # a passthrough (non-OPTARENA) var
    env["OPTARENA_ZED"] = "z"  # dynamic OPTARENA_*, must sort after ABC
    env["OPTARENA_ABC"] = "a"  # dynamic OPTARENA_*, must sort before ZED
    return env


@pytest.mark.parametrize("backend", ["apptainer", "docker", "podman", "udocker"])
@pytest.mark.parametrize("hardware", ["cpu", "nvidia", "amd"])
def test_bash_and_python_fold_identical_argv(backend, hardware, monkeypatch):
    env = controlled_env(backend)
    # bash: --print emits one token per line, no exec/probe.
    proc = subprocess.run(["bash", str(LAUNCHER), hardware, "--print", "--", *AGENT_ARGS],
                          env=env,
                          capture_output=True,
                          text=True,
                          check=True)
    bash_argv = proc.stdout.splitlines()

    # python: same env (so collect_env sees the same vars), same repo_root as bash computes.
    for key in [k for k in os.environ if k.startswith("OPTARENA_") or k in ("OLLAMA_HOST", "ANTHROPIC_API_KEY")]:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        if key.startswith("OPTARENA_") or key in ("OLLAMA_HOST", "ANTHROPIC_API_KEY"):
            monkeypatch.setenv(key, value)
    inner = ["python", "-m", "optarena.cli", "agent", *AGENT_ARGS]
    py_argv = containers.local_run_command(inner, backend=backend, hardware=hardware, repo_root=str(REPO_ROOT))

    assert bash_argv == py_argv
