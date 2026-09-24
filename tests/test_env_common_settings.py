# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""What is COMMON to every model lives in the launcher, and what is per-model lives in its .env.

The client timeouts were duplicated per model and drifted: kimi and glm53 set them, qwen38 and
oss120b set neither and silently ran on the CLI's 15-minute idle default, which ended healthy Qwen
agents mid-prefill. A value that can be spelled in two places will eventually be spelled two ways,
so these tests state which place each one belongs in and that the per-model values agree with the
server arguments they describe.
"""

import importlib.util
import pathlib
import re
import subprocess
import sys
import types

import pytest
from tests.env_render import rendered

REPO = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"
LAUNCHER = EXPERIMENTS / "run_cluster.sh"

#: Settings that are the same for every model and every harness: the launcher owns them, and a .env
#: that repeats one is how two arms end up on different values.
COMMON_VARS = (
    "API_TIMEOUT_MS",
    "CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS",
    "CLAUDE_STREAM_IDLE_TIMEOUT_MS",
    "API_FORCE_IDLE_TIMEOUT",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
)

#: The launcher's default for each of them. The idle watchdog is the wall that fires first in Claude
#: Code 2.1.197; its default is DERIVED (stream_idle_timeout.py, 2026-09-19) from the arm's own
#: CONTEXT_LENGTH and AGENTS_PER_NODE rather than copied, but every arm that named neither still
#: lands on 1800000 ms, the CLI's ceiling for it -- see test_stream_idle_timeout.py.
LAUNCHER_DEFAULTS = {
    "API_TIMEOUT_MS": "3600000",
    "CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS": '$(python3 "${SCRIPT_DIR}/stream_idle_timeout.py")',
    # The byte watchdog above is installed only for api.anthropic.com. Against SGLang/vLLM the walls
    # that fire are the SSE-event watchdog (floor 300 s) and Bun's own ~300 s fetch socket timeout,
    # which the CLI lifts only when API_FORCE_IDLE_TIMEOUT is falsy -- both unset cut qwen38 streams
    # at 4-5 min of silence mid tool_use ("API Error: The operation timed out.", mlscale 649795).
    # The event watchdog takes the SAME derived number, never a second one.
    "CLAUDE_STREAM_IDLE_TIMEOUT_MS": "${CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS}",
    "API_FORCE_IDLE_TIMEOUT": "0",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32768",
}

#: The rungs each model's server accepts, lowest first, whether that server is a local SGLang/vLLM
#: job or a hosted provider API. Qwen's chat template raises on anything outside low/medium/xhigh --
#: it has no `high` -- and GPT-OSS's stops at high; Kimi and GLM have no ladder at all. The launcher
#: resolves AGENT_EFFORT from these, so a .env that also spelled a rung would be a second source of
#: truth for the one thing a ladder exists to decide.
LADDERS = {
    "qwen38": "low medium xhigh",
    "oss120b": "low medium high",
    "kimi27sglang": "",
    "glm53": "",
    "fable51": "low medium high xhigh max",
    "gpt6astra": "low medium high xhigh max",
    "musespark": "low medium high xhigh max",
    "unionalpha": "",
}

#: What the policy resolves each ladder to: xhigh where the ladder has it, else its top rung, else
#: no field at all.
RESOLVED = {
    "qwen38": "xhigh",
    "oss120b": "high",
    "kimi27sglang": "",
    "glm53": "",
    "fable51": "xhigh",
    "gpt6astra": "xhigh",
    "musespark": "xhigh",
    "unionalpha": "",
}

BASE_ENVS = sorted(EXPERIMENTS.glob(".env.base-*"))


def load_effort() -> types.ModuleType:
    """``experiments/effort.py``, loaded by path: it ships in the agent image, not the package."""
    spec = importlib.util.spec_from_file_location("effort", EXPERIMENTS / "effort.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


effort = load_effort()


def env_values(path: pathlib.Path) -> dict[str, str]:
    """``KEY=VALUE`` lines of the rendered env file, quotes stripped."""
    values: dict[str, str] = {}
    for line in rendered(path).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key] = value.strip().strip('"')
    return values


def is_inference_service_env(path: pathlib.Path) -> bool:
    """True for a base env that runs its model behind a hosted provider API (INFERENCE_SOURCE=service)
    rather than an SGLang/vLLM job the launcher starts on a node."""
    return env_values(path).get("INFERENCE_SOURCE", "node") == "service"


#: Base envs whose model is served by an engine the launcher starts on a node. A hosted service env
#: has no such engine to name a context window for.
ENGINE_BASE_ENVS = [path for path in BASE_ENVS if not is_inference_service_env(path)]


def test_the_launcher_carries_every_base_env() -> None:
    """A model whose .env is not in this parametrisation is a model these rules never checked."""
    assert {path.name.removeprefix(".env.base-") for path in BASE_ENVS} == set(LADDERS)


@pytest.mark.parametrize("path", BASE_ENVS, ids=lambda path: path.name)
@pytest.mark.parametrize("name", COMMON_VARS)
def test_a_base_env_sets_none_of_the_common_client_settings(path: pathlib.Path, name: str) -> None:
    assert name not in env_values(path), f"{path.name} repeats the launcher's {name}"


@pytest.mark.parametrize("name", COMMON_VARS)
def test_the_launcher_exports_each_common_setting_with_its_default(name: str) -> None:
    """The .env files no longer carry these, so the launcher's default IS what every arm runs at."""
    default = LAUNCHER_DEFAULTS[name]
    assert f'export {name}="${{{name}:-{default}}}"' in LAUNCHER.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", ENGINE_BASE_ENVS, ids=lambda path: path.name)
def test_a_base_env_names_the_context_window_its_server_is_started_with(path: pathlib.Path) -> None:
    """The harnesses size their prompt budget off CONTEXT_LENGTH; an engine started with a different
    window makes every one of them wrong in the same invisible way. Scoped to envs that start an
    engine: a hosted-service env (INFERENCE_SOURCE=service) serves through a provider API and starts
    no engine to name a window for."""
    values = env_values(path)
    served = re.findall(r"(?:--context-length|--max-model-len)[= ](\d+)", rendered(path))
    assert served, f"{path.name} starts no engine with a context window"
    assert len(set(served)) == 1, f"{path.name} names several context windows: {served}"
    assert values.get("CONTEXT_LENGTH") == served[0]


@pytest.mark.parametrize("path", BASE_ENVS, ids=lambda path: path.name)
def test_a_base_env_declares_the_ladder_its_server_accepts_and_no_rung(path: pathlib.Path) -> None:
    """The .env states what the SERVER accepts; the launcher states which rung of it to take. A .env
    that also spelled the rung is how oss120b and qwen38 came to be compared at rungs nobody had
    written down together. A model with no ladder declares an empty one rather than omitting the key,
    because a MISSING AGENT_EFFORT still defaults to xhigh in agent_driver.py."""
    values = env_values(path)
    assert "AGENT_EFFORT" not in values, f"{path.name} spells a rung the launcher resolves"
    assert values.get("EFFORT_LADDER") == LADDERS[path.name.removeprefix(".env.base-")]


@pytest.mark.parametrize("path", BASE_ENVS, ids=lambda path: path.name)
def test_the_policy_resolves_each_declared_ladder_to_the_rung_that_model_runs_at(path: pathlib.Path) -> None:
    """The ladders are only right if the rung they resolve to is the one the campaign meant to run."""
    model = path.name.removeprefix(".env.base-")
    assert effort.resolve(env_values(path)["EFFORT_LADDER"]) == RESOLVED[model]


def test_every_cli_idle_wall_resolves_to_the_one_derived_value() -> None:
    """Run the launcher's own idle-timeout export lines: the SSE-event watchdog must land on the
    same number as the byte watchdog, and Bun's fetch socket timeout must be switched off (the CLI
    reads "0" as falsy and then passes ``timeout: false`` to fetch). A 262144-token qwen38 arm at 40
    agents per node derives the CLI's 30-minute ceiling."""
    names = ("CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS", "CLAUDE_STREAM_IDLE_TIMEOUT_MS", "API_FORCE_IDLE_TIMEOUT")
    lines = [
        line.strip()
        for line in LAUNCHER.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(tuple(f"export {name}=" for name in names))
    ]
    assert len(lines) == len(names)
    script = "\n".join([*lines, *(f'echo "{name}=${{{name}}}"' for name in names)])
    done = subprocess.run(
        ["bash", "-c", script],
        env={
            "PATH": "/usr/bin:/bin",
            "SCRIPT_DIR": str(EXPERIMENTS),
            "CONTEXT_LENGTH": "262144",
            "AGENTS_PER_NODE": "40",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    resolved = dict(line.split("=", 1) for line in done.stdout.split())
    assert resolved == {
        "CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS": "1800000",
        "CLAUDE_STREAM_IDLE_TIMEOUT_MS": "1800000",
        "API_FORCE_IDLE_TIMEOUT": "0",
    }
