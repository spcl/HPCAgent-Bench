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
import sys
import types

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"
LAUNCHER = EXPERIMENTS / "run_cluster.sh"

#: Settings that are the same for every model and every harness: the launcher owns them, and a .env
#: that repeats one is how two arms end up on different values.
COMMON_VARS = ("API_TIMEOUT_MS", "CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS", "CLAUDE_CODE_MAX_OUTPUT_TOKENS")

#: The launcher's default for each of them. The idle watchdog is the wall that fires first in Claude
#: Code 2.1.197, and 1800000 ms is the CLI's ceiling for it.
LAUNCHER_DEFAULTS = {
    "API_TIMEOUT_MS": "3600000",
    "CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS": "1800000",
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
    """``KEY=VALUE`` lines of a shell-compatible env file, quotes stripped, comments skipped."""
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
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
    served = re.findall(r"(?:--context-length|--max-model-len)[= ](\d+)", path.read_text(encoding="utf-8"))
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
