# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The context window claude-code 2.1.197 is told about, and where it compacts inside it.

300 of 300 episodes never compacted: for a model it does not know the CLI assumes a 200000 window,
leaves proactive compaction off while that window's source is "auto", and waits for Anthropic's
"prompt is too long" to compact reactively -- vLLM and SGLang say "maximum context length" instead,
so Qwen episodes grew to 230674 input tokens and died on the 400. agent_driver.claude_context_env
names the window (CLAUDE_CODE_MAX_CONTEXT_TOKENS, CLAUDE_CODE_AUTO_COMPACT_WINDOW) and places the
trigger (CLAUDE_AUTOCOMPACT_PCT_OVERRIDE); scripts/claude_compaction_stub.py proves it on the binary.

USER 2026-09-22: the limit is min(served window, 262144) for every model, and the trigger leaves a
full reply (CLAUDE_CODE_MAX_OUTPUT_TOKENS) plus one turn of growth under it. The window comes from
keys every arm snapshot ALREADY carries -- CONTEXT_LENGTH and the engine's --context-length /
--max-model-len -- so a pending job picks the fix up at start without being re-rendered.
"""

import importlib.util
import math
import pathlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tests.env_render import rendered

REPO = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"
ARM_ENVS = sorted(EXPERIMENTS.glob(".env.*"))

#: The window each model is served with (the engine's argument, or the provider's published one).
SERVED = {
    "qwen38": 262144,
    "kimi27sglang": 262144,
    "glm53": 262144,
    "unionalpha": 262144,
    "oss120b": 131072,
    "fable51": 1000000,
    "musespark": 1048576,
    "gpt6astra": 1050000,
}

#: What claude is given for each served window at the common 32768-token reply cap. The percentage
#: puts the CLI's trigger floor((window - 20000) * pct / 100) at window - 32768 - 40000.
EXPECTED = {
    262144: {
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "262144",
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "262144",
        "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "78.2080",
    },
    131072: {
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "131072",
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "131072",
        "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "52.4920",
    },
}


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="driver")
def driver_fixture() -> ModuleType:
    return load("agent_driver")


def env_values(path: pathlib.Path) -> dict[str, str]:
    """The flat KEY=VALUE environment a job sources for ``path``, quotes stripped."""
    values: dict[str, str] = {}
    for line in rendered(path).splitlines():
        key, _, value = line.partition("=")
        values[key] = value.strip().strip('"')
    return values


def model_of(path: pathlib.Path) -> str:
    """The model an arm env serves, read off its file name (.env.base-<m>, .env.llrbase-<m>-<lang>, ...)."""
    names = [name for name in SERVED if f"-{name}" in path.name]
    assert len(names) == 1, f"{path.name}: cannot tell its model from {names}"
    return names[0]


def cli_trigger(environment: dict[str, str], max_output: int) -> int:
    """The token count claude-code 2.1.197 compacts at, as its own arithmetic computes it:
    floor(E * (pct / 100)) with E = window - min(max output, 20000), capped at E - 13000."""
    effective = int(environment["CLAUDE_CODE_AUTO_COMPACT_WINDOW"]) - min(max_output, 20000)
    percent = float(environment["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"])
    return min(math.floor(effective * (percent / 100)), effective - 13000)


@pytest.mark.parametrize("path", ARM_ENVS, ids=lambda path: path.name)
def test_every_arm_env_gives_claude_its_models_window_capped_at_256k(driver: ModuleType, path: pathlib.Path) -> None:
    """Each committed arm, rendered the way a snapshot is: the window is the model's own, never the
    cap standing in for a window the arm forgot to name, and never above 262144."""
    values = env_values(path)
    values["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = "32768"  # the launcher's export (run_cluster.sh)
    served = SERVED[model_of(path)]
    assert driver.served_context(values) == served
    assert driver.claude_context_env(values) == EXPECTED[min(served, 262144)]


@pytest.mark.parametrize("window", sorted(EXPECTED))
@pytest.mark.parametrize("max_output", [32768, 16384])
def test_the_trigger_leaves_a_full_reply_and_one_turn_under_the_window(
    driver: ModuleType, window: int, max_output: int
) -> None:
    """A request sent just under the trigger still fits with its whole reply reserved, and so does the
    compaction request after one more turn of up to CLAUDE_TURN_HEADROOM tokens. The truncated
    percentage may place the trigger a token or two early, never late."""
    environment = driver.claude_context_env(
        {"CONTEXT_LENGTH": str(window), "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(max_output)}
    )
    wanted = window - max_output - driver.CLAUDE_TURN_HEADROOM
    assert wanted - 2 <= cli_trigger(environment, max_output) <= wanted


def test_the_smallest_window_any_source_names_wins(driver: ModuleType) -> None:
    """The engine enforces its own argument; a CONTEXT_LENGTH or a second engine's args naming a larger
    window must not raise the limit past it."""
    environment = {
        "CONTEXT_LENGTH": "262144",
        "SGLANG_EXTRA_ARGS": "--trust-remote-code --context-length 262144",
        "VLLM_EXTRA_ARGS": "--dtype auto --max-model-len=131072 --gpu-memory-utilization 0.70",
    }
    assert driver.served_context(environment) == 131072


def test_an_arm_naming_no_window_gets_the_policy_cap(driver: ModuleType) -> None:
    """No committed arm and no pending snapshot does this; the cap is still a limit the policy allows."""
    assert driver.served_context({}) == driver.CLAUDE_CONTEXT_CAP == 262144


def test_claude_env_carries_the_context_variables_over_whatever_the_submitter_exported(
    driver: ModuleType, tmp_path: pathlib.Path
) -> None:
    """claude_env is what the agent process gets: the three variables are in it, and a value the
    submitting shell leaked (the driver copies os.environ) does not survive."""
    context = SimpleNamespace(replica_root="http://n0:8000", workdir=tmp_path)
    base = {"VLLM_EXTRA_ARGS": "--max-model-len 131072", "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "900000"}
    environment = driver.claude_env(context, base)
    assert {name: environment[name] for name in EXPECTED[131072]} == EXPECTED[131072]


def test_the_argv_never_carries_autocompact_even_where_a_cli_would_accept_it(
    driver: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """CLAUDE_AUTOCOMPACT used to become --autocompact, an option 2.1.197 does not have: the probe
    dropped it on every recorded arm. The environment above is the one mechanism now."""
    monkeypatch.setenv("CLAUDE_AUTOCOMPACT", "200144")
    monkeypatch.setattr(driver, "claude_supports_flag", lambda binary, flag: True)
    argv = driver.claude_command(SimpleNamespace(prompt="optimize it", mcp_config=tmp_path / "mcp.json"))
    assert "--autocompact" not in argv
