# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``experiments/retokenize.py``: counting a killed attempt's output with the model's own tokenizer.

The LAST tier of the output precedence (8.2), reached only by an attempt whose server counted
nothing. What it must get right is WHAT counts as generated -- the model's thinking, its answer text
and the arguments of every tool call it made, and nothing the CLI wrote on its behalf -- and what it
must refuse: a model it has no tokenizer for is None, never a zero, because a zero would join the
measurements and pull an arm's cost down.

No test here loads a real tokenizer: that would tie the suite to one cluster's offline HuggingFace
cache. The counting function is injected instead, which is exactly how ``token_cost`` receives it.
"""

import importlib.util
import json
import pathlib
import sys
from types import ModuleType

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def load_retokenize() -> ModuleType:
    spec = importlib.util.spec_from_file_location("retokenize", REPO / "experiments" / "retokenize.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="retokenize")
def retokenize_fixture() -> ModuleType:
    return load_retokenize()


def assistant(message_id: str, block: dict[str, object], model: str = "optarena-llm") -> dict[str, object]:
    return {"type": "assistant", "message": {"id": message_id, "model": model, "content": [block]}}


def test_what_the_model_generated_is_its_thinking_its_text_and_its_tool_arguments(retokenize) -> None:
    """One assistant event carries ONE content block, so the blocks are taken as they come. A tool
    call's arguments are generated tokens like any other -- the model wrote that JSON."""
    events = [
        assistant("m1", {"type": "thinking", "thinking": "weigh it up"}),
        assistant("m1", {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}),
        assistant("m2", {"type": "text", "text": "done"}),
    ]

    assert retokenize.generated_text(events) == ["weigh it up", json.dumps({"command": "ls"}), "done"]


def test_the_clis_synthetic_placeholder_is_not_the_models_work(retokenize) -> None:
    """``API Error: The operation timed out.`` is the CLI writing in the model's place. Counting it
    would bill the agent for the endpoint's failure."""
    events = [
        assistant("m1", {"type": "text", "text": "real"}),
        assistant("syn", {"type": "text", "text": "API Error: The operation timed out."}, model="<synthetic>"),
    ]

    assert retokenize.generated_text(events) == ["real"]


def test_a_model_without_a_local_tokenizer_counts_nothing_rather_than_zero(retokenize) -> None:
    """None and 0 are different answers: 0 says the attempt generated nothing, None says nobody
    could count it, and only the second leaves ``output_source`` at "none"."""
    assert retokenize.counter("no-such-org/no-such-model") is None
    assert retokenize.output_counter("no-such-org/no-such-model")([]) is None


def test_an_arms_short_tag_and_a_repo_id_both_name_a_model(retokenize) -> None:
    """Runs record either: ``VLLM_MODEL`` is the repo id, ``HPCAGENT_BENCH_RECORD_MODEL`` the tag."""
    assert retokenize.repo_of("qwen38") == "Qwen/Qwen3.8-27B-FP8"
    assert retokenize.repo_of("moonshotai/Kimi-K2.7-Code") == "moonshotai/Kimi-K2.7-Code"


def test_the_model_is_read_from_the_env_the_job_was_launched_with(retokenize, tmp_path: pathlib.Path) -> None:
    """``<campaign>/.agent-launch/<job>/.env`` sits beside the run directories, and ``VLLM_MODEL`` in
    it is the only unambiguous answer to whose tokenizer: the served model name is an alias
    (``optarena-vllm``) and the transcript records that alias, not the weights."""
    run_dir = tmp_path / "cpf-llr-focus40-20260914" / "636540"
    worker = run_dir / "agents" / "node-0" / "problem-0-worker-0"
    worker.mkdir(parents=True)
    launch = tmp_path / "cpf-llr-focus40-20260914" / ".agent-launch" / "636540"
    launch.mkdir(parents=True)
    (launch / ".env").write_text(
        'CAMPAIGN_ARM=cpf-llr-focus40-qwen38-c\nVLLM_MODEL="Qwen/Qwen3.8-27B-FP8"\nCLAUDE_MODEL=optarena-llm\n',
        encoding="utf-8",
    )

    assert retokenize.model_for_run(run_dir) == "Qwen/Qwen3.8-27B-FP8"


def test_the_short_tag_answers_when_the_launch_env_names_no_repo(retokenize, tmp_path: pathlib.Path) -> None:
    run_dir = tmp_path / "campaign" / "700001"
    (run_dir / "agents" / "node-0" / "problem-0-worker-0").mkdir(parents=True)
    launch = tmp_path / "campaign" / ".agent-launch" / "700001"
    launch.mkdir(parents=True)
    (launch / ".env").write_text("HPCAGENT_BENCH_RECORD_MODEL=oss120b\n", encoding="utf-8")

    assert retokenize.model_for_run(run_dir) == "oss120b"


def test_a_run_whose_launch_env_was_not_kept_names_no_model(retokenize, tmp_path: pathlib.Path) -> None:
    """Every campaign before .agent-launch existed, which is most of the recorded corpus. Those runs
    need --model on the command line; guessing one would retokenize with the wrong vocabulary."""
    run_dir = tmp_path / "campaign" / "636540"
    worker = run_dir / "agents" / "node-0" / "problem-0-worker-0"
    worker.mkdir(parents=True)

    assert retokenize.model_for_run(run_dir) == ""
    assert retokenize.counter_for(worker) is None
