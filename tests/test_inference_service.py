# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""An arm whose inference comes from a hosted service rather than a server the job starts.

``experiments/inference_service.py`` is the one place that reads an arm's ``INFERENCE_SERVICE_*``
block. It resolves that block into the three values every consumer downstream already reads (the
base URL, the served model name, the key), decides which variable the claude CLI's key belongs in,
refuses a harness that cannot speak the service's wire shape, and writes the run's inference
provenance. The fake-server cases below drive the repo's own HTTP poster and usage parsers against
that resolved endpoint, so a service arm's request shape and its token accounting are both proven
against a server that records what it received.

The secret itself never crosses this boundary: the arm names the VARIABLE its key lives in, and the
launcher copies it by indirection. The cases that matter most here are the ones asserting a key
value reaches the worker and reaches nothing else.
"""

import http.server
import importlib.util
import json
import pathlib
import sys
import threading
import types
from collections.abc import Iterator
from typing import ClassVar

import pytest

from hpcagent_bench.harness.agent import anthropic_usage, http_chat_json

EXPERIMENTS = pathlib.Path(__file__).resolve().parents[1] / "experiments"
AGENT_HARNESS = pathlib.Path(__file__).resolve().parents[1] / "containers" / "agent" / "harness"

#: A key value no other string in these cases spells, so a leak search cannot match by accident.
SECRET = "sk-test-1nf3r3nc3-s3rv1c3-l34k-c4n4ry"


def load(name: str, path: pathlib.Path) -> types.ModuleType:
    """Import a module by path, the way the driver loads the launcher's helpers."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="service")
def service_fixture() -> types.ModuleType:
    if str(EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS))
    return load("inference_service", EXPERIMENTS / "inference_service.py")


@pytest.fixture(name="runner_common")
def runner_common_fixture(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    monkeypatch.syspath_prepend(str(AGENT_HARNESS))
    return load("runner_common", AGENT_HARNESS / "runner_common.py")


def openai_arm(**overrides: str) -> dict[str, str]:
    """An OpenAI-shaped service arm's environment, as its ``.env`` sets it."""
    arm = {
        "INFERENCE_SOURCE": "service",
        "INFERENCE_NODES": "0",
        "INFERENCE_SERVICE_PROVIDER": "openai",
        "INFERENCE_SERVICE_BASE_URL": "https://api.openai.com/v1",
        "INFERENCE_SERVICE_MODEL": "gpt-6-astra",
        "INFERENCE_SERVICE_TIER": "standard",
        "INFERENCE_SERVICE_API": "openai",
        "INFERENCE_SERVICE_AUTH": "bearer",
        "INFERENCE_SERVICE_KEY_ENV": "OPENAI_API_KEY",
        "HARNESS": "openhands",
        "OPENAI_API_KEY": SECRET,
    }
    arm.update(overrides)
    return arm


def anthropic_arm(**overrides: str) -> dict[str, str]:
    """An Anthropic-shaped service arm's environment, as its ``.env`` sets it."""
    arm = {
        "INFERENCE_SOURCE": "service",
        "INFERENCE_NODES": "0",
        "INFERENCE_SERVICE_PROVIDER": "anthropic",
        "INFERENCE_SERVICE_BASE_URL": "https://api.anthropic.com/v1",
        "INFERENCE_SERVICE_MODEL": "claude-fable-5-1",
        "INFERENCE_SERVICE_TIER": "standard",
        "INFERENCE_SERVICE_API": "anthropic",
        "INFERENCE_SERVICE_AUTH": "x-api-key",
        "INFERENCE_SERVICE_KEY_ENV": "ANTHROPIC_API_KEY",
        "HARNESS": "claude",
        "ANTHROPIC_API_KEY": SECRET,
    }
    arm.update(overrides)
    return arm


def muse_arm(**overrides: str) -> dict[str, str]:
    """The Muse Spark contributor-tier arm, Meta's Anthropic-shaped surface with bearer auth."""
    arm = {
        "INFERENCE_SOURCE": "service",
        "INFERENCE_NODES": "0",
        "INFERENCE_SERVICE_PROVIDER": "meta",
        "INFERENCE_SERVICE_BASE_URL": "https://api.meta.ai/v1",
        "INFERENCE_SERVICE_MODEL": "muse-spark-1.3-contributor",
        "INFERENCE_SERVICE_TIER": "contributor",
        "INFERENCE_SERVICE_API": "anthropic",
        "INFERENCE_SERVICE_AUTH": "bearer",
        "INFERENCE_SERVICE_KEY_ENV": "META_MODEL_API_KEY",
        "HARNESS": "claude",
        "META_MODEL_API_KEY": SECRET,
    }
    arm.update(overrides)
    return arm


# which source an arm selects


def test_an_arm_that_names_no_source_still_starts_its_own_server(service: types.ModuleType) -> None:
    """Every arm written before this mode existed declares no source and must keep its server."""
    assert service.source({}) == service.SOURCE_NODE
    assert service.source({"INFERENCE_SOURCE": "node"}) == service.SOURCE_NODE


def test_an_unknown_source_is_refused_rather_than_read_as_a_server_arm(service: types.ModuleType) -> None:
    """A typo in the one key that decides whether a job allocates GPUs must not resolve silently."""
    with pytest.raises(SystemExit, match="INFERENCE_SOURCE"):
        service.source({"INFERENCE_SOURCE": "hosted"})


def test_a_service_arm_points_every_consumer_at_the_service(service: types.ModuleType) -> None:
    """The launcher composes ONE endpoint triple; the service block has to land in that triple or
    the agent driver, the runners and the claude CLI would each need their own knob."""
    exported = service.launcher_env(service.from_environ(openai_arm()))
    assert exported["VLLM_BASE_URL"] == "https://api.openai.com/v1"
    assert exported["VLLM_REPLICA_URLS"] == "https://api.openai.com/v1"
    assert exported["VLLM_SERVED_MODEL"] == "gpt-6-astra"
    # No node serves this arm, so nothing may inherit a hostname that would resolve to one.
    assert exported["VLLM_MASTER_HOST"] == ""


def test_a_service_arm_that_still_claims_an_inference_node_is_refused(service: types.ModuleType) -> None:
    """An arm that asks for a GPU node it will never use sizes its allocation for a server that is
    never started, and the allocation check in beverin.sbatch would pass it."""
    with pytest.raises(SystemExit, match="INFERENCE_NODES"):
        service.from_environ(openai_arm(INFERENCE_NODES="1"))


@pytest.mark.parametrize(
    "missing", ["INFERENCE_SERVICE_BASE_URL", "INFERENCE_SERVICE_MODEL", "INFERENCE_SERVICE_KEY_ENV"]
)
def test_an_incomplete_service_block_names_the_key_it_is_missing(service: types.ModuleType, missing: str) -> None:
    arm = openai_arm()
    del arm[missing]
    with pytest.raises(SystemExit, match=missing):
        service.from_environ(arm)


def test_a_key_variable_that_is_not_set_fails_before_any_agent_starts(service: types.ModuleType) -> None:
    """A whole arm running against a 401 for its wall clock is the failure this check exists for."""
    arm = openai_arm()
    del arm["OPENAI_API_KEY"]
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        service.from_environ(arm)


# the wire shape decides what may run against it


def test_an_anthropic_shaped_service_refuses_an_openai_runner(service: types.ModuleType) -> None:
    """mini-SWE, OpenHands and optimas all speak /v1/chat/completions; against a Messages-only
    service every request 404s, and the arm burns its wall clock discovering that."""
    with pytest.raises(SystemExit, match="miniswe"):
        service.from_environ(anthropic_arm(HARNESS="miniswe"))


def test_an_openai_shaped_service_refuses_the_claude_harness(service: types.ModuleType) -> None:
    """The claude CLI appends /v1/messages, which an OpenAI-only service does not serve."""
    with pytest.raises(SystemExit, match="claude"):
        service.from_environ(openai_arm(HARNESS="claude"))


def test_meta_serves_both_shapes_so_the_arm_chooses(service: types.ModuleType) -> None:
    """Muse Spark answers on both surfaces; the arm's declared shape is what picks one."""
    assert service.from_environ(muse_arm()).api == service.API_ANTHROPIC
    assert service.from_environ(muse_arm(INFERENCE_SERVICE_API="openai", HARNESS="openhands")).api == service.API_OPENAI


def test_a_first_party_anthropic_service_authenticates_with_x_api_key_alone(service: types.ModuleType) -> None:
    """The claude CLI sends Authorization: Bearer whenever ANTHROPIC_AUTH_TOKEN is set, and
    api.anthropic.com answers a bearer-plus-key pairing with 401. Meta's Messages surface is the
    other way round: it is bearer-only."""
    assert service.claude_key_variable(service.from_environ(anthropic_arm())) == "ANTHROPIC_API_KEY"
    assert service.claude_key_variable(service.from_environ(muse_arm())) == "ANTHROPIC_AUTH_TOKEN"


# the key travels by NAME, never by value


def test_the_exported_block_carries_the_key_variable_and_never_the_key(service: types.ModuleType) -> None:
    """run_cluster.sh copies the key by indirection (``${!INFERENCE_KEY_ENV}``), so the secret
    never passes through this process, its stdout, or the eval that reads it."""
    exported = service.launcher_env(service.from_environ(openai_arm()))
    assert exported["INFERENCE_KEY_ENV"] == "OPENAI_API_KEY"
    assert all(SECRET not in value for value in exported.values())


def test_the_shell_block_is_shell_safe_and_holds_no_key(service: types.ModuleType) -> None:
    """The launcher evals this block, so a value must never be able to become shell code -- and the
    key is not in it at all, only the name of the variable holding it."""
    block = service.shell_block(service.launcher_env(service.from_environ(muse_arm())))
    assert SECRET not in block
    assert "VLLM_SERVED_MODEL=muse-spark-1.3-contributor" in block
    assert "VLLM_MASTER_HOST=''" in block
    injected = service.shell_block({"VLLM_SERVED_MODEL": "a; rm -rf $HOME"})
    assert injected == "VLLM_SERVED_MODEL='a; rm -rf $HOME'"


# provenance


def test_the_run_records_the_provider_model_and_tier(service: types.ModuleType, tmp_path: pathlib.Path) -> None:
    """Which service answered, at which tier, is the service arm's counterpart to the engine and
    image a server arm records -- without it a contributor-tier run is indistinguishable from a
    standard-tier one in the archive."""
    service.record(tmp_path, muse_arm())
    written = json.loads((tmp_path / service.RECORD_NAME).read_text(encoding="utf-8"))
    assert written["source"] == "service"
    assert written["provider"] == "meta"
    assert written["model"] == "muse-spark-1.3-contributor"
    assert written["tier"] == "contributor"
    assert written["api"] == "anthropic"
    assert written["base_url"] == "https://api.meta.ai/v1"
    assert written["key_env"] == "META_MODEL_API_KEY"


def test_the_recorded_provenance_never_holds_the_key(service: types.ModuleType, tmp_path: pathlib.Path) -> None:
    service.record(tmp_path, muse_arm())
    assert SECRET not in (tmp_path / service.RECORD_NAME).read_text(encoding="utf-8")


def test_a_server_arm_records_its_engine_instead(service: types.ModuleType, tmp_path: pathlib.Path) -> None:
    """One file answers "what produced these tokens" for both modes, or a reader has to know which
    mode a run used before knowing where to look."""
    service.record(
        tmp_path, {"INFERENCE_ENGINE": "sglang", "INFERENCE_CE_ENV": "sglang-latest", "VLLM_MODEL": "Qwen/Q"}
    )
    written = json.loads((tmp_path / service.RECORD_NAME).read_text(encoding="utf-8"))
    assert written["source"] == "node"
    assert written["engine"] == "sglang"
    assert written["ce_env"] == "sglang-latest"
    assert written["model"] == "Qwen/Q"


# fake services: the request a service arm actually sends, and what its usage folds to


class RecordingHandler(http.server.BaseHTTPRequestHandler):
    """Base for the two fake services: records the last request and answers a canned body."""

    seen: ClassVar[dict[str, object]] = {}
    body: ClassVar[dict[str, object]] = {}
    required: ClassVar[tuple[str, ...]] = ()
    path_wanted: ClassVar[str] = "/"

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        headers = {name.lower(): value for name, value in self.headers.items()}
        type(self).seen = {"path": self.path, "headers": headers, "payload": payload}
        missing = [header for header in type(self).required if not self.headers.get(header)]
        if missing or self.path != type(self).path_wanted:
            self.send_response(400)
            self.end_headers()
            return
        encoded = json.dumps(type(self).body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - the base class's name
        return


class FakeOpenAIService(RecordingHandler):
    """``POST /v1/chat/completions``, refusing anything without a bearer key."""

    required = ("Authorization",)
    path_wanted = "/v1/chat/completions"
    body = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 400,
            "total_tokens": 1400,
            "prompt_tokens_details": {"cached_tokens": 900},
            "completion_tokens_details": {"reasoning_tokens": 250},
        },
    }


class FakeAnthropicService(RecordingHandler):
    """``POST /v1/messages``, refusing anything without the key header and the version header."""

    required = ("x-api-key", "anthropic-version")
    path_wanted = "/v1/messages"
    body = {
        "content": [{"type": "text", "text": "ok"}],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 400,
            "cache_read_input_tokens": 900,
            "cache_creation_input_tokens": 50,
        },
    }


@pytest.fixture(name="fake_openai")
def fake_openai_fixture() -> Iterator[tuple[str, type[FakeOpenAIService]]]:
    yield from serve(FakeOpenAIService)


@pytest.fixture(name="fake_anthropic")
def fake_anthropic_fixture() -> Iterator[tuple[str, type[FakeAnthropicService]]]:
    yield from serve(FakeAnthropicService)


def serve(handler: type[RecordingHandler]) -> Iterator[tuple[str, type[RecordingHandler]]]:
    handler.seen = {}
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class FakeUsage:
    """What the Anthropic SDK hands :func:`anthropic_usage`: counters as instance attributes."""

    def __init__(self, counts: dict[str, int]) -> None:
        self.input_tokens = counts["input_tokens"]
        self.output_tokens = counts["output_tokens"]
        self.cache_read_input_tokens = counts["cache_read_input_tokens"]
        self.cache_creation_input_tokens = counts["cache_creation_input_tokens"]


def test_an_openai_service_arm_sends_its_key_and_its_model(
    service: types.ModuleType, runner_common: types.ModuleType, fake_openai: tuple[str, type[FakeOpenAIService]]
) -> None:
    """The endpoint, the model name and the key all come from the resolved service block, and the
    fake refuses a request that carries no bearer key at all."""
    root, handler = fake_openai
    arm = openai_arm(INFERENCE_SERVICE_BASE_URL=f"{root}/v1")
    exported = service.launcher_env(service.from_environ(arm))
    body = http_chat_json(
        f"{exported['VLLM_BASE_URL']}/chat/completions",
        {"model": exported["VLLM_SERVED_MODEL"], "messages": [{"role": "user", "content": "hi"}]},
        {"Authorization": f"Bearer {arm[exported['INFERENCE_KEY_ENV']]}"},
        10.0,
        "fake service unreachable",
    )
    assert handler.seen["headers"]["authorization"] == f"Bearer {SECRET}"
    assert handler.seen["payload"]["model"] == "gpt-6-astra"
    # The four counts the token watcher sums are disjoint and account for the whole call.
    line = runner_common.openai_usage(body["usage"])
    assert line == {"input": 100, "cached_input": 900, "output": 150, "reasoning": 250}
    assert sum(line.values()) == body["usage"]["total_tokens"]


def test_an_anthropic_service_arm_sends_the_key_header_the_launcher_chose(
    service: types.ModuleType, fake_anthropic: tuple[str, type[FakeAnthropicService]]
) -> None:
    """The fake rejects a request missing either the key header or the version header, so reaching
    it at all proves the arm's auth choice matches the service's."""
    root, handler = fake_anthropic
    arm = anthropic_arm(INFERENCE_SERVICE_BASE_URL=f"{root}/v1")
    resolved = service.from_environ(arm)
    exported = service.launcher_env(resolved)
    body = http_chat_json(
        f"{service.messages_url(exported['VLLM_BASE_URL'])}",
        {"model": exported["VLLM_SERVED_MODEL"], "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
        {service.auth_header(resolved): SECRET, "anthropic-version": service.ANTHROPIC_VERSION},
        10.0,
        "fake service unreachable",
    )
    assert service.claude_key_variable(resolved) == "ANTHROPIC_API_KEY"
    assert handler.seen["headers"]["x-api-key"] == SECRET
    assert handler.seen["payload"]["model"] == "claude-fable-5-1"
    # The Messages API reports the prompt as three disjoint counts; the fold must keep all three.
    folded = anthropic_usage(FakeUsage(body["usage"]))
    assert folded.input_tokens == 1050
    assert folded.cached_tokens == 900
    assert folded.cache_creation_tokens == 50
    assert folded.output_tokens == 400


def test_the_muse_spark_arm_reaches_metas_messages_surface_with_a_bearer_key(
    service: types.ModuleType, fake_anthropic: tuple[str, type[FakeAnthropicService]]
) -> None:
    """Meta's Messages endpoint takes the same wire format behind a bearer key rather than
    x-api-key, which is why the auth spelling is per service and not per shape."""
    root, handler = fake_anthropic
    resolved = service.from_environ(muse_arm(INFERENCE_SERVICE_BASE_URL=f"{root}/v1"))
    assert service.auth_header(resolved) == "Authorization"
    exported = service.launcher_env(resolved)
    http_chat_json(
        service.messages_url(exported["VLLM_BASE_URL"]),
        {"model": exported["VLLM_SERVED_MODEL"], "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
        {"Authorization": f"Bearer {SECRET}", "x-api-key": SECRET, "anthropic-version": service.ANTHROPIC_VERSION},
        10.0,
        "fake service unreachable",
    )
    assert handler.seen["headers"]["authorization"] == f"Bearer {SECRET}"
    assert handler.seen["payload"]["model"] == "muse-spark-1.3-contributor"


def test_an_openai_shaped_usage_block_read_as_anthropic_would_count_nothing(
    runner_common: types.ModuleType,
) -> None:
    """Why the shape gate above exists rather than one parser guessing: the Anthropic usage block
    has no field the OpenAI parser reads, so a mismatched arm would report zero tokens for every
    call instead of failing. The gate refuses the pairing; this records what it prevents."""
    anthropic_block = {"input_tokens": 100, "output_tokens": 400, "cache_read_input_tokens": 900}
    assert runner_common.openai_usage(anthropic_block) == {
        "input": 0,
        "cached_input": 0,
        "output": 0,
        "reasoning": 0,
    }


def test_a_service_arm_writes_no_key_into_the_usage_file(
    service: types.ModuleType, runner_common: types.ModuleType, tmp_path: pathlib.Path
) -> None:
    """The three files a run leaves behind -- provenance, usage, the staged arm env -- are the ones
    an archive keeps, so none of them may carry the key."""
    usage = runner_common.UsageLog(path=tmp_path / "usage.jsonl")
    usage.append(runner_common.openai_usage(FakeOpenAIService.body["usage"]))
    service.record(tmp_path, openai_arm())
    for name in ("usage.jsonl", service.RECORD_NAME):
        assert SECRET not in (tmp_path / name).read_text(encoding="utf-8")


def test_the_example_arms_match_the_model_table(service: types.ModuleType) -> None:
    """models.py is the one source of a model's block; an example env edited without it is exactly
    the drift the table exists to prevent."""
    if str(EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS))
    models = load("models", EXPERIMENTS / "models.py")
    for name in service.EXAMPLE_ARMS:
        text = (EXPERIMENTS / f".env.base-{name}").read_text(encoding="utf-8")
        for key, value in models.MODELS[name].items():
            assert f"{key}={value}" in text, f"{key} in models.py no longer matches .env.base-{name}"


@pytest.mark.parametrize("name", ["musespark", "fable51", "gpt6astra"])
def test_every_example_arm_resolves_when_its_key_is_set(service: types.ModuleType, name: str) -> None:
    """Each shipped example must be a WORKING arm, not a template: reading its env plus the one
    variable it names has to produce a resolved service."""
    text = (EXPERIMENTS / f".env.base-{name}").read_text(encoding="utf-8")
    arm = dict(line.split("=", 1) for line in text.splitlines() if line and not line.startswith("#") and "=" in line)
    arm = {key: value.strip('"') for key, value in arm.items()}
    arm[arm["INFERENCE_SERVICE_KEY_ENV"]] = SECRET
    resolved = service.from_environ(arm)
    assert resolved.tier
    assert SECRET not in service.shell_block(service.launcher_env(resolved))


def test_a_service_arm_never_leaks_its_key_into_the_staged_arm_env() -> None:
    """The arm env is copied into the run tree and read by every role; the key is named there, not
    written there, so rotating it never means editing a committed file."""
    text = (EXPERIMENTS / ".env.base-musespark").read_text(encoding="utf-8")
    assert "META_MODEL_API_KEY=" not in text.replace("INFERENCE_SERVICE_KEY_ENV=META_MODEL_API_KEY", "")


def test_an_unreachable_service_fails_loudly(service: types.ModuleType) -> None:
    """A closed port must raise rather than return an empty body an agent would treat as a reply."""
    arm = openai_arm(INFERENCE_SERVICE_BASE_URL="http://127.0.0.1:1/v1")
    exported = service.launcher_env(service.from_environ(arm))
    with pytest.raises(RuntimeError, match="unreachable"):
        http_chat_json(f"{exported['VLLM_BASE_URL']}/chat/completions", {}, {}, 2.0, "service unreachable")
