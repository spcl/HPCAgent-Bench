# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""inference/alps-endpoint.sh, sourced in a Daint job to use a beverin ACCESS=alps endpoint.

Sources the real script, which runs the real curl against a local server that answers like sglang
with --api-key: the caller's shell gets VLLM_BASE_URL/VLLM_API_KEY/VLLM_MODEL only after /v1/models
lists the model and one chat answers, the key reaches neither curl's argv nor any output, and every
failed check exports nothing.
"""

import collections.abc
import http.server
import json
import pathlib
import shutil
import socket
import subprocess
import sys
import threading
import typing

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "containers" / "cluster" / "ce-images" / "inference" / "alps-endpoint.sh"
KEY = "fedcba9876543210" * 4
MODEL = "hpcagent-bench-vllm"
#: What the test shell prints after sourcing: whether the key arrived, never the key itself.
REPORT = (
    'source "$1" "$2"; rc=$?\n'
    '[ "${VLLM_API_KEY:-}" = "$EXPECTED_KEY" ] && echo key=exported || echo key=absent\n'
    'echo "url=${VLLM_BASE_URL:-} model=${VLLM_MODEL:-} rc=$rc"\n'
)


class FakeSglang(http.server.ThreadingHTTPServer):
    """sglang with --api-key: 401 without the bearer key; /v1/models lists one model; chat answers."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), FakeSglangHandler)
        self.listed_model = MODEL
        self.chat_choices: list[dict[str, object]] = [{"message": {"role": "assistant", "content": "OK"}}]
        self.requests: list[tuple[str, str, str]] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"


class FakeSglangHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def fake(self) -> FakeSglang:
        return typing.cast(FakeSglang, self.server)

    def reply(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authorized(self) -> bool:
        authorization = self.headers.get("Authorization", "")
        self.fake().requests.append((self.command, self.path, authorization))
        if authorization == f"Bearer {KEY}":
            return True
        self.reply(401, {"error": {"message": "Unauthorized"}})
        return False

    def do_GET(self) -> None:
        if self.authorized():
            self.reply(200, {"object": "list", "data": [{"id": self.fake().listed_model, "object": "model"}]})

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.authorized():
            self.reply(200, {"object": "chat.completion", "choices": self.fake().chat_choices})


@pytest.fixture
def server() -> collections.abc.Iterator[FakeSglang]:
    fake = FakeSglang()
    thread = threading.Thread(target=fake.serve_forever, daemon=True)
    thread.start()
    yield fake
    fake.shutdown()
    fake.server_close()
    thread.join()


def source(
    tmp_path: pathlib.Path, url: str | None, key: str = KEY, key_mode: int = 0o600
) -> subprocess.CompletedProcess[str]:
    """Source the script in a clean bash with an endpoint.json for `url` (None: no file) and a key file."""
    stub = tmp_path / "bin"
    stub.mkdir()
    curl = shutil.which("curl")
    assert curl is not None
    (stub / "curl").write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{tmp_path}/curl-argv"\nexec {curl} "$@"\n')
    (stub / "curl").chmod(0o755)
    key_file = tmp_path / "endpoint.key"
    key_file.write_text(key + "\n", encoding="utf-8")
    key_file.chmod(key_mode)
    endpoint = tmp_path / "endpoint.json"
    if url is not None:
        published = {"url": url, "served_model": MODEL, "key_file": str(key_file), "node": "nid002536", "job_id": "1"}
        endpoint.write_text(json.dumps(published) + "\n", encoding="utf-8")
    env = {
        "PATH": f"{stub}:{pathlib.Path(sys.executable).parent}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "EXPECTED_KEY": KEY,
    }
    return subprocess.run(
        ["bash", "-c", REPORT, "bash", str(SCRIPT), str(endpoint)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=tmp_path,
    )


def closed_port_url() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


def test_a_live_endpoint_exports_its_url_model_and_key(tmp_path: pathlib.Path, server: FakeSglang) -> None:
    done = source(tmp_path, server.url)
    assert done.stdout.splitlines()[-2:] == ["key=exported", f"url={server.url} model={MODEL} rc=0"], done.stderr
    assert server.requests == [
        ("GET", "/v1/models", f"Bearer {KEY}"),
        ("POST", "/v1/chat/completions", f"Bearer {KEY}"),
    ]


def test_the_key_reaches_neither_curls_argv_nor_any_output(tmp_path: pathlib.Path, server: FakeSglang) -> None:
    """ps shows every process's argv to every user on the node."""
    done = source(tmp_path, server.url)
    argv = (tmp_path / "curl-argv").read_text(encoding="utf-8").splitlines()
    assert len(argv) == 2, argv
    assert [line for line in argv if KEY in line] == []
    assert KEY not in done.stdout + done.stderr


def test_a_wrong_key_exports_nothing_and_names_the_401(tmp_path: pathlib.Path, server: FakeSglang) -> None:
    done = source(tmp_path, server.url, key="0" * 64)
    assert done.stdout.splitlines()[-2:] == ["key=absent", "url= model= rc=1"]
    assert "answered 401" in done.stderr


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o400])
def test_a_key_file_that_is_not_mode_600_is_refused_before_any_request(
    tmp_path: pathlib.Path, server: FakeSglang, mode: int
) -> None:
    done = source(tmp_path, server.url, key_mode=mode)
    assert done.stdout.splitlines()[-2:] == ["key=absent", "url= model= rc=2"]
    assert "must exist, be owned by you and be mode 600" in done.stderr
    assert server.requests == []


def test_an_endpoint_that_does_not_list_the_served_model_exports_nothing(
    tmp_path: pathlib.Path, server: FakeSglang
) -> None:
    server.listed_model = "some-other-model"
    done = source(tmp_path, server.url)
    assert done.stdout.splitlines()[-2:] == ["key=absent", "url= model= rc=1"]
    assert f"want 200 listing {MODEL}" in done.stderr
    assert [path for _, path, _ in server.requests] == ["/v1/models"]


def test_an_endpoint_whose_chat_returns_no_choice_exports_nothing(tmp_path: pathlib.Path, server: FakeSglang) -> None:
    server.chat_choices = []
    done = source(tmp_path, server.url)
    assert done.stdout.splitlines()[-2:] == ["key=absent", "url= model= rc=1"]
    assert "chat/completions answered 200, want 200 with a choice" in done.stderr


def test_an_unreachable_endpoint_exports_nothing(tmp_path: pathlib.Path) -> None:
    done = source(tmp_path, closed_port_url())
    assert done.stdout.splitlines()[-2:] == ["key=absent", "url= model= rc=1"]
    assert "answered 000" in done.stderr


def test_a_missing_endpoint_file_says_the_serving_job_may_have_ended(tmp_path: pathlib.Path) -> None:
    done = source(tmp_path, None)
    assert done.stdout.splitlines()[-2:] == ["key=absent", "url= model= rc=2"]
    assert "the serving job has ended" in done.stderr
