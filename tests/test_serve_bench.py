# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``hpcagent_bench/cluster/serve_bench.py``: the phase figures from request records, the report table, the load against
a stub OpenAI endpoint (a report with its phases and the KV verdict lands in the run directory), and the submit cap."""

import argparse
import http.server
import json
import pathlib
import tempfile
import threading

import pytest

from hpcagent_bench.cluster import env_spec, serve_bench


def record(ttft: float, latency: float, output: int, prompt: int, cached: int, **extra: int) -> dict[str, float]:
    return {"ttft": ttft, "latency": latency, "output": output, "prompt": prompt, "cached": cached} | extra  # type: ignore[return-value]


def test_summarize_reports_throughput_decode_ttft_and_cache_hit() -> None:
    records = [record(1.0, 11.0, 100, 1000, 900), record(3.0, 13.0, 100, 1000, 700), {"error": 1}]  # type: ignore[list-item]
    s = serve_bench.summarize(records, seconds=10.0, concurrency=2)
    assert (s["requests"], s["errors"], s["output_tok_s"], s["per_agent_output_tok_s"]) == (2, 1, 20.0, 10.0)
    assert (s["decode_tok_s_median"], s["ttft_p50"], s["cache_hit"], s["mean_prompt_tokens"]) == (10.0, 3.0, 0.8, 1000)


class Stub(http.server.BaseHTTPRequestHandler):
    """An OpenAI chat endpoint that streams two chunks and a usage block."""

    def do_POST(self) -> None:  # noqa: N802 -- http.server's name
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        usage = {"prompt_tokens": 100, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 90}}
        for chunk in ({"choices": [{"delta": {"content": "hi"}}]}, {"choices": [], "usage": usage}):
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *_: object) -> None:
        return


def test_run_writes_a_report_with_the_phases_the_pool_and_the_guards_verdict(tmp_path: pathlib.Path) -> None:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    (tmp_path / "serve.env").write_text(env_spec.as_text(env_spec.render("experiment:kimi27sglang")))
    (tmp_path / "server-0.log").write_text("KV Cache is allocated. dtype: x, #tokens: 211200, KV size: 1 GB\n")
    args = argparse.Namespace(
        base=f"http://127.0.0.1:{server.server_port}", model="m", name="cfg", run_dir=tmp_path, api_up_s=12,
        concurrency=[2], seconds=1.0, prefix_tokens=50, turn_tokens=20, output_tokens=4,
    )  # fmt: skip
    try:
        assert serve_bench.run(args) == 0
    finally:
        server.shutdown()
    report = json.loads((tmp_path / serve_bench.report_name("cfg")).read_text())
    assert (report["config"], report["api_up_s"], report["kv_pool_tokens"]) == ("cfg", 12, 211_200)
    assert "below the floor" in report["kv_guard"]
    (phase,) = report["phases"]
    assert phase["summary"]["concurrency"] == 2 and phase["summary"]["requests"] >= 1
    assert phase["summary"]["cache_hit"] == 0.9
    assert "| cfg | 211200 | 12 | 2 |" in serve_bench.table([tmp_path / serve_bench.report_name("cfg")])


def test_submit_takes_one_to_three_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit, match="1 to 3"):
        serve_bench.submit("kimi27sglang", ["a", "b", "c", "d"], [20], 900)
    seen: list[dict[str, str]] = []

    def fake(command: list[str], env: dict[str, str], check: bool) -> object:
        seen.append({key: env[key] for key in ("MODEL", "SERVE_BENCH", "BENCH_CONCURRENCY", "BENCH_SECONDS")})
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(serve_bench.subprocess, "run", fake)
    assert serve_bench.submit("kimi27sglang", ["kimi-base", "kimi-ep4-triton"], [20, 40], 300) == 0
    assert [one["SERVE_BENCH"] for one in seen] == ["kimi-base", "kimi-ep4-triton"]
    assert seen[0] == {
        "MODEL": "kimi27sglang",
        "SERVE_BENCH": "kimi-base",
        "BENCH_CONCURRENCY": "20 40",
        "BENCH_SECONDS": "300",
    }


if __name__ == "__main__":
    test_summarize_reports_throughput_decode_ttft_and_cache_hit()
    test_run_writes_a_report_with_the_phases_the_pool_and_the_guards_verdict(pathlib.Path(tempfile.mkdtemp()))
    with pytest.MonkeyPatch.context() as patch:
        test_submit_takes_one_to_three_configs(patch)
