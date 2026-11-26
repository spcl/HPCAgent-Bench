# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The KV-pool guard of ``hpcagent_bench/cluster/serve_guard.py``: the pool read from the server logs of SGLang and vLLM,
the floor of a model, the verdict written to the run directory, and the watch that cancels a job whose pool is below
it (``KV_GUARD=warn`` only reports). Kimi's configuration must pass at the measured 1.85M tokens and fail at 0.98M."""

import json
import os
import pathlib
from collections.abc import Callable

import pytest

from hpcagent_bench.cluster import env_spec, serve_guard
from tests.bash_stub import bash_stub

SGLANG = (
    "[2026-11-17 16:06:28 PP0 TP0] KV Cache is allocated. dtype: torch.float8_e4m3fnuz, #tokens: {pool}, KV size: 14.88 GB\n"
    "[2026-11-17 16:06:29 PP3 TP0] KV Cache is allocated. dtype: torch.float8_e4m3fnuz, #tokens: {other}, KV size: 3 GB\n"
)
KIMI = env_spec.render("experiment:kimi27sglang")


QWEN = {"KV_TOKENS_PER_AGENT": "100000", "AGENT_NODES": "1", "AGENTS_PER_NODE": "40"}


@pytest.mark.parametrize(
    ("measure", "expected"),
    [
        # the pool is the smallest any pipeline states
        (lambda: serve_guard.pool_tokens(SGLANG.format(pool=1_848_960, other=984_000).splitlines()), 984_000),
        (lambda: serve_guard.pool_tokens(["max_total_num_tokens=211200, chunked_prefill_size=4096"]), 211_200),
        (lambda: serve_guard.pool_tokens(["INFO GPU KV cache size: 1,234,567 tokens"]), 1_234_567),
        (lambda: serve_guard.pool_tokens(["nothing here"]), None),
        # the floor is the pipeline's agents times the per-agent figure, and a model without one has none
        (lambda: serve_guard.floor_tokens(KIMI), 20 * 80_000),
        (lambda: serve_guard.floor_tokens({"AGENTS_PER_PIPELINE": "20"}), None),
        # a replicas server holds every agent of the job
        (lambda: serve_guard.floor_tokens(QWEN), 40 * 100_000),
    ],
    ids=[
        "sglang_smallest",
        "max_total_num_tokens",
        "vllm_kv_cache_size",
        "no_pool",
        "kimi_floor",
        "no_floor",
        "replicas",
    ],
)
def test_the_pool_is_the_smallest_any_pipeline_states(measure: Callable[[], int | None], expected: int | None) -> None:
    assert measure() == expected


def test_kimi_at_fraction_052_passes_and_at_048_fails() -> None:
    floor = serve_guard.floor_tokens(KIMI)
    assert serve_guard.verdict(1_848_960, floor).ok
    for starved in (984_000, 211_200):
        failed = serve_guard.verdict(starved, floor)
        assert not failed.ok and "below the floor" in failed.message
    assert not serve_guard.verdict(None, floor).ok
    assert serve_guard.verdict(1, None).ok


def test_check_writes_the_verdict_into_the_run_directory(tmp_path: pathlib.Path) -> None:
    log = tmp_path / "server-0.log"
    log.write_text(SGLANG.format(pool=1_848_960, other=1_848_960))
    assert serve_guard.check(KIMI, [log], tmp_path).ok
    assert json.loads((tmp_path / "kv-guard.json").read_text())["pool"] == 1_848_960
    log.write_text(SGLANG.format(pool=211_200, other=211_200))
    assert not serve_guard.check(KIMI, [log], tmp_path).ok


def test_the_cli_check_fails_below_the_floor_and_warns_with_kv_guard_warn(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "server-0.log").write_text(SGLANG.format(pool=211_200, other=211_200))
    env = tmp_path / "serve.env"
    env.write_text(env_spec.as_text(KIMI))
    args = ["check", "--env", str(env), "--run-dir", str(tmp_path)]
    monkeypatch.delenv("KV_GUARD", raising=False)
    assert serve_guard.main(args) == 1
    monkeypatch.setenv("KV_GUARD", "warn")
    assert serve_guard.main(args) == 0


def test_watch_cancels_the_job_whose_pool_is_below_the_floor_and_leaves_the_one_that_holds(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cancelled = tmp_path / "cancelled"
    bash_stub(tmp_path / "bin", "scancel", f'echo "$@" >>{cancelled}')
    bash_stub(tmp_path / "bin", "scontrol", "echo JobId=7 StdOut=/nonexistent/out")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")
    run = tmp_path / "run"
    run.mkdir()
    (run / "server-0.log").write_text(SGLANG.format(pool=1_848_960, other=1_848_960))
    assert serve_guard.watch(KIMI, run, "7", timeout=5) == 0
    assert not cancelled.exists()
    (run / "server-0.log").write_text(SGLANG.format(pool=211_200, other=211_200))
    assert serve_guard.watch(KIMI, run, "7", timeout=5) == 1
    assert cancelled.read_text().strip() == "7"
    cancelled.unlink()
    assert serve_guard.watch(KIMI | {"KV_GUARD": "warn"}, run, "7", timeout=5) == 0
    assert not cancelled.exists()
