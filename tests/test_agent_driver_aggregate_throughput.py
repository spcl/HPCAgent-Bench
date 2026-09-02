# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""agent_driver.py: the AGGREGATE generation-throughput probe.

The single-stream probe (tests/test_agent_driver_throughput.py) answers "how fast is one request",
which on a PP=4 pipeline is tens of tok/s because one request leaves three stages idle. The number
the campaign is actually served at is the aggregate over the ~40 agents in flight at once, and it
exists only while they are all running -- so it is taken from the server's own Prometheus counters,
sampled during the workload, and differenced.

Three things make that arithmetic lie if they are not handled, and each has a test here: a counter
that RESET because the server restarted (a negative delta, or a post-restart absolute passed off as
one interval's work), an interval so short that a handful of tokens divides out into the thousands
the real figure lives in, and an aggregate quoted over a window in which the server was never busy.
The fourth invariant is that none of this can fail the run it measures: an endpoint that is down, a
truncated exposition and an unwritable run dir all cost the measurement and nothing else.
"""

import importlib.util
import json
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.request
from types import ModuleType
from typing import Any

import pytest

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "containers/cluster/example-script"

# Restated rather than imported from the driver: these are vLLM's series names, so a test that read
# them off the module under test would keep passing after a typo renamed both at once.
GENERATION = "vllm:generation_tokens_total"
PROMPT = "vllm:prompt_tokens_total"
RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"


def load_example_module(name: str) -> ModuleType:
    """``sys.modules`` must carry the module BEFORE exec, matching tests/test_validate_run.py."""
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="driver")
def driver_fixture() -> ModuleType:
    return load_example_module("agent_driver")


class FakeMetrics:
    """Just enough of urlopen's context manager for a body to be read as bytes."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __enter__(self) -> "FakeMetrics":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self.text.encode("utf-8")


def exposition(generation: float, prompt: float, running: float, waiting: float, model: str = "optarena-vllm") -> str:
    """A cut-down copy of what vLLM actually serves at /metrics, labels and neighbours included."""
    return "\n".join(
        [
            "# HELP vllm:generation_tokens_total Number of generation tokens processed.",
            "# TYPE vllm:generation_tokens_total counter",
            f'vllm:generation_tokens_total{{model_name="{model}"}} {generation}',
            f'vllm:prompt_tokens_total{{model_name="{model}"}} {prompt}',
            f'vllm:num_requests_running{{model_name="{model}"}} {running}',
            f'vllm:num_requests_waiting{{model_name="{model}"}} {waiting}',
            f'vllm:time_to_first_token_seconds_bucket{{model_name="{model}",le="0.1"}} 7.0',
            "",
        ]
    )


def row(elapsed: float, generation: float, running: float = 40.0, waiting: float = 0.0) -> dict[str, float]:
    """One sample as the sampler records it: a timestamp and the counters read at it."""
    return {
        "elapsed_s": elapsed,
        GENERATION: generation,
        PROMPT: generation / 4,
        RUNNING: running,
        WAITING: waiting,
    }


def test_the_probe_scrapes_the_server_root_and_not_the_openai_path(driver: ModuleType) -> None:
    """run_cluster.sh composes every replica as http://<node>:<port>/v1 because that is the base an
    OpenAI client wants. The Prometheus app is mounted BESIDE /v1, so a probe that appended /metrics
    to the replica URL would 404 for the whole campaign and report nothing, which is indistinguishable
    from a server that was simply idle."""
    assert driver.metrics_url("http://nid002994:8000/v1") == "http://nid002994:8000/metrics"
    assert driver.metrics_url("http://nid002994:8000/v1/") == "http://nid002994:8000/metrics"
    # A base that never carried the OpenAI path must not lose its last three characters.
    assert driver.metrics_url("http://nid002994:8000") == "http://nid002994:8000/metrics"
    assert driver.server_root("http://nid002994:8000/v1") == "http://nid002994:8000"


def test_every_label_set_of_a_series_is_summed_and_a_lookalike_name_is_not(driver: ModuleType) -> None:
    """The series carry a model_name whose value is whatever --served-model-name was, so the label
    set cannot be matched on; a server exposing two of them would otherwise have half its tokens
    dropped. Matching on the name has to be exact all the same: prometheus_client emits _created
    beside every counter and histogram buckets beside every latency, and a prefix match would fold
    a bucket count into the token total."""
    text = "\n".join(
        [
            "# TYPE vllm:generation_tokens_total counter",
            'vllm:generation_tokens_total{model_name="kimi"} 1200.0',
            'vllm:generation_tokens_total{model_name="qwen"} 300.0',
            'vllm:generation_tokens_total_created{model_name="kimi"} 1.7e9',
            'vllm:prompt_tokens_total{model_name="kimi"} 50.0',
            'vllm:num_requests_running{model_name="kimi"} 12.0',
            'vllm:num_requests_waiting{model_name="kimi"} 3.0',
            'vllm:time_to_first_token_seconds_bucket{model_name="kimi",le="+Inf"} 999.0',
        ]
    )
    parsed = driver.parse_prometheus(text, driver.AGGREGATE_METRICS)
    assert parsed[GENERATION] == pytest.approx(1500.0)  # both label sets, not the _created epoch
    assert parsed[PROMPT] == pytest.approx(50.0)
    assert parsed[RUNNING] == pytest.approx(12.0)
    assert parsed[WAITING] == pytest.approx(3.0)


def test_a_truncated_or_incomplete_exposition_costs_the_sample_and_not_the_run(
    driver: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The body comes off a server that may be mid-restart, so half an exposition is an ordinary
    thing to read. A row missing one series cannot be differenced against a row that has it, so the
    whole sample is dropped -- and nothing raises, because this runs beside 40 live agents."""
    bodies = iter(
        [
            'vllm:generation_tokens_total{model_name="kimi"} 12',  # truncated: three series missing
            exposition(generation=float("nan"), prompt=10, running=1, waiting=0),  # NaN would poison the rate
            "<html>502 Bad Gateway</html>",
            exposition(generation=100, prompt=10, running=1, waiting=0),
        ]
    )
    monkeypatch.setattr(driver.urllib.request, "urlopen", lambda request, timeout=None: FakeMetrics(next(bodies)))

    assert driver.scrape_metrics("http://vllm:8000/metrics", {}) is None
    assert driver.scrape_metrics("http://vllm:8000/metrics", {}) is None
    assert driver.scrape_metrics("http://vllm:8000/metrics", {}) is None
    assert driver.scrape_metrics("http://vllm:8000/metrics", {})[GENERATION] == pytest.approx(100.0)


def test_a_partial_scrape_is_dropped_rather_than_read_as_a_counter_going_backwards(
    driver: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In replica mode the aggregate is the sum over every serving endpoint. A sum missing one
    endpoint is not a smaller reading of the same quantity -- it is a counter that fell, and the
    next interval would be scored as a server restart. Both answer or the sample does not exist."""
    served = {
        "http://a:8000/metrics": exposition(1000, 100, 20, 2),
        "http://b:8000/metrics": exposition(500, 50, 10, 1),
    }

    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> FakeMetrics:
        url = request.full_url
        if url not in served:
            raise urllib.error.URLError("connection refused")
        return FakeMetrics(served[url])

    monkeypatch.setattr(driver.urllib.request, "urlopen", fake_urlopen)

    both = driver.scrape_aggregate(["http://a:8000/metrics", "http://b:8000/metrics"], {})
    assert both[GENERATION] == pytest.approx(1500.0) and both[RUNNING] == pytest.approx(30.0)
    assert driver.scrape_aggregate(["http://a:8000/metrics", "http://gone:8000/metrics"], {}) is None


def test_a_counter_reset_is_neither_a_negative_rate_nor_a_post_restart_absolute(driver: ModuleType) -> None:
    """A Prometheus counter that decreased did not un-generate tokens: the process behind it
    restarted. Both wrong answers are excluded here -- the negative delta, and the tempting one where
    the post-restart absolute (500 tokens) is taken as the interval's work, which would report a
    plausible 50 tok/s over ten seconds the server mostly spent reloading weights."""
    samples = [row(0.0, 1000.0), row(10.0, 31000.0), row(20.0, 500.0), row(30.0, 30500.0)]

    intervals, resets = driver.aggregate_intervals(samples)

    assert resets == 1
    assert len(intervals) == 2, "the reset interval must be dropped, not scored from the absolute"
    assert [interval["generation_tok_s"] for interval in intervals] == pytest.approx([3000.0, 3000.0])


def test_a_scrape_landing_on_top_of_the_previous_one_is_not_a_rate(driver: ModuleType) -> None:
    """A near-zero interval divides a handful of tokens by nearly nothing and lands in the thousands
    of tok/s -- the exact magnitude the real aggregate lives in, so the noise would be unreadable as
    noise. 200 tokens in 10 ms is 20000 tok/s and must never reach the report."""
    samples = [row(0.0, 0.0), row(0.01, 200.0), row(10.01, 30200.0)]

    intervals, resets = driver.aggregate_intervals(samples)

    assert resets == 0
    assert len(intervals) == 1
    assert intervals[0]["generation_tok_s"] == pytest.approx(3000.0)


def test_the_overall_figure_covers_the_saturated_window_and_not_the_ramp_or_the_drain(driver: ModuleType) -> None:
    """An aggregate averaged over the whole run reports the server slower than it ever was while the
    campaign was running: the first agents are still starting and the last are alone on the machine.
    Here the plateau serves 3000 tok/s and the whole-run average is 1830."""
    samples = [
        row(0.0, 0.0, running=2.0),
        row(10.0, 1_000.0, running=2.0),
        row(20.0, 31_000.0, running=40.0),
        row(30.0, 61_000.0, running=40.0),
        row(40.0, 91_000.0, running=40.0),
        row(50.0, 91_500.0, running=1.0),
    ]
    intervals, _ = driver.aggregate_intervals(samples)
    floor = 40.0 * driver.AGGREGATE_SATURATED_FRACTION
    saturated = [interval for interval in intervals if interval["running"] >= floor]

    overall = driver.window_totals(saturated)
    assert overall["generation_tok_s"] == pytest.approx(3000.0)
    assert driver.window_totals(intervals)["generation_tok_s"] == pytest.approx(1830.0)


def test_the_overall_rate_drops_the_seconds_it_could_not_observe(driver: ModuleType) -> None:
    """The window is summed from its intervals, not taken as last-counter minus first-counter over
    the span. A restart removes its interval's tokens AND its interval's seconds, so the rate stays
    tokens per second-that-was-observed; dividing by the raw span would report the aggregate low by
    exactly the length of the outage."""
    samples = [row(0.0, 0.0), row(10.0, 30_000.0), row(20.0, 5_000.0), row(30.0, 35_000.0)]
    intervals, resets = driver.aggregate_intervals(samples)

    overall = driver.window_totals(intervals)
    assert resets == 1
    assert overall["seconds"] == pytest.approx(20.0)  # not the 30 s the samples span
    assert overall["generation_tok_s"] == pytest.approx(3000.0)  # not 35000/30 = 1166


def test_the_report_states_the_concurrency_every_figure_was_taken_at(
    driver: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An aggregate tok/s is a claim about a saturated server, so the report has to carry the
    evidence for that claim beside it. The raw series is written out too: the plateau, the ramp and
    the drain are only separable if the per-interval rows survive the summary."""
    monkeypatch.setenv("RUN_DIR", str(tmp_path))
    monkeypatch.delenv("AGENT_NODE_RANK", raising=False)
    monkeypatch.setenv("SLURM_PROCID", "0")
    samples = [
        row(0.0, 0.0, running=40.0, waiting=12.0),
        row(10.0, 30_000.0, running=40.0, waiting=9.0),
        row(20.0, 60_000.0, running=40.0, waiting=0.0),
    ]

    driver.report_aggregate_throughput(samples, missed=2)

    out = capsys.readouterr().out
    assert "peak running=40 peak waiting=12" in out
    assert "missed=2 counter_resets=0" in out
    assert "generation=3000.0 tok/s" in out
    written = json.loads((tmp_path / "aggregate-throughput-node0.json").read_text(encoding="utf-8"))
    assert len(written["samples"]) == 3 and len(written["intervals"]) == 2
    assert written["saturated"]["generation_tok_s"] == pytest.approx(3000.0)
    assert written["missed_scrapes"] == 2


def test_a_window_that_was_never_saturated_reports_the_two_requests_it_saw(
    driver: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The failure this guards is a number quoted out of context: 90 tok/s taken while two agents
    were in flight says nothing about a 40-agent campaign, and reads as a catastrophic regression
    next to a figure taken at full load. The peak is printed, so the reader can tell them apart."""
    monkeypatch.delenv("RUN_DIR", raising=False)
    samples = [row(0.0, 0.0, running=2.0), row(10.0, 900.0, running=2.0)]

    driver.report_aggregate_throughput(samples, missed=0)

    out = capsys.readouterr().out
    assert "peak running=2 peak waiting=0" in out
    assert "generation=90.0 tok/s" in out


def test_an_unwritable_run_dir_and_a_single_sample_do_not_raise(
    driver: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reporting happens after every agent has finished; nothing it does may cost the run its exit
    code. One sample is not an interval and must be said, not written out as a rate of zero."""
    monkeypatch.setenv("RUN_DIR", str(tmp_path / "does" / "not" / "exist"))
    driver.report_aggregate_throughput([row(0.0, 0.0), row(10.0, 30_000.0)], missed=0)
    assert "could not write" in capsys.readouterr().out

    driver.report_aggregate_throughput([row(0.0, 5.0)], missed=7)
    out = capsys.readouterr().out
    assert "1 samples, missed=7" in out and "no interval to measure" in out


def test_the_sampler_records_a_series_and_stops_when_the_agents_do(
    driver: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sampler runs beside the ThreadPoolExecutor for the length of the workload, so it has to
    end on the event rather than on a deadline, and it has to timestamp what it read rather than
    trust the nominal interval -- a scrape takes real time and the replicas are walked one by one.

    It is handed the READY REPLICAS main() already waited on, so it is also the place where the
    OpenAI path those carry has to come off; a sampler that scraped them verbatim would spend the
    campaign 404ing against /v1 and report an idle server."""
    counter = {"generation": 0.0}
    scraped: list[str] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> FakeMetrics:
        scraped.append(request.full_url)
        counter["generation"] += 1000.0
        return FakeMetrics(exposition(counter["generation"], counter["generation"] / 4, 40, 3))

    monkeypatch.setattr(driver.urllib.request, "urlopen", fake_urlopen)
    stop = threading.Event()
    state: dict[str, Any] = {"samples": [], "missed": 0}
    thread = threading.Thread(
        target=driver.sample_aggregate_throughput, args=(["http://vllm:8000/v1"], {}, 0.01, stop, state), daemon=True
    )

    thread.start()
    deadline = time.monotonic() + 5.0
    while len(state["samples"]) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    stop.set()
    thread.join(timeout=5.0)

    assert not thread.is_alive(), "the sampler must end on the stop event, not outlive the run"
    assert set(scraped) == {"http://vllm:8000/metrics"}
    samples = state["samples"]
    assert len(samples) >= 3 and state["missed"] == 0
    assert samples[0]["elapsed_s"] < samples[-1]["elapsed_s"]
    assert samples[-1][GENERATION] > samples[0][GENERATION]
    assert samples[0][RUNNING] == pytest.approx(40.0) and samples[0][WAITING] == pytest.approx(3.0)


def test_a_dead_endpoint_costs_samples_and_never_the_workload(
    driver: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sampler is a daemon thread running against a server the agents are hammering. If it could
    raise, the campaign would lose nothing visible and the log would carry a traceback nobody can
    attribute; instead the misses are counted and the report says how many there were."""

    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> FakeMetrics:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(driver.urllib.request, "urlopen", fake_urlopen)
    stop = threading.Event()
    state: dict[str, Any] = {"samples": [], "missed": 0}
    thread = threading.Thread(
        target=driver.sample_aggregate_throughput, args=(["http://vllm:8000/v1"], {}, 0.01, stop, state), daemon=True
    )

    thread.start()
    deadline = time.monotonic() + 5.0
    while state["missed"] < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    stop.set()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert state["missed"] >= 3 and state["samples"] == []


def test_the_probe_is_on_by_default_and_switchable_off_from_the_environment(
    driver: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default-on because a measurement nobody remembers to arm is not taken, and it costs one HTTP
    GET per interval against a server already serving 40 agents. Garbage reads as off rather than as
    a crash: this value is one line in a hand-edited .env file."""
    monkeypatch.delenv("AGGREGATE_PROBE_SECONDS", raising=False)
    assert driver.aggregate_probe_seconds() == pytest.approx(driver.AGGREGATE_PROBE_SECONDS)
    monkeypatch.setenv("AGGREGATE_PROBE_SECONDS", "5")
    assert driver.aggregate_probe_seconds() == pytest.approx(5.0)
    monkeypatch.setenv("AGGREGATE_PROBE_SECONDS", "0")
    assert driver.aggregate_probe_seconds() == 0.0
    monkeypatch.setenv("AGGREGATE_PROBE_SECONDS", "every 5s")
    assert driver.aggregate_probe_seconds() == 0.0
