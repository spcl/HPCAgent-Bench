# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every capability a host can lack reaches the agent as a named ``cause``, through the real
``/profile`` route. CPU group: it runs on CI, where there is no PMU, no PAPI and no GPU, which is
exactly the host these answers are for -- never a crash, never a 500, never an empty profile."""

import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from http.server import ThreadingHTTPServer

import pytest

from hpcagent_bench import perf_reports
from hpcagent_bench.harness import gpu_profiling, papi, profiling
from hpcagent_bench.harness.service import ServiceConfig
from tests.test_gpu_profiling import gpu_submission

JudgeFactory = Callable[..., tuple[ThreadingHTTPServer, str]]

#: A C delivery that builds on any host with a C compiler; nothing here measures it.
TRIVIAL_C = {"language": "c", "source": "void gemm_fp64(void) {}"}

#: What ``perf_reports.perf_check`` raises before anything is built.
PERF_CHECK_CAUSES = ("not_linux", "perf_missing", "no_perf_events", "perf_event_paranoid")

#: What ``papi.check`` raises before anything is built.
PAPI_CHECK_CAUSES = ("not_linux", "papi_missing")

#: What ``gpu_profiling.gpu_check`` raises per device language, before anything is built.
GPU_CHECK_CAUSES = {
    "cuda": ("not_linux", "nsys_missing", "no_gpu", "insufficient_permissions"),
    "hip": ("not_linux", "rocprof_missing", "rocminfo_missing", "no_amd_gpu", "kfd_permission_denied", "timed_out"),
}

#: The routes that sum counts and gate on PAPI before the build.
SUMMED_COUNTING_ROUTES = {"papi": {"tool": "papi"}, "linuxperf-counters": {"tool": "linuxperf", "counters": True}}

#: What the per-thread CHILD can report; the route passes each through inside a 200.
PER_THREAD_CHILD_CAUSES = (
    "not_linux",
    "papi_missing",
    "papi_init_failed",
    "events_unsupported",
    "attach_refused",
    "threads_moved",
    "no_measured_rep",
    "not_openmp",
)


def post_profile(url: str, fields: dict[str, object]) -> tuple[int, dict[str, object]]:
    """``(status, answer)`` of a raw ``POST /profile``; a refusal is an answer, read and closed."""
    body = {"kernel": "gemm", "rank": 0, **TRIVIAL_C, **fields}
    request = urllib.request.Request(
        f"{url}/profile", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as refused:
        with refused:
            return refused.code, json.loads(refused.read())


def refusing(exception: type[Exception], cause: str) -> Callable[..., str]:
    """A check that refuses with ``cause``, whatever arguments the route hands it."""

    def check(*args: object, **kwargs: object) -> str:
        raise exception(cause, f"this host refused: {cause}")

    return check


def per_thread_child(monkeypatch: pytest.MonkeyPatch, script: str) -> None:
    """Replace the counted child with ``script``: the parent still builds, spawns and parses."""
    monkeypatch.setattr(profiling, "child_argv", lambda request_file, **kwargs: [sys.executable, "-c", script])


@pytest.mark.parametrize("cause", PERF_CHECK_CAUSES)
def test_a_host_that_cannot_sample_answers_linuxperf_with_a_503_naming_the_cause(
    make_judge: JudgeFactory, monkeypatch: pytest.MonkeyPatch, cause: str
) -> None:
    monkeypatch.setattr(perf_reports, "perf_check", refusing(perf_reports.PerfUnavailable, cause))
    status, answer = post_profile(make_judge(ServiceConfig())[1], {"tool": "linuxperf"})
    assert (status, answer.get("cause")) == (503, cause), answer


@pytest.mark.parametrize("route", sorted(SUMMED_COUNTING_ROUTES))
@pytest.mark.parametrize("cause", PAPI_CHECK_CAUSES)
def test_a_host_that_cannot_count_answers_a_summed_counting_route_with_a_503_naming_the_cause(
    make_judge: JudgeFactory, monkeypatch: pytest.MonkeyPatch, route: str, cause: str
) -> None:
    """The sampler is present, so a refusal here can only be PAPI's, and it must say so."""
    assert cause in papi.CAUSES, cause
    monkeypatch.setattr(perf_reports, "perf_check", lambda: "/fake/bin/perf")
    monkeypatch.setattr(papi, "check", refusing(papi.PapiUnavailable, cause))
    status, answer = post_profile(make_judge(ServiceConfig())[1], SUMMED_COUNTING_ROUTES[route])
    assert (status, answer.get("cause")) == (503, cause), answer


@pytest.mark.parametrize("cause", PER_THREAD_CHILD_CAUSES)
def test_a_per_thread_child_that_cannot_count_answers_200_with_its_cause_in_the_report(
    make_judge: JudgeFactory, monkeypatch: pytest.MonkeyPatch, cause: str
) -> None:
    """The page promises ``per_thread.cause``; a report without one reads as a balanced kernel."""
    assert cause in papi.CAUSES, cause
    report = papi.missing_report(cause, f"the child refused: {cause}")
    report["text"] = papi.render_thread_report(report)
    line = profiling.RESULT_PREFIX + json.dumps(report)
    per_thread_child(monkeypatch, f"print({line!r})")
    status, answer = post_profile(make_judge(ServiceConfig())[1], {"tool": "papi", "per_thread": True, "threads": 2})
    assert status == 200, answer
    report = answer["per_thread"]
    assert report["cause"] == cause and report["imbalance"] is None, report
    assert f"[{cause}]" in str(answer["text"]), answer["text"]


def test_a_per_thread_child_that_dies_answers_200_with_run_failed_naming_its_exit(
    make_judge: JudgeFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A counting child killed by a missing PMU driver or a segfault is a cause, not a 500 -- and
    the parent renders it, because the child that would have is gone."""
    per_thread_child(monkeypatch, "import sys; sys.stderr.write('PAPI_start: no PMU'); sys.exit(3)")
    status, answer = post_profile(make_judge(ServiceConfig())[1], {"tool": "papi", "per_thread": True, "threads": 2})
    assert status == 200, answer
    report = answer["per_thread"]
    assert report["cause"] == "run_failed", report
    assert "exit 3" in report["missing"] and "no PMU" in report["missing"], report["missing"]
    assert "[run_failed]" in str(answer["text"]), answer["text"]


@pytest.mark.parametrize(
    "language, cause", [(language, cause) for language, causes in GPU_CHECK_CAUSES.items() for cause in causes]
)
def test_a_host_that_cannot_trace_answers_a_device_submission_with_a_503_naming_the_cause(
    make_judge: JudgeFactory, monkeypatch: pytest.MonkeyPatch, language: str, cause: str
) -> None:
    assert cause in gpu_profiling.CAUSES, cause
    monkeypatch.setattr(gpu_profiling, "gpu_check", refusing(gpu_profiling.GpuProfilerUnavailable, cause))
    status, answer = post_profile(make_judge(ServiceConfig())[1], gpu_submission(language).to_json())
    assert (status, answer.get("cause")) == (503, cause), answer
