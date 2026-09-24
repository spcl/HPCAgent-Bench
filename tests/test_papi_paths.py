# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The counting control flow of :mod:`hpcagent_bench.harness.papi`, driven through a scripted libpapi.

No PAPI, no PMU and no kernel: the ctypes library is a fake whose ``PAPI_read`` answers from a
script keyed on how many times the fake kernel has run, and the native call is a loop over the
timed-call seam. What is left is exactly the code between those two boundaries -- arming, the read
bracket, the rep selection, the thread bookkeeping and the reports built from them.
"""

import ctypes
import os
import pathlib
import time
from collections.abc import Callable, Sequence

import pytest

from hpcagent_bench.harness import papi
from hpcagent_bench.harness.native_call import RepTiming

#: Thread ids of the fake process's workers. The calling thread is always ``os.getpid()``.
WORKERS = (101, 102, 103)


def fake_install(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, *, header: bool) -> pathlib.Path:
    """A PAPI prefix whose library is mapped in a fake ``/proc/self/maps``; returns the library file."""
    lib = tmp_path / "prefix" / "lib" / "libpapi.so.7.2.0.0"
    lib.parent.mkdir(parents=True)
    lib.touch()
    if header:
        (tmp_path / "prefix" / "include").mkdir()
        (tmp_path / "prefix" / "include" / "papi.h").touch()
    maps = tmp_path / "maps"
    maps.write_text(
        "7f0000-7f1000 r--p 00000000 00:2a 11 /usr/lib/libc.so.6\n"
        f"7f2000-7f3000 r-xp 00000000 00:2a 12 {lib}\n"
        "7f4000-7f5000 rw-p 00000000 00:00 0\n"
    )
    monkeypatch.setattr(papi, "check", lambda: None)
    monkeypatch.setattr(papi, "MAPS", maps)
    return lib


def test_build_flags_name_the_directory_of_the_libpapi_this_process_loaded(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``find_library`` answers with a soname; the mapping is what names the directory the loader used."""
    lib = fake_install(tmp_path, monkeypatch, header=True)
    include = tmp_path / "prefix" / "include"
    assert papi.build_flags() == ([f"-I{include}"], [f"-L{lib.parent}", f"-Wl,-rpath,{lib.parent}", "-lpapi"])


def test_build_flags_add_no_include_directory_without_papi_h(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ``-I`` naming a directory with no ``papi.h`` would only hide where the header really is."""
    fake_install(tmp_path, monkeypatch, header=False)
    assert papi.build_flags()[0] == []


def test_a_libpapi_that_is_not_mapped_is_papi_missing(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    maps = tmp_path / "maps"
    maps.write_text("7f0000-7f1000 r-xp 00000000 00:2a 11 /usr/lib/libc.so.6\n")
    monkeypatch.setattr(papi, "check", lambda: None)
    monkeypatch.setattr(papi, "MAPS", maps)
    with pytest.raises(papi.PapiUnavailable) as caught:
        papi.build_flags()
    assert caught.value.cause == "papi_missing", caught.value.cause


class Strerror:
    """``PAPI_strerror``: an attribute papi.py sets ``restype`` on, so it cannot be a bound method."""

    restype: object = None

    def __call__(self, code: int) -> bytes:
        return b"simulated refusal"


class ScriptedPapi:
    """A libpapi whose counters read ``values(tid, calls)``, where ``calls`` counts kernel runs.

    ``attach_rc`` is what ``PAPI_attach`` answers, so a host that refuses the attach is one argument.
    """

    def __init__(self, values: Callable[[int, int], Sequence[int]], attach_rc: int = papi.PAPI_OK) -> None:
        self.values = values
        self.attach_rc = attach_rc
        self.calls = 0
        self.owner: dict[int, int] = {}
        self.PAPI_strerror = Strerror()

    def kernel(self) -> None:
        self.calls += 1

    def PAPI_event_name_to_code(self, name: bytes, ref: object) -> int:
        return papi.PAPI_OK

    def PAPI_create_eventset(self, ref: ctypes.c_int) -> int:
        handle = len(self.owner) + 1
        ref._obj.value = handle
        self.owner[handle] = os.getpid()
        return papi.PAPI_OK

    def PAPI_assign_eventset_component(self, eventset: ctypes.c_int, component: int) -> int:
        return papi.PAPI_OK

    def PAPI_attach(self, eventset: ctypes.c_int, tid: ctypes.c_ulong) -> int:
        self.owner[eventset.value] = tid.value
        return self.attach_rc

    def PAPI_add_event(self, eventset: ctypes.c_int, code: object) -> int:
        return papi.PAPI_OK

    def PAPI_start(self, eventset: ctypes.c_int) -> int:
        return papi.PAPI_OK

    def PAPI_read(self, eventset: ctypes.c_int, out: ctypes.Array[ctypes.c_longlong]) -> int:
        values = self.values(self.owner[eventset.value], self.calls)
        for index in range(len(out)):  # one slot per event in the set, which may be fewer than scripted
            out[index] = values[index]
        return papi.PAPI_OK

    def PAPI_stop(self, eventset: ctypes.c_int, out: object) -> int:
        return papi.PAPI_OK

    def PAPI_destroy_eventset(self, ref: object) -> int:
        return papi.PAPI_OK


def install(monkeypatch: pytest.MonkeyPatch, lib: ScriptedPapi, *, slow_first_rep_s: float = 0.0) -> None:
    """Point papi at ``lib`` and replace the native call with ``warmup + reps`` runs of its kernel.

    The fake keeps the real seam's contract: ``_call_native_impl`` reads ``.ns`` off every timed
    call, so a counter that answered a bare int fails here exactly as it failed every /profile."""

    def native_call(
        *args: object, timed_call: Callable[..., RepTiming], reps: int, warmup: int, **kwargs: object
    ) -> None:
        for rep in range(warmup + reps):
            if rep == warmup and slow_first_rep_s:
                timing = timed_call(lambda: (time.sleep(slow_first_rep_s), lib.kernel()), [], lambda: None)
            else:
                timing = timed_call(lib.kernel, [], lambda: None)
            assert timing.ns >= 0 and timing.host_ns == timing.ns, timing

    monkeypatch.setattr(papi, "initialised", lambda: lib)
    monkeypatch.setattr(papi, "_call_native_impl", native_call)
    monkeypatch.setattr(papi, "available_events", lambda: ("PAPI_TOT_CYC", "PAPI_TOT_INS"))
    monkeypatch.setattr(papi, "hardware_counters", lambda: 5)


def count(metric: str = "cycles", *, reps: int = 2, warmup: int = 1) -> papi.MetricRow:
    return papi.counting_worker("/fake.so", None, {}, "c", None, metric, reps, warmup, 1.0, 0)


def per_thread(*, reps: int = 2, warmup: int = 1) -> papi.PerThreadReport:
    return papi.per_thread_report("/fake.so", None, {}, "c", None, reps, warmup, 1.0, 0)


def transient_worker(lib: ScriptedPapi) -> Callable[[], tuple[int, ...]]:
    """Thread listing of a process whose worker 102 exists only between measured rep 1 and rep 2.

    The warmup is call 1, so arming sees (pid, 101); call 2 is rep 1, after which 102 is alive; it
    is gone again once rep 2 (call 3) has run, which is when teardown looks.
    """
    return lambda: (os.getpid(), 101, 102) if lib.calls == 2 else (os.getpid(), 101)


def test_a_thread_that_came_and_went_between_the_reps_fails_the_summed_count(monkeypatch) -> None:
    """Its work is in no event set, so the sum is short by exactly that thread. Sampling the thread
    list only at arm and at teardown never saw it and returned the short sum as a count."""
    lib = ScriptedPapi(lambda tid, calls: (calls * 10,))
    install(monkeypatch, lib)
    monkeypatch.setattr(papi, "thread_ids", transient_worker(lib))
    row = count()
    assert row["count"] is None, row
    assert "1 thread(s) started after the counters armed" in row["missing"], row["missing"]


def test_a_thread_that_came_and_went_between_the_reps_refuses_the_per_thread_report(monkeypatch) -> None:
    """The distribution would be missing exactly the thread it is about, so the report is absent
    with the cause that says so rather than balanced over the threads that were attached."""
    lib = ScriptedPapi(lambda tid, calls: (calls * 10, calls * 20))
    install(monkeypatch, lib)
    monkeypatch.setattr(papi, "thread_ids", transient_worker(lib))
    report = per_thread()
    assert report["cause"] == "threads_moved" and report["imbalance"] is None, report


def one_two_three_four(tid: int, calls: int) -> tuple[int, int]:
    """Cycles and instructions after ``calls`` kernel runs: the calling thread and three workers
    burn 1000, 2000, 3000 and 4000 cycles per call, each at two instructions per cycle."""
    share = (os.getpid(), *WORKERS).index(tid) + 1
    return 1000 * share * calls, 2000 * share * calls


def test_a_known_one_two_three_four_split_comes_back_as_its_rows_and_its_imbalance(monkeypatch) -> None:
    """The distribution is the one number a summed count discards. 1:2:3:4 has a mean of 2.5 and a
    max of 4, so the report must say 1.6 and name the fourth thread, on any host."""
    lib = ScriptedPapi(one_two_three_four)
    install(monkeypatch, lib)
    monkeypatch.setattr(papi, "thread_ids", lambda: (os.getpid(), *WORKERS))
    monkeypatch.setattr(papi, "placement", lambda tid: {"cpus": [0], "pinned": True, "core": str(tid)})
    report = per_thread(reps=3)
    assert "missing" not in report, report
    rows = report["threads"]
    assert [row["cycles"] for row in rows] == [1000, 2000, 3000, 4000], rows
    assert [row["cycle_share"] for row in rows] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert all(row["cpi"] == pytest.approx(0.5) for row in rows), rows
    spread = report["imbalance"]
    assert spread["max_over_mean"] == pytest.approx(1.6) and spread["wasted_fraction"] == pytest.approx(0.375)
    assert spread["critical_tid"] == WORKERS[-1], spread
    assert (report["reps_counted"], report["threads_participating"]) == (3, 4), report


def test_every_counted_rep_reaches_the_native_call_as_a_rep_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_call_native_impl`` reads ``rep.ns`` off each timed call, so a counter that answers a bare
    int fails every /profile PAPI metric with ``counted run failed (AttributeError: 'int' object
    has no attribute 'ns')``."""
    lib = ScriptedPapi(one_two_three_four)
    seen: list[object] = []

    def native_call(
        *args: object, timed_call: Callable[..., RepTiming], reps: int, warmup: int, **kwargs: object
    ) -> None:
        seen.extend(timed_call(lib.kernel, [], lambda: None) for _ in range(warmup + reps))

    install(monkeypatch, lib)
    monkeypatch.setattr(papi, "_call_native_impl", native_call)
    monkeypatch.setattr(papi, "thread_ids", lambda: (os.getpid(), *WORKERS))
    row = count(reps=2, warmup=1)
    assert row["count"] == 10_000, row
    assert len(seen) == 3 and all(isinstance(rep, RepTiming) for rep in seen), seen


def test_a_summed_count_is_one_rep_s_delta_on_every_attached_thread(monkeypatch) -> None:
    """The metric is the kernel's total work, so it is every thread's after-minus-before for one
    call, added -- not the cumulative reading, and not the calling thread's share."""
    lib = ScriptedPapi(one_two_three_four)
    install(monkeypatch, lib)
    monkeypatch.setattr(papi, "thread_ids", lambda: (os.getpid(), *WORKERS))
    row = count(reps=3)
    assert (row["count"], row["threads_counted"], row["scope"]) == (10_000, 4, "all_threads"), row
    assert row["reps_counted"] == 3 and "fallback" not in row, row


def test_a_refused_attach_counts_the_calling_thread_alone_and_says_why(monkeypatch) -> None:
    """A worker nothing is attached to makes the sum wrong without a symptom, so the fallback is the
    calling thread's own count with the refusal attached, never a silent fraction of the work."""
    lib = ScriptedPapi(one_two_three_four, attach_rc=-1)
    install(monkeypatch, lib)
    monkeypatch.setattr(papi, "thread_ids", lambda: (os.getpid(), *WORKERS))
    row = count()
    assert (row["count"], row["threads_counted"], row["scope"]) == (1000, 1, "calling_thread"), row
    assert "cannot attach to thread 101" in row["fallback"] and "simulated refusal" in row["fallback"], row


def test_a_refused_attach_refuses_the_per_thread_report_by_cause(monkeypatch) -> None:
    """The calling thread alone has no distribution, so the per-thread answer is absent, not balanced."""
    lib = ScriptedPapi(one_two_three_four, attach_rc=-1)
    install(monkeypatch, lib)
    monkeypatch.setattr(papi, "thread_ids", lambda: (os.getpid(), *WORKERS))
    report = per_thread()
    assert report["cause"] == "attach_refused" and report["threads"] == [], report
    assert "simulated refusal" in report["missing"], report["missing"]


#: A device metric as gpu_feature_set resolves one.
OCCUPANCY: papi.ResolvedGpuMetric = {
    "metric": "occupancy",
    "vendor": "nvidia",
    "component": "cuda",
    "event": "cuda:::sm__warps_active.pct_of_peak_sustained_active:device=0",
    "matches": ["cuda:::sm__warps_active.pct_of_peak_sustained_active:device=0"],
    "unit": "%",
    "question": "q",
    "reading": "r",
}


def test_a_device_count_is_the_first_measured_rep_s_delta_not_the_fastest_rep_s(monkeypatch) -> None:
    """Under device counters the clock is a replay artifact, so choosing the fastest rep would choose
    by noise. Rep 1 is made the slow one and moves the counter by 5; rep 2 is fast and moves it by 7."""
    cumulative = {1: 100, 2: 105, 3: 112}  # after the warmup call, after rep 1, after rep 2
    lib = ScriptedPapi(lambda tid, calls: (cumulative[calls],))
    install(monkeypatch, lib, slow_first_rep_s=0.05)
    monkeypatch.setattr(
        papi,
        "gpu_feature_set",
        lambda vendor=None, metrics=(): {
            "supported": {"occupancy": OCCUPANCY},
            "unsupported": {},
            "permissions": {"nvidia": None, "amd": None},
        },
    )
    monkeypatch.setattr(papi, "device_barrier", lambda vendor: (lambda: 0, ""))
    monkeypatch.setattr(
        papi,
        "gpu_component",
        lambda name: {
            "index": 2,
            "name": name,
            "short_name": name,
            "description": "",
            "enabled": True,
            "disabled_reason": "",
        },
    )
    row = papi.gpu_counting_worker("/fake.so", None, {}, "cuda", None, "occupancy", "nvidia", False, None, 2, 1, 1.0, 0)
    assert (row["count"], row["reps_counted"]) == (5, 2), row
    assert row["elapsed_ns"] >= 50_000_000, "the elapsed time is not the first measured rep's"
    assert (row["unit"], row["residency"], row["devices_matched"]) == ("%", "host", 1), row
