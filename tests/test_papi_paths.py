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
import time
from collections.abc import Callable, Sequence

import pytest

from hpcagent_bench.harness import papi

#: Thread ids of the fake process's workers. The calling thread is always ``os.getpid()``.
WORKERS = (101, 102, 103)


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
        for index, value in enumerate(self.values(self.owner[eventset.value], self.calls)):
            out[index] = value
        return papi.PAPI_OK

    def PAPI_stop(self, eventset: ctypes.c_int, out: object) -> int:
        return papi.PAPI_OK

    def PAPI_destroy_eventset(self, ref: object) -> int:
        return papi.PAPI_OK


def install(monkeypatch: pytest.MonkeyPatch, lib: ScriptedPapi, *, slow_first_rep_s: float = 0.0) -> None:
    """Point papi at ``lib`` and replace the native call with ``warmup + reps`` runs of its kernel."""

    def native_call(*args: object, timed_call: Callable[..., int], reps: int, warmup: int, **kwargs: object) -> None:
        for rep in range(warmup + reps):
            if rep == warmup and slow_first_rep_s:
                timed_call(lambda: (time.sleep(slow_first_rep_s), lib.kernel()), [], lambda: None)
            else:
                timed_call(lib.kernel, [], lambda: None)

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
