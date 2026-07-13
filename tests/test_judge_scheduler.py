# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Judge device scheduler: slot planning, dynamic dispatch, GPU pinning, config."""
import threading
import time

from optarena.agent_bench import native_call
from optarena.agent_bench.judge_scheduler import DeviceSlot, JudgeConfig, JudgeScheduler, srun_wrap


def test_slots_gpu_then_cpu_per_node():
    cfg = JudgeConfig(gpus_per_node=2, cpu_slots_per_node=1)
    slots = cfg.slots()
    assert [s.label for s in slots] == ["local:gpu0", "local:gpu1", "local:cpu0"]
    assert cfg.nodes == 1


def test_slots_multinode():
    cfg = JudgeConfig(gpus_per_node=4, cpu_slots_per_node=0, nodelist=("nid001", "nid002"))
    slots = cfg.slots()
    assert cfg.nodes == 2
    assert len(slots) == 8
    assert slots[0] == DeviceSlot("gpu", 0, "nid001")
    assert slots[4] == DeviceSlot("gpu", 0, "nid002")
    assert all(not s.is_local for s in slots)


def test_slots_fallback_to_one_cpu():
    assert JudgeConfig(gpus_per_node=0, cpu_slots_per_node=0).slots() == [DeviceSlot("cpu", 0, None)]


def test_from_config_defaults(monkeypatch):
    # No GPUs visible in a CPU test env -> 0 gpu slots, 1 cpu slot.
    monkeypatch.setattr("optarena.agent_bench.judge_scheduler.local_gpu_count", lambda: 0)
    monkeypatch.delenv("OPTARENA_JUDGE_GPUS_PER_NODE", raising=False)
    monkeypatch.delenv("OPTARENA_JUDGE_CPU_SLOTS_PER_NODE", raising=False)
    cfg = JudgeConfig.from_config()
    assert cfg.gpus_per_node == 0 and cfg.cpu_slots_per_node == 1


def test_run_processes_all_in_order():
    sched = JudgeScheduler([DeviceSlot("cpu", 0), DeviceSlot("cpu", 1)])
    out = sched.run(list(range(20)), lambda it, slot: it * it)
    assert [v for _s, v in out] == [i * i for i in range(20)]
    assert all(s == "ok" for s, _v in out)


def test_error_is_captured_not_fatal():

    def run_item(it, slot):
        if it == 3:
            raise ValueError("boom")
        return it

    out = JudgeScheduler([DeviceSlot("cpu", 0)]).run(list(range(5)), run_item)
    assert out[3][0] == "err" and isinstance(out[3][1], ValueError)
    assert [v for s, v in out if s == "ok"] == [0, 1, 2, 4]


def test_gpu_slot_pins_thread_local():
    # A gpu slot exposes its index via native_call.assigned_device() inside run_item;
    # a cpu slot leaves it None. (No real GPU needed -- we only read the thread-local.)
    seen = {}

    def run_item(it, slot):
        seen[it] = native_call.assigned_device()
        return it

    slots = [DeviceSlot("gpu", 0), DeviceSlot("gpu", 1), DeviceSlot("cpu", 0)]
    JudgeScheduler(slots).run([0, 1, 2, 3, 4, 5], run_item)
    # every item saw a valid pin (a gpu index int, or None for a cpu slot); and the
    # pin is cleared after (no leak into the main thread).
    assert set(seen.values()) <= {0, 1, None}
    assert native_call.assigned_device() is None


def test_dynamic_work_stealing():
    # 2 slots, 6 items; one slot's items are slow. A static split would make the
    # fast slot finish 3 quick items then idle; work-stealing keeps it pulling.
    order_lock = threading.Lock()
    finish_order = []

    def run_item(it, slot):
        time.sleep(0.02 if it == 0 else 0.001)  # item 0 is the slow one
        with order_lock:
            finish_order.append(it)
        return it

    JudgeScheduler([DeviceSlot("cpu", 0), DeviceSlot("cpu", 1)]).run(list(range(6)), run_item)
    # The slow item 0 finishes late while the other slot drains the rest -> item 0
    # is NOT among the first finishers (proves the fast slot kept stealing work).
    assert 0 in finish_order[-2:]
    assert len(finish_order) == 6


def test_srun_wrap_remote_vs_local():
    launcher = ("srun", "--nodelist", "{node}", "--gpus", "1", "-n", "1")
    local = srun_wrap(DeviceSlot("gpu", 0, None), ["bench", "in", "out"], launcher)
    remote = srun_wrap(DeviceSlot("gpu", 2, "nid007"), ["bench", "in", "out"], launcher)
    assert local == ["bench", "in", "out"]
    assert remote == ["srun", "--nodelist", "nid007", "--gpus", "1", "-n", "1", "bench", "in", "out"]


def test_native_call_assigned_device_roundtrip():
    assert native_call.assigned_device() is None
    native_call.set_assigned_device(3)
    assert native_call.assigned_device() == 3
    native_call.set_assigned_device(None)
    assert native_call.assigned_device() is None
