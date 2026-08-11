# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Direct tests for :func:`exit_after`, the wall-clock watchdog decorator.

No test file named it (it is only exercised indirectly by ``frameworks/test.py`` wrapping a real
kernel run, where the timeout is normally never hit). Two things are worth pinning directly: the
watchdog timer is always cancelled once the wrapped call returns or raises (a leaked timer would
later fire an unrelated ``KeyboardInterrupt`` into whatever runs next), and a call that truly
overruns really does raise ``KeyboardInterrupt``.
"""
import time
import types

import pytest

from hpcagent_bench.frameworks import timeout_decorator


class _FakeTimer:
    """Records start()/cancel() instead of scheduling a real background alarm."""

    instances = []

    def __init__(self, interval, function, args=None):
        self.interval = interval
        self.function = function
        self.args = args or []
        self.started = False
        self.cancelled = False
        _FakeTimer.instances.append(self)

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True


@pytest.fixture(autouse=True)
def _reset_fake_timer():
    _FakeTimer.instances.clear()
    yield
    _FakeTimer.instances.clear()


def test_fast_call_returns_its_value_and_cancels_the_watchdog(monkeypatch):
    # Rebind the NAME the module looks up, not an attribute on the real (process-wide, shared)
    # threading module -- mutating threading.Timer itself would leak into every other thread.
    monkeypatch.setattr(timeout_decorator, "threading", types.SimpleNamespace(Timer=_FakeTimer))

    @timeout_decorator.exit_after(30)
    def quick():
        return 42

    assert quick() == 42
    assert len(_FakeTimer.instances) == 1
    timer = _FakeTimer.instances[0]
    assert timer.started and timer.cancelled


def test_exception_from_the_wrapped_call_still_cancels_the_watchdog(monkeypatch):
    # The finally-cancel must run on the exception path too, or a raise inside the wrapped
    # function would leak a live timer that fires later, into an unrelated caller.
    monkeypatch.setattr(timeout_decorator, "threading", types.SimpleNamespace(Timer=_FakeTimer))

    @timeout_decorator.exit_after(30)
    def boom():
        raise ValueError("kernel blew up")

    with pytest.raises(ValueError, match="kernel blew up"):
        boom()
    assert _FakeTimer.instances[0].cancelled


def test_cdquit_reports_the_function_name_and_interrupts_the_main_thread(monkeypatch, capsys):
    # Same rebind-the-name reasoning as above: `_thread` is the process-wide C module, so patch
    # the reference `timeout_decorator` looks up rather than the shared module's attribute.
    calls = []
    monkeypatch.setattr(timeout_decorator, "thread", types.SimpleNamespace(interrupt_main=lambda: calls.append(1)))

    timeout_decorator.cdquit("slow_kernel")

    assert calls == [1]
    assert "slow_kernel took too long" in capsys.readouterr().err


def test_a_call_that_overruns_the_deadline_raises_keyboard_interrupt():
    # Real timer, real overrun -- deadline well under the wrapped sleep so this is not a race,
    # and both numbers stay small enough that the test finishes in well under a second either way.
    @timeout_decorator.exit_after(0.05)
    def slow():
        time.sleep(0.4)
        return "unreachable"

    with pytest.raises(KeyboardInterrupt):
        slow()
