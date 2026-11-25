# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
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
from collections.abc import Callable, Iterator
from typing import ClassVar

import pytest

from hpcagent_bench.frameworks import timeout_decorator


class _FakeTimer:
    """Records start()/cancel() instead of scheduling a real background alarm."""

    instances: ClassVar[list["_FakeTimer"]] = []

    def __init__(self, interval: int, function: Callable[..., object], args: list[str] | None = None) -> None:
        self.interval = interval
        self.function = function
        self.args = args or []
        self.started = False
        self.cancelled = False
        _FakeTimer.instances.append(self)

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True


@pytest.fixture(autouse=True)
def reset_fake_timer() -> Iterator[None]:
    _FakeTimer.instances.clear()
    yield
    _FakeTimer.instances.clear()


def test_fast_call_returns_its_value_and_cancels_the_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
    # Rebind the NAME the module looks up, not an attribute on the real (process-wide, shared)
    # threading module -- mutating threading.Timer itself would leak into every other thread.
    monkeypatch.setattr(timeout_decorator, "threading", types.SimpleNamespace(Timer=_FakeTimer))

    @timeout_decorator.exit_after(30)
    def quick() -> int:
        return 42

    assert quick() == 42
    assert len(_FakeTimer.instances) == 1
    timer = _FakeTimer.instances[0]
    assert timer.started
    assert timer.cancelled


def test_exception_from_the_wrapped_call_still_cancels_the_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
    # The finally-cancel must run on the exception path too, or a raise inside the wrapped
    # function would leak a live timer that fires later, into an unrelated caller.
    monkeypatch.setattr(timeout_decorator, "threading", types.SimpleNamespace(Timer=_FakeTimer))

    @timeout_decorator.exit_after(30)
    def boom() -> None:
        raise ValueError("kernel blew up")

    with pytest.raises(ValueError, match="kernel blew up"):
        boom()
    assert _FakeTimer.instances[0].cancelled


def test_a_call_that_overruns_the_deadline_raises_keyboard_interrupt() -> None:
    # Real timer, real overrun -- deadline well under the wrapped sleep so this is not a race,
    # and both numbers stay small enough that the test finishes in well under a second either way.
    @timeout_decorator.exit_after(0.05)
    def slow() -> str:
        time.sleep(0.4)
        return "unreachable"

    with pytest.raises(KeyboardInterrupt):
        slow()
