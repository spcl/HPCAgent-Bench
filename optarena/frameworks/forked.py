# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run a callable in a forked child and SURFACE its failure (signal/traceback/timeout) instead of eating it."""
import multiprocessing
import queue
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

from optarena import osinfo

#: Grace period (seconds) to drain the result queue after the child exits cleanly.
_DRAIN_S = 5.0


@dataclass
class RunResult:
    """Outcome of a forked run: ``ok`` is the success signal; on failure ``signal``/``error`` name the
    cause (see :func:`forked_failure_reason`); ``result`` carries the picklable return value."""
    ok: bool
    exit_code: Optional[int] = None
    signal: Optional[str] = None
    error: Optional[str] = None
    result: Any = None


def forked_failure_reason(r: RunResult) -> str:
    """One-line cause for a failed :class:`RunResult`: signal name, else last traceback line, else "unknown"."""
    return r.signal or (r.error.strip().splitlines()[-1] if r.error else "unknown")


def _child(fn, args, kwargs, q):
    try:
        out = fn(*args, **kwargs)
        try:
            q.put(("ok", out))
        except Exception:  # unpicklable return value -> success without a payload
            q.put(("ok", None))
    except BaseException:  # noqa: BLE001 -- surface EVERY failure, never swallow it
        tb = traceback.format_exc()
        sys.stdout.write(tb)
        sys.stdout.flush()
        q.put(("error", tb))


def _drain(progress_q, current):
    """Return the last item pushed to ``progress_q`` (or ``current``), so a kill preserves the last progress."""
    try:
        while True:
            current = progress_q.get_nowait()
    except queue.Empty:
        pass
    return current


def run_forked(fn: Callable,
               *args,
               label: str = "",
               timeout: Optional[float] = None,
               stream_progress: bool = False,
               mp_context: Optional[str] = None,
               **kwargs) -> RunResult:
    """Run ``fn(*args, **kwargs)`` in a forked child; returns a failed RunResult (cause logged to stdout) on
    a fatal signal, exception, or timeout overrun, else ``ok=True`` with the picklable return value.
    ``stream_progress=True`` preserves the child's last ``progress`` snapshot even if it is later killed."""
    # fork is cheap on Linux/WSL2; spawn on macOS, where forking after numpy/BLAS threads can abort the child.
    ctx = multiprocessing.get_context(mp_context if mp_context is not None else osinfo.mp_context())
    q = ctx.Queue()
    progress_q = ctx.Queue() if stream_progress else None
    if progress_q is not None:
        kwargs = {**kwargs, "progress": progress_q}
    p = ctx.Process(target=_child, args=(fn, args, kwargs, q))
    tag = f"[{label}] " if label else ""
    p.start()
    last_progress = None
    deadline = (time.monotonic() + timeout) if timeout is not None else None
    # Poll so the result queue drains while the child is alive -- a payload bigger than the OS
    # pipe buffer would otherwise block the child's feeder thread forever (join-then-read deadlocks).
    poll = 0.1
    result_item = None  # (status, payload) once the child's single result is received
    while p.is_alive():
        if progress_q is not None:
            last_progress = _drain(progress_q, last_progress)
        if deadline is not None and time.monotonic() >= deadline:
            if result_item is not None:
                break  # child actually finished (payload already drained) -- not a timeout
            p.terminate()  # SIGTERM
            p.join(5.0)
            if p.is_alive():  # a child that ignores/blocks SIGTERM would hang the
                p.kill()  # parent on an unbounded join -- escalate to SIGKILL
                p.join()
            if progress_q is not None:
                last_progress = _drain(progress_q, last_progress)
            msg = f"{tag}timed out after {timeout}s"
            sys.stdout.write(msg + "\n")
            sys.stdout.flush()
            return RunResult(ok=False, signal="TIMEOUT", error=msg, result=last_progress)
        if result_item is None:
            try:
                result_item = q.get(timeout=poll)
            except queue.Empty:
                pass
        else:
            p.join(poll)
    if progress_q is not None:
        last_progress = _drain(progress_q, last_progress)
    ec = p.exitcode
    if ec is not None and ec < 0:  # killed by a fatal signal (segfault, abort, ...)
        try:
            sig = signal.Signals(-ec).name
        except ValueError:
            sig = f"signal {-ec}"
        msg = f"{tag}child killed by {sig}"
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()
        return RunResult(ok=False, exit_code=ec, signal=sig, error=msg, result=last_progress)
    if result_item is None:  # not drained in-loop -- covers the clean-exit race window
        try:
            result_item = q.get(timeout=_DRAIN_S)
        except queue.Empty:
            return RunResult(ok=False,
                             exit_code=ec,
                             error=f"{tag}child exited {ec} with no result",
                             result=last_progress)
    status, payload = result_item
    if status == "ok":
        return RunResult(ok=True, exit_code=ec, result=payload)
    return RunResult(ok=False, exit_code=ec, error=payload, result=last_progress)
