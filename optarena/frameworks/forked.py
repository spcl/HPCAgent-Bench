# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run a callable in a forked child and SURFACE its failure instead of eating it.

Native (no-container) collection runs each kernel in its own child process so a
segfault or a framework exception in one kernel cannot take down the whole sweep --
and, unlike a swallow-everything harness, the cause is reported: a fatal signal
(``SIGSEGV`` / ``SIGABRT`` / ... from a crashing kernel) is decoded to its name, any
Python traceback is printed to stdout by the child before it exits, and a child that
runs past ``timeout`` seconds is terminated and reported as ``TIMEOUT``.

This is the shared isolation primitive for the native framework-baseline collection
and the native agent run (and the per-kernel wall-clock budget).
"""
import multiprocessing
import queue
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

from optarena import osinfo

#: Grace period (seconds) to drain the result queue after the child has exited
#: cleanly -- the process is done but the queue feeder thread may still be flushing.
_DRAIN_S = 5.0


@dataclass
class RunResult:
    """Outcome of a forked run. ``ok`` is the only success signal. On failure the cause
    is either a kill -- ``signal`` names the fatal signal / ``TIMEOUT`` and ``error`` may
    carry a human-readable detail (e.g. the timeout seconds) -- or a non-signal fault,
    where ``error`` alone holds the traceback / message and ``signal`` is ``None``. Use
    :func:`forked_failure_reason` for the one-line cause (it prefers ``signal``).
    ``result`` carries the callable's return value on success (must be picklable;
    ``None`` if it was not)."""
    ok: bool
    exit_code: Optional[int] = None
    signal: Optional[str] = None
    error: Optional[str] = None
    result: Any = None


def forked_failure_reason(r: RunResult) -> str:
    """One-line cause for a failed :class:`RunResult`: the fatal signal name, else the
    last line of the child's traceback, else ``"unknown"``."""
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
    """Return the LAST item the child pushed to ``progress_q`` (or the prior
    ``current`` if it pushed nothing new), so a killed child's most recent reported
    progress survives the kill."""
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
    """Run ``fn(*args, **kwargs)`` in a forked child.

    Returns a failed :class:`RunResult` -- and logs the cause to stdout -- on a fatal
    signal (segfault), an exception (traceback), or a ``timeout`` overrun; otherwise
    ``ok=True`` with the (picklable) return value. ``timeout`` is a per-call wall-clock
    budget in seconds (``None`` waits forever); ``label`` tags the stdout log lines.

    ``mp_context`` forces the multiprocessing start method (e.g. ``"spawn"`` for a
    device call whose CUDA context does not survive ``fork``); when unset the OS default
    (:func:`osinfo.mp_context`, honouring a config/env override) is used.

    With ``stream_progress=True`` the child receives a ``progress`` multiprocessing
    queue (as a keyword arg) it can ``.put()`` best-so-far snapshots on; the last one
    is preserved in ``RunResult.result`` EVEN when the child is later killed by the
    timeout or a signal -- so an overrun still yields its best-so-far (the online-exam
    snapshot) instead of nothing."""
    # fork on Linux/WSL2 (cheap -- the child inherits the parent's inputs); spawn on
    # macOS, where forking after numpy/BLAS/Accelerate threads can abort the child
    # (osinfo.mp_context resolves the OS default, honouring a config/env override).
    # A caller may override the start method via mp_context (e.g. device -> spawn).
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
    # Poll at a fixed interval so the RESULT queue is drained WHILE the child is still
    # alive. A child that puts a payload larger than the OS pipe buffer (~64 KB -- e.g.
    # an L/XL preset's output arrays) blocks in its queue-feeder thread until the parent
    # reads it, so it never exits: joining-then-reading would deadlock (forever with
    # timeout=None, or surface as a false TIMEOUT otherwise). Draining here frees the
    # feeder. A blocking ``q.get(timeout=poll)`` wakes the instant the child puts, so it
    # doubles as the wait and adds no latency to a fast success.
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
            return RunResult(ok=False, exit_code=ec, error=f"{tag}child exited {ec} with no result",
                             result=last_progress)
    status, payload = result_item
    if status == "ok":
        return RunResult(ok=True, exit_code=ec, result=payload)
    return RunResult(ok=False, exit_code=ec, error=payload, result=last_progress)
