# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
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

from hpcagent_bench import osinfo
from hpcagent_bench.isolation import pause_openmp_pools

#: Grace period (seconds) to drain the result queue after the child exits cleanly.
_DRAIN_S = 5.0

#: How long the child may take to say it started before the deadline is armed anyway. An
#: unbounded wait on a child that never runs is worse than a slightly wrong clock.
ARM_GRACE_S = 30.0

#: How long a SIGTERMed child has to exit before the parent escalates to SIGKILL.
TERM_GRACE_S = 5.0

#: Extra time granted to a child the kernel says is DUMPING CORE. Its cause is already decided --
#: it took a fatal signal and the kernel is writing the image -- so a SIGKILL here does not stop a
#: hung child, it relabels a crash as a kill and the caller records the wrong cause. On a distro
#: whose ``core_pattern`` pipes to a helper the dump costs a second even on an idle box (measured:
#: 1.1s for a 20MB python child), and the helper is itself a process that has to be scheduled, so
#: on a loaded runner it is the SIGTERM grace that runs out first. A dump terminates on its own;
#: this only has to be longer than one takes.
COREDUMP_GRACE_S = 60.0


def is_core_dumping(pid: int) -> bool:
    """True when the kernel reports ``pid`` is writing a core image (Linux >= 4.15).

    False everywhere the answer is not knowable -- another OS, a reaped pid, a hidepid mount --
    which is the pre-existing behaviour: escalate.
    """
    try:
        with open(f"/proc/{pid}/status", "r") as fh:
            for line in fh:
                if line.startswith("CoreDumping:"):
                    return line.split(":", 1)[1].strip() == "1"
    except OSError:
        return False
    return False


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
    # First act, before any work: this is what arms the parent's deadline (see run_forked).
    q.put(("started", None))
    try:
        out = fn(*args, **kwargs)
        try:
            # NOTE: put() only enqueues -- pickling happens later in the queue's feeder thread, so an
            # unpicklable or oversized payload is NOT caught here. It surfaces in the parent as
            # "child exited 0 with no result"; callers with large payloads must spill to disk
            # (see native_call.spill_outputs). This except covers only put() itself failing.
            q.put(("ok", out))
        except Exception:  # queue unusable -> success without a payload
            q.put(("ok", None))
    except BaseException:  # noqa: BLE001 -- surface EVERY failure, never swallow it
        tb = traceback.format_exc()
        sys.stdout.write(tb)
        sys.stdout.flush()
        q.put(("error", tb))


def take_result(q, timeout):
    """Next item from ``q`` that is a RESULT, or None within ``timeout``.

    ``started`` is a clock signal rather than an outcome, and a child that starts and finishes
    inside one poll leaves both queued -- so every read has to be able to step past it.
    """
    end = time.monotonic() + timeout
    while True:
        try:
            item = q.get(timeout=max(0.0, end - time.monotonic()))
        except queue.Empty:
            return None
        if item[0] != "started":
            return item


def _drain(progress_q, current):
    """Return the last item pushed to ``progress_q`` (or ``current``), so a kill preserves the last progress."""
    try:
        while True:
            current = progress_q.get_nowait()
    except queue.Empty:
        pass
    return current


def run_forked(
    fn: Callable,
    *args,
    label: str = "",
    timeout: Optional[float] = None,
    stream_progress: bool = False,
    mp_context: Optional[str] = None,
    **kwargs,
) -> RunResult:
    """Run ``fn(*args, **kwargs)`` in a forked child; returns a failed RunResult (cause logged to stdout) on
    a fatal signal, exception, or timeout overrun, else ``ok=True`` with the picklable return value.
    ``stream_progress=True`` preserves the child's last ``progress`` snapshot even if it is later killed."""
    # fork is cheap on Linux/WSL2; spawn on macOS, where forking after numpy/BLAS threads can abort the child.
    ctx = multiprocessing.get_context(mp_context if mp_context is not None else osinfo.mp_context())
    # fork() duplicates only the calling thread, so a child entering a parallel region with the
    # parent's pool live blocks forever -- libgomp installs no pthread_atfork handler. No-op under spawn.
    pause_openmp_pools()
    q = ctx.Queue()
    progress_q = ctx.Queue() if stream_progress else None
    if progress_q is not None:
        kwargs = {**kwargs, "progress": progress_q}
    p = ctx.Process(target=_child, args=(fn, args, kwargs, q))
    tag = f"[{label}] " if label else ""
    p.start()
    last_progress = None
    # The deadline measures the CHILD'S runtime, so the child arms it by reporting that it started
    # -- not p.start(). Fork/spawn latency is the parent's cost (seconds under spawn, and on a
    # loaded box a fork can be slow to schedule too); billing it to the callee means a child that
    # takes longer to reach its first bytecode than its own timeout is SIGTERMed before it runs,
    # and every failure it was about to report is attributed to a clock it never got to start.
    started_at = time.monotonic()
    deadline = None
    # Poll so the result queue drains while the child is alive -- a payload bigger than the OS
    # pipe buffer would otherwise block the child's feeder thread forever (join-then-read deadlocks).
    poll = 0.1
    result_item = None  # (status, payload) once the child's single result is received
    while p.is_alive():
        if progress_q is not None:
            last_progress = _drain(progress_q, last_progress)
        # Until the child reports in, the ceiling is its own timeout plus the arming grace, so a
        # child that never runs at all still ends rather than hanging the parent forever.
        limit = (
            None if timeout is None else (deadline if deadline is not None else (started_at + timeout + ARM_GRACE_S))
        )
        if limit is not None and time.monotonic() >= limit:
            if result_item is not None:
                break  # child actually finished (payload already drained) -- not a timeout
            p.terminate()  # SIGTERM
            p.join(TERM_GRACE_S)
            if p.is_alive() and p.pid is not None and is_core_dumping(p.pid):
                p.join(COREDUMP_GRACE_S)  # already dying of its own signal -- wait, do not relabel it
            if p.is_alive():  # a child that ignores/blocks SIGTERM would hang the
                p.kill()  # parent on an unbounded join -- escalate to SIGKILL
                p.join()
            if progress_q is not None:
                last_progress = _drain(progress_q, last_progress)
            # The child can die of its OWN fatal signal in the window between the deadline check
            # and terminate() -- a segfaulting vendor runtime on a loaded box is exactly that race.
            # Reporting it as TIMEOUT hides the cause the caller is trying to attribute, so the
            # exit code decides: anything other than the signal we just sent is the child's own.
            # Losing that race was never about the DEADLINE (widened 0.5s -> 2s, and CI went red
            # again): the child dies on time and the kernel then takes a second or more to reap it
            # through the core_pattern helper, which is charged against the grace above.
            ec = p.exitcode
            if ec is not None and ec < 0 and -ec not in (signal.SIGTERM, signal.SIGKILL):
                break
            msg = f"{tag}timed out after {timeout}s"
            sys.stdout.write(msg + "\n")
            sys.stdout.flush()
            return RunResult(ok=False, signal="TIMEOUT", error=msg, result=last_progress)
        if result_item is None:
            try:
                item = q.get(timeout=poll)
            except queue.Empty:
                item = None
            if item is not None and item[0] == "started":
                deadline = time.monotonic() + timeout if timeout is not None else None
            elif item is not None:
                result_item = item
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
        result_item = take_result(q, _DRAIN_S)
        if result_item is None:
            return RunResult(
                ok=False,
                exit_code=ec,
                error=(
                    f"{tag}child exited {ec} with no result "
                    "(a payload the queue feeder could not deliver -- oversized or "
                    "unpicklable -- dies exactly this way)"
                ),
                result=last_progress,
            )
    status, payload = result_item
    if status == "ok":
        return RunResult(ok=True, exit_code=ec, result=payload)
    return RunResult(ok=False, exit_code=ec, error=payload, result=last_progress)
