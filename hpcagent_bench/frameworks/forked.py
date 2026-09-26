# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run a callable in a forked child and SURFACE its failure (signal/traceback/timeout) instead of eating it."""

import contextlib
import contextvars
import ctypes
import multiprocessing
import multiprocessing.connection
import multiprocessing.context
import multiprocessing.process
import multiprocessing.queues
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from hpcagent_bench import osinfo
from hpcagent_bench.isolation import pause_openmp_pools
from hpcagent_bench.seal import SealPlan, enter

#: One message on the result queue: the start stamp that arms the parent's deadline, the child's
#: return value (``None`` when the queue could not take the real one), or its traceback text. ``R`` is
#: whatever the callable returns; a progress snapshot stands in for that return value (the best-so-far a
#: killed child would have returned), so it is the same type.
type ChildMessage[R] = tuple[Literal["started"], None] | tuple[Literal["ok"], R | None] | tuple[Literal["error"], str]

#: The subset of :data:`ChildMessage` that ENDS a run; ``started`` is a clock signal, not an outcome.
type ResultMessage[R] = tuple[Literal["ok"], R | None] | tuple[Literal["error"], str]

type ChildQueue[R] = multiprocessing.queues.Queue[ChildMessage[R]]
type ProgressQueue[R] = multiprocessing.queues.Queue[R]

#: The child's LAST-RESORT report channel: a raw pipe, not a queue. Queue.put needs a feeder
#: thread, and a child after a refused seal (user namespace, no id map, uid 65534 at its
#: RLIMIT_NPROC) cannot start one. ``Connection.send_bytes`` writes straight to the fd with no
#: thread and survives being passed to a spawned child.
type ErrorWriter = multiprocessing.connection.Connection
type ErrorReader = multiprocessing.connection.Connection

#: Cap on that report, so a runaway traceback cannot fill the pipe and block the dying child.
ERROR_BYTES = 64 * 1024

#: A start-method context that can fork a process. ``get_context(method)`` is typed as the BaseContext
#: those three derive from, which declares no ``Process``.
type ProcessContext = (
    multiprocessing.context.ForkContext
    | multiprocessing.context.SpawnContext
    | multiprocessing.context.ForkServerContext
)
CONCRETE_CONTEXTS = (
    multiprocessing.context.ForkContext,
    multiprocessing.context.SpawnContext,
    multiprocessing.context.ForkServerContext,
)

#: Grace period (seconds) to drain the result queue after the child exits cleanly.
DRAIN_S = 5.0

#: How long the child may take to say it started before the deadline is armed anyway. An
#: unbounded wait on a child that never runs is worse than a slightly wrong clock.
ARM_GRACE_S = 30.0

#: How long a SIGTERMed child has to exit before the parent escalates to SIGKILL.
TERM_GRACE_S = 5.0

#: Extra time granted to a child the kernel says is DUMPING CORE: it already took a fatal signal,
#: so a SIGKILL would only relabel the crash as a kill. A ``core_pattern`` helper can take seconds
#: on a loaded node; this only has to exceed one dump.
COREDUMP_GRACE_S = 60.0

#: Set by a caller once nobody will read this thread's result: every child :func:`run_forked` or
#: :func:`run_command` has running under it is killed, and each reports itself ABANDONED.
ABANDONED: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar("abandoned", default=None)

#: How often a parent waiting on a child checks :data:`ABANDONED`.
ABANDON_POLL_S = 0.1


@contextlib.contextmanager
def abandoned_by(event: threading.Event) -> Iterator[None]:
    """Kill every child this thread waits on inside the block once ``event`` is set."""
    token = ABANDONED.set(event)
    try:
        yield
    finally:
        ABANDONED.reset(token)


def kill_group(proc: subprocess.Popen[str]) -> None:
    """SIGKILL ``proc`` and everything it started in its process group; a group already gone is fine."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)


def run_command(
    argv: Sequence[str], *, env: Mapping[str, str] | None = None, cwd: str | None = None, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    """``subprocess.run(argv, capture_output=True, text=True, timeout=timeout)`` that, inside an
    :func:`abandoned_by` block, also stops once the event is set, returning what it printed so far.

    Inside the block the command leads its own process group and every early end kills the whole
    group: a profile is ``perf`` over a python child over a forked measurement, and killing ``perf``
    alone leaves the measurement on the cores.
    """
    abandoned = ABANDONED.get()
    if abandoned is None:
        return subprocess.run(
            list(argv), capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout, check=False
        )
    started = time.monotonic()
    with subprocess.Popen(
        list(argv), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=cwd, start_new_session=True
    ) as proc:
        try:
            while True:
                try:
                    out, err = proc.communicate(timeout=ABANDON_POLL_S)
                    break
                except subprocess.TimeoutExpired:
                    if timeout is not None and time.monotonic() - started >= timeout:
                        kill_group(proc)
                        raise subprocess.TimeoutExpired(proc.args, timeout, *proc.communicate()) from None
                    if abandoned.is_set():
                        kill_group(proc)
                        out, err = proc.communicate()
                        break
        except BaseException:
            kill_group(proc)
            raise
    return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)


def is_core_dumping(pid: int) -> bool:
    """True when the kernel reports ``pid`` is writing a core image (Linux >= 4.15); False when
    that is not knowable (another OS, a reaped pid, a hidepid mount), so the caller escalates."""
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("CoreDumping:"):
                    return line.split(":", 1)[1].strip() == "1"
    except OSError:
        return False
    return False


@dataclass(frozen=True, slots=True)
class RunResult[ResultT]:
    """Outcome of a forked run (frozen, so a RunResult of a concrete payload type reads as one of any
    wider type, as forked_failure_reason and the OOM classifier do): ``ok`` is the success signal; on failure ``signal``/``error`` name the
    cause (see :func:`forked_failure_reason`); ``result`` carries the picklable return value, or the
    last streamed progress snapshot when the child was killed before returning."""

    ok: bool
    exit_code: int | None = None
    signal: str | None = None
    error: str | None = None
    result: ResultT | None = None


#: An un-indented ``ExceptionType: message`` line inside a :func:`traceback.format_exc` text: frames
#: are indented, and continuation lines of a multi-line message (SQLAlchemy's ``[SQL: ...]`` dump,
#: a doc-link URL) do not start with an identifier followed by ``:``.
EXCEPTION_HEADER = re.compile(r"^[A-Za-z_][\w.]*:\s")


def exception_header(traceback_text: str) -> str:
    """The raised exception's ``Type: message`` line out of a :func:`traceback.format_exc` text, or
    "" if none matches. Not necessarily the LAST line: a chained exception ("... the direct cause of
    the following exception") prints an earlier header too, so this takes the LAST MATCH, which is
    the header of the exception that actually propagated."""
    header = ""
    for line in traceback_text.splitlines():
        if EXCEPTION_HEADER.match(line):
            header = line
    return header


def forked_failure_reason(r: RunResult[object]) -> str:
    """One-line cause for a failed :class:`RunResult`: signal name, else the raised exception's type
    and message (:func:`exception_header`), else the raw text's last line (a non-traceback message,
    e.g. a timeout or an abandoned-child notice), else "unknown"."""
    if r.signal:
        return r.signal
    if not r.error:
        return "unknown"
    return exception_header(r.error) or r.error.strip().splitlines()[-1]


#: ``prctl`` option number for PR_SET_PDEATHSIG (asm-generic, stable across Linux architectures).
PR_SET_PDEATHSIG = 1


def reparented(parent_at_entry: int, parent_now: int) -> bool:
    """True when the pid that had us at entry is no longer our parent.

    Not ``parent_now == 1``: a sealed worker (:mod:`experiments.seal_worker`) is PID 1 of its own
    PID namespace, so its children legitimately read ``getppid() == 1``.
    """
    return parent_at_entry != parent_now


def die_with_parent() -> None:
    """Ask the kernel to SIGKILL this child when its parent dies. Linux only; best effort.

    An orphaned child keeps every inherited descriptor alive (a judge's, or pytest-xdist's execnet
    pipe, which then hangs the session). The pre/post-prctl getppid comparison (:func:`reparented`)
    covers a parent that died before prctl ran.
    """
    if not osinfo.IS_LINUX:
        return
    parent_at_entry = os.getppid()
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(PR_SET_PDEATHSIG, ctypes.c_ulong(signal.SIGKILL), 0, 0, 0) != 0:
            return
    except (OSError, AttributeError, ValueError):
        return  # no prctl (musl, a sandbox, a non-Linux kernel claiming linux)
    if reparented(parent_at_entry, os.getppid()):  # died in the gap above; the signal just armed never arrives
        os._exit(0)


def process_context(method: str) -> ProcessContext:
    """The start-method context named by ``method``, checked to carry the process factory this
    module forks through. An unknown method raises out of ``get_context`` itself."""
    ctx = multiprocessing.get_context(method)
    if not isinstance(ctx, CONCRETE_CONTEXTS):  # unreachable: the three are every named method
        raise RuntimeError(f"start method {method!r} resolves to a context with no Process")
    return ctx


def report_without_a_thread(err_w: ErrorWriter | None, text: str) -> None:
    """Write ``text`` to the raw error pipe (no queue, no feeder thread). The only reporting path
    after a REFUSED seal (see :data:`ErrorWriter`); never raises, so the real cause is not replaced
    by a failure to report it."""
    if err_w is None:
        return
    try:
        err_w.send_bytes(text.encode("utf-8", "replace")[:ERROR_BYTES])
    except (OSError, ValueError):  # pipe closed, or the parent is already gone
        pass


def child_main[ResultT](
    fn: Callable[..., ResultT],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    q: ChildQueue[ResultT],
    seal: SealPlan | None = None,
    err_w: ErrorWriter | None = None,
) -> None:
    die_with_parent()
    try:
        if seal is not None:  # before the queue's feeder thread starts: a user namespace wants one thread
            enter(seal)
    except BaseException:  # noqa: BLE001 -- a refused seal is surfaced like any other child failure
        tb = traceback.format_exc()
        if err_w is not None:
            # The pipe, not the queue: this child may be unable to start a feeder thread.
            report_without_a_thread(err_w, tb)
        else:
            # No pipe: a long-lived judge built from an older tree calls with five arguments
            # (tests/test_forked.py entry-point ABI test). Best effort through the queue.
            with contextlib.suppress(Exception):
                q.put(("error", tb))
        return
    # First act, before any work: this is what arms the parent's deadline (see run_forked).
    q.put(("started", None))
    try:
        out = fn(*args, **kwargs)
        try:
            # put() only enqueues; pickling runs in the feeder thread, so an unpicklable or oversized
            # payload surfaces in the parent as "child exited 0 with no result" (large payloads
            # spill to disk: native_call.spill_outputs). This except covers put() itself failing.
            q.put(("ok", out))
        except Exception:  # queue unusable -> success without a payload
            q.put(("ok", None))
    except BaseException:  # noqa: BLE001 -- surface EVERY failure, never swallow it
        tb = traceback.format_exc()
        sys.stdout.write(tb)
        sys.stdout.flush()
        q.put(("error", tb))


#: WIRE NAME for :func:`child_main`: forkserver/spawn pickle the target by qualified name, and a
#: long-lived judge started from an older checkout still asks for this spelling. An alias, never a
#: second body.
_child = child_main


def take_result[ResultT](q: ChildQueue[ResultT], timeout: float) -> ResultMessage[ResultT] | None:
    """Next item from ``q`` that is a RESULT, or None within ``timeout``; steps past ``started``,
    which a child that starts and finishes inside one poll leaves queued."""
    end = time.monotonic() + timeout
    while True:
        try:
            item = q.get(timeout=max(0.0, end - time.monotonic()))
        except queue.Empty:
            return None
        if item[0] != "started":
            return item


def take_error(err_r: ErrorReader, timeout: float) -> str:
    """The child's raw-pipe report, or ``""`` when it wrote none within ``timeout``.

    ``poll`` before ``recv_bytes`` because the child may have died without writing, and a bare
    ``recv_bytes`` on a pipe whose only other writer is gone would raise rather than wait."""
    try:
        if not err_r.poll(timeout):
            return ""
        return bytes(err_r.recv_bytes()).decode("utf-8", "replace")
    except (OSError, EOFError, ValueError):
        return ""


def drain_progress[ResultT](progress_q: ProgressQueue[ResultT], current: ResultT | None) -> ResultT | None:
    """Return the last item pushed to ``progress_q`` (or ``current``), so a kill preserves the last progress."""
    try:
        while True:
            current = progress_q.get_nowait()
    except queue.Empty:
        pass
    return current


def poll_queue[ResultT](q: ChildQueue[ResultT], timeout: float) -> ChildMessage[ResultT] | None:
    """The next message on ``q``, or ``None`` when none arrives within ``timeout``."""
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return None


def stop_child(p: multiprocessing.process.BaseProcess) -> None:
    """SIGTERM ``p``, then SIGKILL it once the grace period passes: a child that ignores or blocks
    SIGTERM would hang the parent on an unbounded join. A child the kernel says is dumping core already
    took a fatal signal, so it gets the core-dump grace instead of a kill that would relabel the crash."""
    p.terminate()
    p.join(TERM_GRACE_S)
    if p.is_alive() and p.pid is not None and is_core_dumping(p.pid):
        p.join(COREDUMP_GRACE_S)
    if p.is_alive():
        p.kill()
        p.join()


def finished_result[ResultT](
    ec: int | None,
    q: ChildQueue[ResultT],
    err_r: ErrorReader,
    tag: str,
    result_item: ResultMessage[ResultT] | None,
    last_progress: ResultT | None,
) -> RunResult[ResultT]:
    """The outcome of a child that exited with code ``ec``: its fatal signal, else its result message
    (drained here when the loop did not see it), else its raw-pipe report, else "no result"."""
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
        result_item = take_result(q, DRAIN_S)
    if result_item is None:
        # The raw pipe first: a child that could not use the queue (a refused seal, see ErrorWriter)
        # reported there, and native_call routes seal faults by the SealError name.
        reported = take_error(err_r, DRAIN_S)
        if reported:
            return RunResult(ok=False, exit_code=ec, error=f"{tag}{reported}", result=last_progress)
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
    if result_item[0] == "ok":
        return RunResult(ok=True, exit_code=ec, result=result_item[1])
    return RunResult(ok=False, exit_code=ec, error=result_item[1], result=last_progress)


def run_forked[**P, ResultT](
    fn: Callable[P, ResultT],
    *args: P.args,
    label: str = "",
    timeout: float | None = None,
    stream_progress: bool = False,
    mp_context: str | None = None,
    seal: SealPlan | None = None,
    **kwargs: P.kwargs,
) -> RunResult[ResultT]:
    """Run ``fn(*args, **kwargs)`` in a forked child; returns a failed RunResult (cause logged to stdout) on
    a fatal signal, exception, or timeout overrun, else ``ok=True`` with the picklable return value.
    ``stream_progress=True`` preserves the child's last ``progress`` snapshot even if it is later killed.
    ``seal`` runs the child sealed (:func:`hpcagent_bench.seal.enter`) before ``fn`` sees it."""
    tag = f"[{label}] " if label else ""
    abandoned = ABANDONED.get()
    if abandoned is not None and abandoned.is_set():
        return RunResult(ok=False, signal="ABANDONED", error=f"{tag}abandoned before it started")
    # fork is cheap on Linux/WSL2; spawn on macOS, where forking after numpy/BLAS threads can abort the child.
    ctx = process_context(mp_context if mp_context is not None else osinfo.mp_context())
    # fork() duplicates only the calling thread, so a child entering a parallel region with the
    # parent's pool live blocks forever -- libgomp installs no pthread_atfork handler. No-op under spawn.
    pause_openmp_pools()
    q: ChildQueue[ResultT] = ctx.Queue()
    progress_q: ProgressQueue[ResultT] | None = ctx.Queue() if stream_progress else None
    call_kwargs: dict[str, object] = dict(kwargs)
    if progress_q is not None:
        call_kwargs["progress"] = progress_q
    err_r, err_w = ctx.Pipe(duplex=False)
    p = ctx.Process(target=child_main, args=(fn, args, call_kwargs, q, seal, err_w))
    p.start()
    # The parent's copy of the write end goes NOW, so the read end sees EOF as soon as the child is
    # gone rather than blocking on a writer that only this process still holds.
    err_w.close()
    last_progress: ResultT | None = None
    # The deadline measures the CHILD'S runtime: the child arms it by reporting that it started,
    # so fork/spawn latency is not billed to the callee. Until it reports in, the ceiling is its own
    # timeout plus the arming grace, so a child that never runs at all still ends.
    limit = None if timeout is None else time.monotonic() + timeout + ARM_GRACE_S
    # Poll so the result queue drains while the child is alive -- a payload bigger than the OS
    # pipe buffer would otherwise block the child's feeder thread forever (join-then-read deadlocks).
    poll = 0.1
    #: The child's single result message, once received.
    result_item: ResultMessage[ResultT] | None = None
    while p.is_alive():
        if abandoned is not None and abandoned.is_set():
            p.kill()
            p.join()
            return RunResult(ok=False, signal="ABANDONED", error=f"{tag}abandoned: nobody reads its result")
        if progress_q is not None:
            last_progress = drain_progress(progress_q, last_progress)
        if limit is not None and time.monotonic() >= limit:
            if result_item is not None:
                break  # child actually finished (payload already drained) -- not a timeout
            stop_child(p)
            if progress_q is not None:
                last_progress = drain_progress(progress_q, last_progress)
            # The child can die of its OWN fatal signal between the deadline check and terminate();
            # the exit code decides: any signal other than the one just sent is the child's own.
            ec = p.exitcode
            if ec is not None and ec < 0 and -ec not in (signal.SIGTERM, signal.SIGKILL):
                break
            msg = f"{tag}timed out after {timeout}s"
            sys.stdout.write(msg + "\n")
            sys.stdout.flush()
            return RunResult(ok=False, signal="TIMEOUT", error=msg, result=last_progress)
        if result_item is not None:
            p.join(poll)
            continue
        item = poll_queue(q, poll)
        if item is not None and item[0] == "started":
            limit = None if timeout is None else time.monotonic() + timeout
        elif item is not None:
            result_item = item
    if progress_q is not None:
        last_progress = drain_progress(progress_q, last_progress)
    return finished_result(p.exitcode, q, err_r, tag, result_item, last_progress)
