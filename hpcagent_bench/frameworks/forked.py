# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run a callable in a forked child and SURFACE its failure (signal/traceback/timeout) instead of eating it."""

import contextlib
import contextvars
import ctypes
import enum
import multiprocessing
import multiprocessing.connection
import multiprocessing.context
import multiprocessing.forkserver
import multiprocessing.process
import multiprocessing.queues
import multiprocessing.reduction
import io
import os
import pathlib
import pickle
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, cast

if TYPE_CHECKING:
    import numpy as np

from hpcagent_bench import osinfo
from hpcagent_bench.openmp.isolation import pause_openmp_pools
from hpcagent_bench.seal import SealPlan, enter

__all__ = [
    "ABANDONED",
    "ABANDON_POLL_S",
    "ARM_GRACE_S",
    "CHILD_ENV_LOCK",
    "CONCRETE_CONTEXTS",
    "COREDUMP_GRACE_S",
    "DRAIN_S",
    "ERROR_BYTES",
    "EXCEPTION_HEADER",
    "PR_SET_PDEATHSIG",
    "RESULT_POLL_S",
    "SHM_ARG_BYTES",
    "TERM_GRACE_S",
    "ChildMessage",
    "ChildQueue",
    "ErrorReader",
    "ErrorWriter",
    "Message",
    "ProcessContext",
    "ProgressQueue",
    "ResultMessage",
    "RunResult",
    "ShmArgs",
    "abandoned_by",
    "attach_shm_array",
    "restore_call",
    "not_abandoned",
    "child_environment",
    "child_main",
    "die_with_parent",
    "drain_progress",
    "exception_header",
    "finished_result",
    "forget_inherited_forkserver",
    "forked_failure_reason",
    "is_core_dumping",
    "kill_group",
    "poll_queue",
    "process_context",
    "reparented",
    "report_without_a_thread",
    "run_command",
    "run_forked",
    "stop_child",
    "take_error",
    "take_result",
]


#: An argument array at or above this size reaches a spawned or forkserver child as a file in :func:`shm_root`, mapped
#: copy-on-write, instead of through the start pipe (a pickled copy in the parent, then another in the child).
SHM_ARG_BYTES = 64 * 1024 * 1024


def shm_root() -> str:
    """Where argument arrays are staged: ``/dev/shm`` (memory-backed) when writable, else the temp directory."""
    return "/dev/shm" if os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()


def attach_shm_array(path: str, dtype: "np.dtype", shape: tuple[int, ...], order: Literal["C", "F"]) -> "np.ndarray":
    """The staged array at ``path``, mapped copy-on-write (writes stay in this process); the file is unlinked, the
    mapping keeps its pages."""
    # Deferred: numpy loads an OpenMP runtime, which a spawned child must load only after its context is set (the
    # pickled arguments import it at this same point).
    import numpy as np  # noqa: PLC0415

    mapped = np.memmap(path, dtype=dtype, mode="c", shape=shape, order=order)
    os.unlink(path)
    return mapped.view(np.ndarray)


class ShmPickler(multiprocessing.reduction.ForkingPickler):
    """The start pickler, with every large plain array written to a :func:`shm_root` file (its path in ``staged``)."""

    def __init__(self, file: io.BytesIO, staged: list[str]) -> None:
        super().__init__(file, pickle.HIGHEST_PROTOCOL)
        self.staged = staged

    def reducer_override(
        self, obj: object
    ) -> tuple[Callable[[str, "np.dtype", tuple[int, ...], Literal["C", "F"]], "np.ndarray"], tuple[object, ...]]:
        numpy_module = sys.modules.get("numpy")  # an array means numpy is loaded; this module never imports it
        if numpy_module is None or type(obj) not in (numpy_module.ndarray, numpy_module.memmap):
            return NotImplemented
        array = cast("np.ndarray", obj)
        if array.nbytes < SHM_ARG_BYTES or array.dtype.hasobject:
            return NotImplemented
        # Fortran order is kept as the C order of the transpose; any other layout is written C-ordered.
        fortran = bool(array.flags.f_contiguous and not array.flags.c_contiguous)
        try:
            handle, path = tempfile.mkstemp(prefix="hb-arg-", suffix=".bin", dir=shm_root())
            self.staged.append(path)
            with os.fdopen(handle, "wb") as fh:
                (array.T if fortran else array).tofile(fh)
        except OSError:  # no room: the pipe carries it
            return NotImplemented
        # The dtype itself: an extension type (ml_dtypes bfloat16) spells "<V2" and would arrive as raw void.
        return attach_shm_array, (path, array.dtype, array.shape, "F" if fortran else "C")


def restore_call(
    fn: Callable[..., object], args_pickle: bytes, kwargs: dict[str, object]
) -> tuple[Callable[..., object], object, dict[str, object]]:
    """The child's ``(fn, args, kwargs)`` with ``args`` unpickled (its staged arrays mapped)."""
    return fn, pickle.loads(args_pickle), kwargs


class ShmArgs:
    """A child's ``(fn, args, kwargs)``: ``args`` pickled on construction with large arrays staged by
    :class:`ShmPickler` (before any start lock is held), the rest when the process starts; :meth:`release`
    removes what the child did not attach."""

    __slots__ = ("args_pickle", "fn", "kwargs", "staged")

    def __init__(self, fn: Callable[..., object], args: object, kwargs: dict[str, object]) -> None:
        self.fn = fn
        self.kwargs = kwargs
        self.staged: list[str] = []
        buffer = io.BytesIO()
        try:
            ShmPickler(buffer, self.staged).dump(args)
        except BaseException:
            self.release()
            raise
        self.args_pickle = buffer.getvalue()

    def __reduce__(
        self,
    ) -> tuple[
        Callable[..., tuple[Callable[..., object], object, dict[str, object]]],
        tuple[Callable[..., object], bytes, dict[str, object]],
    ]:
        # Runs inside Process.start, so the queues among the kwargs may be pickled.
        return restore_call, (self.fn, self.args_pickle, self.kwargs)

    def release(self) -> None:
        for path in self.staged:
            with contextlib.suppress(OSError):
                os.unlink(path)


class Message(enum.Enum):
    """What a message on the result queue says: the child STARTED (the stamp that arms the parent's
    deadline), it returned OK, or it raised an ERROR."""

    STARTED = enum.auto()
    OK = enum.auto()
    ERROR = enum.auto()


#: One message on the result queue: the start stamp, the child's return value (or a progress snapshot of the same
#: type; ``None`` when the queue could not take it), or its traceback text.
type ChildMessage[R] = (
    tuple[Literal[Message.STARTED], None] | tuple[Literal[Message.OK], R | None] | tuple[Literal[Message.ERROR], str]
)

#: The subset of :data:`ChildMessage` that ENDS a run; :attr:`Message.STARTED` is a clock signal, not an outcome.
type ResultMessage[R] = tuple[Literal[Message.OK], R | None] | tuple[Literal[Message.ERROR], str]

type ChildQueue[R] = multiprocessing.queues.Queue[ChildMessage[R]]
type ProgressQueue[R] = multiprocessing.queues.Queue[R]

#: The child's LAST-RESORT report channel: a raw pipe needing no feeder thread (a child after a refused seal may be
#: unable to start one).
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

#: Seconds between the parent's looks at a running child's result queue.
RESULT_POLL_S: float = 0.1

#: How long the child may take to say it started before the deadline is armed anyway. An
#: unbounded wait on a child that never runs is worse than a slightly wrong clock.
ARM_GRACE_S = 30.0

#: How long a SIGTERMed child has to exit before the parent escalates to SIGKILL.
TERM_GRACE_S = 5.0

#: Extra time for a child the kernel says is DUMPING CORE (a SIGKILL would relabel the crash as a kill).
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


@contextlib.contextmanager
def not_abandoned() -> Iterator[None]:
    """Run every child inside the block to its end even after the caller is abandoned: a reference timing it
    yields is cached for the next request."""
    token = ABANDONED.set(None)
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

    The command leads its own process group and every early end (the timeout, the abandon event, an
    exception in the caller) kills the whole group: a profile is ``perf`` over a python child over a
    forked measurement, and killing ``perf`` alone leaves the measurement on the cores.
    """
    abandoned = ABANDONED.get()
    poll = ABANDON_POLL_S if abandoned is not None else timeout
    started = time.monotonic()
    with subprocess.Popen(
        list(argv), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=cwd, start_new_session=True
    ) as proc:
        try:
            while True:
                try:
                    out, err = proc.communicate(timeout=poll)
                    break
                except subprocess.TimeoutExpired:
                    if timeout is not None and (abandoned is None or time.monotonic() - started >= timeout):
                        kill_group(proc)
                        raise subprocess.TimeoutExpired(proc.args, timeout, *proc.communicate()) from None
                    if abandoned is not None and abandoned.is_set():
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
        with pathlib.Path(f"/proc/{pid}/status").open() as fh:
            for line in fh:
                if line.startswith("CoreDumping:"):
                    return line.split(":", 1)[1].strip() == "1"
    except OSError:
        return False
    return False


@dataclass(frozen=True, slots=True)
class RunResult[ResultT]:
    """Outcome of a forked run (frozen): ``ok`` is the success signal; on failure ``signal``/``error`` name the
    cause; ``result`` is the return value, or the last progress snapshot of a killed child."""

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
    """The ``Type: message`` line of the exception that propagated (the LAST match, since chained exceptions print
    earlier headers), or ""."""
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

    Not ``parent_now == 1``: a sealed worker (:mod:`hpcagent_bench.cluster.seal_worker`) is PID 1 of its own
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


#: Serializes every child start in :func:`run_forked` with :func:`child_environment`: a child (and a forkserver
#: started on first use) copies ``os.environ`` at ``p.start()``, and the judge runs grades on several threads.
CHILD_ENV_LOCK = threading.Lock()


@contextlib.contextmanager
def child_environment(env: Mapping[str, str]) -> Iterator[None]:
    """``env`` in ``os.environ`` for the span of one child START, then the previous values again.

    A spawned child execs a fresh interpreter, and the dynamic loader reads ``LD_LIBRARY_PATH`` at exec:
    this is how a child gets an OpenMP context (:mod:`hpcagent_bench.openmp.context`) without changing the
    parent, whose own libraries are already mapped. ``Process.start`` returns only after the child has
    exec'd, so the environment is restored before anything else can see it; the lock keeps another
    thread's start from inheriting it."""
    with CHILD_ENV_LOCK:
        saved = {key: os.environ.get(key) for key in env}
        os.environ.update(env)
        try:
            yield
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class ForkServerHandle(Protocol):
    """The private handle fields of ``multiprocessing.forkserver.ForkServer`` (absent from its type stubs)."""

    _forkserver_pid: int | None
    _forkserver_address: str | None
    _forkserver_alive_fd: int | None


def forget_inherited_forkserver() -> None:
    """Forget the parent's forkserver handle inherited by a forked child, so its first forkserver start spawns its
    own server instead of ``waitpid``-ing a stranger."""
    server = cast("ForkServerHandle", multiprocessing.forkserver._forkserver)  # the module's one ForkServer
    server._forkserver_pid = None
    server._forkserver_address = None
    server._forkserver_alive_fd = None


def release_inherited_child_env_lock() -> None:
    """A fresh :data:`CHILD_ENV_LOCK` in a forked child: the parent forks while holding it."""
    global CHILD_ENV_LOCK  # noqa: PLW0603 -- the inherited copy stays locked forever
    CHILD_ENV_LOCK = threading.Lock()


os.register_at_fork(after_in_child=forget_inherited_forkserver)
os.register_at_fork(after_in_child=release_inherited_child_env_lock)


def process_context(method: str) -> ProcessContext:
    """The start-method context named by ``method``, checked to carry the process factory this
    module forks through. An unknown method raises out of ``get_context`` itself."""
    ctx = multiprocessing.get_context(method)
    if not isinstance(ctx, CONCRETE_CONTEXTS):  # unreachable: the three are every named method
        raise RuntimeError(f"start method {method!r} resolves to a context with no Process")
    return ctx


def report_without_a_thread(err_w: ErrorWriter, text: str) -> None:
    """Write ``text`` to the raw error pipe (no queue, no feeder thread). The only reporting path
    after a REFUSED seal (see :data:`ErrorWriter`); never raises, so the real cause is not replaced
    by a failure to report it."""
    with contextlib.suppress(OSError, ValueError):  # pipe closed, or the parent is already gone
        err_w.send_bytes(text.encode("utf-8", "replace")[:ERROR_BYTES])


def child_main[ResultT](
    call: tuple[Callable[..., ResultT], tuple[object, ...], dict[str, object]],  # a ShmArgs arrives as this
    q: ChildQueue[ResultT],
    seal: SealPlan | None,
    err_w: ErrorWriter,
    tag: str = "",
) -> None:
    fn, args, kwargs = call
    die_with_parent()
    try:
        if seal is not None:  # before the queue's feeder thread starts: a user namespace wants one thread
            enter(seal)
    except BaseException:  # noqa: BLE001 -- a refused seal is surfaced like any other child failure
        # The pipe, not the queue: this child may be unable to start a feeder thread.
        report_without_a_thread(err_w, traceback.format_exc())
        return
    # First act, before any work: this is what arms the parent's deadline (see run_forked).
    q.put((Message.STARTED, None))
    try:
        out = fn(*args, **kwargs)
        try:
            # put() only enqueues; pickling runs in the feeder thread, so an unpicklable or oversized
            # payload surfaces in the parent as "child exited 0 with no result" (large payloads
            # spill to disk: native_call.spill_outputs). This except covers put() itself failing.
            q.put((Message.OK, out))
        except Exception:  # noqa: BLE001 -- queue unusable -> success without a payload
            q.put((Message.OK, None))
    except BaseException:  # noqa: BLE001 -- surface EVERY failure, never swallow it
        tb = traceback.format_exc()
        sys.stdout.write(f"{tag}{tb}")  # the label names which call failed in a shared log
        sys.stdout.flush()
        q.put((Message.ERROR, tb))


def take_result[ResultT](q: ChildQueue[ResultT], timeout: float) -> ResultMessage[ResultT] | None:
    """Next item from ``q`` that is a RESULT, or None within ``timeout``; steps past :attr:`Message.STARTED`,
    which a child that starts and finishes inside one poll leaves queued."""
    end = time.monotonic() + timeout
    while True:
        try:
            item = q.get(timeout=max(0.0, end - time.monotonic()))
        except queue.Empty:
            return None
        if item[0] is not Message.STARTED:
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
    if result_item[0] is Message.OK:
        return RunResult(ok=True, exit_code=ec, result=result_item[1])
    return RunResult(ok=False, exit_code=ec, error=result_item[1], result=last_progress)


def run_forked[ResultT](
    fn: Callable[..., ResultT],
    *args: object,
    label: str = "",
    timeout: float | None = None,
    stream_progress: bool = False,
    mp_context: str | None = None,
    seal: SealPlan | None = None,
    env: Mapping[str, str] | None = None,
    on_start: Callable[[int], None] | None = None,
    held_s: Callable[[], float] | None = None,
    **kwargs: object,
) -> RunResult[ResultT]:
    """Run ``fn(*args, **kwargs)`` in a forked child; a failed RunResult (cause logged) on a fatal signal, exception
    or timeout overrun, else ``ok=True`` with the picklable return value.

    ``stream_progress`` keeps the last ``progress`` snapshot of a killed child; ``seal`` seals the child first; a
    non-empty ``env`` forces ``spawn`` (only a fresh interpreter rereads ``LD_LIBRARY_PATH``); ``on_start`` gets
    the pid; ``held_s`` reports seconds the caller held the child stopped, which extend the deadline."""
    tag = f"[{label}] " if label else ""
    abandoned = ABANDONED.get()
    if abandoned is not None and abandoned.is_set():
        return RunResult(ok=False, signal="ABANDONED", error=f"{tag}abandoned before it started")
    # fork is cheap on Linux/WSL2; spawn on macOS, where forking after numpy/BLAS threads can abort the child.
    ctx = process_context("spawn" if env else mp_context if mp_context is not None else osinfo.mp_context())
    # fork() duplicates only the calling thread, so a child entering a parallel region with the
    # parent's pool live blocks forever -- libgomp installs no pthread_atfork handler. No-op under spawn.
    pause_openmp_pools()
    q: ChildQueue[ResultT] = ctx.Queue()
    progress_q: ProgressQueue[ResultT] | None = ctx.Queue() if stream_progress else None
    call_kwargs: dict[str, object] = dict(kwargs)
    if progress_q is not None:
        call_kwargs["progress"] = progress_q
    err_r, err_w = ctx.Pipe(duplex=False)
    call: tuple[object, ...] | ShmArgs = (fn, args, call_kwargs)
    try:
        # Large arrays reach a spawned or forkserver child as files, not through the start pipe; staged here, so
        # another slot's start never waits on this write under the lock below.
        if not isinstance(ctx, multiprocessing.context.ForkContext):
            call = ShmArgs(fn, args, call_kwargs)
        p = ctx.Process(target=child_main, args=(call, q, seal, err_w, tag))
        # Every start holds the lock: a fork, or a forkserver started on first use, copies ``os.environ`` too, and
        # another thread's context environment (an llvm ``LD_LIBRARY_PATH``) would otherwise stick to it.
        with child_environment(env) if env else CHILD_ENV_LOCK:
            p.start()
    except BaseException:
        # No child: its pipe and queues go now (the finally below unlinks staged arrays only after a start).
        err_r.close()
        err_w.close()
        for queue in (q, progress_q):
            if queue is not None:
                queue.close()
        if isinstance(call, ShmArgs):
            call.release()
        raise
    try:
        # The parent's copy of the write end goes NOW, so the read end sees EOF as soon as the child is
        # gone rather than blocking on a writer that only this process still holds.
        err_w.close()
        if on_start is not None and p.pid is not None:
            on_start(p.pid)
        last_progress: ResultT | None = None
        # The deadline measures the CHILD'S runtime: the child arms it by reporting that it started,
        # so fork/spawn latency is not billed to the callee. Until it reports in, the ceiling is its own
        # timeout plus the arming grace, so a child that never runs at all still ends.
        limit = None if timeout is None else time.monotonic() + timeout + ARM_GRACE_S
        # Poll so the result queue drains while the child is alive -- a payload bigger than the OS
        # pipe buffer would otherwise block the child's feeder thread forever (join-then-read deadlocks).
        poll = RESULT_POLL_S
        #: The child's single result message, once received.
        result_item: ResultMessage[ResultT] | None = None
        while p.is_alive():
            if abandoned is not None and abandoned.is_set():
                p.kill()
                p.join()
                return RunResult(ok=False, signal="ABANDONED", error=f"{tag}abandoned: nobody reads its result")
            if progress_q is not None:
                last_progress = drain_progress(progress_q, last_progress)
            if limit is not None and time.monotonic() >= limit + (held_s() if held_s is not None else 0.0):
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
            if item is not None and item[0] is Message.STARTED:
                limit = None if timeout is None else time.monotonic() + timeout
            elif item is not None:
                result_item = item
        if progress_q is not None:
            last_progress = drain_progress(progress_q, last_progress)
        return finished_result(p.exitcode, q, err_r, tag, result_item, last_progress)
    finally:
        if isinstance(call, ShmArgs):
            call.release()
