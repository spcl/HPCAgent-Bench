# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep one judge rank's grading upstream alive: run it, and start it again when it dies.

WHY. The upstream (``hpcagent_bench serve`` on loopback) is a plain background child of the judge
step's shell, started once and never looked at again. Nothing notices when it goes: the router in
front of it keeps answering ``/health`` with 200 and turns every grade behind it into a 502. In
641799 the judge node's memory reached 513546 MiB of 513546 MiB -- four judges on one node, each
grading an XL scientific kernel whose references live in the PARENT -- and the kernel OOM killer
took rank 4's upstream at 10:44. That rank then refused ~2000 calls over the next fourteen hours
and recorded not one row, while its three siblings on the same node kept grading: the node had
memory again one second after the kill, and only the dead process was missing.

So a restart is the fix: the rank loses the grade that was in flight and serves the next one.

A crash LOOP is a different failure and must not be papered over -- a judge that cannot bind its
port, or whose seal the host refuses, dies in seconds every time. ``--min-uptime-seconds`` splits
the two: an upstream that never reaches that age counts as a quick failure, and
``--max-quick-restarts`` of them in a row ends the supervisor non-zero, which is what the launcher
already reads as "this judge did not come up".

Standard library only, and the command is passed through verbatim: this process is a supervisor,
not a second place where the judge's argv is decided.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from types import FrameType

#: Exit code for a supervisor that gave up on a crash-looping upstream.
GAVE_UP = 1
#: How long the supervisor waits for a signalled child before it escalates to SIGKILL.
TERM_GRACE_S = 10.0


def announce(message: str) -> None:
    """One supervisor line in the rank's log, flushed: the log is a file, so it is block-buffered
    and an unflushed line is exactly the evidence a post-mortem needs and does not get."""
    print(message, flush=True)


def describe(returncode: int) -> str:
    """``rc=N`` for an exit, ``signal=NAME`` for a death -- an OOM kill reads as SIGKILL, which is
    the whole point of printing this rather than a bare number."""
    if returncode >= 0:
        return f"rc={returncode}"
    try:
        return f"signal={signal.Signals(-returncode).name}"
    except ValueError:
        return f"signal={-returncode}"


class Supervisor:
    """Runs ``command`` until it is told to stop, restarting it when it ends on its own."""

    __slots__ = ("command", "label", "min_uptime", "max_quick", "backoff", "child", "stopping")

    def __init__(self, command: Sequence[str], label: str, min_uptime: float, max_quick: int, backoff: float) -> None:
        self.command = list(command)
        self.label = label
        self.min_uptime = min_uptime
        self.max_quick = max_quick
        self.backoff = backoff
        self.child: subprocess.Popen[bytes] | None = None
        self.stopping = False

    def stop(self, signum: int, _frame: FrameType | None) -> None:
        """Pass the launcher's signal to the upstream and stop supervising.

        The shell's teardown kills this process, and a child left behind would hold the loopback
        port (and its grading children) for as long as the allocation lasts."""
        self.stopping = True
        child = self.child
        if child is not None and child.poll() is None:
            child.send_signal(signum if signum in (signal.SIGINT, signal.SIGTERM) else signal.SIGTERM)

    def wait(self) -> int:
        """The child's exit status, or its terminating signal as a negative number."""
        child = self.child
        if child is None:  # only called between spawn and reap
            raise RuntimeError("judge_upstream: waited with no child running")
        while True:
            try:
                return child.wait()
            except KeyboardInterrupt:  # SIGINT reached the wait, not the handler
                self.stop(signal.SIGINT, None)

    def reap(self) -> None:
        """After a stop request: give the child the grace it was promised, then SIGKILL it."""
        child = self.child
        if child is None or child.poll() is not None:
            return
        try:
            child.wait(timeout=TERM_GRACE_S)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()

    def run(self) -> int:
        """Supervise until stopped (0) or until the upstream crash-loops (:data:`GAVE_UP`)."""
        quick = 0
        while not self.stopping:
            started = time.monotonic()
            self.child = subprocess.Popen(self.command)
            status = self.wait()
            uptime = time.monotonic() - started
            if self.stopping:
                self.reap()
                return 0
            quick = quick + 1 if uptime < self.min_uptime else 0
            announce(
                f"judge upstream {self.label} exited {describe(status)} after {uptime:.0f}s "
                f"(quick failures in a row: {quick})"
            )
            if quick >= self.max_quick:
                announce(f"judge upstream {self.label} keeps failing within {self.min_uptime:.0f}s; giving up")
                return GAVE_UP
            announce(f"judge upstream {self.label} restarting in {self.backoff:.0f}s")
            time.sleep(self.backoff)
        return 0


def parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a judge upstream and restart it when it dies.")
    parser.add_argument("--label", default="", help="how this rank is named in the log lines")
    parser.add_argument("--min-uptime-seconds", type=float, default=60.0)
    parser.add_argument("--max-quick-restarts", type=int, default=3)
    parser.add_argument("--backoff-seconds", type=float, default=5.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = parse(argv)
    command = [str(word) for word in args.command]
    command = command[1:] if command[:1] == ["--"] else command
    if not command:
        raise SystemExit("judge_upstream: no command after --")
    supervisor = Supervisor(
        command,
        args.label or f"pid {os.getpid()}",
        float(args.min_uptime_seconds),
        int(args.max_quick_restarts),
        float(args.backoff_seconds),
    )
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, supervisor.stop)
    return supervisor.run()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
