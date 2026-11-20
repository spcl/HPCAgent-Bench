# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Slurm reservations a job must not run into: a job that is still running when a reservation it is not part of
starts is killed, and one whose ``--time`` reaches into it never starts ("ReqNodeNotAvail, Reserved for maintenance").

``fit`` gives a job the time it can have before the next such reservation: as requested when it fits, shortened down to a
minimum (a prepare or golden job resumes from its caches, so a shorter job is useful), else started after the
reservation (``--begin``). An experiment job's minimum is its requested time: its limits are fixed and a job that ends
before an agent had its wall clock would make the episode its own control, so it is deferred, never shortened.
Reservations longer than :data:`STANDING_DAYS` (a standing partition reservation) and those the submitter may use are
ignored; so is one already running, which no job of ours starts into.

    python -m hpcagent_bench.cluster.reservations fit --time HH:MM:SS --min HH:MM:SS [--partition P] [--account A]
        prints  walltime=<HH:MM:SS>  and, when the job must wait,  begin=<YYYY-MM-DDTHH:MM:SS>
"""

import argparse
import dataclasses
import datetime
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence

__all__ = ["MARGIN_SECONDS", "STANDING_DAYS", "Plan", "Reservation", "fit", "main", "parse", "plan", "read"]

#: A reservation longer than this holds nodes for a partition's life, not for a maintenance window.
STANDING_DAYS = 7
#: Left between the end of a job and the start of a reservation.
MARGIN_SECONDS = 600
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
FIELD = re.compile(r"(\w+)=(\S*)")


@dataclasses.dataclass(frozen=True, slots=True)
class Reservation:
    name: str
    start: datetime.datetime
    end: datetime.datetime
    partition: str
    users: frozenset[str]
    accounts: frozenset[str]

    def usable_by(self, user: str, accounts: Iterable[str]) -> bool:
        return user in self.users or bool(self.accounts & set(accounts))

    def holds(self, partition: str) -> bool:
        """Whether it may hold nodes of ``partition``: it names it, or names none."""
        return self.partition in ("", "(null)") or not partition or partition in self.partition.split(",")


@dataclasses.dataclass(frozen=True, slots=True)
class Plan:
    """The time a job gets and when it starts: ``begin`` None = at once; ``blocked_by`` names the reservation that
    shortened or deferred it."""

    seconds: int
    begin: datetime.datetime | None
    blocked_by: str | None


def parse(text: str) -> list[Reservation]:
    """The reservations of ``scontrol --oneliner show reservation`` output (``SLURM_TIME_FORMAT=standard``)."""
    found: list[Reservation] = []
    for line in text.splitlines():
        fields = dict(FIELD.findall(line))
        if "ReservationName" not in fields:
            continue
        found.append(
            Reservation(
                name=fields["ReservationName"],
                start=datetime.datetime.strptime(fields["StartTime"], TIME_FORMAT),
                end=datetime.datetime.strptime(fields["EndTime"], TIME_FORMAT),
                partition=fields.get("PartitionName", ""),
                users=frozenset(fields.get("Users", "").split(",")),
                accounts=frozenset(fields.get("Accounts", "").split(",")),
            )
        )
    return found


def read() -> list[Reservation]:
    """The cluster's reservations; none when ``scontrol`` is absent or fails (no Slurm, nothing to avoid)."""
    try:
        shown = subprocess.run(
            ["scontrol", "--oneliner", "show", "reservation"],
            capture_output=True,
            text=True,
            check=False,
            env=os.environ | {"SLURM_TIME_FORMAT": "standard"},
        )
    except FileNotFoundError:
        return []
    return parse(shown.stdout) if shown.returncode == 0 else []


def blocking(
    reservations: Iterable[Reservation], now: datetime.datetime, partition: str, user: str, accounts: Sequence[str]
) -> list[Reservation]:
    """The reservations a job of ``user`` on ``partition`` must avoid, earliest first."""
    return sorted(
        (
            one
            for one in reservations
            if one.start > now
            and one.end - one.start <= datetime.timedelta(days=STANDING_DAYS)
            and one.holds(partition)
            and not one.usable_by(user, accounts)
        ),
        key=lambda one: one.start,
    )


def plan(
    requested_s: int,
    minimum_s: int,
    avoid: Sequence[Reservation],
    now: datetime.datetime,
    begin: datetime.datetime | None = None,
    margin_s: int = MARGIN_SECONDS,
) -> Plan:
    """The time and start of a job asking for ``requested_s`` seconds, at least ``minimum_s``, that must not run into
    ``avoid`` (:func:`blocking`), starting no earlier than ``begin``."""
    start = max(now, begin) if begin else now
    deferred = False
    blocked: str | None = None
    for one in avoid:
        if one.end <= start or one.start >= start + datetime.timedelta(seconds=requested_s + margin_s):
            continue
        room = int((one.start - start).total_seconds()) - margin_s
        blocked = one.name
        if room >= minimum_s:
            return Plan(min(requested_s, room), start if deferred else None, blocked)
        start, deferred = one.end, True
    return Plan(requested_s, start if deferred else None, blocked)


def fit(
    requested_s: int,
    minimum_s: int,
    partition: str,
    user: str,
    accounts: Sequence[str],
    begin: datetime.datetime | None = None,
) -> Plan:
    """:func:`plan` against the cluster's reservations as they are now."""
    now = datetime.datetime.now()  # noqa: DTZ005 -- scontrol prints local time, as does this
    return plan(requested_s, minimum_s, blocking(read(), now, partition, user, accounts), now, begin)


def seconds_of(clock: str) -> int:
    """``HH:MM:SS`` (hours may pass 24) or ``D-HH:MM:SS`` as seconds."""
    days, _, rest = clock.rpartition("-")
    hours, minutes, seconds = (int(part) for part in rest.split(":"))
    return (int(days or 0) * 24 + hours) * 3600 + minutes * 60 + seconds


def clock_of(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reservations", description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("fit", help="the --time and --begin a job gets")
    command.add_argument("--time", required=True, help="the requested HH:MM:SS")
    command.add_argument("--min", required=True, help="the least it may be given, HH:MM:SS")
    command.add_argument("--partition", default="")
    command.add_argument("--account", action="append", default=[])
    command.add_argument("--begin", help="the earliest start, YYYY-MM-DDTHH:MM:SS")
    args = parser.parse_args(argv)
    begin = datetime.datetime.strptime(args.begin, TIME_FORMAT) if args.begin else None
    result = fit(
        seconds_of(args.time), seconds_of(args.min), args.partition, os.environ.get("USER", ""), args.account, begin
    )
    print(f"walltime={clock_of(result.seconds)}")
    if result.begin is not None:
        print(f"begin={result.begin.strftime(TIME_FORMAT)}")
    if result.blocked_by is not None:
        print(f"reservation {result.blocked_by} limits this job", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
