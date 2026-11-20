# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``hpcagent_bench/cluster/reservations.py``: reservations parsed from ``scontrol`` output, the ones a job must avoid
(not its own, not standing, still to come), and the time and start a job gets: as asked, shortened to a minimum, or
deferred past the reservation (an experiment's minimum is its whole time)."""

import datetime
import os
import pathlib
import tempfile

import pytest

from hpcagent_bench.cluster import reservations
from tests.bash_stub import bash_stub

NOW = datetime.datetime(2026, 11, 17, 18, 0, 0)
SCONTROL = """\
ReservationName=cscs StartTime=2026-10-29T16:59:00 EndTime=2027-10-29T16:59:00 Duration=365-00:00:00 Nodes=nid[1-9] PartitionName=(null) Users=(null) Accounts=root
ReservationName=maintenance StartTime=2026-11-18T07:00:00 EndTime=2026-11-18T19:00:00 Duration=12:00:00 Nodes=nid[1-9] PartitionName=mi300 Users=root Accounts=(null)
ReservationName=mine StartTime=2026-11-17T20:00:00 EndTime=2026-11-17T21:00:00 Duration=01:00:00 Nodes=nid1 PartitionName=mi300 Users=ybudanaz Accounts=(null)
ReservationName=other-partition StartTime=2026-11-17T19:00:00 EndTime=2026-11-17T20:00:00 Duration=01:00:00 Nodes=nid1 PartitionName=mi200 Users=root Accounts=(null)
"""


def hours(count: float) -> int:
    return int(count * 3600)


def avoided(partition: str = "mi300") -> list[reservations.Reservation]:
    return reservations.blocking(reservations.parse(SCONTROL), NOW, partition, "ybudanaz", ["test-account"])


def test_only_the_reservations_a_job_could_run_into_are_avoided() -> None:
    assert [one.name for one in avoided()] == ["maintenance"], "not the standing, not mine, not another partition's"
    assert [one.name for one in avoided("mi200")] == ["other-partition"]
    root = reservations.blocking(reservations.parse(SCONTROL), NOW, "mi300", "root", [])
    assert [one.name for one in root] == ["mine"], "a reservation its user may use is no obstacle, another's is"


def test_a_job_that_ends_before_the_reservation_is_given_what_it_asked() -> None:
    result = reservations.plan(hours(6), hours(1), avoided(), NOW)  # 18:00 + 6 h + margin < 07:00 tomorrow
    assert (result.seconds, result.begin, result.blocked_by) == (hours(6), None, None)


def test_a_job_that_reaches_into_it_is_shortened_down_to_its_minimum() -> None:
    result = reservations.plan(hours(20), hours(1), avoided(), NOW)
    assert result.seconds == hours(13) - reservations.MARGIN_SECONDS and result.begin is None
    assert result.blocked_by == "maintenance"


def test_an_experiment_whose_minimum_is_its_whole_time_waits_for_the_reservation_to_end() -> None:
    result = reservations.plan(hours(16), hours(16), avoided(), NOW)
    assert result.seconds == hours(16) and result.begin == datetime.datetime(2026, 11, 18, 19, 0, 0)


def test_a_later_reservation_defers_past_the_first_too() -> None:
    chained = [
        *avoided(),
        reservations.Reservation(
            "again",
            datetime.datetime(2026, 11, 19, 6, 0),
            datetime.datetime(2026, 11, 19, 12, 0),
            "mi300",
            frozenset({"root"}),
            frozenset(),
        ),
    ]
    result = reservations.plan(hours(16), hours(16), chained, NOW)
    assert result.begin == datetime.datetime(2026, 11, 19, 12, 0, 0)


def test_the_cli_reads_scontrol_and_prints_walltime_and_begin(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listing = tmp_path / "listing"
    listing.write_text(
        SCONTROL.replace("2026-11-18T07:00:00", "2099-01-01T07:00:00").replace(
            "2026-11-18T19:00:00", "2099-01-01T19:00:00"
        )
    )
    bash_stub(tmp_path / "bin", "scontrol", f"cat {listing}")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")
    monkeypatch.setenv("USER", "ybudanaz")
    assert [one.name for one in reservations.read()] == ["cscs", "maintenance", "mine", "other-partition"]
    assert reservations.fit(hours(6), hours(6), "mi300", "ybudanaz", []).seconds == hours(6)
    assert reservations.seconds_of("1-02:00:00") == hours(26) and reservations.clock_of(hours(6)) == "06:00:00"


if __name__ == "__main__":
    test_only_the_reservations_a_job_could_run_into_are_avoided()
    test_a_job_that_ends_before_the_reservation_is_given_what_it_asked()
    test_a_job_that_reaches_into_it_is_shortened_down_to_its_minimum()
    test_an_experiment_whose_minimum_is_its_whole_time_waits_for_the_reservation_to_end()
    test_a_later_reservation_defers_past_the_first_too()
    with pytest.MonkeyPatch.context() as patch:
        test_the_cli_reads_scontrol_and_prints_walltime_and_begin(pathlib.Path(tempfile.mkdtemp()), patch)
