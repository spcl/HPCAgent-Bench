# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One experiment's observations, from judge databases to the file a figure reads.

    db(s) --extract--> frame --fuse(csv...)--> frame --> .db  (and .csv)  --load--> figure

`extract` reads only the rows :mod:`hpcagent_bench.campaigns` says belong to the experiment.
`fuse` joins those live rows with frozen CSVs of jobs whose directories are gone, LIVE WINNING job
by job. `build` is the two of them plus the write, which is what a caller normally wants.

Every step is read-only with respect to its inputs, and every run stamps :data:`EXTRACTED_AT` so
two extractions of the same experiment are told apart by more than a file mtime.
"""

import argparse
import dataclasses
import datetime
import logging
import pathlib
import sqlite3
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

from hpcagent_bench import campaigns, experiments, frozen_observations, observations_extract, paths

if TYPE_CHECKING:
    import pandas as pd

LOG = logging.getLogger(__name__)

#: Column stamped with the UTC time the rows were read, ISO-8601 to the second.
EXTRACTED_AT: str = "extracted_at"

#: Column naming the experiment the rows were selected for.
EXPERIMENT_COLUMN: str = "experiment_key"

#: Columns this module adds to whatever the extractor recorded.
PROVENANCE: tuple[str, ...] = (EXTRACTED_AT, EXPERIMENT_COLUMN)


def now() -> str:
    """The extraction timestamp, UTC, seconds resolution."""
    return datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat()


@dataclasses.dataclass(frozen=True, slots=True)
class Provenance:
    """What went into a fused frame, and what did not."""

    experiment: str
    live_rows: int
    frozen_rows: int
    dropped_retired: int
    dropped_foreign: int
    arms: tuple[str, ...]
    frozen_jobs: tuple[str, ...]
    extracted_at: str

    def report(self) -> str:
        """One block naming every count, for a caller to print beside the file it just wrote."""
        lines = [
            f"experiment     {self.experiment}",
            f"extracted_at   {self.extracted_at}",
            f"live rows      {self.live_rows}",
            f"frozen rows    {self.frozen_rows} from {len(self.frozen_jobs)} job(s) whose directory is gone",
            f"dropped        {self.dropped_retired} retired, {self.dropped_foreign} not this experiment's",
            f"arms ({len(self.arms)})      {', '.join(self.arms)}",
        ]
        return "\n".join(lines)


def stamp(frame: "pd.DataFrame", experiment: str, extracted_at: str) -> "pd.DataFrame":
    """``frame`` with the provenance columns set."""
    return frame.assign(**{EXTRACTED_AT: extracted_at, EXPERIMENT_COLUMN: experiment})


def keep_owned(frame: "pd.DataFrame", selection: campaigns.Selection) -> tuple["pd.DataFrame", int, int]:
    """``frame`` cut to the arms ``selection`` owns, with the two drop counts.

    Retired and foreign are counted apart because they mean different things: a retired arm ran and
    the user took it out, a foreign one belongs to another experiment that shares a run root."""
    if frame.empty or "arm" not in frame.columns:
        return frame, 0, 0
    arms = frame["arm"].astype(str)
    mine = arms.map(lambda arm: campaigns.prefix_of(arm) in selection.prefixes)
    retired = arms.map(campaigns.dropped)
    return frame[mine & ~retired], int((mine & retired).sum()), int((~mine).sum())


def extract(selection: campaigns.Selection, frozen: pathlib.Path | None = None, **options: object) -> "pd.DataFrame":
    """Every row of the experiment: judge rows, task rows with their token totals, and the frozen
    rows of jobs whose directories are gone or unreadable.

    One extractor (:mod:`hpcagent_bench.observations_extract`), because there were two and they
    disagreed: the other wrote the plural table name into ``record`` and no ``task`` rows at all,
    so a frame from it carried no token cost and every ``record == "task"`` rule silently did
    nothing."""
    import pandas as pd

    got = observations_extract.extract(
        observations_extract.Options(
            runs=selection.run_globs(),
            benchmarks=paths.BENCHMARKS,
            focus_tag=selection.tag,
            frozen_dir=frozen,
            **options,  # type: ignore[arg-type]
        )
    )
    return pd.DataFrame(got.observations)


def check_columns(live: "pd.DataFrame", frozen: "pd.DataFrame") -> None:
    """Raise when the frozen rows do not carry the columns the live ones do.

    Concatenating mismatched frames silently fills the gap with NaN, and a NaN speed-up reads as a
    kernel nobody ran rather than as a column that was never extracted."""
    if live.empty or frozen.empty:
        return
    missing = sorted(set(live.columns) - set(frozen.columns) - set(PROVENANCE))
    if missing:
        raise ValueError(f"frozen rows lack {len(missing)} live column(s): {missing}")


def fuse(
    selection: campaigns.Selection,
    live: "pd.DataFrame",
    frozen: "pd.DataFrame",
    extracted_at: str = "",
) -> tuple["pd.DataFrame", Provenance]:
    """One frame for the experiment, with a count of everything the selection left behind."""
    import pandas as pd

    extracted_at = extracted_at or now()
    check_columns(live, frozen)
    live, live_retired, live_foreign = keep_owned(live, selection)
    frozen, frozen_retired, frozen_foreign = keep_owned(frozen, selection)
    parts = [part for part in (live, frozen) if not part.empty]
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    # Stamped HERE, on the result, not on the way in: a caller that built its own frame still gets
    # provenance, and re-stamping an already-stamped frame is the same value.
    if not frame.empty:
        frame = stamp(frame, selection.experiment, extracted_at)
    jobs = tuple(sorted(frozen["job"].astype(str).unique())) if not frozen.empty else ()
    arms = tuple(sorted(frame["arm"].astype(str).unique())) if not frame.empty else ()
    return frame, Provenance(
        experiment=selection.experiment,
        live_rows=len(live),
        frozen_rows=len(frozen),
        dropped_retired=live_retired + frozen_retired,
        dropped_foreign=live_foreign + frozen_foreign,
        arms=arms,
        frozen_jobs=jobs,
        extracted_at=extracted_at,
    )


def write_db(frame: "pd.DataFrame", path: pathlib.Path) -> pathlib.Path:
    """``frame`` as the ``observations`` table of a fresh SQLite file at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as connection:
        frame.to_sql(experiments.OBSERVATIONS_TABLE, connection, index=False)
    return path


def write_csv(frame: "pd.DataFrame", path: pathlib.Path) -> pathlib.Path:
    """``frame`` as a CSV at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def load(path: pathlib.Path) -> "pd.DataFrame":
    """An observations ``.db`` or ``.csv`` as the frame a figure draws.

    One entry point for both, so a figure never learns which it was handed, and the cleaning rules
    (foreign kernel, pre-relaunch, cancelled, ``-clean`` superseded) run exactly once, here."""
    return experiments.read_observations(path)


def build(
    experiment: str,
    out: pathlib.Path,
    csv_out: pathlib.Path | None = None,
    frozen: pathlib.Path | None = None,
    root: pathlib.Path | None = None,
    csvs: Sequence[pathlib.Path] = (),
) -> tuple["pd.DataFrame", Provenance]:
    """Extract ``experiment``, fuse any extra CSVs in, write ``out`` (and ``csv_out``)."""
    import pandas as pd

    selection = campaigns.resolve(experiment, root)
    extracted_at = now()
    live = extract(selection, frozen)
    extra = pd.concat([load(path) for path in csvs], ignore_index=True) if csvs else pd.DataFrame()
    frame, provenance = fuse(selection, live, extra, extracted_at)
    if frame.empty:
        raise SystemExit(f"no observations for experiment {experiment!r} under {list(selection.run_globs())}")
    write_db(frame, out) if out.suffix == ".db" else write_csv(frame, out)
    if csv_out is not None:
        write_csv(frame, csv_out)
    return frame, provenance


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help=f"one of: {', '.join(campaigns.experiments_available())}")
    parser.add_argument("--out", type=pathlib.Path, required=True, help="observations .db (or .csv) to write")
    parser.add_argument("--csv", dest="csv_out", type=pathlib.Path, help="also write the frame as a CSV here")
    parser.add_argument(
        "--fuse-csv",
        type=pathlib.Path,
        action="append",
        default=[],
        help="extra observations CSV to fuse in; repeatable",
    )
    parser.add_argument("--runs-root", type=pathlib.Path, help=f"default {campaigns.runs_root()}")
    parser.add_argument(
        "--frozen-observations",
        default=None,
        help=f"frozen extraction dir; default ${frozen_observations.ENV}, '' to read none",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    frame, provenance = build(
        args.experiment,
        args.out,
        csv_out=args.csv_out,
        frozen=frozen_observations.resolve(args.frozen_observations),
        root=args.runs_root,
        csvs=tuple(args.fuse_csv),
    )
    print(provenance.report())
    LOG.debug("fused frame: %d rows", len(frame))
    print(f"wrote          {args.out}" + (f" and {args.csv_out}" if args.csv_out else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
