# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Read the results DB for the speedup figures: the shared selector / filter path
(:func:`load_results`), the per-machine partition (:func:`machine_groups`) and the per-cell median
with its bootstrap CI (:func:`cell_summary`). ``statistics/plot_speedup.py`` draws from these.

The DB is the ``results`` table of the SQLite results DB (``results/hpcagent_bench.db`` by default,
written by the collection sweeps in :mod:`hpcagent_bench.support.collect`), read through the stdlib
``sqlite3`` so reporting never pulls in the framework stack. The plot renders headless (``Agg``);
``text.usetex`` is set per call (:func:`set_usetex`).
"""

import dataclasses
import math
import pathlib
import re
import sqlite3
from dataclasses import dataclass
from typing import cast

import matplotlib
import pandas as pd  # pyright: ignore[reportMissingTypeStubs] -- pandas ships none
from matplotlib.figure import Figure

from hpcagent_bench.harness import recording
from hpcagent_bench.spec import select_short_names
from hpcagent_bench.stats import style, summary

__all__ = [
    "CELL_COLUMNS",
    "CI_SEED",
    "DEFAULT_BASELINE",
    "CellSummary",
    "cell_summary",
    "filter_datatype",
    "fold_build_axes",
    "fold_variant",
    "load_results",
    "machine_groups",
    "machine_label",
    "machine_output",
    "read_results_table",
    "save_figure",
    "set_usetex",
]


#: Seed for every per-cell bootstrap so the same DB yields the same published figure.
CI_SEED: int = 0

#: The speedup denominator. Named here because it is not just another series: every ratio
#: divides by it, so it has to survive :func:`load_results` under its own name.
#:
#: Overridable because numpy is not always AVAILABLE as one: a reference with a loop-carried
#: dependence is a Python loop, too slow to time at XL, so most llr40 kernels have no numpy
#: XL row.
#:
#: The default is ``numba``, the loop-level tracks' graded denominator. Every function that divides
#: takes a ``baseline`` argument: the denominator is a property of the figure, not the process.
DEFAULT_BASELINE: str = "numba"


def set_usetex(usetex: bool) -> None:
    """Toggle LaTeX text rendering for the process. ``False`` keeps mathtext (``$...$``)
    working, so the CI superscripts still render without a LaTeX install."""
    matplotlib.rcParams["text.usetex"] = usetex


def save_figure(output: str, fig: Figure) -> str:
    """Write ``fig`` to ``output`` through :func:`style.save` (undated, fixed dpi) in ``output``'s format."""
    path = pathlib.Path(output)
    style.save(fig, path, formats=(path.suffix.lstrip(".") or "png",))
    return output


def load_results(
    db: str | None,
    benchmark: str = "all",
    preset: str = "S",
    datatype: str = "float64",
    variant: str | None = None,
    baseline: str = DEFAULT_BASELINE,
) -> pd.DataFrame:
    """Read + filter the ``results`` table into the per-sample frame both figures consume.

    Applies the shared selection (kernel / track / dwarf / ``@lvl<n>`` via
    :func:`select_short_names`), drops undomained / unvalidated rows, filters to ``datatype``
    (legacy NULL treated float64) and ``preset``, folds the sparse ``variant`` axis into the
    ``benchmark`` name (``benchmark/variant``) and the ``flavor`` / ``build`` axes into the
    ``framework`` name (``dace_cpu/canonicalize/extended``). One row per timed sample survives,
    with columns ``benchmark``, ``domain``, ``framework``, ``time``, plus the ``cpu`` / ``gpu``
    machine axes, which are deliberately NOT folded -- see :func:`machine_groups`.
    """
    data = read_results_table(db)
    data = data.drop(["timestamp"], axis=1).reset_index(drop=True)

    if benchmark != "all":
        keep: set[str] = set(select_short_names(benchmark))
        data = data.loc[data["benchmark"].isin(list(keep))].reset_index(drop=True)

    data = data.loc[data["domain"] != ""]
    data = data.loc[data["validated"].eq(True)]
    data = data.drop(["validated"], axis=1).reset_index(drop=True)

    data = fold_build_axes(fold_variant(filter_datatype(data, datatype, db), variant), baseline)
    data = data.loc[data["preset"] == preset]
    data = data.drop(["preset"], axis=1).reset_index(drop=True)
    return data


def read_results_table(db: str | None) -> pd.DataFrame:
    """The raw ``results`` table of ``db`` (default: the base DB), aggregated from its shards first."""
    # A distributed run leaves one DB per rank and no merged file until something asks for it; this
    # is that ask, so plotting a sharded run needs no separate aggregation step.
    target = db if db is not None else recording.base_db_path()
    aggregate = recording.ensure_aggregated(target)
    # sqlite3.connect CREATES an absent file, so a run that recorded nothing reaches the query with
    # an empty DB and dies on a bare "no such table: results" naming neither the path it opened nor
    # the shards it looked for. The shards are the authoritative writes (recording.db_path) and the
    # base is the cache built from them, so "no shard beside the base" is the diagnosis worth
    # printing: it says the RUN leg wrote nothing, which is never a plotting bug.
    if not recording.table_exists(aggregate, "results"):
        shards = recording.shard_paths(target)
        raise RuntimeError(
            f"no results table in {aggregate!r}. Shards beside it: {shards or 'none'}. "
            f"A run records into its own shard (hpcagent_bench<N>.db) and the base is "
            f"rebuilt from those, so an absent table means the run leg recorded no rows "
            f"-- check that leg, not the plot."
        )
    conn = sqlite3.connect(aggregate)
    data: pd.DataFrame = pd.read_sql_query("SELECT * FROM results", conn)
    conn.close()
    return data


def filter_datatype(data: pd.DataFrame, datatype: str, db: str | None) -> pd.DataFrame:
    """``data``'s ``datatype`` rows (legacy NULL read as float64), the column dropped."""
    if "datatype" in data.columns:
        legacy_mask = data["datatype"].isna()
        data.loc[legacy_mask, "datatype"] = "float64"
        data = data.loc[data["datatype"] == datatype]
        return data.drop(["datatype"], axis=1).reset_index(drop=True)
    if datatype != "float64":
        raise RuntimeError(f"{db} predates the datatype column; cannot filter to --datatype={datatype}.")
    return data


def fold_variant(data: pd.DataFrame, variant: str | None) -> pd.DataFrame:
    """``data`` restricted to ``variant`` (plus variant-less rows), the variant folded into ``benchmark``."""
    if "variant" not in data.columns:
        return data
    if variant is not None:
        data = data.loc[(data["variant"].isna()) | (data["variant"] == variant)]
    sparse_mask = data["variant"].notna()
    data.loc[sparse_mask, "benchmark"] = (
        data.loc[sparse_mask, "benchmark"].astype(str) + "/" + data.loc[sparse_mask, "variant"].astype(str)
    )
    return data.drop(["variant"], axis=1).reset_index(drop=True)


def fold_build_axes(data: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """``data`` with the ``flavor`` and ``build`` axes folded into every non-baseline ``framework``."""
    # `flavor` and `build` fold into `framework` exactly as `variant` folds into `benchmark` above:
    # they are stored apart so the DB can be queried on either axis, and joined here because a
    # figure plots one series per column. Without the fold, dace_cpu's three optimizers -- and the
    # same optimizer measured on two DaCe trees -- would silently average into one line.
    #
    # The baseline never folds. `record.build` is a property of the deployment, so the launcher sets
    # it once and every framework measured under it gets stamped, baseline included -- but the
    # baseline is the DIVISOR, not a series. Fold it and one job's reference becomes `numpy/main`,
    # which is no longer the name every speedup is divided by; fold two builds and there are two
    # references and no defined denominator at all. It is also semantically empty: numpy does not
    # depend on which DaCe tree was checked out.
    for axis in ("flavor", "build"):
        if axis in data.columns:
            mask = data[axis].notna() & (data["framework"] != baseline)
            data.loc[mask, "framework"] = (
                data.loc[mask, "framework"].astype(str) + "/" + data.loc[mask, axis].astype(str)
            )
            data = data.drop([axis], axis=1).reset_index(drop=True)
    return data


def machine_label(cpu: object, gpu: object) -> str:
    """Filename-safe identity of one machine: the CPU, plus the device when one was used."""
    parts = [str(cpu)] + ([str(gpu)] if isinstance(gpu, str) and gpu else [])
    return re.sub(r"[^A-Za-z0-9._-]+", "-", "-".join(parts)).strip("-") or "unknown"


def machine_groups(data: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Split rows into one frame per ``(cpu, gpu)``: a figure may only compare one machine's runs.

    Every other axis in :func:`load_results` FOLDS -- flavor and build join the framework name so
    two pipelines read as two series in one figure. Hardware is the opposite. A speedup is a ratio
    against the numpy baseline, so a candidate timed on one node over a baseline timed on another
    is not a speedup at all, it is a hardware comparison; and nothing downstream can notice,
    because both rows are perfectly well-formed. Partition, never fold.

    Sorted by label, so one DB always yields the same files in the same order.
    """
    # Normalized FIRST: a machine with no device records it as NULL in some rows and "" in others,
    # which would be two group keys (and two figures overwriting one filename) for one machine.
    normalized: pd.DataFrame = data.assign(
        cpu=data["cpu"].fillna("").astype(str),
        gpu=data["gpu"].fillna("").astype(str),
    )
    grouped: list[tuple[str, pd.DataFrame]] = []
    for keys, rows in normalized.groupby(["cpu", "gpu"], dropna=False):
        cpu, gpu = cast("tuple[str, str]", keys)
        grouped.append((machine_label(cpu, gpu), rows.drop(["cpu", "gpu"], axis=1).reset_index(drop=True)))
    return sorted(grouped, key=lambda pair: pair[0])


def machine_output(output: str, label: str) -> str:
    """``plots/heatmap.pdf`` -> ``plots/heatmap.<machine>.pdf``.

    Suffixed ALWAYS, even when the DB holds one machine: a fixed filename would silently change
    meaning the day a second machine's rows land beside the first, which is exactly the mix-up the
    split exists to prevent.
    """
    path = pathlib.Path(output)
    return str(path.with_name(f"{path.stem}.{label}{path.suffix}"))


@dataclass(frozen=True, slots=True)
class CellSummary:
    """One ``(benchmark, domain, framework)`` cell of the :func:`cell_summary` frame.

    :ivar time: the outlier-cleaned median, which is both the plotted value and the one
        best-selection reads.
    :ivar ci_perc: the bootstrap CI width as a percent of that median.
    """

    benchmark: str
    domain: str
    framework: str
    time: float
    ci_low: float
    ci_high: float
    ci_perc: float


#: Column order of the :func:`cell_summary` frame, which is :class:`CellSummary`'s field order.
CELL_COLUMNS: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(CellSummary))


def cell_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Per ``(benchmark, domain, framework)`` cell: the outlier-cleaned median and its
    bootstrap CI (:func:`hpcagent_bench.stats.summary.median_ci`, which warns -- naming the cell -- on
    each dropped sample). Returns columns ``benchmark, domain, framework, time, ci_low,
    ci_high, ci_perc`` where ``time`` is the cleaned median (used for best-selection AND the
    plotted value) and ``ci_perc`` is the CI width as a percent of that median."""
    rows: list[CellSummary] = []
    for keys, g in data.groupby(["benchmark", "domain", "framework"], dropna=False):
        b, dom, fw = cast("tuple[str, str, str]", keys)
        med, lo, hi = summary.median_ci(g["time"].to_numpy(), label=f"{b}@{fw}", seed=CI_SEED)[:3]
        perc = ((hi - lo) / med * 100.0) if (med != 0.0 and not math.isnan(med)) else 0.0
        rows.append(CellSummary(b, dom, fw, med, lo, hi, perc))
    return pd.DataFrame([dataclasses.asdict(r) for r in rows], columns=list(CELL_COLUMNS))
