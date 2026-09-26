# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Render the two report figures from the results DB: a speedup heatmap and a per-kernel
distribution grid.

Both read the ``results`` table from the SQLite results DB (``results/hpcagent_bench.db`` by default,
written by the collection sweeps in :mod:`hpcagent_bench.support.collect`), share the one
selector / filter path (:func:`load_results`), and lay their rows out with the one ordering
scheme (:mod:`hpcagent_bench.reporting_order`): scientific_computing grouped by dwarf, then loop_level_reasoning,
then machine_learning.

* :func:`plot_heatmap` -- the NPBench-style ``RdYlGn_r`` speedup table, now OPT-IN: no default
  flow emits it, because its ratio axis reads a 0.5x regression as a smaller event than a 1.5x
  win (``statistics/plot_speedup.py`` is the speedup figure a run plots). The per-cell median
  used for best-selection AND the bootstrap-CI superscript both come from OUTLIER-CLEANED
  samples via :func:`hpcagent_bench.stats.summary.median_ci` (which warns, naming the cell, on every
  dropped sample); NumPy's own column shows absolute runtimes.
* :func:`plot_distribution_grid` -- the full per-sample distribution per kernel as a grid of
  violin or box plots, sized to a two-column scientific-paper width.

The plot renders headless (``Agg``). ``text.usetex`` is set per call (``usetex=True`` default;
pass ``usetex=False`` where LaTeX is unavailable -- the CI superscripts still render via
matplotlib mathtext). matplotlib/pandas/SciPy are imported on demand (never at CLI ``--help``
time); the DB is read through the stdlib ``sqlite3`` so reporting never pulls in the framework
stack.
"""

import dataclasses
import logging
import math
import pathlib
import re
import sqlite3
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd  # pyright: ignore[reportMissingTypeStubs] -- pandas ships none

import matplotlib.pyplot as plt  # noqa: E402 -- must follow the package's backend setup

from matplotlib.axes import Axes  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

# scipy ships no type stubs, so what it hands back is converted explicitly at each call site.

from hpcagent_bench.stats import population
from hpcagent_bench.stats import summary  # noqa: E402
from hpcagent_bench.harness import recording  # noqa: E402
from hpcagent_bench.paths import PLOTS_DIR  # noqa: E402
from hpcagent_bench.reporting_order import BY_DWARF, GroupSpan, order_rows, row_meta_for  # noqa: E402
from hpcagent_bench.spec import select_short_names  # noqa: E402
from hpcagent_bench.stats import palette, style  # noqa: E402

LOG = logging.getLogger(__name__)

#: One timing sample, or one plotted number, per element.
FloatArray = npt.NDArray[np.float64]


class NoBaselineRows(ValueError):
    """A slice of the data holds no rows for the speedup denominator, so it has no ratios."""


#: Seed for every per-cell bootstrap so the same DB yields the same published figure.
CI_SEED: int = 0

#: These dense, many-cell figures (a heatmap, a per-kernel grid, a two-panel diagnostic) are drawn
#: at paper size, so their type and strokes are the shared print scale.
TYPE: style.TypeScale = style.PRINT_SCALE

#: The speedup denominator. Named here because it is not just another series: every ratio in the
#: heatmap divides by it, so it has to survive :func:`load_results` under its own name.
#:
#: Overridable because numpy is not always AVAILABLE as one: a reference with a loop-carried
#: dependence is a Python loop, too slow to time at XL, so most llr-focus40 kernels have no numpy
#: XL row.
#:
#: The default is ``numba``, the loop-level tracks' graded denominator. Every function that divides
#: takes a ``baseline`` argument: the denominator is a property of the figure, not the process.
DEFAULT_BASELINE: str = "numba"


def set_usetex(usetex: bool) -> None:
    """Toggle LaTeX text rendering for the process. ``False`` keeps mathtext (``$...$``)
    working, so the CI superscripts still render without a LaTeX install."""
    matplotlib.rcParams["text.usetex"] = usetex


def format_fixed(x: float, width: int) -> str:
    float_format = "{:." + f"{width}" + "f}"
    return float_format.format(x)


def column_geomean(x: pd.Series) -> float:
    """The column's geomean over the cells that carry a ratio; NaN when none does.

    NOT :func:`scipy.stats.mstats.gmean`: its ``log(0)`` sends the WHOLE column to 0.0, so one
    unmeasured cell entered as a zero would read as a total collapse.
    :func:`~hpcagent_bench.stats.summary.usable_ratios` drops that cell.

    A dropped cell is NAMED, because the useful question is which kernel produced a zero and why.
    ``x`` is indexed by kernel, so the warning quotes ``kernel=value`` for each one.
    """
    present = x.dropna()
    values: FloatArray = present.to_numpy(dtype=np.float64)
    usable = summary.usable_ratios(values, warn=False)
    if usable.size != values.size:
        rejected = present[(~np.isfinite(values)) | (values <= 0.0)]
        named = ", ".join(f"{kernel}={value:g}" for kernel, value in rejected.items())
        message = (
            f"geomean: dropped {rejected.size} cell(s) that are not finite positive ratios: {named}. "
            f"A zero here is an UNMEASURED cell, not a total regression -- find out why it is zero "
            f"rather than plotting it. It is excluded from this geomean."
        )
        LOG.warning("%s", message)
        warnings.warn(message, stacklevel=2)
    return summary.geomean(usable) if usable.size else float("nan")


def abbreviate_speedup(x: float) -> str:
    """Short speedup label with an up/down indicator."""
    if math.isnan(x):
        return ""
    prefix = "^" if x < 1 else ("v" if x > 1 else "")
    value = 1 / x if x < 1 else x
    if value > 100:
        value = float(int(value))  # above 100x the fraction is noise, so the label drops it
    if value > 1000:
        return prefix + format_fixed(value / 1000, 1) + "k"
    return prefix + format_fixed(value, 1)


def abbreviate_runtime(x: float) -> str:
    """Short runtime label; DB times are in milliseconds."""
    if math.isnan(x):
        return ""
    if x >= 1000:
        return format_fixed(x / 1000, 2) + " s"
    return format_fixed(x, 2) + " ms"


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


def one_node_per_kernel(data: pd.DataFrame) -> None:
    """Refuse a kernel whose candidate and baseline rows name two different nodes.

    :func:`machine_groups` partitions on ``(cpu, gpu)``, which cannot separate two nodes of one
    homogeneous cluster; every speedup here divides one kernel's framework cells by its baseline
    cell, so all of a kernel's rows must come from one node (:func:`population.one_node`). A frame
    without the column predates it and is not checked."""
    if "node" not in data.columns:
        return
    for kernel, rows in data.groupby("benchmark"):
        population.one_node(rows["node"].tolist(), label=str(kernel))


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


def reorder_rows(names: Sequence[str], order: str) -> tuple[list[str], list[GroupSpan]]:
    """Ordered short_names + group spans for a set of plotted benchmark names."""
    return order_rows(row_meta_for(list(names)), order)


def draw_group_labels(ax: Axes, spans: Sequence[GroupSpan], x_right: float) -> None:
    """Draw a separator line at each internal group boundary and the group's y-axis text to
    the right of the heatmap (``clip_on=False``; the caller saves with ``bbox_inches='tight'``
    so the outside text is kept)."""
    for span in spans:
        if span.start > 0:
            ax.axhline(span.start - 0.5, color=style.REFERENCE, linewidth=TYPE.line_width)
        mid = (span.start + span.end - 1) / 2.0
        ax.text(  # pyright: ignore[reportUnknownMemberType] -- matplotlib takes untyped **kwargs
            x_right,
            mid,
            span.label,
            ha="left",
            va="center",
            rotation=90,
            fontsize=TYPE.annotation_pt,
            clip_on=False,
        )


def plot_heatmap(
    benchmark: str = "all",
    preset: str = "S",
    datatype: str = "float64",
    variant: str | None = None,
    order: str = BY_DWARF,
    db: str | None = None,
    output: str = PLOTS_DIR + "/heatmap.pdf",
    usetex: bool = True,
    baseline: str = DEFAULT_BASELINE,
) -> list[str]:
    """Read ``db`` and emit ONE speedup heatmap PER MACHINE; returns the paths written.

    A plural return, because a results DB may hold rows from more than one node and those may
    never share a figure (:func:`machine_groups`). ``output`` names the family, not a file: each
    machine gets ``<stem>.<cpu>[-<gpu>]<ext>``.

    :param benchmark: selector (kernel / track / dwarf / ``@lvl<n>``) matched against
        the ``benchmark`` (short_name) column; ``all`` keeps every row.
    :param preset: data-size preset to plot (rows with a different preset are dropped).
    :param datatype: precision to plot; legacy NULL-datatype rows are treated float64.
    :param variant: restrict to a single sparse variant; ``None`` keeps every
        (benchmark, variant) as its own ``benchmark/variant`` row.
    :param order: row ordering, ``by_dwarf`` (default) or ``by_level`` (see
        :mod:`hpcagent_bench.reporting_order`).
    :param db: SQLite results DB path; ``None`` uses the configured ``record.db_path``.
    :param output: PDF path FAMILY (default under ``results/plots``); each machine's file is this
        name with the machine label inserted before the extension.
    :param usetex: render text with LaTeX (default); ``False`` for a LaTeX-free box.
    """
    set_usetex(usetex)
    everything = load_results(db, benchmark, preset, datatype, variant, baseline)
    groups = machine_groups(everything)
    # An empty selection must FAIL: the comprehension below would otherwise write no file and exit 0.
    if not groups:
        raise RuntimeError(
            f"no rows to plot: benchmark={benchmark!r} preset={preset!r} "
            f"datatype={datatype!r} variant={variant!r} db={db!r}. The DB has no "
            f"validated, domained rows matching that selection."
        )
    written: list[str] = []
    for label, rows in groups:
        try:
            written.append(heatmap_figure(rows, order, machine_output(output, label), baseline))
        except NoBaselineRows as exc:
            # Named in the log rather than swallowed: a missing machine in the output is a fact
            # about the data and the reader has to be able to find out which one and why.
            LOG.warning("plotting: skipping machine %s -- %s", label, exc)
    if not written:
        raise NoBaselineRows(f"no machine in scope has {baseline} rows to divide by")
    return written


def ink_for(ratio: float) -> str:
    """Cell text colour: ink inside the pale middle of the ramp, white on its saturated ends."""
    magnitude = 1 / ratio if ratio < 1 else ratio
    return style.INK if magnitude < 1.3 else "white"


def ci_superscript(summary: pd.DataFrame, benchmark: str, framework: str) -> str:
    """The cell's CI width as a mathtext superscript percent; empty when the frame has no cell."""
    cell = summary[(summary["framework"] == framework) & (summary["benchmark"] == benchmark)]
    perc_col = cast("pd.Series", cell["ci_perc"])
    perc = int(perc_col.to_numpy()[0]) if len(perc_col) != 0 else 0
    return f"$^{{({perc})}}$" if perc > 0 else ""


def heatmap_figure(data: pd.DataFrame, order: str, output: str, baseline: str = DEFAULT_BASELINE) -> str:
    """Draw ONE machine's speedup heatmap to ``output``; returns the path written.

    Split from :func:`plot_heatmap` so the per-machine partition happens once, above the drawing,
    rather than being threaded through it.
    """
    one_node_per_kernel(data)
    # Per-cell cleaned median + CI (the median drives best-selection AND the plotted value).
    summary = cell_summary(data)
    best = summary[["benchmark", "domain", "framework", "time"]].copy()

    frmwrks = cast("list[str]", list(data["framework"].unique()))
    # Raised, not asserted, and the CALLER decides: figures are emitted one per machine, and a
    # machine that ran only one framework has nothing to divide by -- a thin slice, not a broken run.
    if baseline not in frmwrks:
        raise NoBaselineRows(f"no {baseline} rows to divide by; frameworks present: {sorted(frmwrks)}")
    frmwrks.remove(baseline)
    frmwrks.append(baseline)
    lfilter: list[str] = ["benchmark", "domain"] + frmwrks

    # Wide form: normalise every framework's median to NumPy's; keep the raw times for the
    # NumPy column and the geomean Total.
    best_wide: pd.DataFrame = best.pivot_table(
        index=["benchmark", "domain"], columns="framework", values="time"
    ).reset_index()
    best_wide = best_wide[lfilter].reset_index(drop=True)
    best_wide_time: pd.DataFrame = best_wide.copy(deep=True)
    for f in frmwrks:
        best_wide[f] = best_wide[f] / best_wide_time[baseline]

    # Row ordering: reindex both the ratio and the raw-time frames identically.
    ordered_names, spans = reorder_rows(cast("list[str]", best_wide["benchmark"].tolist()), order)
    rank: dict[str, int] = {n: i for i, n in enumerate(ordered_names)}

    def by_rank(column: pd.Series) -> pd.Series:
        return column.map(rank)

    best_wide = best_wide.sort_values("benchmark", key=by_rank).reset_index(drop=True)
    best_wide_time = best_wide_time.sort_values("benchmark", key=by_rank).reset_index(drop=True)

    # Indexed by kernel so a dropped cell can be NAMED rather than counted.
    overall: pd.DataFrame = pd.melt(best_wide.drop(["domain"], axis=1), ["benchmark"]).set_index("benchmark")
    overall = overall.groupby(["framework"]).value.apply(column_geomean).reset_index()
    overall_wide: pd.DataFrame = overall.pivot_table(columns="framework", values="value", dropna=False).reset_index(
        drop=True
    )
    overall_wide = overall_wide[frmwrks]

    overall_time: pd.DataFrame = pd.melt(best_wide_time.drop(["domain"], axis=1), ["benchmark"]).set_index("benchmark")
    overall_time = overall_time.groupby(["framework"]).value.apply(column_geomean).reset_index()
    overall_time_wide: pd.DataFrame = overall_time.pivot_table(
        columns="framework", values="value", dropna=False
    ).reset_index(drop=True)

    plt.style.use("classic")
    figsz = (len(frmwrks) + 1, 12)
    fig, (ax2, ax1) = plt.subplots(2, 1, figsize=figsz, sharex=True, gridspec_kw={"height_ratios": [0.1, 5.7]})

    totals = cast("FloatArray", overall_wide.to_numpy())
    total_baseline = cast("FloatArray", overall_time_wide[baseline].to_numpy())
    ax2.imshow(totals, cmap="RdYlGn_r", interpolation="nearest", vmin=0, vmax=2, aspect="auto")
    ax2.set_yticks(np.arange(1))
    ax2.set_yticklabels(["Total"])
    for j in range(len(overall_wide.columns)):
        if j < len(overall_wide.columns) - 1:
            ratio = totals[0, j]
            ax2.text(
                j,
                0,
                abbreviate_speedup(ratio),
                ha="center",
                va="center",
                color=ink_for(ratio),
                fontsize=TYPE.label_pt,
            )
        else:
            ax2.text(
                j,
                0,
                abbreviate_runtime(total_baseline[0]),
                ha="center",
                va="center",
                color="white",
                fontsize=TYPE.label_pt,
            )

    hm_data: pd.DataFrame = best_wide.drop(["benchmark", "domain"], axis=1)
    ratios = cast("FloatArray", hm_data.to_numpy())
    base_times = cast("FloatArray", best_wide_time[baseline].to_numpy())
    names = cast("list[str]", best_wide["benchmark"].tolist())
    columns = cast("list[str]", hm_data.columns.tolist())
    ax1.imshow(ratios, cmap="RdYlGn_r", interpolation="nearest", vmin=0, vmax=2, aspect="auto")

    ax1.set_xticks(np.arange(len(columns)))
    ax1.set_yticks(np.arange(len(names)))
    ax1.set_xticklabels(columns)
    ax1.set_yticklabels(names)
    plt.setp(ax1.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    for i in range(len(names)):
        for j in range(len(columns)):
            if j == len(columns) - 1:
                ax1.text(
                    j,
                    i,
                    abbreviate_runtime(base_times[i]),
                    ha="center",
                    va="center",
                    color=style.INK,
                    fontsize=TYPE.label_pt,
                )
                continue
            ratio = ratios[i, j]
            if math.isnan(ratio):
                continue  # NaN cell renders blank
            ci = ci_superscript(summary, names[i], columns[j])
            ax1.text(
                j,
                i,
                abbreviate_speedup(ratio) + ci,
                ha="center",
                va="center",
                color=ink_for(ratio),
                fontsize=TYPE.label_pt,
            )

    # Group separators + right-side y-axis group text (structured grids / tsvc2 / machine_learning / ...).
    draw_group_labels(ax1, spans, x_right=len(columns) - 0.35)

    ax1.set_ylabel("Benchmarks", labelpad=0)

    plt.tight_layout()
    return save_figure(output, fig)


def grid_shape(n: int) -> tuple[int, int]:
    """rows, cols for ``n`` per-kernel cells: a single kernel is 1x1, otherwise up to 4
    columns (``ceil(sqrt(n))`` capped) so each cell stays >= ~1.6in wide at a two-column
    paper width."""
    if n <= 1:
        return 1, 1
    ncols = min(4, math.ceil(math.sqrt(n)))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def framework_slots(data: pd.DataFrame, baseline: str = DEFAULT_BASELINE) -> list[str]:
    """The FULL framework set across the plotted scope, in a fixed slot order (numpy first as
    the reference, then alphabetical). Every panel reserves one slot per framework here, so a
    kernel missing a framework leaves an empty gap instead of re-packing the present ones."""
    present = cast("list[str]", list(data["framework"].unique()))
    return sorted(present, key=lambda f: (f != baseline, f))


def plot_distribution_grid(
    benchmark: str = "all",
    preset: str = "S",
    datatype: str = "float64",
    variant: str | None = None,
    framework: str | None = None,
    kind: str = "violin",
    order: str = BY_DWARF,
    db: str | None = None,
    baseline: str = DEFAULT_BASELINE,
    output: str = PLOTS_DIR + "/distribution.pdf",
    col_width_in: float = 3.4,
    usetex: bool = True,
) -> list[str]:
    """Emit ONE per-kernel distribution grid (violin or box) PER MACHINE; returns the paths written.

    Plural for the same reason as :func:`plot_heatmap`: rows from two nodes may not share a figure,
    so ``output`` names the family and each machine's file carries its label.

    Modelled on npbench's per-kernel subplot grid (framework-coloured, one shared legend), but
    with FIXED per-framework slots: every panel reserves one slot per framework in
    :func:`framework_slots`, each violin/box drawn at its framework's CONSTANT slot index and
    CONSTANT width. A kernel missing a framework leaves an empty gap at that slot -- the present
    ones are never re-packed -- so glyph widths stay uniform whether or not a framework ran (the
    bug that made sparse panels render thick, ugly bars). ``xlim`` and ``xticks`` are constant
    across panels.

    Each cell shows the full outlier-cleaned sample spread
    (:func:`hpcagent_bench.stats.summary.drop_outliers`, which warns on a drop). Scope is the same
    selector grammar as :func:`plot_heatmap` (single kernel = 1x1, an explicit list, a whole
    track, a subtrack-per-level via ``@lvl<n>``), kernels ordered by the shared scheme. The
    figure is sized to a two-column paper width (``col_width_in`` per paper column, ~3.4in).

    :param framework: restrict to one framework, else every framework in scope.
    :param kind: ``violin`` (default) or ``box``.
    :param order: ``by_dwarf`` (default) or ``by_level``.
    """
    if kind not in ("violin", "box"):
        raise ValueError(f"kind must be 'violin' or 'box' (got {kind!r})")
    set_usetex(usetex)
    everything = load_results(db, benchmark, preset, datatype, variant, baseline)
    if framework is not None:
        everything = everything[everything["framework"] == framework].reset_index(drop=True)
    if bool(everything.empty):
        raise RuntimeError(f"no rows to plot for benchmark={benchmark!r} preset={preset!r} datatype={datatype!r}")
    return [
        distribution_figure(rows, kind, order, machine_output(output, label), col_width_in, baseline)
        for label, rows in machine_groups(everything)
    ]


def distribution_figure(
    data: pd.DataFrame, kind: str, order: str, output: str, col_width_in: float, baseline: str = DEFAULT_BASELINE
) -> str:
    """Draw ONE machine's distribution grid to ``output``; returns the path written.

    Split from :func:`plot_distribution_grid` for the same reason as :func:`heatmap_figure`: the
    per-machine partition belongs above the drawing, not threaded through it.
    """

    kernels = list(dict.fromkeys(cast("list[str]", data["benchmark"].tolist())))  # unique, insertion order
    ordered = reorder_rows(kernels, order)[0]

    slots = framework_slots(data, baseline)  # FIXED slot per framework, shared by every panel
    colors = palette.framework_colors(slots)
    nslots = len(slots)

    nrows, ncols = grid_shape(len(ordered))
    fig_w = col_width_in if ncols == 1 else min(2 * col_width_in, ncols * col_width_in)
    fig_h = max(2.1, nrows * 2.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    for idx, kernel in enumerate(ordered):
        ax = axes[idx // ncols][idx % ncols]
        sub = data[data["benchmark"] == kernel]
        for slot, fw in enumerate(slots):
            times = cast("pd.Series", sub[sub["framework"] == fw]["time"])
            samples = cast("FloatArray", times.to_numpy())
            if samples.size == 0:
                continue  # empty gap at this framework's fixed slot; never re-pack
            kept = summary.drop_outliers(samples, label=f"{kernel}@{fw}")[0]
            if kept.size == 0:
                continue
            if kind == "violin":
                draw_violin(ax, kept, slot, colors[fw], 0.75)
            else:
                draw_sample_box(ax, kept, slot, colors[fw])
        ax.set_xlim(-0.6, nslots - 0.4)  # CONSTANT across panels
        ax.set_xticks(range(nslots))
        ax.set_xticklabels([])  # framework identity lives in the shared legend, not per panel
        ax.set_title(kernel, fontsize=TYPE.annotation_pt)
        ax.tick_params(axis="y", labelsize=TYPE.tick_pt)
        if idx % ncols == 0:
            ax.set_ylabel("Time (ms)", fontsize=TYPE.annotation_pt)

    # Blank any unused cells in the last row.
    for idx in range(len(ordered), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    # One shared framework legend (colour -> framework), above the grid -- the same wrap-to-fit
    # helper every other figure's key uses, rather than a second hand-rolled ``fig.legend`` call.
    handles = [Rectangle((0, 0), 1, 1, color=colors[fw], label=fw) for fw in slots]
    style.legend_below(fig, handles, ncol=min(nslots, 6), y=1.02, fontsize=TYPE.legend_pt)

    plt.tight_layout()
    return save_figure(output, fig)


#: A distribution glyph's CONSTANT width, independent of how many frameworks a kernel has.
GLYPH_WIDTH: float = 0.8


def draw_violin(ax: Axes, samples: FloatArray, position: float, color: str, alpha: float) -> None:
    """One violin of ``samples`` at ``position`` in ``color``, its median in ink."""
    parts = ax.violinplot([samples], positions=[position], widths=GLYPH_WIDTH, showmedians=True, showextrema=False)
    for body in cast("list[PolyCollection]", parts["bodies"]):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(alpha)
    if "cmedians" in parts:
        medians = cast("LineCollection", parts["cmedians"])
        medians.set_color(style.INK)
        medians.set_linewidth(TYPE.line_width)


def draw_sample_box(ax: Axes, samples: FloatArray, position: float, color: str) -> None:
    """One box of ``samples`` at ``position`` in ``color``, no fliers, its median in ink."""
    bp = ax.boxplot([samples], positions=[position], widths=GLYPH_WIDTH * 0.75, showfliers=False, patch_artist=True)
    for patch in cast("list[Patch]", bp["boxes"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for med in cast("list[Line2D]", bp["medians"]):
        med.set_color(style.INK)
