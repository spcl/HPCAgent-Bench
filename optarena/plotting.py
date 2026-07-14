# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render the framework-benchmark ``results`` table as a speedup heatmap PDF.

Reads the ``results`` table from the SQLite results DB (``optarena.db`` by default,
written by the collection sweeps in :mod:`optarena.support.collect`), keeps the best
(median-fastest) implementation per (framework, benchmark), normalises every
framework's runtime to NumPy's, and lays the result out as a ``RdYlGn_r`` heatmap
with per-cell bootstrap-CI superscripts and a geomean ``Total`` row. NumPy's own
column shows absolute runtimes instead of a ratio.

The plot renders headless (``Agg``) with ``text.usetex`` -- it needs a LaTeX install.
This is imported on demand (never at CLI ``--help`` time) so matplotlib/pandas/SciPy
are only required when a plot is actually produced; the DB is read through the stdlib
``sqlite3`` so reporting never pulls in the heavy framework stack.
"""
import math
import sqlite3

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')  # headless: save to file, never open a window
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt  # noqa: E402 -- must follow the backend/rcParams setup

from scipy.stats.mstats import gmean  # noqa: E402

from optarena.spec import select_short_names  # noqa: E402


def my_round(x, width):
    float_format = "{:." + f"{width}" + "f}"
    return float_format.format(x)


def my_geomean(x):
    """Geomean that ignores NA values."""
    x = x.dropna()
    return gmean(x)


def my_speedup_abbr(x):
    """Short speedup label with an up/down indicator."""
    prefix = ""
    label = ""
    if math.isnan(x):
        return ""
    if x < 1:
        prefix = u"↑"
        x = 1 / x
    elif x > 1:
        prefix = u"↓"
    if x > 100:
        x = int(x)
    if x > 1000:
        label = prefix + str(my_round(x / 1000, 1)) + "k"
    else:
        label = prefix + str(my_round(x, 1))
    return str(label)


def my_runtime_abbr(x):
    """Short runtime label; DB times are in milliseconds."""
    if math.isnan(x):
        return ""
    if x >= 1000:
        return str(my_round(x / 1000, 2)) + " s"
    return str(my_round(x, 2)) + " ms"


def bootstrap_ci(data, statfunction=np.median, alpha=0.05, n_samples=300, seed=0):
    """inspired by https://github.com/cgevans/scikits-bootstrap.

    Resamples with a SEEDED local RNG so the published CI is reproducible: the same
    ``optarena.db`` yields the same per-cell superscript on every run. The old global
    ``np.random`` stream made an identical figure vary run-to-run."""
    rng = np.random.default_rng(seed)

    def bootstrap_ids(data, n_samples):
        for _ in range(n_samples):
            yield rng.integers(data.shape[0], size=(data.shape[0], ))

    alphas = np.array([alpha / 2, 1 - alpha / 2])
    nvals = np.round((n_samples - 1) * alphas).astype(int)

    data = np.array(data)
    if np.prod(data.shape) != max(data.shape):
        raise ValueError("Data must be 1D")
    data = data.ravel()

    boot_indexes = bootstrap_ids(data, n_samples)
    stat = np.asarray([statfunction(data[ids]) for ids in boot_indexes])
    stat.sort(axis=0)

    return stat[nvals][1] - stat[nvals][0]


def plot_heatmap(benchmark="all", preset="S", datatype="float64", variant=None, db="optarena.db", output="heatmap.pdf"):
    """Read ``db`` and emit the speedup heatmap to ``output`` (a PDF).

    :param benchmark: selector (kernel / track / dwarf / ``@lvl<n>``) matched against
        the ``benchmark`` (short_name) column; ``all`` keeps every row.
    :param preset: data-size preset to plot (rows with a different preset are dropped).
    :param datatype: precision to plot; legacy NULL-datatype rows are treated float64.
    :param variant: restrict to a single sparse variant; ``None`` keeps every
        (benchmark, variant) as its own ``benchmark/variant`` row.
    :param db: SQLite results DB path (default ``optarena.db`` in the cwd).
    :param output: PDF path to write (default ``heatmap.pdf`` in the cwd).
    """
    conn = sqlite3.connect(db)
    data = pd.read_sql_query("SELECT * FROM results", conn)
    conn.close()

    # timestamp only groups the rows of one run; the plot does not use it
    data = data.drop(['timestamp'], axis=1).reset_index(drop=True)

    # Selector: restrict to a kernel / track / dwarf / @level selection (reuses the
    # KERNELS.select grammar, keyed on the short_name in the benchmark column).
    if benchmark != 'all':
        keep = set(select_short_names(benchmark))
        data = data[data['benchmark'].isin(keep)].reset_index(drop=True)

    # Remove everything that does not have a domain
    data = data[data["domain"] != ""]

    # remove everything that does not validate, then get rid of validated column
    data = data[data['validated'] == True]
    data = data.drop(['validated'], axis=1).reset_index(drop=True)

    # Filter by precision (NULL = legacy row, treated as float64)
    if 'datatype' in data.columns:
        legacy_mask = data['datatype'].isna()
        data.loc[legacy_mask, 'datatype'] = 'float64'
        data = data[data['datatype'] == datatype]
        data = data.drop(['datatype'], axis=1).reset_index(drop=True)
    elif datatype != 'float64':
        raise RuntimeError(f"{db} predates the datatype column; cannot filter to --datatype={datatype}.")

    # Variant axis: every (benchmark, variant) becomes its own row. Dense rows have
    # variant=NULL and keep their plain benchmark name; sparse rows are renamed to
    # `benchmark/variant` so the heatmap shows the storage-format / distribution
    # combinations side by side.
    if 'variant' in data.columns:
        if variant is not None:
            data = data[(data['variant'].isna()) | (data['variant'] == variant)]
        sparse_mask = data['variant'].notna()
        data.loc[sparse_mask, 'benchmark'] = (data.loc[sparse_mask, 'benchmark'].astype(str) + '/' +
                                              data.loc[sparse_mask, 'variant'].astype(str))
        data = data.drop(['variant'], axis=1).reset_index(drop=True)

    # Filter by preset
    data = data[data['preset'] == preset]
    data = data.drop(['preset'], axis=1).reset_index(drop=True)

    # for each (framework, benchmark) take the median runtime across its samples
    aggdata = data.groupby(["benchmark", "domain", "framework"], dropna=False).agg({"time": "median"}).reset_index()
    best = aggdata.sort_values("time").groupby(["benchmark", "domain", "framework"], dropna=False).first().reset_index()
    bestgroup = best.drop(["time"], axis=1)  # remove time, we don't need it and it is actually a median
    data = pd.merge(left=bestgroup, right=data, on=["benchmark", "domain", "framework"],
                    how="inner")  # do a join on data and best

    frmwrks = list(data['framework'].unique())
    assert ('numpy' in frmwrks)
    frmwrks.remove('numpy')
    frmwrks.append('numpy')
    lfilter = ['benchmark', 'domain'] + frmwrks

    # get improvement over numpy (keep times in best_wide_time for numpy column), reorder columns
    best_wide = best.pivot_table(index=["benchmark", "domain"], columns="framework",
                                 values="time").reset_index()  # pivot to wide form
    best_wide = best_wide[lfilter].reset_index(drop=True)
    best_wide_time = best_wide.copy(deep=True)
    for f in frmwrks:
        best_wide[f] = best_wide[f] / best_wide_time['numpy']

    # compute ci-size for each
    cidata = data.groupby(["benchmark", "domain", "framework"], dropna=False).agg({
        "time": [bootstrap_ci, "median"]
    }).reset_index()
    cidata.columns = ['_'.join(col).strip() for col in cidata.columns.values]
    cidata['perc'] = (cidata['time_bootstrap_ci'] / cidata['time_median']) * 100

    overall = best_wide.drop(['domain'], axis=1)
    overall = pd.melt(overall, [
        'benchmark',
    ])
    overall = overall.groupby(['framework'
                               ]).value.apply(my_geomean).reset_index()  # throws warnings if NA is found, which is ok
    overall_wide = overall.pivot_table(columns="framework", values="value", dropna=False).reset_index(drop=True)
    overall_wide = overall_wide[frmwrks]

    overall_time = best_wide_time.drop(['domain'], axis=1)
    overall_time = pd.melt(overall_time, ['benchmark'])
    overall_time = overall_time.groupby(
        ['framework']).value.apply(my_geomean).reset_index()  # throws warnings if NA is found, which is ok
    overall_time_wide = overall_time.pivot_table(columns="framework", values="value",
                                                 dropna=False).reset_index(drop=True)

    plt.style.use('classic')
    figsz = (len(frmwrks) + 1, 12)
    fig, (ax2, ax1) = plt.subplots(2, 1, figsize=figsz, sharex=True, gridspec_kw={'height_ratios': [0.1, 5.7]})

    hm_data_all = overall_wide
    ax2.imshow(hm_data_all.to_numpy(), cmap='RdYlGn_r', interpolation='nearest', vmin=0, vmax=2, aspect="auto")
    ax2.set_yticks(np.arange(1))
    ax2.set_yticklabels(["Total"])
    for j in range(len(overall_wide.columns)):
        if j < len(overall_wide.columns) - 1:
            label = hm_data_all.to_numpy()[0, j]
            t = label
            if t < 1:
                t = 1 / t
            if t < 1.3:
                ax2.text(j, 0, my_speedup_abbr(label), ha="center", va="center", color="grey", fontsize=8)
            else:
                ax2.text(j, 0, my_speedup_abbr(label), ha="center", va="center", color="white", fontsize=8)
        else:
            # Last column: render numpy's absolute runtime in the Total row.
            label = overall_time_wide['numpy'].to_numpy()[0]
            ax2.text(j, 0, my_runtime_abbr(label), ha="center", va="center", color="white", fontsize=8)

    # plot benchmark heatmap
    hm_data = best_wide.drop(['benchmark', 'domain'], axis=1)
    ax1.imshow(hm_data.to_numpy(), cmap='RdYlGn_r', interpolation='nearest', vmin=0, vmax=2, aspect="auto")

    # We want to show all ticks...
    ax1.set_xticks(np.arange(len(hm_data.columns)))
    ax1.set_yticks(np.arange(len(best_wide['benchmark'])))
    # ... and label them with the respective list entries
    ax1.set_xticklabels(hm_data.columns)
    ax1.set_yticklabels(best_wide['benchmark'])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax1.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    for i in range(len(best_wide['benchmark'])):
        # annotate with improvement over numpy
        for j in range(len(hm_data.columns)):
            b = best_wide['benchmark'][i]
            f = hm_data.columns[j]
            if j < len(hm_data.columns) - 1:
                label = hm_data.to_numpy()[i, j]
                if math.isnan(label):
                    pass  # NaN cell renders blank
                else:
                    p = cidata[(cidata['framework_'] == f) & (cidata['benchmark_'] == b)]['perc']
                    ci = int(p.to_numpy()[0])
                    if ci > 0:
                        ci = "$^{(" + str(ci) + ")}$"
                    else:
                        ci = ""
                    t = label
                    if t < 1:
                        t = 1 / t
                    if t < 1.3:
                        ax1.text(j, i, my_speedup_abbr(label) + ci, ha="center", va="center", color="grey", fontsize=8)
                    else:
                        ax1.text(j, i, my_speedup_abbr(label) + ci, ha="center", va="center", color="white", fontsize=8)
            else:
                label = best_wide_time['numpy'].to_numpy()[i]
                ci = ""  # numpy runtime column shows no CI superscript
                ax1.text(j, i, my_runtime_abbr(label) + ci, ha="center", va="center", color="black", fontsize=8)

    ax1.set_ylabel("Benchmarks", labelpad=0)

    plt.tight_layout()
    plt.savefig(output, dpi=600)
    plt.close(fig)
    return output
