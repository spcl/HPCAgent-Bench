import argparse
import math
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # headless: save to file, never open a window
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

from scipy.stats.mstats import gmean
from optarena.infrastructure import utilities as util
from optarena.spec import PRESET_CHOICES, select_short_names


def my_round(x, width):
    float_format = "{:." + f"{width}" + "f}"
    return float_format.format(x)


# geomean which ignores NA values
def my_geomean(x):
    x = x.dropna()
    res = gmean(x)
    return res


# make nice/short numbers with up/down indicator
def my_speedup_abbr(x):
    prefix = ""
    label = ""
    if math.isnan(x):
        return ""
    if x < 1:
        prefix = u"\u2191"
        x = 1 / x
    elif x > 1:
        prefix = u"\u2193"
    if x > 100:
        x = int(x)
    if x > 1000:
        label = prefix + str(my_round(x / 1000, 1)) + "k"
    else:
        label = prefix + str(my_round(x, 1))
    return str(label)


# make nice/short runtime numbers; DB times are milliseconds
def my_runtime_abbr(x):
    if math.isnan(x):
        return ""
    if x >= 1000:
        return str(my_round(x / 1000, 2)) + " s"
    return str(my_round(x, 2)) + " ms"


def bootstrap_ci(data, statfunction=np.median, alpha=0.05, n_samples=300):
    """inspired by https://github.com/cgevans/scikits-bootstrap"""

    # import warnings

    def bootstrap_ids(data, n_samples=100):
        for _ in range(n_samples):
            yield np.random.randint(data.shape[0], size=(data.shape[0], ))

    alphas = np.array([alpha / 2, 1 - alpha / 2])
    nvals = np.round((n_samples - 1) * alphas).astype(int)
    # if np.any(nvals < 10) or np.any(nvals >= n_samples - 10):
    #     warnings.warn(
    #         "Some values used extremal samples; results are probably unstable. "
    #         "Try to increase n_samples")

    data = np.array(data)
    if np.prod(data.shape) != max(data.shape):
        raise ValueError("Data must be 1D")
    data = data.ravel()

    boot_indexes = bootstrap_ids(data, n_samples)
    stat = np.asarray([statfunction(data[_ids]) for _ids in boot_indexes])
    stat.sort(axis=0)

    return stat[nvals][1] - stat[nvals][0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b",
                        "--benchmark",
                        default="all",
                        help="selector: a single kernel, a track (hpc/ml/foundation), a dwarf "
                        "(hpc/structured_grids), or a level (hpc@lvl1, hpc/structured_grids@lvl2, "
                        "lvl2). Default: all.")
    parser.add_argument("-p", "--preset", choices=list(PRESET_CHOICES), nargs="?", default='S')
    parser.add_argument("-d",
                        "--datatype",
                        choices=['float32', 'float64'],
                        default='float64',
                        help="Precision to plot (default: float64). "
                        "Legacy NULL-datatype rows are treated as float64.")
    parser.add_argument("-V",
                        "--variant",
                        default=None,
                        help="Restrict to a single sparse variant (e.g. "
                        "csr_uniform). Default: every (benchmark, "
                        "variant) appears as its own row in the "
                        "heatmap, labelled `benchmark/variant`. Dense "
                        "rows keep their plain benchmark name.")
    args = vars(parser.parse_args())

# create a database connection
database = r"optarena.db"
conn = util.create_connection(database)
data = pd.read_sql_query("SELECT * FROM results", conn)

# timestamp only groups the rows of one run; the plot does not use it
data = data.drop(['timestamp'], axis=1).reset_index(drop=True)

# Selector: restrict to a kernel / track / dwarf / @level selection (reuses the
# KERNELS.select grammar, keyed on the short_name in the benchmark column).
if args['benchmark'] != 'all':
    keep = set(select_short_names(args['benchmark']))
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
    data = data[data['datatype'] == args['datatype']]
    data = data.drop(['datatype'], axis=1).reset_index(drop=True)
elif args['datatype'] != 'float64':
    raise RuntimeError(f"optarena.db predates the datatype column; cannot filter "
                       f"to --datatype={args['datatype']}.")

# Variant axis: every (benchmark, variant) becomes its own row. Dense
# rows have variant=NULL and keep their plain benchmark name; sparse
# rows are renamed to `benchmark/variant` so the heatmap shows the
# storage-format / distribution combinations side by side.
if 'variant' in data.columns:
    if args['variant'] is not None:
        data = data[(data['variant'].isna()) | (data['variant'] == args['variant'])]
    sparse_mask = data['variant'].notna()
    data.loc[sparse_mask, 'benchmark'] = (data.loc[sparse_mask, 'benchmark'].astype(str) + '/' +
                                          data.loc[sparse_mask, 'variant'].astype(str))
    data = data.drop(['variant'], axis=1).reset_index(drop=True)

# Filter by preset
data = data[data['preset'] == args['preset']]
data = data.drop(['preset'], axis=1).reset_index(drop=True)

# for each (framework, benchmark) take the median runtime across its samples
aggdata = data.groupby(["benchmark", "domain", "framework"], dropna=False).agg({"time": "median"}).reset_index()
best = aggdata.sort_values("time").groupby(["benchmark", "domain", "framework"], dropna=False).first().reset_index()
bestgroup = best.drop(["time"], axis=1)  # remove time, we don't need it and it is actually a median
data = pd.merge(left=bestgroup, right=data, on=["benchmark", "domain", "framework"],
                how="inner")  # do a join on data and best

frmwrks = list(data['framework'].unique())
print(frmwrks)
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
                           ]).value.apply(my_geomean).reset_index()  #this throws warnings if NA is found, which is ok
overall_wide = overall.pivot_table(columns="framework", values="value", dropna=False).reset_index(drop=True)
overall_wide = overall_wide[frmwrks]

overall_time = best_wide_time.drop(['domain'], axis=1)
overall_time = pd.melt(overall_time, ['benchmark'])
overall_time = overall_time.groupby(
    ['framework']).value.apply(my_geomean).reset_index()  #this throws warnings if NA is found, which is ok
overall_time_wide = overall_time.pivot_table(columns="framework", values="value", dropna=False).reset_index(drop=True)

plt.style.use('classic')
figsz = (len(frmwrks) + 1, 12)
fig, (ax2, ax1) = plt.subplots(2, 1, figsize=figsz, sharex=True, gridspec_kw={'height_ratios': [0.1, 5.7]})

hm_data_all = overall_wide
im0 = ax2.imshow(hm_data_all.to_numpy(), cmap='RdYlGn_r', interpolation='nearest', vmin=0, vmax=2, aspect="auto")
ax2.set_yticks(np.arange(1))
ax2.set_yticklabels(["Total"])
for j in range(len(overall_wide.columns)):
    if j < len(overall_wide.columns) - 1:
        label = hm_data_all.to_numpy()[0, j]
        t = label
        if t < 1:
            t = 1 / t
        if t < 1.3:
            text = ax2.text(j, 0, my_speedup_abbr(label), ha="center", va="center", color="grey", fontsize=8)
        else:
            text = ax2.text(j, 0, my_speedup_abbr(label), ha="center", va="center", color="white", fontsize=8)
    else:
        # Last column: render numpy's absolute runtime in the Total row
        # (this previously assigned label but never called ax2.text(...),
        # leaving the cell blank — issue #32 "Total is empty").
        label = overall_time_wide['numpy'].to_numpy()[0]
        ax2.text(j, 0, my_runtime_abbr(label), ha="center", va="center", color="white", fontsize=8)

# plot benchmark heatmap
hm_data = best_wide.drop(['benchmark', 'domain'], axis=1)
im = ax1.imshow(hm_data.to_numpy(), cmap='RdYlGn_r', interpolation='nearest', vmin=0, vmax=2, aspect="auto")

# We want to show all ticks...
ticks = ax1.set_xticks(np.arange(len(hm_data.columns)))
ticks = ax1.set_yticks(np.arange(len(best_wide['benchmark'])))
# ... and label them with the respective list entries
ticks = ax1.set_xticklabels(hm_data.columns)
ticks = ax1.set_yticklabels(best_wide['benchmark'])

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
                    text = ax1.text(j,
                                    i,
                                    my_speedup_abbr(label) + ci,
                                    ha="center",
                                    va="center",
                                    color="grey",
                                    fontsize=8)
                else:
                    text = ax1.text(j,
                                    i,
                                    my_speedup_abbr(label) + ci,
                                    ha="center",
                                    va="center",
                                    color="white",
                                    fontsize=8)
        else:
            label = best_wide_time['numpy'].to_numpy()[i]
            ci = ""  # numpy runtime column shows no CI superscript
            text = ax1.text(j, i, my_runtime_abbr(label) + ci, ha="center", va="center", color="black", fontsize=8)

ax1.set_ylabel("Benchmarks", labelpad=0)

plt.tight_layout()
plt.savefig("heatmap.pdf", dpi=600)
