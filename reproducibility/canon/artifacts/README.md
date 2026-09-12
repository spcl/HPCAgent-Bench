# canon artifacts

The dace canonicalization ablation: TSVC kernels built with the canonicalization pass on and off,
against serial `gcc -O3`. No agents and no models are involved, which is why it is its own
experiment rather than a panel of llr8.

Both figures live in `hpcagent_bench.stats.figures.signed`, with the rest of the plotting code, and
their consumers are `tests/test_figures_signed.py`, which CI collects.

`data/` and `figures/` are empty in the repository. The builder takes a SWEEP DIRECTORY -- the
per-rank timing CSVs a sweep leaves on the cluster -- rather than a committed table, because a sweep
is machine-specific and a frozen copy of one would be read as a portable result:

```bash
python -m hpcagent_bench.stats.figures.signed <sweep-dir> --out reproducibility/canon/figures
```

It writes both figures, and beside each a `-kernels.csv` and a `-summary.csv`: the ratios with the
milliseconds behind them, and the per-arm geomean with its interval. Those tables are what a rerun
is diffed on, never the images. It refuses rather than draws an empty axis when the sweep has no
baseline arm.
