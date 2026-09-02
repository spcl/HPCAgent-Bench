# canon artifacts

The dace canonicalization ablation: TSVC kernels built with the canonicalization pass on and off,
against serial `gcc -O3`. No agents and no models are involved, which is why it is its own
experiment rather than a panel of llr8.

The tests beside the plot scripts import `benchlib`, which lives in the paper repository at
`ICLR26Reproducibility/paper_artifacts/`. Put that directory on `PYTHONPATH` to run them:
without it they fail at COLLECTION with `No module named 'benchlib'`, which reads like a
broken test and is a missing path. CI never collects them -- it lists `tests/test_*.py`.

`data/` and `figures/` are empty in the repository. Both plot scripts take a SWEEP DIRECTORY -- the
per-rank timing CSVs a sweep leaves on the cluster -- rather than a committed table, because a sweep
is machine-specific and a frozen copy of one would be read as a portable result:

```bash
python3 ../plot_tsvc_speedup.py <sweep-dir> --out ../figures
python3 ../plot_canon_vs.py     <sweep-dir> --out ../figures
```

Both refuse rather than draw an empty axis when the sweep has no baseline arm.
