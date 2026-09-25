# canon artifacts

The compiler-baseline ablation (DaCe canonicalization on and off, against `cc`, `cc_autopar` and
Numba, no agents) is reproduced by
[ICLR26Reproducibility](https://github.com/ThrudPrimrose/ICLR26Reproducibility)
`experiments/canon/reproduce.sh`.

Its input is a sweep directory. `experiments/submit-canon-llr40.sh` submits one Slurm job per
column (`experiments/canon_column.sh`) and writes per-rank timing CSVs under the sweep's `OUT_ROOT`
([experiments/README.md](../../../experiments/README.md#canon-compiler-baselines)).
`reproduce.sh --extract` turns that directory into `data/canon.db`:

```bash
python3 scripts/collect_canon.py --run-dir "$CANON_SWEEP" --db data/canon.db
```
