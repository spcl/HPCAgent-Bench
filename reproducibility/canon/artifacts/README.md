# canon artifacts

The canon ablation (DaCe canonicalization on and off, against `cc`, `cc_autopar` and Numba, no
agents) is reproduced from
[ICLR26Reproducibility](https://github.com/ThrudPrimrose/ICLR26Reproducibility),
`experiments/canon/reproduce.sh`; that repository is canonical for the figures, tables and numbers.

The one step the artifact repo cannot run itself: producing the sweep `--extract` reads.
`experiments/submit-canon-llr40.sh` submits one Slurm job per compiler column
(`experiments/canon_column.sh`) on CSCS Beverin and writes the per-rank timing CSVs under
`$CANON_SWEEP`; `reproduce.sh --extract` then turns that directory into `data/canon.db` with
`scripts/collect_canon.py`.
