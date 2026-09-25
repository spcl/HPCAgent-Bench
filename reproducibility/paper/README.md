# Reproducibility

Everything needed to regenerate the figures and tables of the paper from its measurements. Scores,
statistics and plots come from HPCAgent-Bench; this repository holds only the paper's pipeline.

## Run

```sh
export HPCAGENT_BENCH=/path/to/hpcagent-bench      # checkout at the paper-experiments tag
export PYTHON=/path/to/venv/bin/python              # pip install -e "$HPCAGENT_BENCH[<hw>]" --group dace
export DATA_URL=<data archive of the release>

./download.sh      # 1. data/: observations per experiment, compiler sweep, GH200 regrade (verified)
./run_all.sh       # 2. statistics, every figure and table, then the checksum check
```

`./run_all.sh` ends with `OK: N files match SHA256SUMS`, or names every file that differs. Each
figure also has its own script, so one figure can be redrawn after `./stats.sh`:

| Script | Output | Paper |
|---|---|---|
| `stats.sh` | `work/*.db` (pooled answers), `tables/*.csv` (paired tests) | all figures |
| `fig2.sh` | `figures/efficacy_packets_and_scope.pdf` | Figure 3 |
| `fig3.sh` | `figures/scope_row.pdf` | Figure 4 |
| `fig4.sh` | `figures/cost_weighting.pdf` | Figure 5 |
| `fig5.sh` | `figures/ml_scaling.pdf` | Figure 6 |
| `fig8.sh` | `figures/cheating_per_kernel.pdf` | Figure 7 (appendix) |
| `tab3.sh` | `tables/gh200-transfer-table.tex` | Table 3 |

`./run_all.sh --record` rewrites `SHA256SUMS`. No step needs a GPU or model access.

## Folders

| Folder | Contents |
|---|---|
| `lib/` | the paper's own helpers: pooling rules, compiler comparators, the PyTorch anchor, the GH200 join and table |
| `figures/`, `tables/` | the committed outputs that `SHA256SUMS` covers |
| `case-studies/` | full submission sources behind Sections 4.4 and 4.5 (`INDEX.md`; `MISSING.md` lists sources that no longer exist) |
| `skill_histories/` | every version of the skill packets the agents read |
| `tools/` | `pull.sh` (authors: mirror the cluster runs), `make_archive.sh` (build the anonymized data archive), `anonymize.py` (the terms of `.anonymize-terms.txt` over databases, text and figures), `make_zenodo.sh` (the anonymized Zenodo package) |

## Data from the cluster (authors)

`./download.sh --from-cluster` mirrors the run directories with `tools/pull.sh` (settings in the
untracked `tools/cluster.env`, template `cluster.env.example`) into `mirror/` and rebuilds `data/`
from them: every latest answer at its final grade, answers of deleted jobs from their frozen
observations. `tools/make_archive.sh` then packs `data/` into the release archive and writes
`DATA_SHA256SUMS`.
