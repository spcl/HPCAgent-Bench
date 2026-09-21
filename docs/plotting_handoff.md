# Plotting handoff: the three paper figures

How to regenerate the figures the agentbench paper uses, what each one shows, and what to watch
for in the data behind them. [plotting.md](plotting.md) is the design contract every figure
follows; this page is the working recipe.

Every command below writes three files: the PDF for the paper, a PNG beside it, and a CSV holding
the numbers behind every mark. Hand over all three. A figure without its table cannot be checked.

## Setup

Run from the repository root with the tree and the translator package on the path:

```bash
export HB=$PWD                                   # this repository
export PYTHONPATH="$HB:$HB/hpcagent_bench/numpy_translators/src"
export MPLBACKEND=Agg PYTHONHASHSEED=0           # headless and byte-reproducible
export AR=/path/to/ICLR26Reproducibility          # per-track observations + pair tables
export CANON_DB=/path/to/results/canon.db         # the canon sweep (compiler timings)
```

The inputs:

| input | what it is | where it comes from |
|---|---|---|
| observations (`$AR/experiments/<track>/data/<track>.csv` or `.db`) | one row per graded submission and per task, per arm | `python -m hpcagent_bench.experiments --runs ... --out ...` ([plotting.md](plotting.md#extract-once-plot-from-the-csv)) |
| pair tables (`$AR/experiments/<track>/tables/*_billed.csv`) | which control arm pairs with which treated arm | `experiments/paired_arms.py` |
| canon DB (`$CANON_DB`) | median time per (compiler column, kernel), validated only | the canon sweep; table `canon` |
| roster file | the kernels a track is scored over, one per line | derived from the observations, below |

The llr-focus40 roster is the set of kernels its control arms were served. Derive it from the same
data the figure reads rather than typing it:

```bash
python3 -c "
import pandas as pd
d = pd.read_csv('$AR/experiments/llr-cpu/data/llr-cpu.csv', low_memory=False)
print('\n'.join(sorted(set(d[d.arm == 'cpf-llr-focus40-kimi27sglang-c'].benchmark.astype(str)))))
" > roster-llr-focus40.txt        # 40 kernels
```

## 1. Compilers, per kernel: Pluto, Numba, DaCe canon CPU, DaCe canon GPU, PPCG-HIP

![compilers per kernel](figures/example-compilers-per-kernel.png)

```bash
python3 statistics/plot_llr40_compilers.py \
    --canon-db "$CANON_DB" --roster-file roster-llr-focus40.txt \
    --canon-columns pluto,dace_cpu_canonicalize,dace_gpu_canonicalize,ppcg_hip \
    --offset 0.6 --out figures/compilers-per-kernel
```

Drawn by `hpcagent_bench.stats.figures.signed.llr40_two_row_figure`. Omitting `--observations`
draws the compiler columns alone; passing it adds every model's CPF arm.

How to read it:

- **Numba is the denominator**, not a series: it is the orange 1x line, and the Y axis says
  "Speed-up over Numba" (`--baseline` changes it). This is the 2026-09-20 decision; Pluto and PPCG
  are comparators drawn beside it, never the reference.
- **Filled mark = a measured result. Hollow mark = no verified result, scored 1x.** A kernel a
  compiler declined (non-affine, emission refused) or never ran enters at 1x and is counted in the
  geomean, never dropped (`canon.roster_speedups`).
- The rightmost column is the geomean over the roster with its 95% log-t interval, value printed
  to one decimal.
- `--offset` spreads a kernel's series across its slot so marks at the same height stay readable;
  0 stacks them.

**Caveat on the current data.** PPCG has a validated result on 6 of the 40 kernels and Pluto on
22, so their geomeans (0.9x and 1.4x) are mostly placeholders at 1x. Read them with
`compilers-per-kernel-kernels.csv`, where an empty `denominator_ms` is a placeholder. The DaCe
columns are complete (40 of 40).

## 2. Optimizers, one row, speed-up only

![optimizer row](figures/example-optimizer-row.png)

```bash
python3 statistics/plot_optimizer_row.py --canon-db "$CANON_DB" \
    --panel "title=Loop Reasoning CPU (LLR);observations=$AR/experiments/llr-cpu/data/llr-cpu.csv;arms=cpf-llr-focus40-{model}-c;compilers=dace_cpu_canonicalize,pluto;baseline=numba;roster=roster-llr-focus40.txt" \
    --panel "title=Loop Reasoning GPU (LLR);observations=$AR/experiments/llr-gpu/data/llr-gpu.csv;arms=gpu-llr-focus40-{model}-hip;compilers=dace_gpu_canonicalize,ppcg_hip;baseline=numba;roster=roster-llr-focus40.txt" \
    --panel "title=Repository Formulation;observations=$AR/experiments/git-scicomp/data/git-scicomp.csv;arms=git-scicomp-{model}-repo;baseline=c-autopar;repeats=median" \
    --out figures/optimizer-row.pdf
```

Drawn by `hpcagent_bench.stats.figures.optimizers.figure_optimizer_row`. One `--panel` per column;
each is a `key=value;...` spec:

| key | meaning |
|---|---|
| `title` | panel subtitle (required) |
| `observations` | observations file holding the LLM arms; omit for a compilers-only panel |
| `arms` | arm name template with `{model}`, filled for each model |
| `models` | comma list of model tags; default `--models` (`qwen38,oss120b,kimi27sglang`) |
| `compilers` | comma list of canon columns; needs `--canon-db` and `roster=` |
| `baseline` | the denominator column (`numba`, `c-autopar`); printed under the ticks as "1x = ..." |
| `baseline_name` | override the text of that note |
| `repeats` | `latest` (a rerun supersedes, the default) or `median` (designed repeats, e.g. git-scicomp) |
| `roster` | roster file; without it an arm is scored over the kernels it was served |

How to read it: one column per optimizer, one mark per column, the geomean speed-up over the
panel's baseline with its 95% log-t interval. Colour and shape name the optimizer, the X tick
gives a short name, the legend the full one. LLM arms and compilers are scored over the **same**
roster, with an unanswered kernel at 1x for both, so an LLM that solved 13 kernels and a compiler
that declined 34 are compared on the same 40. The CSV carries `solved` and `kernels` for every
mark, which the figure cannot show, and the script prints them:

```
Loop Reasoning GPU (LLR)   Qwen3.8-27B               2.6x  solved 13/40
Loop Reasoning GPU (LLR)   PPCG (CUDA via hipify)    0.9x  solved  6/40
```

Panels share one log2 axis but not one denominator: the loop-level tracks are timed against
Numba and the repository track against auto-parallelized C, which is why each panel names its own
baseline under the ticks instead of the Y title saying "over Numba".

Caveats on the current data:

- **Repository panel: hold the numbers.** 12.8% of git-scicomp graded rows are unstamped legacy
  rows that have not been through the regrade migration (`scripts/regrade.py`); until that wave
  runs, do not quote them.
- **GPU panel uses the HIP arms.** Triton and OpenMP offload are out of scope for this figure.

## 3. The LLR efficacy figure (`efficacy-packets-and-scope`)

![efficacy packets and scope](figures/example-efficacy-packets-and-scope.png)

The exact command that produced the committed figure. Re-running it on the current tree
reproduces the PNG byte for byte:

```bash
python3 statistics/plot_score_change.py "$AR/experiments/llr-gpu/data/llr-gpu.db" \
  --comparison "title=Loop Reasoning CPU (LLR);intervention=lang-skills;pairs=$AR/experiments/llr-cpu/tables/skills_billed.csv;observations=$AR/experiments/llr-cpu/data/llr-cpu.db;placeholders=Fortran" \
  --comparison "title=Loop Reasoning GPU (LLR);intervention=lang-skills;pairs=$AR/experiments/llr-gpu/tables/skills_billed.csv;observations=$AR/experiments/llr-gpu/data/llr-gpu.db;difference=HIP:qwen38,HIP:kimi27sglang" \
  --comparison "title=Repository Context;intervention=repo;pairs=$AR/experiments/git-scicomp/tables/repo-vs-kernel_billed.csv;observations=$AR/experiments/git-scicomp/data/git-scicomp.db;repeats=median;control-label=Kernel Formulation" \
  --cost-model billed --out figures/efficacy-packets-and-scope.pdf --table figures/efficacy-packets-and-scope.csv
```

Drawn by `hpcagent_bench.stats.figures.efficacy.figure_dot_row` (the default `--mode dots`). Two
rows share one set of columns: geomean speed-up on top, billed token cost below. Each column is one
model and delivery; its two marks are the control (hollow circle) and the treated arm (the packet's
shape). Per-comparison options go inside each `--comparison` spec:

| key | effect |
|---|---|
| `placeholders=Fortran` | draws an empty column for a leg with no data yet, so the spacing does not change when it lands |
| `difference=HIP:qwen38,...` | a grey bar between a named pair's two marks, labelled with the factor |
| `repeats=median` | median over designed repeats instead of latest-run-wins |
| `control-label=...` | what the control is called in the legend |

`--cost-model billed` weights tokens as 1 x fresh + 0.1 x re-sent + 1 x output. `*` marks a
speed-up change and `+` a token-cost change significant after Benjamini-Hochberg correction within
the figure. The solve rate, which this figure cannot show, ships as a LaTeX table from
`statistics/table_solve_rate.py` built from the same pair tables.

## Where the code lives, and how to extend it

The rule from [plotting.md](plotting.md): every figure is a function in `hpcagent_bench/stats/`,
and `statistics/plot_*.py` only parses arguments. A missing capability goes into the library, with
a test, never into a script or a paper repository.

| figure | library function | script |
|---|---|---|
| compilers per kernel | `stats/figures/signed.py: llr40_two_row_figure` | `statistics/plot_llr40_compilers.py` |
| optimizer row | `stats/figures/optimizers.py: figure_optimizer_row` | `statistics/plot_optimizer_row.py` |
| LLR efficacy | `stats/figures/efficacy.py: figure_dot_row` | `statistics/plot_score_change.py` |

Shared behaviour these figures rely on, all in the library:

- **Device-variant labels.** The registry maps `dace_cpu_canonicalize` and `dace_gpu_canonicalize`
  to one optimizer, so both are "Canonical Parallel Form". When a figure draws both, each row falls
  back to its `frameworks` name, which carries the device (`signed.distinct_canon_labels`).
- **Placeholders are hollow.** `style.point_mark(..., delivered=False)` always draws an empty face;
  a cross in the series colour on a filled mark of that colour is invisible.
- **Summary values do not overprint.** Value labels tagged `style.SPREAD_GID` are moved apart at
  save time, after the axis limits are final (`style.settle_spread_labels`, called from
  `style.save`).
- **One decimal.** Speed-up values print as `6.3x`, `0.9x`
  (`kernel_comparison.speedup_value_text`); below 0.1x they keep one significant figure so a real
  slowdown never reads `0.0x`.

To add an optimizer to the row: give it a `SHORT_NAMES` entry in `stats/figures/optimizers.py` and
make sure the registry (`hpcagent_bench/envs/registry.yaml`) names it under `optimizers` (shape) and
`frameworks` (colour and display name). To add a track, add a `--panel`.

## Before handing a figure over

- Open the PNG. The legend, tick labels and value labels must not collide.
- Check the CSV's `solved` column against what the text claims.
- Every number is a geomean with a 95% log-t interval over the roster, with an unanswered kernel
  at 1x. If a caption says anything else, the caption is wrong.
