# Plotting

How a campaign's run directories become a paper figure: extract once, then draw every figure from
the extracted observations. Statistics behind the corpus figures (speed-up heatmap, per-kernel
distribution grid): [measurement_statistics.md](measurement_statistics.md). Token cost and cards:
[token_accounting.md](token_accounting.md).

## Design contract

Every figure in the HPCAgent-Bench papers follows these rules. A figure that breaks one is wrong.

1. **Library, not script.** Every figure is a function in `hpcagent_bench.stats` (`figures.efficacy`,
   `figures.per_kernel`, `figures.signed`, `figures.optimizers`, `figures.scaling`,
   `figures.kernel_comparison`, `summary`, `palette`, `style`). `statistics/plot_*.py` only parse
   arguments. A missing capability goes into the library with a test, never into a script or a paper
   repository.
2. **Speed-up axis = log2 of the ratio** (`summary.log2_change`): 2x at +1, 0.5x at -1, 0 = no change,
   ticks labeled back in ratios (`style.ratio_tick_label`). Never `signed_change` (ratio - 1), never a
   bare ratio axis.
3. **Statistics per Hoefler and Belli (SC15)**, as `stats.rules` encodes them: paired per kernel, the
   geomean of per-kernel ratios with the log-space t 95% interval (`summary.geomean_ci`) for speed-up
   and cost. Nothing is joined by a trend line (rule 12); the emitted table carries raw milliseconds
   and token counts (rule 4). Speed-up and token cost never share one axis.
4. **Colour = intervention, shape = optimizer.** Colour is the skill, tool, harness or packet
   (`palette.color`; control is a hollow mark in `palette.control_color`). Shape is the optimizer
   (`palette.marker`): an LLM or a standalone optimizer (registry `optimizers`, e.g. DaCe, CPF). CPF
   given to an agent (`cpf`, `cpfsrc`) is a packet, so a colour. Where only one entity varies, colour
   is that entity: `palette.framework_color` (compiler figures), `palette.harness_color`,
   `palette.model_color`. Exception: `figures.efficacy` and `figures.kernel_comparison` colour by
   model (`palette.model_color`) and give the whole panel one packet shape (`palette.packet_marker`),
   since one panel already belongs to one packet.
5. **Efficacy figure.** The paper figure (`--mode dots`, default) stacks one column per model and
   delivery: geomean speed-up, tasks completed, token cost. The 2D form (`--mode paired`) puts log2
   speed-up on X and token cost on Y, one mark per arm with its interval on both axes. Several
   comparisons join as one row of panels (up to three square panels fit a single column).
6. **Per-kernel figure** (MPR/CPF): wide, two rows on one kernel axis: log2 speed-up per kernel with
   its interval on top, tokens per kernel below. Past a dashed separator, one summary slot per
   series: speed-up = geomean with 95% log-t interval over the SOLVED kernels, tokens = median over
   the served kernels. The tokens row is omitted when no series carries tokens.
7. **Missing value = hollow cross.** A kernel without a verified answer enters at 1x
   (`population.NOT_DELIVERED`), is drawn crossed, named `style.NOT_DELIVERED_LABEL` in the key, and
   left out of every summary. Hollow alone means control. A `?` marks data not run yet
   (`--mark-pending`).
8. **Sizing.** Physical inches come from the paper template (`style.ICLR_TEXT_WIDTH_IN`,
   `ACM_COLUMN_WIDTH_IN`, `ACM_TEXT_WIDTH_IN`), never from `\includegraphics` scaling. The data box is
   fixed and the chrome is measured around it (`style.*_protrusion_in`, `per_kernel.fit_canvas`), never
   `tight_layout`; save with `style.save(..., fixed=True)`.
9. **One legend per figure**, below it (`style.legend_below`), never `ax.legend`. It names the
   interval method, n and the undelivered cross.
10. **Minor ticks.** Log axes, linear axes in log2 units and the tasks-completed row carry unlabelled
    minor ticks and a faint minor grid (`style.minor_ticks`, `style.MinorLocator`). Other linear axes
    and category axes carry none. A count over a fixed roster is a census: a mark with no interval.
11. **Deliverable** = PDF, 150 dpi PNG beside it, the CSV behind every mark, and the exact CLI. A bad
    figure is saved, shown with bad-vs-good, and asked about; never silently redrawn.

Further conventions:

- **Identity keys by name.** `palette.color(name)`, `palette.model_color(name)`,
  `palette.framework_color(name)` key by entity, never by list position. Key order in
  `hpcagent_bench/envs/registry.yaml` is append-only; a mid-list insert recolours every published
  figure (`tests/test_palette.py` pins colours).
- **Names come from the registry** through `hpcagent_bench.experiment_tags` (`display_name`,
  `model_name`, `packet_name`, `framework_name`), never literals. Serving details (`sglang`, `-FP8`)
  stay out of names.
- **Baseline is a property of the data**: `figures.results.baseline_of(frame)` reads the column the
  judge stamped; `DEFAULT_BASELINE` (`numba`) is only the fallback.
- **Costs add.** A kernel's tokens are the sum over the tasks the arm ran on it
  (`population.kernel_tokens`); medians are taken over kernels, never over episodes in a cell.
- **Intervals.** `summary.geomean_interval` (one arm's own kernels) uses log-t at or above
  `summary.LOG_T_MIN_SAMPLES` (20) samples and a log-space bootstrap below. Paired figures always use
  `summary.geomean_ci`. No interval under `summary.MIN_INTERVAL_SAMPLES` kernels.
- **Labels.** Title Case (identifiers keep their spelling). Ticks at 0 or 90 degrees. Values print
  with one decimal (`style.ratio_label`: `6.3x`, `0.04x` below 0.1x); tokens with
  `style.decade_label` (`35.5K`). Labels beside marks are tagged `style.CLEAR_GID` and settled clear
  at save (`style.settle_clear_labels`). If a figure caps coverage, it prints what was dropped.
- **A connector is a pair link**, never a trend: it joins one arm's control and treated marks, and the
  legend names it `Pair Link`.

## Extract once, plot from the observations

```bash
python -m hpcagent_bench.experiments \
    --runs "$RUN_ROOT/llrblind-*" --runs "$RUN_ROOT/6[0-9][0-9][0-9][0-9][0-9]" \
    --experiment llrblind --out data/observations.csv
```

`--runs` is a run-root glob and `--experiment` an arm prefix; both repeat. Keep waves in a suffix
(`<campaign>-w2`) so one prefix matches every wave. Check the printed summary: a missing arm means a
wrong prefix. From Python: `hpcagent_bench.experiments.observations(globs, experiment=[...])`.

Registered experiments (`hpcagent_bench.campaigns`) extract by name, and fuse regrades:

```bash
python -m hpcagent_bench.dataset --experiment llr-focus40-blind \
    --regrades "$RUN_ROOT/regrades/regrade-*.db" --out data/llrblind.db --csv data/llrblind.csv
```

Without `--regrades`, an unstamped row is refused rather than mixed with the current timing rule.
An answer scored correct but never submitted counts once promoted: `hpcagent-bench regrade worklist
--scope unpromoted`, then `regrade run`, then `regrade promote-apply` (or extraction with
`--regrades`) adds it as a `promoted-unsubmitted` submission.

Speed-up comes from `submission` rows, cost from `task` rows, both reduced by
`hpcagent_bench.stats.population` (latest valid submission per kernel; the task's final-attempt
tokens). A predicate over both columns at once keeps neither record type.

## The figures

| script | figure | library |
|---|---|---|
| `plot_score_change.py` | efficacy: speed-up, tasks completed and token cost per comparison | `figures.efficacy.figure_dot_row`, `figure_row` |
| `plot_optimizer_row.py` | one row of 1-D panels, speed-up only, LLM arms beside compilers over one roster | `figures.optimizers.figure_optimizer_row` |
| `plot_llr40_compilers.py` | llr-focus40 per kernel: canon columns, Pluto, PPCG-HIP, optional CPF arms | `figures.signed.llr40_two_row_figure` |
| `plot_kernel_comparison.py` | llr-focus40 per kernel: DaCe canon CPU and every complete agent arm | `figures.kernel_comparison` + `per_kernel` |
| `plot_per_kernel.py` | one selection's per-kernel speed-up and tokens (`--style ci\|box`, `--layout separate\|stacked`) | `figures.per_kernel.figure_panels` |
| `plot_repo_vs_kernel.py` | one pair's per-kernel ratio, speed-up over tokens | `figures.per_kernel` |
| `plot_arm_summary.py` | per-arm geomean speed-up and median spend, one slot per language | `stats.summary`, `palette` |
| `plot_tokens.py` | tokens per kernel, per model | `population.kernel_tokens` |
| `plot_single_shot_score.py` | blind arm funnel: reached, correct, faster | `palette`, `style` |
| `plot_scaling.py` | distributed track: eta(P), sigma(P), per-kernel, per-arm summary | `figures.scaling` |
| `plot_canon_speedup.py` | median speed-up per framework from one canon sweep (`--db`) | `stats.canon` |
| `plot_parallelism.py` | SDFG parallelism taxonomy per DaCe column (`--db`) | `metrics.parallelism` |
| `plot_speedup.py`, `plot_results.py` | corpus figures from the results DB | see [measurement_statistics.md](measurement_statistics.md) |

`table_solve_rate.py` writes the solve-rate LaTeX table that goes beside the efficacy figure.
Run any script with `-h` for its flags.

Quick looks at one campaign:

```bash
python statistics/plot_arm_summary.py data/observations.csv --experiment llr40v11 \
    --out figures/arm.pdf --table data/arm.csv
python statistics/plot_tokens.py data/observations.csv --experiment llr40v11 \
    --out figures/tokens.pdf --table data/tokens.csv
python statistics/plot_per_kernel.py data/observations.csv --experiment llr40v11 \
    --style ci --layout stacked --out figures/per-kernel.pdf --table data/per-kernel.csv
```

## The paper figures, end to end

Every command writes the PDF, a PNG beside it and the CSV behind every mark. Before using a figure,
open the PNG (legend, ticks and value labels must not collide) and check the CSV's `solved` column
against what the text claims.

### Setup and inputs

```bash
export HPCAGENT_BENCH_REPO=$PWD
export PYTHONPATH="$HPCAGENT_BENCH_REPO:$HPCAGENT_BENCH_REPO/hpcagent_bench/numpy_translators/src"
export MPLBACKEND=Agg PYTHONHASHSEED=0            # headless, byte-reproducible
export AR=/path/to/reproducibility-artifact       # per-track observations + pair tables
export CANON_DB=/path/to/results/canon.db         # canon sweep, table `canon`
```

| input | what | from |
|---|---|---|
| `$AR/experiments/<track>/data/<track>.{csv,db}` | observations | extraction, above |
| `$AR/experiments/<track>/tables/*_billed.csv` | pair tables | `statistics/paired_arms.py` |
| `$CANON_DB` | median time per (compiler column, kernel), validated only | canon sweep |
| roster file | kernels a track is scored over, one per line | derived below |

Derive the llr-focus40 roster from the kernels its control arm was served:

```bash
python3 -c "
import pandas as pd
d = pd.read_csv('$AR/experiments/llr-cpu/data/llr-cpu.csv', low_memory=False)
print('\n'.join(sorted(set(d[d.arm == 'cpf-llr-focus40-kimi27sglang-c'].benchmark.astype(str)))))
" > roster-llr-focus40.txt
```

Build a pair table (one per comparison; `--policy solved` is the default and is stamped on the CSV,
and the figure refuses a table built under another policy or card):

```bash
python statistics/paired_arms.py --observations "$AR/experiments/llr-cpu/data/llr-cpu.csv" \
    --pair cpf-llr-focus40-qwen38-c,cpf-llr-focus40-qwen38-c-skills --family skills \
    --cost-model billed --out "$AR/experiments/llr-cpu/tables/skills_billed.csv"
```

### 1. Compilers per kernel

![compilers per kernel](figures/example-compilers-per-kernel.png)

```bash
python3 statistics/plot_llr40_compilers.py \
    --canon-db "$CANON_DB" --roster-file roster-llr-focus40.txt \
    --canon-columns pluto,dace_cpu_canonicalize,dace_gpu_canonicalize,ppcg_hip \
    --offset 0.6 --out figures/compilers-per-kernel
```

- Numba is the denominator (the 1x line, `--baseline` changes it). Pluto and PPCG are comparators.
- Filled mark = measured. Hollow crossed mark = no validated result, drawn at 1x, kept as a row of
  `-kernels.csv` (`canon.roster_speedups`), left out of the summary; read the `n` column of
  `-summary.csv` before quoting a geomean.
- `--observations` adds every model's CPF arm. `--offset` spreads a kernel's series across its slot;
  0 stacks them.
- When both DaCe device columns appear, each falls back to its `frameworks` name, which carries the
  device (`signed.distinct_canon_labels`).

### 2. Optimizer row, speed-up only

![optimizer row](figures/example-optimizer-row.png)

```bash
python3 statistics/plot_optimizer_row.py --canon-db "$CANON_DB" \
    --panel "title=Loop Reasoning CPU (LLR);observations=$AR/experiments/llr-cpu/data/llr-cpu.csv;arms=cpf-llr-focus40-{model}-c;compilers=dace_cpu_canonicalize,pluto;baseline=numba;roster=roster-llr-focus40.txt" \
    --panel "title=Loop Reasoning GPU (LLR);observations=$AR/experiments/llr-gpu/data/llr-gpu.csv;arms=gpu-llr-focus40-{model}-hip;compilers=dace_gpu_canonicalize,ppcg_hip;baseline=numba;roster=roster-llr-focus40.txt" \
    --panel "title=Repository Formulation;observations=$AR/experiments/git-scicomp/data/git-scicomp.csv;arms=git-scicomp-{model}-repo;baseline=c-autopar;repeats=median" \
    --out figures/optimizer-row.pdf
```

| `--panel` key | meaning |
|---|---|
| `title` | panel subtitle (required) |
| `observations` | observations file with the LLM arms; omit for a compilers-only panel |
| `arms` | arm template with `{model}` |
| `models` | comma list of model tags; default `--models` |
| `compilers` | canon columns; needs `--canon-db` and `roster` |
| `baseline` | denominator column (`numba`, `c-autopar`), printed as "1x = ..." |
| `baseline_name` | override that note's text |
| `repeats` | `latest` (default) or `median` (designed repeats) |
| `roster` | roster file; without it an arm is scored over the kernels it was served |

One mark per optimizer: geomean speed-up over the panel's baseline with its 95% log-t interval. LLM
arms and compilers are scored over the same roster, an unanswered kernel at 1x for both; the script
prints and the CSV carries `solved` and `kernels` per mark. Panels share one log2 axis but not one
denominator, so each names its own baseline. To add an optimizer, give it a `SHORT_NAMES` entry in
`stats/figures/optimizers.py` and register it under `optimizers` (shape) and `frameworks` (colour,
name) in `registry.yaml`.

### 3. Efficacy figure (`efficacy-packets-and-scope`)

![efficacy packets and scope](figures/example-efficacy-packets-and-scope.png)

```bash
python3 statistics/plot_score_change.py "$AR/experiments/llr-gpu/data/llr-gpu.db" \
  --comparison "title=Loop Reasoning CPU (LLR);intervention=lang-skills;pairs=$AR/experiments/llr-cpu/tables/skills_billed.csv;observations=$AR/experiments/llr-cpu/data/llr-cpu.db;placeholders=Fortran" \
  --comparison "title=Loop Reasoning GPU (LLR);intervention=lang-skills;pairs=$AR/experiments/llr-gpu/tables/skills_billed.csv;observations=$AR/experiments/llr-gpu/data/llr-gpu.db;difference=HIP:qwen38,HIP:kimi27sglang" \
  --comparison "title=Blind (LLR CPU);intervention=lang-skills;pairs=$AR/experiments/llrblind/tables/skills_billed.csv;observations=$AR/experiments/llrblind/data/llrblind.db" \
  --comparison "title=Repo. Context;intervention=repo;pairs=$AR/experiments/git-scicomp/tables/repo-vs-kernel_billed.csv;observations=$AR/experiments/git-scicomp/data/git-scicomp.db;repeats=median;control-label=Kernel Formulation" \
  --cost-model billed --row-width acm-text \
  --out figures/efficacy-packets-and-scope.pdf --table figures/efficacy-packets-and-scope.csv
```

Drawn by `figures.efficacy.figure_dot_row`. Each column is one model and delivery: control = hollow
circle, treated = the packet's shape. Rows:

- **Speed-up**: geomean over the kernels both arms solved; a wrong answer is left out, a correct
  slower answer keeps its sub-1 ratio. `--speedup-over served` draws every kernel with a failure at 1x.
- **Tasks completed**: kernels solved per arm on a 0..N axis, no interval (census).
  `--no-success-row` drops it.
- **Token cost**: every served kernel, failed ones included, priced with `--cost-model`.

`*` marks a significant speed-up change and `+` a significant token-cost change after
Benjamini-Hochberg correction within the figure. An interval is cut at
`FigureConfig.interval_reach` past the outermost mark with an arrowhead; a mark over fewer than
`FigureConfig.min_interval_kernels` kernels has none.

`--comparison` is a `key=value;...` spec, one per panel. Either `treatment=<packet>` (split one
campaign on a recorded packet) or `pairs=<csv>` (pairs from `paired_arms.py`, whose corrected
verdicts are the stars; the figure recomputes only the drawn point through
`figures.efficacy.reduce_pair`). Other keys:

| key | effect |
|---|---|
| `title`, `intervention` | panel title; registered packet key the treated side wears (hue, name) |
| `observations=a.db,b.db` | observations for this panel; default the positional files |
| `control-label=...` | legend name of a control that is not "no packet" |
| `repeats=median` | median over designed repeats instead of latest run |
| `placeholders=Fortran` | empty column for a leg with no data yet |
| `pending=kimi27sglang,qwen38` | empty column per model with no pair yet; `?` with `--mark-pending` |
| `difference=HIP:qwen38,...` | grey bar between a named pair's two marks, with its factor |

A single comparison can also use top-level flags:

```bash
python statistics/plot_score_change.py scored.csv blind.csv \
    --pairs-csv blind_vs_scored.csv --intervention no-score --control-label "Score Tool" \
    --out figures/blind.pdf --table data/blind.csv
```

`--row-width {natural,iclr,acm-column,acm-text}` sizes a joined row to a page budget. `--mode paired`
draws the 2D form; `--show-cloud` adds the per-kernel paired cloud behind it.

The solve-rate table from the same pair tables:

```bash
python statistics/table_solve_rate.py "$AR/experiments/llr-cpu/data/llr-cpu.db" \
    --pairs-csv "$AR/experiments/llr-cpu/tables/skills_billed.csv" --intervention lang-skills \
    --out tables/solve-rate.tex
```

## Scaling figures

`statistics/plot_scaling.py` draws the distributed track from the same observations, rows with
`record == "scaling"`. Required columns: `ranks` (P), `ranked_ns` (T(P)), `single_rank_ns` (T(1));
optional: `scaling_mode` (`weak`/`strong`), `nodes`, `work_ratio` (r; missing on a weak row means
r = P), `scaling_note`. The judge persists `scaling_points` and `scaling_curves`
(`harness.recording.record_scaling`); extraction turns them into scaling rows.

Every point goes through `harness.metric.scaling_point`, the function the grade uses:
eta(P) = T(1)/(P T(P)) strong, r T(1)/(P T(P)) weak. A row whose recorded `efficiency` disagrees is
refused (`figures.scaling.disagreements`). A P the sweep could not measure is a hole, never a zero,
and is listed with its reason in `<table>-dropped.csv`. P is a log2 axis with ticks at the rank
counts run and no grid. Weak and strong are panels; colour and shape are the model.

```bash
OBS="$AR/data/mlscale_observations.csv"
python statistics/plot_scaling.py "$OBS" --experiment mlscale --out figures/scaling --table data/scaling.csv
python statistics/plot_scaling.py "$OBS" --experiment mlscale --figure efficiency --out figures/scaling
python statistics/plot_scaling.py "$OBS" --experiment mlscale --figure speedup --out figures/scaling
python statistics/plot_scaling.py "$OBS" --experiment mlscale --figure per-kernel --mode strong \
    --quantity efficiency --out figures/scaling
python statistics/plot_scaling.py "$OBS" --experiment mlscale --figure summary --out figures/scaling
python statistics/plot_scaling.py "$OBS" --arm 'mlscale-qwen38-hip' --width 5.5 --out figures/scaling-qwen38
```

## A new figure

Add a function under `hpcagent_bench/stats/figures/` with a test, then a thin script in `statistics/`:

```python
import pathlib

from hpcagent_bench import experiment_tags
from hpcagent_bench.stats import palette
from hpcagent_bench.stats import style as plotstyle

plotstyle.apply()                       # before importing pyplot
import matplotlib.pyplot as plt

models = sorted(frame.model.unique())
hues, shapes = palette.model_colors(models), palette.model_markers(models)
fig, ax = plt.subplots(figsize=(8.4, 5.2))
for model in models:
    part = frame[frame.model == model]
    ax.scatter(part.x, part.y, color=hues[model], marker=shapes[model], s=130,
               label=experiment_tags.model_name(model))
ax.set_ylabel("Median Tokens per Task")
plotstyle.value_axis(ax, "y", log_base=10.0)     # grid on the measured axis only
plotstyle.despine(ax)
plotstyle.legend_below(fig, ax.get_legend_handles_labels()[0], y=0.02)
plotstyle.title(fig, experiment_tags.display_name("llr40v11"))
plotstyle.save(fig, pathlib.Path("figures/out"), fixed=True)   # writes .pdf and .png
```
