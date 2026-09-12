# Plotting

How to turn a campaign's run directories into a figure: the shared modules every figure script
imports, the extraction step, and the drawing rules each one follows. For the statistics behind
the framework/kernel corpus figures (the speedup heatmap and the per-kernel distribution grid),
see [measurement_statistics.md](measurement_statistics.md).

Three shared modules, all under `hpcagent_bench/stats/` except `experiment_tags.py`. Each rule
below exists because the failure named beside it happened here.

| module | owns |
|---|---|
| `hpcagent_bench/stats/palette.py` | which colour and which SHAPE an entity wears |
| `hpcagent_bench/stats/style.py` | everything that is never data: type, grid, legend, ticks, title |
| `hpcagent_bench/experiment_tags.py` | how a campaign, model or language is SPELLED |

## Extract once, plot from the CSV

```bash
python -m hpcagent_bench.experiments \
    --runs '/scratch/.../hpcagent-bench-runs/llrblind-*' \
    --runs '/scratch/.../hpcagent-bench-runs/6[0-9][0-9][0-9][0-9][0-9]' \
    --experiment llrblind \
    --out data/observations.csv
```

`--experiment` is an arm PREFIX and repeatable: pass every label the campaign used. A campaign
that spelled its completion waves with a different PREFIX than its first wave keeps only half of
itself under one spelling -- which is the reason a wave belongs in a suffix (`<campaign>-w2`), so
one prefix still matches every wave. `--runs` is repeatable for the same reason -- arms spread over named wave roots and
per-job Slurm-id roots. Read the summary line it prints; a missing arm means a wrong prefix, not a
missing campaign. From Python, `experiments.observations(globs, experiment=[...])` returns the same
thing as a DataFrame.

```bash
python scripts/plot_arm_summary.py  data/llr40_observations.csv --experiment llr40v11 \
    --out figures/arm.pdf   --table data/arm.csv      # -speedup, -tokens, -pair
python scripts/plot_score_change.py data/llr40_observations.csv --experiment llr40v11 \
    --out figures/skills.pdf --table data/skills.csv
python scripts/plot_tokens.py       data/llr40_observations.csv --experiment llr40v11 \
    --out figures/tokens.pdf --table data/tokens.csv
```

Each writes a PDF, a PNG, and the TABLE behind the figure -- a figure nobody can check is a claim.
`--experiment` also sets the title, through `experiment_tags.display_name`.

## A new figure

```python
from hpcagent_bench import experiment_tags
from hpcagent_bench.stats import palette
from hpcagent_bench.stats import style as plotstyle

plotstyle.apply()                       # BEFORE importing pyplot
import matplotlib.pyplot as plt

models = sorted(frame.model.unique())
hues, shapes = palette.model_colors(models), palette.model_markers(models)

fig, ax = plt.subplots(figsize=(8.4, 5.2))
for model in models:
    part = frame[frame.model == model]
    ax.scatter(part.x, part.y, color=hues[model], marker=shapes[model], s=130,
               label=experiment_tags.model_name(model))

ax.set_ylabel("Median Tokens per Task")          # Title Case
plotstyle.value_axis(ax, "y", log_base=10.0)     # ticks + grid on the MEASURED axis only
plotstyle.despine(ax)
fig.subplots_adjust(left=0.17, right=0.975, top=0.855, bottom=0.30)
plotstyle.legend_below(fig, ax.get_legend_handles_labels()[0], y=0.02)
plotstyle.title(fig, experiment_tags.display_name("llr40v11"))
fig.savefig("out.pdf", bbox_inches=fig.bbox_inches)
```

## Rules

**Identity is the entity's, not the figure's.** `palette.color(name)` / `palette.colors(names)`
(packets), `palette.model_color(name)` / `palette.model_colors(names)`, and
`palette.framework_color(name)` / `palette.framework_colors(names)` all key by the entity's name,
not by its position in whatever list one figure happened to hold. Indexing by position meant
dropping a column repainted the survivors, and a framework wore one hue in the heatmap and another
in the speed-up chart. Shape is a second channel (`palette.model_markers(names)`) because colour
alone does not survive greyscale or a column-width shrink. Key order in
`hpcagent_bench/envs/registry.yaml` is **append only** -- inserting a name mid-list re-colours
every published figure already drawn (`tests/test_palette.py` pins each entity's colour so a
reorder fails loudly), so an unused name keeps its slot.

**Names come from `hpcagent_bench/envs/registry.yaml`**, not literals. Spelled at three call sites
they disagreed, one figure saying `qwen38` where its neighbour said `Qwen3.8-27B`. Unknown tags
fall back unchanged so a new campaign never breaks a plot; `tests/test_display_names.py` stops that
fallback going unnoticed by checking each model tag against the checkpoint its arms served. Serving
details stay out of names -- `sglang` and `-FP8` are an engine and a precision, and on an axis they
read as different models.

**Log space for ratios.** A speed-up is multiplicative; on a linear axis every slow-down is crushed
into `[0, 1)` and `0.5x` reads as smaller than `1.5x`.

**The baseline is a property of the data.** `hpcagent_bench.stats.figures.results.baseline_of(frame)`
reads the column the judge stamped; `DEFAULT_BASELINE` (`numba`) is only the fallback. v9/v10 graded
against C, v11 against numba -- a figure that picks its own denominator plots a ratio nobody scored.

**Connectors are elbows, never diagonals.** The straight segment passes through coordinates that
were never measured, and on a plot about where something landed a reader takes the path for data.
`plot_score_change.py` draws the horizontal leg first, so the corner sits under the "with skills"
mark and the vertical leg reads as the change in spend.

**Draw no interval you cannot support.** `plot_tokens.py`'s cells hold a handful of episodes drawn
from several different arms, not repeats of one condition -- on llr40v11 every one of its 120 token
cells mixed the skills and no-skills arms. A bootstrap interval or a scatter of those episodes would
say nothing about sampling uncertainty there, since the spread is mostly the treatment, so it draws
the median alone. `plot_arm_summary.py` and `plot_score_change.py` plot one value per KERNEL, so
each median there carries its percentile bootstrap interval over kernels
(`population.kernel_medians`) as a whisker beside the mark, withheld below
`summary.MIN_INTERVAL_SAMPLES` kernels, and the table carries the two median times behind the
speed-up. Whether a difference is real stays `plot_score_change.py`'s paired test.

**Costs add.** A kernel's token spend is the sum over every episode and attempt the arm ran on it
(`population.kernel_tokens`), the cost behind that kernel's answer. A statistic over episodes, such
as `plot_tokens.py`'s `median_episode_tokens`, is a per-episode quantity and says so in its name.

**Rank statistics on these samples.** Per-kernel speed-ups are heavy-tailed and a mean in log space
still lets one 40x kernel carry the estimate. `plot_score_change.py` uses Hodges-Lehmann with a
distribution-free signed-rank interval, paired by kernel -- Mann-Whitney is the unpaired sibling
and throws away most of the precision.

**Paired figures must be the same size.** Fixed figsize, fixed `subplots_adjust` (not
`tight_layout`), legend inside the canvas, and `bbox_inches=fig.bbox_inches`. `bbox_inches=None`
means "use the rcParam", which here is `"tight"` -- so a figure was sized by its own legend.

**Title Case**, except articles, conjunctions and short prepositions; identifiers keep their
spelling (`numba`, `lang-c`). Ticks rotate 0 or 90 degrees, never an angle. If a figure caps
coverage, print what was dropped -- silent truncation reads as "this is everything".

`value_axis()` handles four matplotlib traps once: `AutoMinorLocator` refuses log scales; the
default log locator labels a single tick on a panel under two decades; `LogFormatterSciNotation`
returns `""` for 5x10^n; and integer minor subs leave the 1-to-2 interval empty while 2-to-5 gets
several. `log_base` is passed, never sniffed -- matplotlib keeps it private and a wrong guess puts
minor lines at wrong ratios.

## The figures

| script | figure |
|---|---|
| `plot_arm_summary.py` | per-arm median speed-up and spend; one x slot per LANGUAGE, models dodged inside |
| `plot_score_change.py` | speed-up against spend, two marks per arm joined by an elbow, quadrants named |
| `plot_tokens.py` | median tokens per episode, per kernel, per model |
| `plot_speedup.py` | per-kernel signed speed-up in magnitude bands, per machine (see [measurement_statistics.md](measurement_statistics.md)) |

The first three read the CSV this page's extraction step produces. The speed-up in each comes from
the `submission` rows and the cost from the `call` rows, both reduced by
`hpcagent_bench.stats.population`: the last verified submission per episode then the max across
episodes for score, and the per-episode maximum of the cumulative token counter for cost. One
predicate over both columns keeps only the rows that carry both, which is the call rows alone.

The framework/kernel corpus figures (the speedup heatmap and the per-kernel distribution grid) read
the results DB instead of this CSV; they are documented in
[measurement_statistics.md](measurement_statistics.md).
