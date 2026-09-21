# Design contract for statistics and figures

Set by the user on 2026-09-16. Every figure in the HPCAgent-Bench, agentbench and MPR/CPF papers follows it;
a figure that does not is wrong, and the drawing agent returns a self-check table against these items.

1. API: every figure is drawn by a function in `hpcagent_bench.stats` (`figures.signed`, `figures.efficacy`,
   `figures.per_kernel`, `summary`, `palette`, `style`); `statistics/plot_*.py` only parse arguments. A
   missing capability is added to the library, never worked around in a script or a paper repository.
2. Speed-up axis = log2 of the speed-up (`summary.log2_change`): 2x at +1, 0.5x at -1, 0 = no change, ticks
   labeled back in ratios (0.25x .. 16x). Never `signed_change` (ratio - 1) and never a bare ratio axis.
3. Statistics follow Hoefler and Belli (SC15) as `hpcagent_bench.stats.rules` encodes them: paired per kernel,
   the GEOMEAN of per-kernel ratios with the log-space t 95% interval (`summary.geomean_ci`) for speed-up AND
   cost; every series is a scatter of its per-kernel values plus that summary as an error bar; nothing is joined
   by a line (rule 12); the emitted table carries the raw milliseconds and token counts (rule 4).
4. Channels: colour = the skill / tool / harness / packet (the intervention, registry hue via `palette.color`;
   compiler columns `palette.framework_color`; control = hollow mark in `palette.control_color`);
   shape = the OPTIMIZER (`palette.marker`): an LLM, or a standalone optimizer such as DaCe or CPF
   (registry `optimizers`, shapes after the models). CPF given to an agent (`cpf`, `cpfsrc`) is a packet, a colour.
5. Efficacy figure = 2D: X = log2 speed-up geomean with its interval, Y = token-cost geomean with its interval,
   one mark per arm; n comparisons = one row of n square panels (up to 3). The agentbench paper's row is
   kernel formulation | language skill packet | three languages.
6. Per-kernel figure (MPR/CPF) = wide, two rows sharing the kernel axis: log2 speed-up per kernel with its
   interval on top, tokens per kernel as a scatter below; past a dashed separator one summary slot per series
   on each row (2026-09-21: speed-up = geomean with its interval over the SOLVED kernels, tokens = median over
   the served kernels); the tokens row is omitted when no series carries tokens; the hollow cross marks only a
   missing value and is excluded from the summary.
7. Sizing for A4/letter papers: a 3-square row or a 2-long-row figure spans the full text width, a single
   square one column; physical inches come from the paper template (`style` constants), never from
   `\includegraphics` scaling. One shared legend per figure, deduplicated across panels, naming the interval
   method and n and the undelivered cross.
8. Deliverable = the 150 dpi PNG beside the PDF, the exact CLI, and the CSV beside the figure. A bad figure
   is saved and shown to the user with bad-vs-good, then asked about; it is never silently redrawn.

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
python statistics/plot_arm_summary.py  data/llr40_observations.csv --experiment llr40v11 \
    --out figures/arm.pdf   --table data/arm.csv      # -speedup, -tokens, -pair
python statistics/plot_score_change.py data/llr40_observations.csv --experiment llr40v11 \
    --out figures/skills.pdf --table data/skills.csv
python statistics/plot_tokens.py       data/llr40_observations.csv --experiment llr40v11 \
    --out figures/tokens.pdf --table data/tokens.csv
```

Each writes a PDF, a PNG, and the TABLE behind the figure -- a figure nobody can check is a claim.
`--experiment` also sets the title, through `experiment_tags.display_name`.

## A comparison whose pairs are not a packet suffix

`--treatment` splits ONE campaign on a recorded packet, which is all it can do. Two comparisons do
not fit that and still get the same two panels:

* **llrblind against the scored arms**: the two sides are two CAMPAIGNS with different arm prefixes.
* **git-scicomp**: the condition is a `kernel`/`repo` scope, not a packet the arm staged.

Both go through `--pairs-csv`, which reads the family CSV `statistics/paired_arms.py` already
writes. Its `arm_a,arm_b` rows ARE the pairs and its corrected verdicts ARE the stars; the figure
computes only the drawn point, through
`hpcagent_bench.stats.figures.efficacy.reduce_pair` every other panel uses. The statistic therefore
keeps ONE definition: a figure that re-derived it could star a pair the paper's table calls not
significant, and a reader would have no way to tell which of the two is the finding.

```bash
python statistics/plot_score_change.py scored.csv blind.csv \
    --pairs-csv blind_vs_scored.csv --intervention no-score --label "No Score Tool" \
    --out figures/blind.pdf --table data/blind.csv

python statistics/plot_score_change.py git_scicomp.csv \
    --pairs-csv git_scicomp_pairs.csv --intervention repo --label "Whole Repository" \
    --control-label "Bare Kernel" --out figures/scicomp.pdf --table data/scicomp.csv
```

`--intervention` is the registered packet key the TREATED side wears, so the hue and the display
name come from `registry.yaml` like any other intervention. `--control-label` names a control that
is not the absence of a packet: git-scicomp's is the bare kernel and llrblind's kept its score tool,
and "No Packet" names neither. The per-arm label is the language plus every packet BOTH arms carried
(`C`, `Fortran`, `C +skills`, `Fortran +skills`) -- never the intervention, which the title and the
legend already say once.

`--comparison` joins several such comparisons -- a packet-suffix split and an explicit pair list
alike -- into ONE row in ONE call: `'title=...;intervention=...;treatment=...'` or
`'title=...;intervention=...;pairs=<csv>[;control-label=...][;observations=a.csv,b.csv]'`, repeated
once per panel. `--row-width {natural,iclr,acm-column,acm-text}` sizes the joined row to a panel's
own natural width or to a paper's page budget
(`hpcagent_bench.stats.style.ICLR_TEXT_WIDTH_IN`/`ACM_COLUMN_WIDTH_IN`/`ACM_TEXT_WIDTH_IN`) so the
PDF drops into the page at scale 1.0 instead of being shrunk by `\includegraphics`.

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
fig.subplots_adjust(left=0.17, right=0.975, top=0.855, bottom=0.30)  # a paper figure measures these: Layout
plotstyle.legend_below(fig, ax.get_legend_handles_labels()[0], y=0.02)
plotstyle.title(fig, experiment_tags.display_name("llr40v11"))
fig.savefig("out.pdf", bbox_inches=fig.bbox_inches)
```

## The drawing conventions

These seven are not preferences. A figure that breaks one is wrong, and the test named beside each
one fails when it does.

**1. The efficacy figure is 2D: log2 speed-up on X, paired token cost on Y.** X is the paired
speed-up geomean as `log2(ratio)` (`summary.log2_change` of `summary.geomean_ci`): 0 is no change,
+1 is 2x faster, -1 is 2x slower, +2 is 4x -- a LINEAR scale in the exponent, so a 74x kernel does
not drag a modest win halfway across the panel, with the ticks read back in ratios
(`hpcagent_bench.stats.style.ratio_tick_label`) exactly like every other speed-up
axis here -- never a bare ratio axis ticked in raw ratios. Y is the paired token-cost geomean,
treated over control, ALSO a `geomean_ci` interval, on its own log scale (SC15 Rule 4: a speed-up
and a spend are different measurements and never share one). ONE mark per arm, crossed with its 95%
interval on both axes; the per-kernel paired cloud behind it is opt-in (`--show-cloud`, default
off -- one comparison's cloud already crowds a square panel past legibility once every kernel is a
dot), and nothing is ever joined by a line (SC15 Rule 12). A bar or a slope figure elsewhere in this repo keeps its measured VALUE on Y and its categories
on X -- that older rule still holds for `plot_arm_summary.py` and the per-kernel figures, which are
not paired ratios. Pinned by `tests/test_plot_score_change.py`'s
`test_x_is_log2_of_the_speed_up_and_zero_is_the_no_change_line` and
`test_y_is_the_paired_token_cost_ratio_with_one_at_the_control`.

**2. Colour is the INTERVENTION, shape is the OPTIMIZER** (an LLM, or a standalone optimizer from the
registry's `optimizers` block, e.g. DaCe and CPF on the MPR compiler figure). `palette.color(packet)` for the treated side
and `palette.control_color()` for the hollow control reference; `palette.marker(model)` for the
shape. An intervention is anything an arm was given or denied, not only a skill packet: `kernel`
(the bare kernel), `repo` (the whole repository) and `no-score` (the blind condition) are registered
packet keys and wear their own global hues, and the blind comparison is drawn by the same paired
figure as every other packet rather than by a figure of its own. Colouring by MODEL wherever a
packet also varies is the bug: it spends the intervention's channel on the entity the shape already
carries, so one arm reads as a different treatment in every figure it appears in. Where only ONE
entity varies the colour is that entity: `palette.framework_color` for the canon compiler figure (no
agent in it), `palette.harness_color` for the harness comparison (claude / miniswe / openhands /
optimas), and `palette.model_color` for a figure whose only axis is which LLM ran.

**This inverts on `hpcagent_bench.stats.figures.efficacy` and `kernel_comparison`** (2026-09-19):
one panel already belongs to one packet (rule 3), so its shape carries no information, while the
few models sharing that panel need telling apart when their summary marks overlap -- checked by
rendering both orders on the same llr40 figure, where a shared hue and only a circle-vs-square edge
read far worse. Those two modules read colour off `palette.model_color` and shape off
`palette.packet_marker` (one shape for the whole panel) instead of `palette.color`/`palette.marker`
-- see `palette`'s own module docstring. Pinned by `tests/test_plot_score_change.py`'s
`test_the_filled_mark_wears_the_model_colour_and_the_hollow_control_wears_the_control_colour`
and `test_the_marker_shape_is_the_packet_and_nothing_else`.

**3. Several comparisons join as ONE ROW of square panels.** `hpcagent_bench.stats.figures.
efficacy.figure_row`/`panel_side`: every comparison is a square panel against its own control, and a
row is the alternatives-not-a-sequence shape (up to three fit a paper's single column at scale 1.0;
`--row-width` scales a wider row to a paper's own text width). EVERY intervention draws through the
same panel, whichever way its pairs were formed -- see "A comparison whose pairs are not a packet
suffix" below. Pinned by `tests/test_plot_score_change.py`'s
`test_n_comparisons_draw_one_row_of_n_square_panels`.

**4. Major grid only.** `style.value_axis` draws it, on both axes here (each carries a measured
ratio), and switches every minor line and minor label off. A minor line is a second grid at a second
weight, and once a figure is reduced for print the panel reads as a texture the marks sit on rather
than a reference they sit against. Pinned by `tests/test_plot_score_change.py`'s
`test_neither_axis_enables_a_minor_grid`.

**5. One legend, on the FIGURE.** `style.legend_below(fig, handles, ...)`, once per figure, never
`ax.legend`. A key on each panel of a multi-panel figure invites reading the panels as different
sets of series when they draw the same ones. Pinned by `tests/test_plot_score_change.py`'s
`test_the_legend_is_drawn_once_on_the_figure_and_never_on_an_axes`.

**6. A kernel the arm never delivered carries a cross.** It enters the paired ratio at
`population.NOT_DELIVERED` and its tokens still count, so it is a placeholder and not a measurement.
The per-kernel cloud draws it as a small cross in the arm's own colour instead of a dot, and the key
names it `style.NOT_DELIVERED_LABEL` (`No Verified Answer (Scored 1x)`). Hollow alone will not do --
hollow is this repo's spelling for the control. Pinned by `tests/test_plot_score_change.py`'s
`test_an_undelivered_kernel_draws_a_cross_and_the_legend_names_it`, and by `tests/test_style_save.py`'s
`test_a_point_mark_that_never_delivered_an_answer_carries_a_cross_on_the_model_shape` for the shared
mark itself.

**7. A per-arm label moves to the first clear place around its mark, never on top of another.**
`efficacy.untangle_labels` tries a ring of candidate offsets (right of the mark first) and keeps the
first whose rendered box touches no mark and no label settled before it. Call it once the layout is
final: a label's offset is in points, so a place clear before `subplots_adjust` need not be clear
after it. Pinned by `tests/test_plot_score_change.py`'s
`test_no_two_arm_labels_overprint_each_other_however_close_the_arms_land`.

## Layout

- **Chrome is measured, never a fixed fraction.** `style.left_protrusion_in`, `right_protrusion_in`,
  `above_protrusion_in` and `below_protrusion_in` read how far an axes draws past its frame;
  `figures.per_kernel.fit_canvas` and `figures.efficacy.figure_dot_row` size the margins, the gaps,
  the names band and the key from them. The data box is the fixed quantity; the canvas grows to fit.
- **Labels beside marks settle at save.** An annotation tagged `style.CLEAR_GID` is moved clear of
  the marks and of the labels settled before it, inside its frame, by `style.settle_clear_labels`
  (called from `style.save`), once every limit is final.
- **Crowded names step down.** `style.shrink_crowded_ticks` shrinks X tick labels together to a
  floor and warns there; a Y label taller than its panel is broken onto two lines, then shrunk
  (`per_kernel.fit_ylabels`). Kernel names are the manifest short name, folded, never cut
  (`per_kernel.kernel_tick_label`).
- **One per-kernel API.** `figures.per_kernel` draws every figure with a kernel axis: cells and
  their status marks (measured, undelivered crossed, pending `?`, flagged `*`), the dodge, one
  summary slot per series, the canvas. `figures.kernel_comparison`, `figures.signed.llr40_rows` and
  `plot_repo_vs_kernel.py` are data sides that hand it `KernelCell`s.
- **Values beside marks: one decimal,** `style.ratio_label` (`6.3x`, `0.9x`; `0.04x` below 0.1x);
  token counts `style.decade_label` (`35.5K`).
- **Summaries: speed-up = geomean with its 95% log-t interval over the SOLVED kernels** (a
  placeholder is drawn crossed, never summarized); **tokens = median over the served kernels**. The
  table written beside the figure carries the same numbers.

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

**A connector is a PAIR LINK, never a trend** (SC15 Rule 12). The segment from an arm's control
mark to its treated mark says the two marks are one arm; its length is the size of the effect and
its direction the sign. It claims nothing about the space between the two conditions, and the legend
names it `Pair Link` so a reader is not left to guess.

**Draw no interval you cannot support.** `plot_tokens.py`'s cells hold a handful of episodes drawn
from several different arms, not repeats of one condition -- on llr40v11 every one of its 120 token
cells mixed the skills and no-skills arms. A bootstrap interval or a scatter of those episodes would
say nothing about sampling uncertainty there, since the spread is mostly the treatment, so it draws
one mark per cell and nothing else. `plot_arm_summary.py` and `plot_score_change.py` plot one value per KERNEL, so
each median there carries its percentile bootstrap interval over kernels
(`population.kernel_medians`) as a whisker beside the mark, withheld below
`summary.MIN_INTERVAL_SAMPLES` kernels, and the table carries the two median times behind the
speed-up. Whether a difference is real stays `plot_score_change.py`'s paired test.

**Costs add.** A kernel's token spend is the sum over the tasks the arm ran on it
(`population.kernel_tokens`), the cost behind that kernel's answer, and every figure and table here
costs a kernel at that one number -- `plot_tokens.py` included, whose cells are `kernel_tokens`.
A median is then taken over KERNELS, never over the episodes inside a cell: those are a different
quantity with a different unit, and one figure using them made a kernel read 400 on this page and
800 on every other.

**Rank statistics on these samples.** Per-kernel speed-ups are heavy-tailed and a mean in log space
still lets one 40x kernel carry the estimate. `plot_score_change.py` pairs by kernel -- Mann-Whitney
is the unpaired sibling and throws away most of the precision.

**Every ratio figure shows the geomean AND its interval,** and says which interval it is showing.
`summary.geomean_interval` picks the log-t interval at or above `summary.LOG_T_MIN_SAMPLES` (20)
samples and a log-space bootstrap below it, for `plot_arm_summary.py`, which reduces ONE arm's own
kernels. The efficacy figure (`plot_score_change.py`) is a PAIRED comparison instead, and both of
its axes draw `summary.geomean_ci` -- always the log-space t interval, at any n, the same estimator
every per-kernel figure's summary slot (`figures.per_kernel.summary_point_speedup`) and
`hpcagent_bench.stats.figures.signed`'s TSVC rows draw. The method and the n go in the legend
text either way, because two differently derived intervals drawn the same way are two claims a
reader cannot separate. See [measurement_statistics.md](measurement_statistics.md).

**Paired figures must be the same size.** A fixed data box (panel width and height) with its bands
measured around it (Layout, below), never `tight_layout`; legend inside the canvas; saved at its own
canvas (`style.save(..., fixed=True)`, i.e. `bbox_inches=fig.bbox_inches`). `bbox_inches=None` means
"use the rcParam", which here is `"tight"` -- so a figure was sized by its own legend.

**Title Case**, except articles, conjunctions and short prepositions; identifiers keep their
spelling (`numba`, `lang-c`). Ticks rotate 0 or 90 degrees, never an angle. If a figure caps
coverage, print what was dropped -- silent truncation reads as "this is everything".

`value_axis()` handles four matplotlib traps once: `AutoMinorLocator` refuses log scales; the
default log locator labels a single tick on a panel under two decades; `LogFormatterSciNotation`
returns `""` for 5x10^n; and integer minor subs leave the 1-to-2 interval empty while 2-to-5 gets
several. `log_base` is passed, never sniffed -- matplotlib keeps it private and a wrong guess puts
minor lines at wrong ratios.

## The figures

Exact commands for the paper figures: [plotting_handoff.md](plotting_handoff.md).

| script | figure |
|---|---|
| `plot_arm_summary.py` | per-arm median speed-up and spend; one x slot per LANGUAGE, models dodged inside |
| `plot_score_change.py` | one comparison as one square panel: log2 speed-up on X, paired token cost on Y, one mark per arm |
| `plot_tokens.py` | tokens per kernel, per model |
| `plot_repo_vs_kernel.py` | one pair's per-kernel RATIO, speed-up over tokens, on one kernel axis (drawn by `per_kernel`) |
| `plot_speedup.py` | per-kernel signed speed-up in magnitude bands, per machine (see [measurement_statistics.md](measurement_statistics.md)) |
| `plot_optimizer_row.py` | one row of 1-D panels, speed-up only: LLM arms beside compilers (canon columns), one mark per optimizer, geomean with its 95% interval over one roster, each panel naming its own baseline (`hpcagent_bench.stats.figures.optimizers.figure_optimizer_row`; recipe in [plotting_handoff.md](plotting_handoff.md)) |
| `plot_per_kernel.py` | one selection's per-kernel speed-up and tokens (ci or box), separate or stacked (`hpcagent_bench.stats.figures.per_kernel`) |
| `plot_kernel_comparison.py` | llr-focus40: DaCe canon CPU and every complete agent arm, log2 speed-up over tokens on one kernel axis, all models in one panel, a summary slot per series (`kernel_comparison` picks the values, `per_kernel` draws them) |
| `plot_llr40_compilers.py` | llr-focus40: DaCe's own canon-sweep columns, the polyhedral compiler baselines (Pluto, `ppcg_hip`; a roster kernel either has no validated result for enters at 1x, crossed, left out of the summary -- `hpcagent_bench.stats.canon.roster_speedups`), and every model's CPF arm, log2 speed-up over tokens on one kernel axis, a summary slot per row on both panels (`hpcagent_bench.stats.figures.signed.llr40_two_row_figure`, drawn by `per_kernel`) |

The first three read the CSV this page's extraction step produces. The speed-up in each comes from
the `submission` rows and the cost from the `task` rows, both reduced by
`hpcagent_bench.stats.population`: the last verified submission per episode then the max across
episodes for score, and the task's own effective total for cost (`population.kernel_tokens`; see
[token_accounting.md](token_accounting.md) for why that total is the final attempt's). The two come
off DIFFERENT record types, so one predicate over both columns keeps only the rows that carry both,
which is neither of them.

The framework/kernel corpus figures (the speedup heatmap and the per-kernel distribution grid) read
the results DB instead of this CSV; they are documented in
[measurement_statistics.md](measurement_statistics.md).
