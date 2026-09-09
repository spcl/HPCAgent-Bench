# Plotting

Every figure in this repo is drawn through three shared modules. They exist because the failures
they prevent all actually happened here, and each one is recorded beside the rule it motivated.

| module | owns |
|---|---|
| `hpcagent_bench/palette.py` | which colour and which SHAPE an entity wears |
| `hpcagent_bench/plotstyle.py` | everything that is never data: type, grid, legend, ticks, title |
| `hpcagent_bench/experiment_tags.py` | how a campaign, model or language is SPELLED |

## Identity: colour and shape belong to the entity

`palette.register(kind, name)` returns `(colour, marker)` and is idempotent. `color()`, `marker()`,
`colors()` and `markers()` read the same registry.

A colour is a pure function of the NAME, never of a series' position in the figure. `plotting.py`
used to index the palette by a framework's position in whatever list a figure happened to hold, so
dropping one column repainted every survivor and the same framework wore one hue in the heatmap and
another in the speed-up chart.

Identity is encoded **twice**, colour and shape, because colour alone does not survive a greyscale
print, a figure shrunk to a column, or a reader who cannot separate two hues.

`FRAMEWORK_ORDER` and `MODEL_ORDER` are **append only**. Inserting a name shifts every colour after
it and silently re-colours every already-published figure. A name that is no longer used still
holds its slot for the same reason.

## Names: `envs/display_names.yaml`

Short tags (`llr40v11`, `qwen38`, `c`) are what the runner routes on; they are wrong on an axis.
`experiment_tags` maps them, from a data file rather than literals, because these names are edited
by whoever is writing the paper and a name spelled at three call sites will disagree with itself --
which it did, one figure saying `qwen38` where its neighbour said `Qwen3.8-27B`.

Every lookup falls back to the tag unchanged, so a new campaign never breaks a figure.
`tests/test_display_names.py` is what stops that fallback from going unnoticed: it reads
`OPTARENA_OPTIMIZER` out of every generated arm `.env` and fails if a model tag served a checkpoint
other than the one the registry names. A swapped checkpoint cannot keep its old label.

Serving details are deliberately NOT in these names. `kimi27sglang` names SGLang only because the
runner had to tell two arms apart, and `-FP8` is a precision; neither is the model, and on an axis
they read as if they were. Where the engine or precision IS the variable, the caption says it once.

## Style: `plotstyle`

`apply()` sets the process-wide rcParams; call it before importing `pyplot`.

**Type is sized for PRINT.** A figure is reproduced at roughly half its authored width, so the
scale is set from the SMALLEST text surviving that: 14pt ticks reach the page near 7pt.

**`title(fig, text)` is centred, and there are no subtitles.** The argument is still accepted and
ignored, so a caller passing one is not silently dropping information it believes is displayed. The
how-to-read sentence belongs in the caption a paper already gives every figure; above the axes it
competes with the title and eats the panel.

**`legend_below(fig, handles, ncol, y)`** puts one legend under the figure. An in-axes legend has
to be placed, and every placement bets that one corner stays empty -- a bet that loses when the
data changes. Pass `y` to keep it INSIDE the canvas whenever two figures must come out the same
size (see below).

**`value_axis(ax, axis, log_base)`** does ticks and grid for the axis carrying the measured
quantity, and only that axis -- a grid line per category on the other one is noise. It is the one
place three matplotlib traps are handled:

* `AutoMinorLocator` refuses a log scale outright, so log axes get a `LogLocator`.
* On a base-10 axis the majors go at 1, 2 and 5 per decade. The default labels only decades, and a
  panel spanning less than two of them ends up with a SINGLE labelled tick.
* `LogFormatterSciNotation` returns the empty string for a 5x10^n tick even with
  `labelOnlyBase=False`, leaving a rule where its label belongs. `decade_label` replaces it and
  writes plain numbers with magnitude suffixes (`500K`, `1M`, `2.5M`) -- a token count is a
  quantity a reader quotes, and scientific notation makes them do arithmetic first.

Minor positions come from `LOG10_MINOR_SUBS`. They are not whole numbers: with majors at 1, 2 and
5, the integer subs `3,4,6..9` leave the 1-to-2 interval with NO minor line while 2-to-5 and 5-to-10
get several, and a grid finer in some bands than others is worse than a coarse one.

`log_base` is passed, never sniffed off the axis -- matplotlib keeps it private and a wrong guess
puts minor lines at wrong ratios, which looks like a grid and reads as a lie.

## Rules that are not in code

**Log space for ratios.** A speed-up is multiplicative. On a linear axis every slow-down is crushed
into `[0, 1)` and every win gets an unbounded tail, so `0.5x` reads as smaller than `1.5x` when they
are the same magnitude. Plot `log2(speedup)`, or a log axis labelled in ratios.

**The baseline is a property of the DATA.** `plotting.baseline_of(frame)` reads the `baseline`
column the judge stamped on each row; `DEFAULT_BASELINE` (`numba`) is only the fallback. llr40v9
and v10 graded against single-core C and everything from v11 grades against numba, so a figure that
picks its own denominator is plotting a ratio nobody scored.

**A connector between two marks is an ELBOW, never a diagonal.** The straight segment passes
through coordinates that were never measured, and on a plot about where something landed a reader
takes the path for data. The right angle is visibly a connector. Same reason a Pareto front is a
step -- and it steps `where="pre"`, since both axes are good-is-up and `"post"` draws the front
through the dominating corner, claiming trade-offs no point achieves.

**Draw no interval you cannot support.** A percentile bootstrap of a MEDIAN at n=2 returns the two
data points; at n=3 barely more. Measured here: median CI width was 0.16 at n=2 against 0.64 at
n=3-4 -- it did not shrink with n, because it was never measuring uncertainty. Cells under
`MIN_INTERVAL_SAMPLES` are drawn hollow and assert nothing. Where an "interval" would span two
CONDITIONS rather than repeats of one, draw none at all: on llr40v11 all 120 token cells mixed the
skills and no-skills arms, so the bar was mostly the treatment wearing an error bar.

**Prefer rank statistics on these samples.** Per-kernel speed-ups are heavy-tailed, and a mean in
log space still lets one 40x kernel carry the estimate. `plot_score_change.py` uses the
Hodges-Lehmann estimator with a distribution-free interval from the Wilcoxon signed-rank test, so
point, interval and p all describe one thing. Paired, by kernel -- Mann-Whitney is the unpaired
sibling and would throw away the pairing that carries most of the precision.

**Two figures meant to load side by side must be the same size.** Fix the figure size, use fixed
`subplots_adjust` margins rather than `tight_layout`, keep the legend inside the canvas, and save
with `bbox_inches=fig.bbox_inches`. `bbox_inches=None` means "use the rcParam", which here is
`"tight"` -- so a figure with a five-entry legend saved wider than its three-entry pair.

**Text case is Title Case**, except articles, conjunctions and short prepositions. Identifiers keep
their own spelling: `numba` and `lang-c` are names, not words. Tick labels rotate 0 or 90 degrees,
never an angle.

**Say what is missing.** If a figure caps coverage -- top-N, a dropped band, a sampled subset --
print what was dropped. Silent truncation reads as "this is everything".

## The figures

| script | figure |
|---|---|
| `scripts/plot_arm_summary.py` | per-arm median speed-up and median tokens, one x slot per LANGUAGE with the models dodged inside it; `-speedup`, `-tokens` and a `-pair` |
| `scripts/plot_score_change.py` | speed-up against spend, two marks per arm (hollow = no skills) joined by an elbow, four quadrants named |
| `scripts/plot_tokens.py` | median tokens per task, per kernel, per model |
| `scripts/plot_speedup.py` | per-kernel signed speed-up in magnitude bands, per machine |
| `hpcagent_bench/plotting.py` | the framework heatmap and the reader everything else loads through |
