# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The sampling-temperature study (temperature3): every run per model, kernel and temperature.

One group per temperature, dashed rules between them, the kernels inside each group and the models side
by side in their colours; three rows: final-grade speedup per run, token cost per run (both a dot per run
over a box, or with ``--style violin`` a violin and the median's bootstrap interval) and solved/graded runs (:func:`hpcagent_bench.stats.figures.temperature.temperature_figure`).
Writes the PDF, a PNG beside it and the CSV of every run drawn.

Owed runs (a final grade still to come, or a run owed a rerun) are refused; ``--allow-owed`` draws them as
a "?" for a look at an unfinished study.

    python -m hpcagent_bench.dataset --study temperature3 --out data/temperature3.db
    python statistics/plot_temperature.py data/temperature3.db --out figures/temperature3.pdf
"""

import argparse
import pathlib

from hpcagent_bench import studies, tags
from hpcagent_bench.stats import cost, population, reliability
from hpcagent_bench.stats import style as plotstyle
from hpcagent_bench.stats.figures import efficacy, per_kernel, temperature


def cost_label(card: cost.CostModel) -> str:
    """The token row's Y title, the card's (fresh input, cached input, output) weights under it; no-break
    spaces keep each line whole when the axis folds a label that is too tall."""
    weights = f"({card.fresh_input:g}, {card.cached_input:g}, {card.output:g})"
    return f"{efficacy.cost_row_label(card.key)}\n{weights}".replace(" ", "\N{NO-BREAK SPACE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("observations", type=pathlib.Path, help="observations db (python -m hpcagent_bench.dataset)")
    parser.add_argument("--tag", default="temperature3", help="the kernels and their order")
    parser.add_argument("--experiment", default="temperature3", help="setup prefix selecting the study")
    parser.add_argument("--setups", default="", help="regex; keep only setups whose full name matches")
    parser.add_argument("--allow-owed", action="store_true", help="draw owed runs (final grade or rerun) as '?'")
    parser.add_argument(
        "--style",
        choices=[mode.value for mode in temperature.Mode],
        default=temperature.Mode.BOX.value,
        help="box: median, quartiles, whiskers to min/max; violin: run density and the median's bootstrap interval",
    )
    parser.add_argument("--title", default="")
    parser.add_argument("--width", type=float, default=plotstyle.ACM_TEXT_WIDTH_IN, help="print width in inches")
    parser.add_argument("--out", type=pathlib.Path, required=True, help="figure .pdf (a .png and .csv beside it)")
    cost.add_arguments(parser)
    args = parser.parse_args()

    card = cost.resolve(args.cost_model, args.cost_models)
    frame = population.select_setups(studies.read_observations(args.observations), args.experiment, args.setups)
    kernels = list(tags.kernels_of(args.tag))
    runs = population.designed_runs(frame)
    runs = temperature.run_costs(cost.priced(frame, card), runs.loc[runs["kernel"].isin(kernels)])
    if runs.empty:
        raise SystemExit("no runs selected")
    try:
        mode = temperature.Mode(args.style)
        fig = temperature.temperature_figure(
            runs, kernels, args.title, args.width, args.allow_owed, cost_label(card), mode
        )
    except reliability.OwedRunsError as refused:
        raise SystemExit(str(refused)) from refused
    args.out.parent.mkdir(parents=True, exist_ok=True)
    per_kernel.save(fig, args.out, print_size=True)
    states = runs[population.RUN_STATE_COLUMN].map(lambda state: state.value)
    table = runs.assign(
        temperature=runs["setup"].map(temperature.temperature_of), **{population.RUN_STATE_COLUMN: states}
    )
    table.to_csv(args.out.with_suffix(".csv"), index=False)
    print(f"{len(runs)} runs -> {args.out} (+ .png, .csv)")


if __name__ == "__main__":
    main()
