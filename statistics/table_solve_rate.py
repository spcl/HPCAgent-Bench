# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""How many kernels each arm actually answered, as the table that goes beside the efficacy figure.

THE FIGURE CANNOT SAY THIS. Every speedup it draws is a geomean over the kernels an arm was
SERVED, with an unanswered kernel held at 1x (the 2026-09-16 rule), so two arms can sit at the same
height having solved twelve kernels and thirty. A reader without the solve rate reads the first as
the second.

One row per (experiment, model, delivery), one column per condition, each cell ``solved/served``.
Built from the same ``--pairs-csv`` the figure is, through the same ``pair_frame``, so the table and
the panel are over one population by construction.

Usage::

    python statistics/table_solve_rate.py obs.db --pairs-csv tables/skills_billed.csv \\
        --intervention lang-skills --experiment "Loop Reasoning CPU (LLR)" --out tables/solve.tex
"""

import argparse
import importlib.util
import pathlib
import sys

import pandas as pd

from hpcagent_bench import experiment_tags
from hpcagent_bench.stats import latex, population

#: ``statistics/plot_score_change.py`` is a script, not a module of the package; the pair-frame
#: logic lives there and is loaded by path rather than copied.
PLOT_SCRIPT: str = "plot_score_change"


def plot_module() -> object:
    """``plot_score_change`` loaded from this directory."""
    path = pathlib.Path(__file__).resolve().parent / f"{PLOT_SCRIPT}.py"
    spec = importlib.util.spec_from_file_location(PLOT_SCRIPT, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[PLOT_SCRIPT] = module
    spec.loader.exec_module(module)
    return module


def solved(frame: pd.DataFrame, repeats: population.RepeatPolicy) -> tuple[int, int]:
    """``(answered, served)`` for one arm's rows: a kernel counts as answered when a real grade
    happened on it, served when the arm was handed it at all."""
    answers = population.kernel_answers(frame, repeats=repeats)
    if answers.empty:
        return 0, 0
    delivered = answers[population.DELIVERED_COLUMN] if population.DELIVERED_COLUMN in answers else None
    return (int(delivered.sum()) if delivered is not None else len(answers)), len(answers)


def rate(frame: pd.DataFrame, repeats: population.RepeatPolicy) -> str:
    """One cell: ``28/40``, or an em dash where the arm has no rows at all."""
    answered, served = solved(frame, repeats)
    return f"{answered}/{served}" if served else "--"


def first_rate(frame: pd.DataFrame, repeats: population.RepeatPolicy) -> str:
    """``21/40``: answered on the FIRST episode, over the whole roster the arm was served.

    The denominator stays the full roster on purpose. Counting it over the sliced rows instead
    shrinks it to the kernels that happen to have a task row under the first episode's run, which
    reports 19/19 for an arm that answered 19 of 40.
    """
    served = len(population.kernel_answers(frame, repeats=repeats))
    if not served:
        return "--"
    sliced = first_episode(frame)
    answers = population.kernel_answers(sliced, repeats=repeats) if not sliced.empty else sliced
    column = population.DELIVERED_COLUMN
    answered = int(answers[column].sum()) if len(answers) and column in answers else 0
    return f"{answered}/{served}"


def first_episode(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` cut to each kernel's FIRST episode.

    An arm that failed a kernel may be rerun (the owed rule allows two), and an arm that was not
    rerun as often has fewer chances at the same roster. Comparing the totals then compares
    attempt counts as much as ability: on the GPU track one model's control legs carry 46, 48 and
    54 episodes against exactly 40 in its packet legs, and every kernel that separates them was
    won on a rerun. The first episode is the one both sides always had.
    """
    if "row_kind" not in frame.columns or "run_id" not in frame.columns:
        return frame
    episodes = frame[frame.row_kind == "task"].sort_values(["benchmark", "ts_ms"])
    first = episodes.drop_duplicates("benchmark", keep="first")
    keys = set(zip(first.benchmark, first.run_id, strict=True))
    # A kernel with NO episode row cannot be attributed to an episode at all -- the 2026-09-19
    # reducer left some arms without their token rows -- so it passes through rather than being
    # filtered out. Dropping it instead undercounts the arm: one CPU arm read 19/40 where it had
    # answered 38, because 21 of its kernels had no task row to key on.
    attributed = set(first.benchmark)
    keep = [
        (b in keys_ok) or (b not in attributed)
        for b, keys_ok in ((b, {b} if (b, r) in keys else set()) for b, r in zip(frame.benchmark, frame.run_id))
    ]
    return frame[keep]


def rows(frame: pd.DataFrame, experiment: str, control: str, treated: str, repeats: population.RepeatPolicy,
         leg_labels: object) -> list[dict[str, str]]:  # fmt: skip
    """One row per (model, delivery) the frame carries."""
    records: list[dict[str, str]] = []
    for (model, leg), pair in frame.assign(leg=leg_labels(frame)).groupby(["model", "leg"]):
        records.append(
            {
                "Experiment": experiment,
                "Model": experiment_tags.model_name(str(model)),
                "Delivery": str(leg),
                control: rate(pair[~pair.skills], repeats),
                treated: rate(pair[pair.skills], repeats),
                f"{control} (1st)": first_rate(pair[~pair.skills], repeats),
                f"{treated} (1st)": first_rate(pair[pair.skills], repeats),
            }
        )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=pathlib.Path, nargs="+", help="extracted observations; repeatable")
    parser.add_argument("--pairs-csv", type=pathlib.Path, required=True, help="the family CSV the figure reads")
    parser.add_argument("--intervention", required=True, help="the packet spec the pairs differ in")
    parser.add_argument("--experiment", default="", help="what to call this comparison in the table's first column")
    parser.add_argument("--control-label", default="No Skill Packet")
    parser.add_argument("--treated-label", default="Language Skill Packet")
    parser.add_argument("--repeats", default="latest", choices=population.REPEAT_POLICIES)
    parser.add_argument(
        "--caption",
        default="Kernels answered, of kernels served. (1st) counts each kernel's first episode "
        "only, which is the attempt both arms always had.",
    )
    parser.add_argument("--label", default="tab:solve-rate")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("tables/solve_rate.tex"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot = plot_module()
    from hpcagent_bench.stats import cost  # noqa: PLC0415 -- after the by-path script load
    from hpcagent_bench.stats.figures import efficacy  # noqa: PLC0415

    table = pd.read_csv(args.pairs_csv)
    pairs = plot.family_pairs(table)  # pyright: ignore[reportAttributeAccessIssue]
    if not pairs:
        raise SystemExit(f"{args.pairs_csv} names no pairs")
    card = cost.resolve(cost.DEFAULT_COST_MODEL, None)
    frame_all = plot.load_all(args.observations, card)  # pyright: ignore[reportAttributeAccessIssue]
    frame = plot.pair_frame(frame_all, pairs, args.intervention)  # pyright: ignore[reportAttributeAccessIssue]
    if frame.empty:
        raise SystemExit(f"no observations for the arms {args.pairs_csv} names")
    records = rows(
        frame, args.experiment or args.intervention, args.control_label, args.treated_label, args.repeats,
        efficacy.leg_labels,
    )  # fmt: skip
    written = latex.write(
        pd.DataFrame(records), args.out, caption=args.caption, label=args.label
    )  # fmt: skip
    print(f"table  -> {written} (+ .csv)")
    for record in records:
        print("  " + " | ".join(f"{key}={value}" for key, value in record.items()))


if __name__ == "__main__":
    main()
