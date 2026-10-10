# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Did an intervention buy speedup, and what did it cost in tokens? One mark per setup, paired against
its own control: treated setups against the no-packet setups of the same experiment, or an explicit pair
list (``--pairs-csv``) for llrblind or gitscicomp10 style comparisons. Both routes feed the same raw
tagged frame :mod:`hpcagent_bench.stats.figures.efficacy` draws from.

Significance is corrected once per figure: :func:`points` tests every (model, leg) on both axes with the
configured paired test (``statistics.paired_test``, sign-flip by default,
:func:`hpcagent_bench.stats.significance.paired`), the p values are adjusted across the whole family by the
configured correction (``statistics.correction``, Benjamini-Hochberg by default), and the star is gated on
the adjusted value. The stats table names both in its ``test`` and ``correction`` columns.

``--per-kernel`` draws the other view: every kernel of a tag, canon-sweep columns (compilers) and an
experiment's setups as rows of speedups over one baseline (:func:`hpcagent_bench.stats.figures.signed.kernel_comparison`).

Several comparisons join as one row of columns: repeat ``--treatment``, or give several
``--comparison`` specs (``title=...;intervention=...;treatment=...`` or
``title=...;intervention=...;pairs=<csv>[;control-label=...][;observations=a.csv,b.csv]``).
"""

import argparse
import dataclasses
import math
import pathlib
import sys
from collections.abc import Sequence

import numpy as np
import pandas as pd

from hpcagent_bench import packets, studies, study_tags
from hpcagent_bench.stats import cost, population, score_rule, significance
from hpcagent_bench.stats import style as plotstyle
from hpcagent_bench.stats.figures import efficacy as efficacy_figures
from hpcagent_bench.stats.figures import setup_names, signed

#: :func:`points`' row shape, so an empty family still carries these columns for ``.dropna`` to use.
POINT_COLUMNS: tuple[str, ...] = (
    "model",
    "language",
    "leg",
    "score",
    "cost",
    "kernels",
    "test",
    "score_p",
    "cost_p",
)


def compare_slice(
    model: str,
    language: str,
    leg: str,
    control: pd.DataFrame,
    treated: pd.DataFrame,
    over: population.KernelPolicy = efficacy_figures.SPEEDUP_OVER,
    card: cost.CostModel | None = None,
) -> dict[str, float | str | int] | None:  # fmt: skip
    """One comparison's two geomean ratios (treated over control) and their raw significance p
    values, over the kernels :func:`~hpcagent_bench.stats.figures.efficacy.paired_kernels` covers."""
    graded = pd.concat([control, treated])
    graded = graded[graded.row_kind == "submission"]
    population.one_denominator(graded.baseline.tolist(), label=f"{model}/{leg}")
    paired = efficacy_figures.paired_kernels(control, treated, card)
    if paired.empty:
        return None
    timed = efficacy_figures.speedup_mask(paired, over)
    log_score = np.log((paired.treated_speedup / paired.control_speedup).to_numpy(dtype=float)[timed])
    log_cost = np.log((paired.treated_tokens / paired.control_tokens).to_numpy(dtype=float))
    score, cost = significance.paired(log_score), significance.paired(log_cost)
    return {
        "model": model,
        "language": language,
        "leg": leg,
        "score": math.exp(score.estimate) if math.isfinite(score.estimate) else math.nan,
        "cost": math.exp(cost.estimate) if math.isfinite(cost.estimate) else math.nan,
        "kernels": int(timed.sum()),
        # Raw p values; the corrected verdict columns come from the whole family at once.
        "test": score.label,
        "score_p": score.pvalue,
        "cost_p": cost.pvalue,
    }


def points(
    control: pd.DataFrame,
    treated: pd.DataFrame,
    over: population.KernelPolicy = efficacy_figures.SPEEDUP_OVER,
    card: cost.CostModel | None = None,
) -> pd.DataFrame:
    """One row per (model, language) present in both sides, with the flags corrected. A setup that ran
    but never had a graded submission is absent rather than entered at zero."""
    graded_control, graded_treated = (
        control[control.row_kind == "submission"],
        treated[treated.row_kind == "submission"],
    )
    keys = sorted(
        set(map(tuple, graded_control[["model", "language"]].drop_duplicates().to_numpy()))
        & set(map(tuple, graded_treated[["model", "language"]].drop_duplicates().to_numpy()))
    )
    rows = [
        compare_slice(
            str(model),
            str(language),
            study_tags.language_name(str(language)),
            control[(control.model == model) & (control.language == language)],
            treated[(treated.model == model) & (treated.language == language)],
            over,
            card,
        )
        for model, language in keys
    ]
    return corrected([row for row in rows if row is not None])


def corrected(rows: Sequence[dict[str, float | str | int]]) -> pd.DataFrame:
    """``rows`` as the stats table, with the configured correction run ONCE over the whole family."""
    frame = pd.DataFrame(list(rows), columns=list(POINT_COLUMNS)).dropna(subset=["score", "cost"])
    if frame.empty:
        return frame
    # Interleaved score, cost, score, cost ... so each row's pair of verdicts comes back adjacent.
    family = [value for row in frame.itertuples(index=False) for value in (row.score_p, row.cost_p)]
    verdicts = significance.verdicts(family)
    return frame.assign(
        score_p_adjusted=[v.adjusted for v in verdicts[0::2]],
        cost_p_adjusted=[v.adjusted for v in verdicts[1::2]],
        score_verdict=[v.finding.value for v in verdicts[0::2]],
        cost_verdict=[v.finding.value for v in verdicts[1::2]],
        correction=[v.correction for v in verdicts[0::2]],
        family_size=sum(1 for v in verdicts if math.isfinite(v.adjusted)),
    )


def load_all(paths: Sequence[pathlib.Path], card: cost.CostModel | None = None) -> pd.DataFrame:
    """Every observations file as one frame, tokens priced by ``card``."""
    frame = pd.concat([studies.read_observations(path) for path in paths], ignore_index=True)
    return population.condition_rows(cost.priced(frame, card or cost.resolve()))


def control_rows(frame_all: pd.DataFrame) -> pd.DataFrame:
    """The control side: the setup recording no packet at all (canonical packet ``""``)."""
    return frame_all[frame_all.packet == ""]


def treatment_frame(frame_all: pd.DataFrame, treatment: str) -> pd.DataFrame:
    """``frame_all``'s control and ``treatment`` rows, tagged ``skills`` True/False -- which is all
    the figure needs, not the packet's name."""
    control = control_rows(frame_all)
    treated = frame_all[frame_all.packet.map(lambda p: packets.has_part(p, treatment))]
    return pd.concat([control.assign(skills=False), treated.assign(skills=True)], ignore_index=True)


def complete_side_setups(
    control: pd.DataFrame, treated: pd.DataFrame, tag_kernels: Sequence[str], treatment: str, include_incomplete: bool
) -> set[str]:
    """The setups of ``control`` and ``treated`` that cover every kernel of ``tag``. A setup short
    of the tag is dropped and named on stderr with its coverage, never silently."""
    combined = pd.concat([control, treated], ignore_index=True)
    if include_incomplete:
        return set(combined["setup"].dropna().astype(str).unique())
    kept, dropped = population.complete_setups(combined, tag_kernels)
    for setup in sorted(dropped):
        print(f"{treatment}: dropping {setup} ({dropped[setup]}/{len(tag_kernels)} tag kernels)", file=sys.stderr)
    return set(kept)


def one_treatment_panel(
    frame_all: pd.DataFrame,
    control: pd.DataFrame,
    treatment: str,
    tag_kernels: Sequence[str],
    include_incomplete: bool = False,
    over: population.KernelPolicy = efficacy_figures.SPEEDUP_OVER,
    card: cost.CostModel | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """``(stats, frame)`` for one treatment against ``control``; ``None`` when either side is empty
    or the two share no (model, language)."""
    treated = frame_all[frame_all.packet.map(lambda p: packets.has_part(p, treatment))]
    if control.empty or treated.empty:
        return None
    keep = complete_side_setups(control, treated, tag_kernels, treatment, include_incomplete)
    control = control[control["setup"].astype(str).isin(keep)]
    treated = treated[treated["setup"].astype(str).isin(keep)]
    if control.empty or treated.empty:
        return None
    stats = points(control, treated, over, card)
    if stats.empty:
        return None
    frame = treatment_frame(frame_all, treatment)
    frame = frame[frame["setup"].astype(str).isin(keep)]
    return stats, frame


def shared_spelling(pair: tuple[str, str], packet: str) -> str:
    """``packet``'s own setup-name token when BOTH setups of ``pair`` carry it, else "" -- the token, not
    the registry key, since a setup reads ``...-c-lang-skills``."""
    suffixes = [study_tags.setup_suffix(setup) for setup in pair]
    for key, spelling in study_tags.packet_spellings():
        if key == packet and all(spelling in suffix for suffix in suffixes):
            return spelling.strip("-")
    return ""


def setup_languages(frame: pd.DataFrame) -> dict[str, str]:
    """``{setup: recorded language}``.

    The language column is the identity the extractor STAMPED; the setup name is a fallback for rows
    that predate it. gitscicomp10's setups are ``git-scicomp-<model>-repo`` and carry no language token
    at all, so reading the name there gives an empty leg -- a blank tick and an unnamed shape.
    """
    if "setup" not in frame.columns or "language" not in frame.columns:
        return {}
    known = frame[["setup", "language"]].dropna().astype(str)
    return dict(zip(known["setup"], known["language"], strict=True))


#: The ``intervention=`` of a panel whose pairs differ in the agent harness rather than a packet.
HARNESS_INTERVENTION: str = "harness"

#: The ``intervention=`` of a panel whose pairs are SEVERAL packets against one control (a merged
#: pair list): each column is "<delivery>-<packet short name>" and wears that packet's shape.
PACKETS_INTERVENTION: str = "packets"


def treated_harness(setup: str) -> str:
    """What a harness comparison's treated setup changed: its packet when it has one (AutoKernel on
    Claude Code), else its harness's display name."""
    packet = study_tags.packet_of(setup)
    if packet:
        return study_tags.packet_name(packet)
    tokens = setup.split("-")
    # A packet named mid-setup ("harness20-caveman-qwen38-c") still names the column.
    for key in study_tags.order("packets"):
        if key and key in tokens:
            return study_tags.packet_name(key)
    for harness in study_tags.order("harnesses"):
        if harness and harness in tokens:
            return study_tags.harness_name(harness)
    return setup


def pair_leg_label(pair: tuple[str, str], intervention: str, recorded_language: str = "") -> str:
    """One pair's LEG: what it DELIVERED, plus every packet BOTH its setups carried -- never the
    intervention the two sides differ in, which the title and the legend already say once.

    ``recorded_language`` is :func:`setup_languages`' answer, used when the setup name has none. A
    HARNESS comparison names each column by the treated setup's harness (or its packet, AutoKernel on
    Claude Code): every setup delivers C, and the harness is what the columns compare.
    """
    if intervention == HARNESS_INTERVENTION:
        return treated_harness(pair[0])
    language = study_tags.setup_delivery_name(pair[0]) or study_tags.language_name(recorded_language)
    if intervention == PACKETS_INTERVENTION:
        # Several packets in one panel: the column names its delivery AND its packet ("C-CPF").
        return f"{language}-{study_tags.packet_short_name(study_tags.packet_of(pair[0]))}"
    resolved = packets.canonical(intervention)
    shared = [
        token for key in study_tags.order("packets") if key and key != resolved if (token := shared_spelling(pair, key))
    ]
    # A packet whose key is a dash-bounded part of a longer shared one (``lang`` in ``lang-skills``) is that one.
    extra = [token for token in shared if not any(token != other and f"-{token}-" in f"-{other}-" for other in shared)]
    return " ".join([language, *[f"+{token}" for token in extra]])


def same_card(table: pd.DataFrame, card: cost.CostModel, source: pathlib.Path) -> None:
    """Refuse a family CSV priced with a different cost card: its stars would describe one cost model
    and the Y axis another. A CSV written before the column existed was priced ``effective``."""
    recorded = set(table["cost_model"].dropna().astype(str)) if "cost_model" in table.columns else set()
    recorded = recorded or {"effective"}
    if recorded != {card.key}:
        raise SystemExit(
            f"{source} was priced with {sorted(recorded)}, the figure with {card.key!r}; pass --cost-model"
        )


def same_rule(table: pd.DataFrame, source: pathlib.Path) -> None:
    """Refuse a family CSV scored under another S_i rule: its stars would test one score and the
    points plot another. A CSV written before the column existed was scored under ``s-v1``."""
    column = score_rule.SCORE_RULE_COLUMN
    recorded = set(table[column].dropna().astype(str)) if column in table.columns else set()
    recorded = recorded or {"s-v1"}
    if recorded != {score_rule.SCORE_RULE}:
        raise SystemExit(
            f"{source} was scored under {sorted(recorded)}, the figure under {score_rule.SCORE_RULE!r}; "
            "rebuild it with statistics/paired_setups.py"
        )


#: Column the family CSV carries its speedup population under (``statistics/paired_setups.py``).
KERNEL_POLICY_COLUMN: str = "kernel_policy"


def same_policy(table: pd.DataFrame, source: pathlib.Path, over: population.KernelPolicy) -> None:
    """Refuse a family CSV whose speedup leg was taken over another kernel population: its stars
    would test failures-at-1x while the marks leave failures out, or the reverse. A CSV written
    before the column existed was taken over every served kernel."""
    recorded = set(table[KERNEL_POLICY_COLUMN].dropna().astype(str)) if KERNEL_POLICY_COLUMN in table else set()
    if (recorded or {"served"}) != {over.value}:
        raise SystemExit(
            f"{source} took its speedup over {sorted(recorded or {'served'})}, the figure over {over.value!r}; "
            f"rebuild it with statistics/paired_setups.py --policy {over.value}"
        )


def family_pairs(table: pd.DataFrame) -> list[tuple[str, str]]:
    """Every ``(treatment, control)`` the family CSV names, in the order it declared them."""
    seen: dict[tuple[str, str], None] = {}
    for row in table.itertuples(index=False):
        seen.setdefault((str(row.setup_a), str(row.setup_b)), None)
    return list(seen)


#: What ``statistics/paired_setups.py`` calls each leg of a pair in the family CSV it writes.
SPEEDUP_LEG: str = "speedup"
TOKENS_LEG: str = "tokens"


def family_stats(table: pd.DataFrame, intervention: str, languages: dict[str, str] | None = None) -> pd.DataFrame:
    """The family CSV's OWN corrected verdicts, as the stats table the figure stars from -- NEVER
    recomputed here (see module docstring)."""
    verdicts = {(str(row.setup_a), str(row.setup_b), str(row.leg)): row for row in table.itertuples(index=False)}
    known = languages or {}
    rows: list[dict[str, float | str | int]] = []
    for pair in family_pairs(table):
        score, cost = verdicts.get((*pair, SPEEDUP_LEG)), verdicts.get((*pair, TOKENS_LEG))
        recorded = known.get(pair[0], "") or known.get(pair[1], "")
        rows.append(
            {
                "model": study_tags.model_of(pair[1]),
                "language": study_tags.language_of(pair[1]) or recorded,
                "leg": pair_leg_label(pair, intervention, recorded),
                "score_verdict": str(score.verdict) if score is not None else "",
                "cost_verdict": str(cost.verdict) if cost is not None else "",
                "kernels": int(score.n_pairs) if score is not None else 0,
            }
        )
    tested = [row for row in table.itertuples(index=False) if math.isfinite(float(row.p_adjusted))]
    # the tests that produced the CSV's p values, carried beside the verdicts drawn from them
    named = {column: str(table[column].iloc[0]) for column in ("test", "correction") if column in table and len(table)}
    return pd.DataFrame(rows).assign(family_size=len(tested), **named)


def pair_frame(frame_all: pd.DataFrame, pairs: Sequence[tuple[str, str]], intervention: str) -> pd.DataFrame:
    """The RAW rows of every setup ``pairs`` names, tagged ``model``/``language``/``leg``/``skills`` --
    the same shape :func:`treatment_frame` produces, keyed by explicit setup identity instead of a
    packet suffix (llrblind's two experiments, gitscicomp10's kernel/repo scope)."""
    known = setup_languages(frame_all)
    parts = []
    for pair in pairs:
        recorded = known.get(pair[0], "") or known.get(pair[1], "")
        leg = pair_leg_label(pair, intervention, recorded)
        for setup, skills in zip(pair, (True, False), strict=True):
            part = frame_all[frame_all["setup"].astype(str) == setup]
            if part.empty:
                continue
            parts.append(
                part.assign(
                    model=study_tags.model_of(setup),
                    language=study_tags.language_of(setup) or known.get(setup, ""),
                    leg=leg,
                    skills=skills,
                ))  # fmt: skip
    return pd.concat(parts, ignore_index=True) if parts else frame_all.iloc[0:0].assign(leg="", skills=False)


def dot_measures(args: argparse.Namespace) -> tuple[str, ...]:
    """The stacked rows ``--success-row`` asks for, in :data:`efficacy_figures.MEASURES` order."""
    return tuple(measure for measure in efficacy_figures.MEASURES if args.success_row or measure != "success")


def write_dot_rows(
    args: argparse.Namespace,
    config: efficacy_figures.FigureConfig,
    frame: pd.DataFrame,
    stats: pd.DataFrame,
    treatment: str,
    card: cost.CostModel,
) -> pathlib.Path:
    """ONE comparison to ``--out``: speedup over the baseline, the solved rate, then what it cost,
    one column per (LLM, delivery)."""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    return efficacy_figures.figure_setup_dots(
        frame, stats, treatment, args.out, control_name=args.control_label,
        config=config, differences=args.difference, over=args.speedup_over, measures=dot_measures(args), card=card,
        **({"row_height_in": args.dots_row_height} if args.dots_row_height else {}),
        labels={
            "speedup": efficacy_figures.speedup_row_label(args.speedup_over),
            "cost": efficacy_figures.cost_row_label(card.key),
        },
    )  # fmt: skip


def write_row(
    args: argparse.Namespace,
    config: efficacy_figures.FigureConfig,
    panels: Sequence[efficacy_figures.Panel],
    card: cost.CostModel,
    row_width: float | None,
    comparators: Sequence[Sequence[efficacy_figures.Comparator]] = (),
    **columns: Sequence[str],
) -> pathlib.Path:
    """Several comparisons to ``--out`` as one row of columns; ``columns`` are
    :func:`~hpcagent_bench.stats.figures.efficacy.figure_dot_row`'s per-column lists."""
    return efficacy_figures.figure_dot_row(
        panels, args.out, config=config,
        row_width_in=row_width or plotstyle.ACM_TEXT_WIDTH_IN,
        **({"row_height_in": args.dots_row_height} if args.dots_row_height else {}),
        over=args.speedup_over, measures=dot_measures(args), card=card, comparators=comparators,
        labels={
            "speedup": efficacy_figures.speedup_row_label(args.speedup_over),
            "cost": efficacy_figures.cost_row_label(card.key),
        },
        **columns,
    )  # fmt: skip


def figure_from_pairs(args: argparse.Namespace, config: efficacy_figures.FigureConfig) -> None:
    """The ``--pairs-csv`` route: an EXPLICIT pair list drawn as the same figure every packet
    comparison goes through.

    ``config`` is passed on EXPLICITLY: left to its default, the figure and the table beside it would be
    drawn under two configurations."""
    table = pd.read_csv(args.pairs_csv)
    pairs = family_pairs(table)
    if not pairs:
        raise SystemExit(f"{args.pairs_csv} names no pairs")
    card = cost.resolve(args.cost_model, args.cost_models)
    same_card(table, card, args.pairs_csv)
    same_rule(table, args.pairs_csv)
    same_policy(table, args.pairs_csv, args.speedup_over)
    frame_all = load_all(args.observations, card)
    frame = pair_frame(frame_all, pairs, args.intervention)
    if frame.empty:
        raise SystemExit(f"no observations for the setups {args.pairs_csv} names")
    stats = family_stats(table, args.intervention, setup_languages(frame_all))
    args.table.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(args.table, index=False)
    efficacy_figures.pairs_table(frame, args.speedup_over, card).to_csv(
        args.table.with_name(f"{args.table.stem}-absolute{args.table.suffix}"), index=False
    )
    written = write_dot_rows(args, config, frame, stats, args.intervention, card)
    report(args.intervention, stats)
    print(f"table  -> {args.table}")
    print(f"figure -> {written} (+ .png)")


def safe_pairs_table(
    frame: pd.DataFrame,
    label: str,
    over: population.KernelPolicy = efficacy_figures.SPEEDUP_OVER,
    card: cost.CostModel | None = None,
) -> pd.DataFrame:
    """:func:`~hpcagent_bench.stats.figures.efficacy.pairs_table`, but a raw-row population that
    mixes timing-reduction stamps is named on stderr
    and skipped -- an extraction issue in the SOURCE data, never this figure's to silently paper
    over. The drawn marks are unaffected: they come from the caller's own pre-corrected ``stats``
    table, never from this recompute, which exists only for the informational per-point CSV."""
    try:
        return efficacy_figures.pairs_table(frame, over, card)
    except population.MixedPopulationError as error:
        print(f"{label}: -absolute table skipped ({error})", file=sys.stderr)
        return pd.DataFrame()


def write_panel_tables(
    table: pathlib.Path,
    suffix: str,
    stats: pd.DataFrame | dict[str, pd.DataFrame],
    frame: pd.DataFrame | dict[str, pd.DataFrame],
    over: population.KernelPolicy = efficacy_figures.SPEEDUP_OVER,
    card: cost.CostModel | None = None,
) -> None:
    """One panel's stats/points CSVs beside the figure, ``stats``/``frame`` either the single-
    treatment shape or the ``{treatment: table}`` one :func:`build_multi_comparison` returns -- a
    multi-treatment panel writes ONE combined CSV per file, a ``packet`` column telling the rows
    apart (the same shape a caller merging several single-treatment CSVs by hand would build)."""
    if isinstance(frame, pd.DataFrame) and frame.empty:
        return  # a stub panel holds the slot and records nothing

    if isinstance(stats, dict):
        combined_stats = pd.concat(
            [one.assign(packet=name) for name, one in stats.items() if not one.empty], ignore_index=True
        )
        combined_points = pd.concat(
            [
                safe_pairs_table(one_frame, name, over, card).assign(packet=name)
                for name, one_frame in frame.items()
                if not one_frame.empty
            ],
            ignore_index=True,
        )
    else:
        combined_stats, combined_points = stats, safe_pairs_table(frame, suffix or "panel", over, card)
    combined_stats.to_csv(table.with_name(f"{table.stem}{suffix}{table.suffix}"), index=False)
    combined_points.to_csv(table.with_name(f"{table.stem}{suffix}-absolute{table.suffix}"), index=False)


def report(treatment: str, stats: pd.DataFrame) -> None:
    """One line: how many of the family's tests fired, and how many were underpowered."""
    if stats.empty or "score_verdict" not in stats.columns:
        print(f"{treatment or '(stub)'}: placeholder panel, nothing drawn")
        return
    score_hits = int((stats.score_verdict == significance.Finding.SIGNIFICANT.value).sum())
    cost_hits = int((stats.cost_verdict == significance.Finding.SIGNIFICANT.value).sum())
    withheld = int((stats.score_verdict == significance.Finding.UNDERPOWERED.value).sum())
    correction = str(stats.correction.iloc[0]) if "correction" in stats else "corrected"
    of_test = f" ({stats.test.iloc[0]})" if "test" in stats else ""
    print(
        f"{treatment}: {len(stats)} points; {correction} over {efficacy_figures.family_size(stats)} tests{of_test}: "
        f"{score_hits} score-significant, {cost_hits} cost-significant, "
        f"{withheld} underpowered"
    )


def parse_spec(spec: str) -> dict[str, str]:
    """``key=value;key=value`` -> a plain dict, for one ``--comparison``."""
    fields: dict[str, str] = {}
    for raw_token in spec.split(";"):
        token = raw_token.strip()
        if not token or "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def spec_observations(spec: dict[str, str], default: Sequence[pathlib.Path]) -> Sequence[pathlib.Path]:
    """A spec's own ``observations=a,b``, else ``default``."""
    return [pathlib.Path(p) for p in spec["observations"].split(",")] if "observations" in spec else default


def spec_experiment(
    spec: dict[str, str], default_observations: Sequence[pathlib.Path], default_experiment: str, card: cost.CostModel
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]] | None:
    """``(every row, the no-packet control, the tag)`` of the spec's ONE experiment, the tag
    being every kernel any of its setups touched; ``None`` without a control."""
    observations = spec_observations(spec, default_observations)
    frame_all = studies.setup_rows(observations[0], spec.get("experiment", default_experiment), card)
    control = control_rows(frame_all)
    if control.empty:
        return None
    return frame_all, control, sorted(frame_all["kernel"].dropna().astype(str).unique())


def build_multi_comparison(
    spec: dict[str, str],
    default_observations: Sequence[pathlib.Path],
    default_experiment: str,
    include_incomplete: bool,
    card: cost.CostModel,
    over: population.KernelPolicy = efficacy_figures.SPEEDUP_OVER,
) -> tuple[str, Sequence[str], dict[str, pd.DataFrame], dict[str, pd.DataFrame]] | None:
    """``treatments=a,b,c`` as ONE panel of several packets against their shared no-packet
    control -- every llr40 skill packet against C at once, say, instead of a row of one-packet
    panels. Packet-suffix only:
    an explicit ``pairs=`` figure is already one panel per pair list, and mixing the two routes in
    one panel would need a control this function has no way to reconcile."""
    treatments = [t.strip() for t in spec["treatments"].split(",") if t.strip()]
    title = spec.get("title") or " / ".join(study_tags.packet_name(t) for t in treatments)
    loaded = spec_experiment(spec, default_observations, default_experiment, card)
    if loaded is None:
        return None
    frame_all, control, tag_kernels = loaded
    stats_by_treatment: dict[str, pd.DataFrame] = {}
    frame_by_treatment: dict[str, pd.DataFrame] = {}
    for treatment in treatments:
        built = one_treatment_panel(frame_all, control, treatment, tag_kernels, include_incomplete, over, card)
        if built is None:
            continue
        stats_by_treatment[treatment], frame_by_treatment[treatment] = built
    if not frame_by_treatment:
        return None
    return title, treatments, stats_by_treatment, frame_by_treatment


def build_comparison(
    spec: dict[str, str],
    default_observations: Sequence[pathlib.Path],
    default_experiment: str,
    include_incomplete: bool,
    card: cost.CostModel,
    over: population.KernelPolicy = efficacy_figures.SPEEDUP_OVER,
) -> efficacy_figures.Panel | None:
    """One ``--comparison`` spec as a panel (:data:`~hpcagent_bench.stats.figures.efficacy.Panel`)
    -- an explicit pair list (``pairs=``), several packets sharing one panel (``treatments=``,
    :func:`build_multi_comparison`), or a single packet-suffix split (``treatment=``) of its own or
    the default experiment. ``treatment`` is the registry key(s) the panel is SHAPED by."""
    if spec.get("stub", "").lower() in ("1", "true", "yes"):
        # A PLACEHOLDER panel: the box, the axes and the caption, with nothing plotted. It keeps a
        # slot in the row for a comparison that has not finished running, so the figure can go into
        # the paper at its final width and the panel fills in later without re-laying out the page.
        return spec.get("title", ""), spec.get("intervention", ""), pd.DataFrame(), pd.DataFrame()
    if "treatments" in spec:
        return build_multi_comparison(spec, default_observations, default_experiment, include_incomplete, card, over)
    intervention = spec["intervention"]
    title = spec.get("title") or study_tags.packet_name(intervention)
    observations = spec_observations(spec, default_observations)
    if "pairs" in spec:
        table = pd.read_csv(pathlib.Path(spec["pairs"]))
        same_card(table, card, pathlib.Path(spec["pairs"]))
        same_rule(table, pathlib.Path(spec["pairs"]))
        same_policy(table, pathlib.Path(spec["pairs"]), over)
        pairs = family_pairs(table)
        if not pairs:
            return None
        frame_all = load_all(observations, card)
        frame = pair_frame(frame_all, pairs, intervention)
        if frame.empty:
            return None
        return title, intervention, family_stats(table, intervention, setup_languages(frame_all)), frame
    loaded = spec_experiment(spec, default_observations, default_experiment, card)
    if loaded is None:
        return None
    frame_all, control, tag_kernels = loaded
    treatment = spec.get("treatment", intervention)
    built = one_treatment_panel(frame_all, control, treatment, tag_kernels, include_incomplete, over, card)
    if built is None:
        return None
    stats, frame = built
    return title, treatment, stats, frame


def build_parser() -> argparse.ArgumentParser:
    """The command line: the observations, the route (``--comparison``, ``--pairs-csv`` or
    ``--experiment`` + ``--treatment``) and the figure's look."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "observations",
        type=pathlib.Path,
        nargs="*",
        help="extracted observations; repeatable (optional with --per-kernel)",
    )
    population.add_selection_arguments(
        parser, experiment_help="setup prefix naming ONE experiment; required without --pairs-csv/--comparison"
    )
    parser.add_argument(
        "--pairs-csv",
        type=pathlib.Path,
        default=None,
        help=
        "a family CSV from statistics/paired_setups.py. Its setup_a,setup_b rows ARE the pairs and "
        "its corrected verdicts ARE the stars, so the figure and the paper's table cannot disagree",
    )  # fmt: skip
    parser.add_argument(
        "--intervention",
        default="",
        help=
        "with --pairs-csv: the registered packet key whose hue and display name the TREATED side wears",
    )  # fmt: skip
    parser.add_argument(
        "--control-label",
        default="",
        help=
        "with --pairs-csv: the hollow mark's legend text, for a control that is not the "
        "absence of a packet. Default: packets.control_label",
    )  # fmt: skip
    parser.add_argument(
        "--treatment",
        action="append",
        default=[],
        help=
        "packet naming a TREATED side (skills, cpf, cpf-src, ...); repeatable -- each is read "
        "against the SAME no-packet control, one at a time. Default: skills. Two or more join as "
        "columns of one row",
    )  # fmt: skip
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        help="'title=...;intervention=...;treatment=...' or "
        "'title=...;intervention=...;pairs=<csv>[;control-label=...][;observations=a.csv,b.csv]"
        "[;experiment=...][;comparators=<csv>;comparator-set=pluto:C,jax_cpu:C]'; repeatable -- joins "
        "into ONE row alongside --treatment, mixing a packet-suffix comparison and an explicit-pairs one "
        "in the same figure. comparators= draws compilers/frameworks (comparators.py's CSV) beside the "
        "models of a delivery on the speedup and solved rows",
    )  # fmt: skip
    parser.add_argument(
        "--row-width",
        choices=("natural", "iclr", "iclr-wrap", "acm-column", "acm-text"),
        default="acm-text",
        help=
        "the joined row's target width: a paper's page budget (style.ACM_TEXT_WIDTH_IN, the full "
        "width of a two-column page, by default; also ICLR_TEXT_WIDTH_IN / ACM_COLUMN_WIDTH_IN) so "
        "the PDF drops in at scale 1.0, or 'natural' for the authoring type scale at text width. A page "
        "budget also sets the type to PAPER_CONFIG, which is the two-column convention",
    )  # fmt: skip
    parser.add_argument(
        "--mark-pending",
        action="store_true",
        default=False,
        help="draw a '?' in every category with no measurement yet: a comparison's 'pending=' models "
        "and 'placeholders=' deliveries (default: the slot stays empty); with --per-kernel, a kernel a row "
        "has not attempted yet, left out of its geomean",
    )
    parser.add_argument(
        "--speedup-over",
        default=efficacy_figures.SPEEDUP_OVER,
        type=population.KernelPolicy,
        choices=population.POLICIES,
        help="solved (default): speedup over the kernels both setups answered correctly, failures shown as "
        "the success-rate row; served: every kernel, a failure at 1x (the fallback reading)",
    )
    parser.add_argument(
        "--difference",
        default="",
        help="draw an ARROW across these comparisons, labelled with the factor between the two "
        "marks: 'HIP:qwen38,HIP:kimi27sglang' (delivery:model, comma separated). On a joined row "
        "say it per panel instead, as a --comparison spec's own 'difference=' key",
    )
    parser.add_argument(
        "--success-row",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="draw the half-height Solved row between speedup and cost (default); "
        "--no-success-row drops it, which shortens the canvas and leaves every other box unchanged",
    )
    parser.add_argument(
        "--dots-row-height",
        type=float,
        default=None,
        help="one stacked row's height, inches. Default: the figure's own -- a single comparison "
        "gets a tall row, a joined row of comparisons a short one, since the joined figure spends "
        "its height on two rows across the whole page",
    )
    kernel = parser.add_argument_group(
        "per-kernel view",
        "--per-kernel draws every kernel of a tag: canon-sweep columns (compilers) and the --experiment setups, "
        "each a row of speedups over --baseline, with tokens spent below",
    )
    kernel.add_argument(
        "--per-kernel", action="store_true", help="draw the per-kernel view instead of the efficacy rows"
    )
    kernel.add_argument("--canon-db", type=pathlib.Path, default=None, help="the canon table a baseline sweep records")
    kernel.add_argument(
        "--canon-columns", default="", help="comma-separated canon columns drawn as rows, e.g. pluto,dace_cpu"
    )
    kernel.add_argument("--baseline", default=signed.BASELINE, help="speedup denominator (default: numba)")
    kernel.add_argument(
        "--baseline-fallback", default="cc_autopar", help="canon column timing a kernel the baseline did not verify"
    )
    kernel.add_argument(
        "--language", default="c", help="setup language suffix, <experiment>-<model>-<language>[-<packet>]"
    )
    kernel.add_argument(
        "--conditions", default="", help="comma-separated setup conditions to draw ('' control); default all"
    )
    kernel.add_argument(
        "--tag-file", type=pathlib.Path, default=None, help="one kernel per line; default every canon kernel"
    )
    kernel.add_argument(
        "--series-label",
        action="append",
        default=[],
        metavar="KEY=LABEL",
        help="rename a row by column or setup; repeatable",
    )
    kernel.add_argument(
        "--offset", type=float, default=0.0, help="spread a kernel's rows over this fraction of its slot"
    )
    kernel.add_argument("--title", default="", help="figure title; default none")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("figures/score_change.pdf"))
    parser.add_argument("--table", type=pathlib.Path, default=pathlib.Path("data/score_change.csv"))
    cost.add_arguments(parser)
    return parser


#: ``--row-width`` choices -> the joined row's width in inches; ``None`` is the authoring scale.
ROW_WIDTHS: dict[str, float | None] = {
    "natural": None,
    "iclr": plotstyle.ICLR_TEXT_WIDTH_IN,
    "iclr-wrap": plotstyle.ICLR_WRAP_WIDTH_IN,
    "acm-column": plotstyle.ACM_COLUMN_WIDTH_IN,
    "acm-text": plotstyle.ACM_TEXT_WIDTH_IN,
}


def figure_config(args: argparse.Namespace, row_width: float | None) -> efficacy_figures.FigureConfig:
    """The figure's config from the command line. A ``--row-width`` is a promise to drop the figure
    in at scale 1.0, so it is drawn at its printed size under the two-column type convention."""
    config = efficacy_figures.PAPER_CONFIG if row_width is not None else efficacy_figures.DEFAULT_CONFIG
    return dataclasses.replace(config, mark_pending=True) if args.mark_pending else config


def comparison_panel(
    raw: str, args: argparse.Namespace, card: cost.CostModel
) -> tuple[efficacy_figures.Panel, dict[str, str]] | None:
    """One ``--comparison`` spec as ``(panel, its spec)``; ``None`` (named on stdout) when it draws nothing."""
    spec = parse_spec(raw)
    built = build_comparison(spec, args.observations, args.experiment, args.include_incomplete, card, args.speedup_over)
    if built is None and spec.get("pending"):
        # A comparison whose setups have not run yet is a STUB: its box, its axes and a "?" per
        # pending model, so the row keeps its final layout until the data lands.
        title = spec.get("title", spec.get("intervention", ""))
        built = (title, spec.get("intervention", spec.get("treatment", "")), pd.DataFrame(), pd.DataFrame())
    if built is None:
        print(f"skipping comparison {raw!r}: empty side, or no (model, language) shared with control")
        return None
    return built, spec


def comparator_entries(spec: dict[str, str]) -> list[tuple[str, str]]:
    """A spec's ``comparator-set=pluto:C,jax_cpu:C`` as ``(comparator, delivery group)`` pairs; a
    comparator without ``:group`` sits under the panel's first delivery."""
    entries = [part.strip() for part in spec.get("comparator-set", "").split(",") if part.strip()]
    return [(parts[0].strip(), parts[2].strip()) for parts in (entry.partition(":") for entry in entries)]


def spec_comparators(spec: dict[str, str]) -> list[efficacy_figures.Comparator]:
    """The compilers and frameworks a spec draws beside its models (``comparators=<csv>``, a table of
    ``comparators.py``'s shape: kernel, comparator, device, numba_ms, ms, speedup), restricted to its
    ``comparator-set``; every comparator of the table when the set is absent. A named comparator the
    table lacks, or has no valid kernel for, is named on stderr."""
    if "comparators" not in spec:
        return []
    table = pd.read_csv(pathlib.Path(spec["comparators"]))
    entries = comparator_entries(spec) or [(str(name), "") for name in table["comparator"].dropna().unique()]
    built = efficacy_figures.comparators_from_table(table, entries)
    drawn = {comparator.name for comparator in built if comparator.speedups}
    for name in dict.fromkeys(entry[0] for entry in entries):
        if name not in drawn:
            print(f"comparator {name!r}: no valid kernel in {spec['comparators']}, not drawn", file=sys.stderr)
    return built


def spec_column(specs: Sequence[dict[str, str]], key: str, default: str = "") -> list[str]:
    """One per-column list of a joined row: each spec's ``key``, else ``default``."""
    return [spec.get(key, default) for spec in specs]


def figure_from_comparisons(
    args: argparse.Namespace, config: efficacy_figures.FigureConfig, card: cost.CostModel, row_width: float | None
) -> None:
    """The ``--comparison`` route: every spec one column of ONE row, each with its own control (a
    bare kernel is not the absence of a packet), arrows, placeholder deliveries (a slot kept so the
    column's spacing is final) and pending models."""
    built = [one for one in (comparison_panel(raw, args, card) for raw in args.comparison) if one is not None]
    if not built:
        raise SystemExit(f"no --comparison of {args.comparison} produced a panel")
    panels, specs = (list(part) for part in zip(*built, strict=True))
    args.table.parent.mkdir(parents=True, exist_ok=True)
    for title, _treatment, stats, frame in panels:
        # The CSV is keyed by title, not by the packet(s) shaping the panel.
        suffix = f"-{title.lower().replace(' ', '-')}"
        write_panel_tables(args.table, suffix, stats, frame, args.speedup_over, card)
    comparators = [spec_comparators(spec) for spec in specs]
    drawn = [comparator for column in comparators for comparator in column]
    if drawn:
        table = efficacy_figures.comparator_table(drawn)
        table.to_csv(args.table.with_name(f"{args.table.stem}-comparators{args.table.suffix}"), index=False)
        print(table.to_string(index=False))
    written = write_row(
        args, config, panels, card, row_width, comparators,
        control_names=spec_column(specs, "control-label"),
        differences=spec_column(specs, "difference", args.difference),
        placeholders=spec_column(specs, "placeholders"), pending=spec_column(specs, "pending"),
    )  # fmt: skip
    for title, treatment, stats, _frame in panels:
        for name, one_stats in stats.items() if isinstance(stats, dict) else ((treatment, stats),):
            report(f"{title}/{name}", one_stats)
    print(f"table  -> {args.table}")
    print(f"figure -> {written} (+ .png)")


def figure_from_treatments(
    args: argparse.Namespace, config: efficacy_figures.FigureConfig, card: cost.CostModel, row_width: float | None
) -> None:
    """The ``--experiment`` route: each ``--treatment`` against the experiment's own no-packet control,
    one figure for one treatment, a joined row for several."""
    if not args.experiment:
        raise SystemExit("--experiment names the experiment to split; pass it, or --pairs-csv/--comparison")
    treatments = args.treatment or ["lang-skills"]
    frame_all = studies.setup_rows(args.observations[0], args.experiment, card, args.setups)
    control = control_rows(frame_all)
    if control.empty:
        raise SystemExit(f"no no-packet control rows for experiment {args.experiment!r}")
    # Every kernel ANY setup of this experiment touched -- the tag :func:`complete_side_setups` gates
    # coverage against.
    tag_kernels = sorted(frame_all["kernel"].dropna().astype(str).unique())
    args.table.parent.mkdir(parents=True, exist_ok=True)
    panels: list[tuple[str, str, pd.DataFrame, pd.DataFrame]] = []
    for treatment in treatments:
        built = one_treatment_panel(
            frame_all, control, treatment, tag_kernels, args.include_incomplete, args.speedup_over, card
        )
        if built is None:
            print(f"skipping {treatment!r}: empty side, or no (model, language) shared with control")
            continue
        stats, frame = built
        # Two or more treatments are suffixed by treatment so nothing overwrites its sibling.
        suffix = "" if len(treatments) == 1 else f"-{treatment}"
        stats.to_csv(args.table.with_name(f"{args.table.stem}{suffix}{args.table.suffix}"), index=False)
        efficacy_figures.pairs_table(frame, args.speedup_over, card).to_csv(
            args.table.with_name(f"{args.table.stem}{suffix}-absolute{args.table.suffix}"), index=False
        )
        panels.append((packets.label(treatment), treatment, stats, frame))
    if not panels:
        raise SystemExit(f"no treatment of {treatments} produced a comparison for experiment {args.experiment!r}")
    if len(panels) == 1:
        # A single comparison draws no name; the caption is its title.
        _title, treatment, stats, frame = panels[0]
        written = write_dot_rows(args, config, frame, stats, treatment, card)
    else:
        written = write_row(args, config, panels, card, row_width)
    for _title, treatment, stats, _frame in panels:
        report(treatment, stats)
    print(f"table  -> {args.table}")
    print(f"figure -> {written} (+ .png)")


def figure_per_kernel(args: argparse.Namespace, card: cost.CostModel) -> None:
    """The ``--per-kernel`` route: one row per canon column and per ``--experiment`` setup over the tag."""
    if args.canon_db is None:
        raise SystemExit("--per-kernel needs --canon-db")
    if len(args.observations) > 1:
        raise SystemExit("--per-kernel reads one observations file")
    if args.observations and not args.experiment:
        raise SystemExit("--per-kernel with observations needs --experiment (the setup prefix)")
    canon_frame = studies.read_table(args.canon_db, "canon")
    tag_kernels = (
        [line.strip() for line in args.tag_file.read_text().splitlines() if line.strip()]
        if args.tag_file is not None
        else setup_names.tag_of(canon_frame)
    )
    if not tag_kernels:
        raise SystemExit("no tag kernel named: pass --tag-file or a --canon-db with rows")
    observations = None
    if args.observations:
        observations = population.select_setups(
            studies.read_observations(args.observations[0]), args.experiment, args.setups
        )
        observations = cost.priced(observations, card)
    stem = signed.kernel_comparison(
        canon_frame,
        observations,
        tag_kernels,
        args.out.with_suffix(""),
        canon_columns=[c for c in args.canon_columns.split(",") if c],
        pattern=setup_names.setup_pattern(args.experiment, args.language) if observations is not None else None,
        conditions=[study_tags.canonical("packets", c) for c in args.conditions.split(",")]
        if args.conditions
        else None,
        baseline=args.baseline,
        title=args.title,
        labels=dict(item.split("=", 1) for item in args.series_label),
        offset=args.offset,
        mark_pending=args.mark_pending,
        baseline_fallback=args.baseline_fallback,
    )
    print(f"figure -> {stem}.pdf (+ .png)")
    print(f"tables -> {stem}-kernels.csv, {stem}-summary.csv")


def main() -> None:
    args = build_parser().parse_args()
    significance.configured()  # an unknown test name stops here, before any data is read
    card = cost.resolve(args.cost_model, args.cost_models)
    if args.per_kernel:
        figure_per_kernel(args, card)
        return
    if not args.observations:
        raise SystemExit("pass at least one observations file")
    row_width = ROW_WIDTHS[args.row_width]
    config = figure_config(args, row_width)
    if args.comparison:
        figure_from_comparisons(args, config, card, row_width)
    elif args.pairs_csv is not None:
        figure_from_pairs(args, config)
    else:
        figure_from_treatments(args, config, card, row_width)


if __name__ == "__main__":
    main()
