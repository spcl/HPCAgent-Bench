# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tables from analysis.py's output: the 48 comparisons, the pooled interventions, the counts, the re-analyses and the
fidelity check against the release's own tables.

    python report.py --out OUT_DIR --release ~/.cache/zenodo-iclr27/reproducibility
"""

import argparse
import math
import pathlib

import pandas as pd

from helpers.scripts.paper_analysis import analysis

ALPHA = 0.05
KEY = ["family", "setup_a", "setup_b", "leg"]


def family_tables(variant: pathlib.Path) -> pd.DataFrame:
    """Main's pair tables of every family, in stats.sh order, with the comparison's kind."""
    frames = []
    for family in analysis.families():
        frame = pd.read_csv(variant / f"{family.name}.csv").assign(kind=family.kind)
        frame["intervention"] = [analysis.intervention_of(family.name, a) for a in frame.setup_a]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def wide(pairs: pd.DataFrame) -> pd.DataFrame:
    """One row per comparison: coverage, then the speed (S) and cost (C) legs."""
    cols = ["rho", "ci_low", "ci_high", "p_value", "p_adjusted", "conclusion", "mde_ratio", "n_pairs"]
    ident = ["family", "kind", "intervention", "setup_a", "setup_b", "n_both_run", "n_b", "n_a", "n_only_a", "n_only_b"]
    ident += ["coverage_p", "coverage_p_adjusted", "rate_difference", "coverage_conclusion", "limits_match"]
    speed = pairs[pairs.leg == "speedup"].set_index(["setup_a", "setup_b"])
    tokens = pairs[pairs.leg == "tokens"].set_index(["setup_a", "setup_b"])[cols].add_suffix("_C")
    return (
        speed[[c for c in ident if c not in ("setup_a", "setup_b")] + cols]
        .add_suffix("")
        .rename(columns={c: f"{c}_S" for c in cols})
        .join(tokens)
        .reset_index()
    )


def fmt(value: object, digits: int = 2) -> str:
    if isinstance(value, float):
        return "--" if math.isnan(value) else (f"{value:.3f}" if abs(value) < 0.01 else f"{value:.{digits}f}")
    return str(value)


def short(setup: str) -> str:
    for prefix in (
        "cpf-llr-focus40-",
        "gpu-llr-focus40-",
        "llrblind-cmp-",
        "git-scicomp-",
        "scicomp-perf-playbook-",
        "harness20-",
    ):
        setup = setup.removeprefix(prefix)
    return setup


def latex(table: pd.DataFrame) -> str:
    """The 48 comparisons as a booktabs table."""
    head = (
        "\\begin{tabular}{llrrrrlllrlllr}\n\\toprule\n"
        "Family & Treated vs.\\ control & $n$ & before & after & $g$/$l$ & $\\rho_S$ [95\\% CI] & $p_S$ & $q_S$ & "
        "$\\rho_C$ [95\\% CI] & $p_C$ & $q_C$ & verdict $S$/$C$ & MDE$_S$ \\\\\n\\midrule\n"
    )
    lines = []
    for r in table.itertuples():
        lines.append(
            f"{r.family} & {short(r.setup_a)} vs.\\ {short(r.setup_b)} & {r.n_both_run} & {r.n_b} & {r.n_a} & "
            f"{r.n_only_a}/{r.n_only_b} & {fmt(r.rho_S)} [{fmt(r.ci_low_S)}, {fmt(r.ci_high_S)}] & {fmt(r.p_value_S, 3)} & "
            f"{fmt(r.p_adjusted_S, 3)} & {fmt(r.rho_C)} [{fmt(r.ci_low_C)}, {fmt(r.ci_high_C)}] & {fmt(r.p_value_C, 3)} & "
            f"{fmt(r.p_adjusted_C, 3)} & {r.conclusion_S}/{r.conclusion_C} & {fmt(r.mde_ratio_S)} \\\\"
        )
    return head + "\n".join(lines).replace("_", "\\_") + "\n\\bottomrule\n\\end{tabular}\n"


def counts(table: pd.DataFrame) -> pd.DataFrame:
    """The headline counts of (c)."""
    sig_s, sig_c = table.p_adjusted_S < ALPHA, table.p_adjusted_C < ALPHA
    raw_s = table.p_value_S < ALPHA
    inter = table.kind == "intervention"
    rows = {
        "significant speed (q<0.05) among 40 interventions": int((sig_s & inter).sum()),
        "significant speed (q<0.05) among all 48": int(sig_s.sum()),
        "speed p<0.05 uncorrected among 40": int((raw_s & inter).sum()),
        "significant cost (q<0.05) among 40": int((sig_c & inter).sum()),
        "significant cost (q<0.05) among all 48": int(sig_c.sum()),
        "significant solve rate (McNemar q<0.05) among 48": int((table.coverage_p_adjusted < ALPHA).sum()),
        "equivalent speed (TOST 1.2x) among 48": int((table.conclusion_S == "equivalent").sum()),
        "equivalent cost (TOST 1.2x) among 48": int((table.conclusion_C == "equivalent").sum()),
        "equivalent solve rate (TOST 10 points) among 48": int((table.coverage_conclusion == "equivalent").sum()),
        "non-significant speed AND cost among 40": int((~sig_s & ~sig_c & inter).sum()),
        "inconclusive speed among 40": int(((table.conclusion_S == "inconclusive") & inter).sum()),
    }
    return pd.DataFrame({"count": rows.values()}, index=pd.Index(list(rows), name="measure"))


def kept(base: pd.DataFrame, other: pd.DataFrame, label: str) -> pd.DataFrame:
    """Per effect (comparison x leg): does ``other`` keep the base's direction and significance."""
    joined = base[[*KEY, "rho", "p_adjusted"]].merge(
        other[[*KEY, "rho", "p_adjusted"]], on=KEY, suffixes=("", "_other")
    )
    sig, sig_other = joined.p_adjusted < ALPHA, joined.p_adjusted_other < ALPHA
    return joined.assign(
        analysis=label,
        significant=sig,
        significant_other=sig_other,
        same_direction=(joined.rho > 1) == (joined.rho_other > 1),
        same_significance=sig == sig_other,
    )


def old_tables(release: pathlib.Path) -> pd.DataFrame:
    """The release's own pair tables (the paper's tests: paired t, BH), renamed to today's keys."""
    frames = [pd.read_csv(release / "tables" / f"{f.name}.csv") for f in analysis.families()]
    return pd.concat(frames, ignore_index=True).rename(columns={"arm_a": "setup_a", "arm_b": "setup_b"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--release", type=pathlib.Path, required=True)
    args = ap.parse_args()
    base = family_tables(args.out / "base")
    table = wide(base)
    table.to_csv(args.out / "table48.csv", index=False)
    (args.out / "table48.tex").write_text(latex(table))
    counts(table).to_csv(args.out / "counts.csv")
    pooled = pd.read_csv(args.out / "base" / "pooled.csv")
    pooled.to_csv(args.out / "pooled.csv", index=False)
    checks = [kept(old_tables(args.release), base, "revised tests (paper tests -> main's)")]
    for variant, label in (
        ("final", "final-timing answers only"),
        ("finalv2", "final-v2 timing only (v1 excluded)"),
        ("latest", "latest launch only"),
        ("cut", "one limit (48M tokens, 16 h)"),
    ):
        if (args.out / variant / "pooled.csv").is_file():
            checks.append(kept(base, family_tables(args.out / variant), label))
    effects = pd.concat(checks, ignore_index=True)
    effects.to_csv(args.out / "reanalysis_effects.csv", index=False)
    summary = effects.groupby("analysis", sort=False).apply(
        lambda g: pd.Series(
            {
                "effects": len(g),
                "significant_before": int(g.significant.sum()),
                "significant_after": int(g.significant_other.sum()),
                "significant_kept": int((g.significant & g.significant_other).sum()),
                "significant_lost": int((g.significant & ~g.significant_other).sum()),
                "newly_significant": int((~g.significant & g.significant_other).sum()),
                "significant_reversed": int((g.significant & ~g.same_direction).sum()),
                "any_direction_flip": int((~g.same_direction).sum()),
            }
        ),
        include_groups=False,
    )
    summary.to_csv(args.out / "reanalysis_summary.csv")
    fidelity = effects[effects.analysis.str.startswith("revised")]
    gap = (fidelity.rho / fidelity.rho_other).map(math.log).abs()
    print(
        f"fidelity: {int((gap < 1e-9).sum())} of {len(gap)} effects reproduce the release's rho exactly; max |log gap| {gap.max():.3g}"
    )
    print(counts(table).to_string())
    print(summary.to_string())


if __name__ == "__main__":
    main()
