"""Table 18 rows (tables/all_comparisons.tex body) from the paired_setups.py CSVs of pairs50*.sh.

Usage: python gen_table18.py PAIRS_DIR RCCL_CSV > rows.tex. CPF rows are v1 (never rerun) and copied verbatim."""
import csv
import pathlib
import sys

MODEL = {"qwen38": "Qwen3.8-27B", "oss120b": "OSS-120B", "kimi27sglang": "Kimi-K2.7"}
LANG = {"c": "C", "fortran": "Fortran", "hip": "HIP", "triton-device": "Triton"}
CPF_ROWS = r"""CPF & LLR CPU & Qwen3.8-27B & C & 40 & 40/37 & -7.5 & 1.11 [0.90, 1.37] & 2.05 [1.45, 2.92] & i/i/d & 1.41 & 1.59 \\
CPF & LLR CPU & OSS-120B & C & 40 & 38/38 & 0 & 1.23 [1.08, 1.39] & 0.97 [0.68, 1.41] & i/d/i & 1.26 & 1.62 \\
CPF & LLR CPU & Kimi-K2.7 & C & 40 & 40/39 & -2.5 & 0.92 [0.73, 1.15] & 0.98 [0.74, 1.29] & i/i/i & 1.42 & 1.48 \\"""


def model_lang(setup: str) -> tuple[str, str]:
    """Model and language of a setup name such as rev-llr40-qwen38-triton-device-lang-skills."""
    for key, name in MODEL.items():
        if f"-{key}-" in setup:
            rest = setup.split(f"-{key}-", 1)[1]
            for lk in ("triton-device", "fortran", "hip", "c"):
                if rest == lk or rest.startswith(lk + "-") or rest.startswith(lk + "+"):
                    return name, LANG[lk]
    raise ValueError(setup)


def cells(path: pathlib.Path) -> list[tuple[str, str, str]]:
    """Per pair in file order: (model, language, the row cells from n on)."""
    legs: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            legs.setdefault((r["setup_a"], r["setup_b"]), {})[r["leg"]] = r
    out = []
    for (a, _b), leg in legs.items():
        sp, tk = leg["speedup"], leg["tokens"]
        dr = 100.0 * float(sp["rate_difference"])
        verdicts = "/".join(x[0] for x in (sp["coverage_conclusion"], sp["conclusion"], tk["conclusion"]))
        f = lambda x: f"{float(x):.2f}"  # noqa: E731
        body = (f"{sp['n_roster']} & {sp['n_b']}/{sp['n_a']} & {'0' if dr == 0 else f'{dr:+.1f}'} & "
                f"{f(sp['rho'])} [{f(sp['ci_low'])}, {f(sp['ci_high'])}] & {f(tk['rho'])} [{f(tk['ci_low'])}, {f(tk['ci_high'])}] & "
                f"{verdicts} & {f(sp['mde_ratio'])} & {f(tk['mde_ratio'])} \\\\")
        model, lang = model_lang(a)
        out.append((model, lang, body))
    return out


def rows(intervention: str, experiment: str, path: pathlib.Path, label: dict[str, str] | None = None) -> list[str]:
    out = []
    for model, lang, body in cells(path):
        name = (label or {}).get(model, intervention)
        out.append(f"{name} & {experiment} & {model} & {lang} & {body}")
    return out


def main() -> None:
    d, rccl = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    harness = [r.replace("Harness20 &", "Harness20 &", 1) for r in []]
    lines = []
    lines += rows("Language packet", "LLR CPU", d / "llr-cpu.csv")
    lines += CPF_ROWS.splitlines()
    lines += rows("Language packet", "LLR GPU", d / "llr-gpu.csv")
    lines += rows("Language packet", "LLR blind", d / "llr-blind-packets.csv")
    lines += rows("Blind", "LLR", d / "blind-vs-open.csv")
    lines += rows("Git framing", "Git-SciComp", d / "git.csv")
    lines += rows("Perf.\\ toolkit", "SciComp40", d / "sc40-toolkit.csv")
    names = ["mini-SWE-agent", "OpenHands", "AutoKernel"]
    for i, r in enumerate(rows("Harness", "Harness20", d / "harness20.csv")):
        lines.append(names[i % 3] + r[len("Harness"):])
    lines += rows("Terse", "LLR", d / "terse-llr.csv")
    lines += rows("Terse", "Harness20", d / "terse-harness20.csv")
    lines += rows("RCCL hints", "MLScale", rccl)
    del harness
    print("\n".join(lines))
    print(f"% {len(lines)} comparisons", file=sys.stderr)


if __name__ == "__main__":
    main()
