"""Table 19 body (pooled, tab:pooled) from all_rows.py's all-rows.csv. Usage: python gen_table19.py all-rows.csv"""
import csv
import sys

NAMES = [
    ("language-packet", "Language packet"), ("blind-vs-open", "Blind vs.\\ open"),
    ("blind-vs-single-sc40", "Blind vs.\\ single, SciComp40"), ("blind-vs-single-llr40", "Blind vs.\\ single, Focus40"),
    ("git-framing", "Git framing"), ("perf-toolkit", "Perf.\\ toolkit"), ("CPF", "CPF"),
    ("harnesses", "Harnesses"), ("autokernel", "AutoKernel plugin"), ("terse", "Terse"),
]
CPF = r"CPF$^{\dagger}$ & 3 & $1.07\times$ ($0.96$--$1.20$) & incon. & $-3.3$ ($-8.3$ to $-0.1$) & equiv. & $1.29\times$ ($1.05$--$1.59$) & diff.  \\"
SHORT = {"different": "diff.", "equivalent": "equiv.", "inconclusive": "incon."}


def ratio(r: dict[str, str], m: str) -> str:
    return f"${float(r[m + '_ratio']):.2f}\\times$ (${float(r[m + '_ci_low']):.2f}$--${float(r[m + '_ci_high']):.2f}$)"


def points(r: dict[str, str]) -> str:
    p, lo, hi = (float(r[k]) for k in ("solve_points", "solve_ci_low", "solve_ci_high"))
    sign = "+" if p > 0 else ""
    return f"${sign}{p:.1f}$ (${lo:.1f}$ to ${hi:.1f}$)"


rows = {r["row"]: r for r in csv.DictReader(open(sys.argv[1], newline=""))}
for key, name in NAMES:
    if key == "CPF":
        print(CPF)
        continue
    r = rows[key]
    print(f"{name} & {r['comparisons']} & {ratio(r, 'speed')} & {SHORT[r['speed_verdict']]} & {points(r)} & "
          f"{SHORT[r['solve_verdict']]} & {ratio(r, 'cost')} & {SHORT[r['cost_verdict']]}  \\\\")
