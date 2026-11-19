"""sec45.py OBS[,OBS...] TAG_DIR COMPARATORS_CSV OUT_DIR: the Section 4.5 numbers from one set of observation databases.

Speed per (setup, kernel) is paired_setups.py's speedup leg (best_by_setup_kernel -> setup_aggregates, SOLVED policy,
tag-restricted). paired_setups.py refuses cross-model / cross-language pairs, so the pairs are taken here with its own
score_leg (= significance.paired on per-kernel log ratios, the configured sign-flip test and interval).
  1. Spearman rho of per-kernel speedups between model pairs over kernels both solved (LLR C, LLR Fortran, SC C).
  2. Model gaps (Qwen/Kimi, Qwen/OSS) in LLR C and HIP; HIP-minus-C difference of log(Qwen/OSS), same paired test.
  3. HIP vs Triton per model on LLR.
  4. Agent / comparator (Pluto vs C, PPCG vs HIP): GM over shared kernels (comparator_ratios.py) + paired test.
"""

import pathlib
import sys

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "statistics")
import paired_setups as ps  # noqa: E402

from hpcagent_bench.stats import population, significance  # noqa: E402

OBS = [pathlib.Path(x) for x in sys.argv[1].split(",")]
TAGS, COMP, OUT = pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4])
MODELS = ("qwen38", "kimi27sglang", "oss120b")
R, S = "rev-llr40", "rev-scicomp40"


def tables(tag: str, setups: list[str]) -> dict[str, dict[str, float]]:
    """{setup: {kernel: speedup}} over the kernels each setup solved, as paired_setups.main builds them."""
    obs = observations[observations.setup.isin(setups)]
    kernels = ps.declared_tag(TAGS / tag, obs)
    obs = obs[obs.kernel.isin(kernels)]
    kept, dropped = population.complete_setups(obs, kernels)
    for s in dropped:
        print(f"note: {s} incomplete on {tag}", file=sys.stderr)
    baseline = population.rows_denominator(ps.graded_rows(obs, setups), label="family")
    table = ps.setup_aggregates(ps.best_by_setup_kernel(obs), ps.served_by_setup(obs), baseline, ps.POLICY)
    return {s: dict(zip(a.kernels, a.values)) for s, a in table.items()}


def paired(logs: list[float]) -> dict[str, object]:
    import math

    r = significance.paired(logs)
    return {"n": len(logs), "ratio": math.exp(r.estimate), "ci_low": math.exp(r.low), "ci_high": math.exp(r.high),
            "p": ps.tested_p(r), "method": r.method}


def pair(t: dict[str, dict[str, float]], a: str, b: str) -> dict[str, object]:
    import math

    shared = sorted(t[a].keys() & t[b].keys())
    return {"a": a, "b": b, "solved_a": len(t[a]), "solved_b": len(t[b]),
            **paired([math.log(t[a][k] / t[b][k]) for k in shared])}


observations = ps.load_observations(OBS)
OUT.mkdir(parents=True, exist_ok=True)
langs = ("c", "fortran", "hip", "triton-device")
llr = tables("llr40.txt", [f"{R}-{m}-{l}" for m in MODELS for l in langs])
sc = tables("scicomp40.txt", [f"{S}-{m}-c" for m in MODELS])
pd.DataFrame([{"setup": s, "kernel": k, "speedup": v} for t in (llr, sc) for s, d in t.items() for k, v in d.items()]
             ).to_csv(OUT / "per-kernel.csv", index=False)

rows = []
for name, t, fmt in (("LLR C", llr, f"{R}-{{}}-c"), ("LLR Fortran", llr, f"{R}-{{}}-fortran"), ("SC C", sc, f"{S}-{{}}-c")):
    for i, m1 in enumerate(MODELS):
        for m2 in MODELS[i + 1:]:
            a, b = t[fmt.format(m1)], t[fmt.format(m2)]
            shared = sorted(a.keys() & b.keys())
            rho, p = spearmanr([a[k] for k in shared], [b[k] for k in shared])
            rows.append({"track": name, "models": f"{m1}/{m2}", "n": len(shared), "spearman": rho, "p": p})
pd.DataFrame(rows).to_csv(OUT / "1-spearman.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))

import math  # noqa: E402

rows = [pair(llr, f"{R}-qwen38-{l}", f"{R}-{m}-{l}") for l in ("c", "hip") for m in ("kimi27sglang", "oss120b")]
q, o = (lambda l: llr[f"{R}-qwen38-{l}"]), (lambda l: llr[f"{R}-oss120b-{l}"])
both = sorted(q("c").keys() & o("c").keys() & q("hip").keys() & o("hip").keys())
diff = [math.log(q("hip")[k] / o("hip")[k]) - math.log(q("c")[k] / o("c")[k]) for k in both]
rows.append({"a": "qwen/oss HIP", "b": "qwen/oss C", **paired(diff)})
pd.DataFrame(rows).to_csv(OUT / "2-model-gaps.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))

rows = [pair(llr, f"{R}-{m}-hip", f"{R}-{m}-triton-device") for m in MODELS]
pd.DataFrame(rows).to_csv(OUT / "3-hip-vs-triton.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))

comp = pd.read_csv(COMP).dropna(subset=["speedup"])
comp = comp[comp.speedup > 0]
rows = []
for name, lang in (("pluto", "c"), ("ppcg_hip", "hip")):
    ref = comp[comp.comparator == name].set_index("kernel").speedup
    for m in MODELS:
        agent = llr[f"{R}-{m}-{lang}"]
        shared = sorted(agent.keys() & set(ref.index))
        rows.append({"comparator": name, "model": m, "compiled": len(ref), "tag_compiled": len(set(ref.index) & set(
            ps.declared_tag(TAGS / "llr40.txt", observations))),
                     **paired([math.log(agent[k] / ref[k]) for k in shared])})
pd.DataFrame(rows).to_csv(OUT / "4-comparators.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
