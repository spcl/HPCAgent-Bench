# `statistics/` -- analyze a campaign that already ran

Everything here reads a results DB / observations CSV a campaign already produced and computes or
plots a number from it. Nothing here submits a job, drives an agent, or is imported by
`experiments/run_cluster.sh` or any other live driver -- that is the dividing line from
`experiments/` (run a campaign) and `scripts/` (pre-commit gates, setup, dev tooling). The shared
statistics engine itself (`palette.py`, `style.py`, `summary.py`, `figures/`, geomean/CI, signed-rank)
stays a package at `hpcagent_bench/stats/`; everything below imports it, none of it re-implements it.

    plot_*.py                12 figures -- one entry point per figure, CLI args only, no logic of
                              their own (see docs/plotting.md for which figure answers which question)
    ablation_stats.py         paired within-kernel ablation stats over merged campaign DBs
    paired_arms.py            paired-arm geomean speedup + token-ratio extraction (CPF/CPFsrc pairs)
    iteration_counts.py       per-agent turn/tool-call counts from transcripts, feeds paired_arms.py

Run any of them with `-h`; `docs/plotting.md` and `docs/measurement_statistics.md` explain the
statistics each one applies (geomean + CI, Mann-Whitney/signed-rank, BH correction) and why.

Moved here 2026-09-19 from `scripts/` (the 12 `plot_*.py`) and `experiments/` (`ablation_stats.py`,
`paired_arms.py`, `iteration_counts.py`) -- confirmed via repo-wide grep that none of the three has a
live-driver import (unlike `experiments/token_report.py` and `experiments/token_cost.py`, which
`run_cluster.sh`/`agent_driver.py` import at run time and which therefore stay in `experiments/`).
