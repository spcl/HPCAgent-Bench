# Owed kernels and checkpointing

A campaign is done when every (arm, kernel) of its roster has an answer. What is missing is
**owed** and gets rerun; what already ran is never run again. This page is the overview; the full
rules are in `experiments/README.md` ("Owed kernels") and the step-by-step runbook in
`experiments/LAUNCH.md` section 9. Collecting and extracting the finished data is
[data_collection.md](data_collection.md).

## What "owed" means

`experiments/remaining_kernels.py` reads every run root of the campaign and, per arm identity
(`X` and `X-clean` are one arm), marks a roster kernel **delivered** when a job of that arm holds a
real grade for it: a `submissions` row, or a genuine `attempts` row graded after the kernel's
manifest last changed, inside the episode's FINAL attempt (a crashed attempt's `/submit` is no
answer). Rows filed under the `adhoc` run id belong to no episode and deliver nothing. Frozen rows
of deleted job directories (`$HPCAGENT_BENCH_FROZEN_OBSERVATIONS`) count as coverage.

Every other roster kernel is owed, classed by how its latest episode ended:

| class | latest episode (`tokens.json` exit code) | rerun budget |
|---|---|---|
| `budget` | 124 / 125: the harness's own timeout or token cap | the arm's 1x scaled by `TOKEN_SCALE`/`TIME_SCALE` |
| `infra` | cancelled, the job died (wall clock, node failure, judge crash), unknown exit, no episode at all, or a clean exit (0, 123, 126, context overflow) that left no grade | 1x |

The 1x is `owed_wave.rerun_base`: the larger of the experiment's policy budget and the arm's own
unscaled budget, so a second budget rerun does not compound; time is capped at 20 h. Nothing
counts reruns: a kernel stays owed until it is delivered. Inside one episode a crashed agent is
relaunched from an empty workspace up to `AGENT_CRASH_ATTEMPTS` (3) times; a timeout is not.

Hand-kept lists override the databases, which are never edited to force a rerun:

| file | row | effect |
|---|---|---|
| `experiments/rerun-kernels.tsv` | (arm, kernel, jobs, reason, status, class) | kernel owed whatever its rows say (a judge rank died mid-run) |
| `experiments/rerun-lost.tsv` | (arm, deleted_jobs, reason, status) | a setup whose job dirs were deleted; `RERUN_LOST=1` plans it over its whole roster |
| `experiments/final-grade-exempt.tsv` | a submission whose source is gone | keeps its live grade as final (written by `regrade_rest.py --exempt-out`) |

A rerun's rows supersede the old ones under the latest-run rule; the old rows stay.

## Recovering before rerunning

A crashed episode can hold a correct `/score` it never promoted to `/submit`. Grading those
(`hpcagent-bench regrade worklist --scope unpromoted`, then `regrade.sbatch ... run`) is cheaper
than a second agent, and the planner leaves out every (arm, kernel) such a promotion worklist
answers (`PROMOTING=`).

## Planning and submitting owed waves

`experiments/submit-owed-wave.sh` (planner `experiments/owed_wave.py`) turns the owed kernels into
fused jobs, one per (experiment, model, harness), and is a dry run unless `SUBMIT=1`. Each owed arm
is rebuilt from its newest launch env with the `-clean` arm name and the class budget; a plan that
would change an arm's contract (any key but budget, identity, images and the model's serving keys)
is refused. Every rerun, a budget repeat included, runs in its arm's own submission mode
(`AGENT_SINGLE_SUBMISSION`, `AGENT_SUBMISSION_POLICY_FILE` as the arm's own submitter launched it),
so an open-mode arm's repeat stays open and its rows pool with the arm's under one mode. Arms with a queued or running job are skipped, so planning twice never double-submits;
when `squeue` does not answer, the dry run still plans and `SUBMIT=1` refuses.

```bash
cd experiments
export PYTHONPATH=$PWD/..
RUNS=$SCRATCH/hpcagent-bench-runs WORK=$SCRATCH/owed/llr-focus40-qwen38

# 1. the owed list, one <arm>.txt per arm that owes anything
python remaining_kernels.py --run-root "$RUNS"/cpf-llr-focus40-<date> \
    --run-root "$RUNS"/owed-llr-focus40-<date> --tag llr-focus40 \
    --arm-prefix cpf-llr-focus40-qwen38 --out-dir "$WORK/owed"

# 2. dry run: env, problems and setups files under OUT, a contract PASS/FAIL per wave, no sbatch
SUBMIT=0 ./submit-owed-wave.sh MODEL=qwen38 EXPERIMENTS=llr-focus40 \
    KERNELS_FILE="$WORK/owed/cpf-llr-focus40-qwen38-c.txt" TOKEN_SCALE=2 TIME_SCALE=2 OUT="$WORK/wave"

# 3. submit the same plan (the account comes from scripts/cscs/account_env.sh)
. ../scripts/cscs/account_env.sh
SUBMIT=1 ./submit-owed-wave.sh MODEL=qwen38 EXPERIMENTS=llr-focus40 \
    KERNELS_FILE="$WORK/owed/cpf-llr-focus40-qwen38-c.txt" TOKEN_SCALE=2 TIME_SCALE=2 OUT="$WORK/wave"

# 4. health of the new wave after 30-45 minutes
python check_job.py <job id>
```

A wave's job is `owed-<experiment>-<model>-<harness>-w<N>` and writes the run root
`owed-<experiment>-<date>`; extraction and `remaining_kernels.py` credit its rows to each arm it
served. After the waves end, the same dry run prints `no owed kernels for <model>`.

## Checkpointing and resume

There is no in-job resume: a job either finishes its problems or its unfinished (arm, kernel)
pairs become owed, and the owed wave is the resume path.

- **Run directory.** Each job writes `$RUN_ROOT/<job id>/`: `judge/rank-N/*.db` (the grades),
  `agents/node-N/problem-P-worker-W/` (`tokens.json` with the exit code and final-attempt start,
  a `cancelled` marker, the transcript), `shared/` and `monitor/`. The launch env sits beside it in
  `$RUN_ROOT/.agent-launch/<job id>/`, which is what the owed planner rebuilds an arm from.
- **No requeue.** Every job is submitted `--no-requeue`: a NODE_FAIL requeue would keep the job id
  and so reuse the run directory, stacking the second run's rows on the first's. A failed job is
  resubmitted by hand (or through an owed wave).
- **Cancelled episodes.** An episode with a `cancelled` marker is owed as `infra`, and analysis
  drops every row of a cancelled task (spec X8), so its partial work is never credited.
- **Final attempt only.** Only rows graded inside an episode's final attempt deliver a kernel; the
  analysis keeps the latest valid submission per (arm, kernel) (`population.latest_runs`), so a
  rerun replaces only the kernels it ran.
- **Regrade shards resume.** `hpcagent-bench regrade run` writes `regrade-<shard>.db` and skips
  every (db, run id, benchmark, ts) it already holds; `cells` writes `regrade-cells-<shard>.db` and,
  under `--migrate`, re-times every row not yet graded under the final rule. Resubmit the same
  `regrade.sbatch` call with the SAME node count (items are dealt `items[shard::shards]`) and it
  continues where it stopped. A regrade row links to its submission by that key.
- **mlscale grade claims.** Auto-mode grade jobs claim submissions in `<out>/scaling-claims.db`
  under `BEGIN IMMEDIATE`, heartbeat every 60 s, and take over a claim whose heartbeat is older
  than 600 s, so chunk jobs run side by side without grading one item twice and a dead job's items
  are picked up again. The done set is what the `scaling-grade-*.db` files hold;
  `python -m hpcagent_bench.harness.scaling_grade pending` counts the rest.

```bash
# resume a killed regrade shard: the same call, the same --nodes
sbatch --no-requeue --nodes=2 --time=07:00:00 regrade.sbatch "$WORK/final.jsonl" "$WORK/final-out" cells 1

# how much mlscale grading is left
python -m hpcagent_bench.harness.scaling_grade pending --runs "$RUNS"/mlscale-<stamp> \
    --env-dir . --out-dir "$SCRATCH/mlscale-grade/<stamp>"
```
