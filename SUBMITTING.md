# Submitting a campaign on Beverin

Node budget, arm sizing, and how to watch and cancel a running campaign. Every command below runs
from `experiments/`.

```bash
cd experiments
```

An arm is one `.env.<arm>` file: role sizes, the problems list, the language and the treatment all
come from it. See [`experiments/README.md`](experiments/README.md) for what an arm is and how the
role split works, and [`experiments/AMD-SUBMISSION.md`](experiments/AMD-SUBMISSION.md) for the
current `submit-<family>.sh` scripts and `run_campaign.sh`, which this page assumes.

## Node budget

Hard ceiling: **36 nodes in flight**, agreed with the team sharing the machine.

| model | nodes per arm | why |
|---|---|---|
| qwen38 | 6 | 1 inference + 1 agent + 4 judge |
| oss120b | 8 | 1 inference + 1 agent + 6 judge |

`JUDGE_NODES` is sized from the measured grading rate, not picked, and the unit is nodes, not
judges: a node runs `JUDGES_PER_NODE` judges, one per socket.

    JUDGE_NODES = ceil(peak grades-per-hour / (170 x JUDGES_PER_NODE)), minimum 1

170 is one rank's measured rate with headroom: a grade compiles, runs and times a submission in
16-21s, so a rank sustains around 200 grades per hour.

**Never pass `--nodes` yourself.** `arm_nodes.sh` derives it from the arm's own `.env`, and
`beverin.sbatch` exits 2 before the run starts if the allocation disagrees with
`INFERENCE_NODES + AGENT_NODES + JUDGE_NODES`.

**No `--account` on beverin.** Every association carries the same QOS, so naming one costs nothing
in scheduling and only risks jobs silently splitting across two project accounts depending on
which command line was typed.

## Submitting

Each campaign family owns a `submit-<family>.sh` script in `experiments/` that builds or points at
a problem list, picks the `.env.<arm>` files for its arms, and submits each through
`beverin.sbatch`; `run_campaign.sh <variant> [sbatch args...]` is the generic single-arm entry
point several of them use. Read the header comment of the script you are running for its exact
knobs (model, language, leg) and see
[`experiments/AMD-SUBMISSION.md`](experiments/AMD-SUBMISSION.md) for the current family list.

To submit one arm directly against an existing `.env.<arm>` file:

```bash
sbatch --nodes="$(. ./arm_nodes.sh; arm_nodes .env.<arm>)" \
    --time=08:00:00 --partition=mi300 --job-name=<arm> \
    --export=ALL,CLUSTER_ENV_FILE="$PWD/.env.<arm>" beverin.sbatch
```

A smoke run is the same command with the walltime cut, which answers "does a task reach an agent,
get graded, and come back" before you commit a wave.

## Before you submit

**The skills packet is FROZEN INTO the problems file.** `make_problems.py` inlines the `SKILL.md`
bodies at generation time; a running arm never re-reads the pages. The submitter refuses a stale
list rather than grading a treatment nobody meant to run, so an edited page shows up as a refused
submit; regenerate the problems list and re-run the arm's `submit-*.sh` when a skills page changes.

Then confirm nothing is already running and the budget has room:

```bash
squeue -u "$USER" -o "%.10i %.30j %.9T %.10M %.5D"
```

## Watching a run

```bash
squeue -u "$USER" -o "%.10i %.30j %.9T %.10M %.5D %R"
sacct -j <jobid> -o JobID,JobName%30,State,Elapsed,ExitCode --parsable2

# the job's own logs
tail -f "${SCRATCH}/hpcagent-bench-runs/slurm/beverin-services-<jobid>.err"

# is the engine actually decoding?  zero of these after requests arrive = a wedged engine
grep -c 'Avg generation throughput' "${SCRATCH}/hpcagent-bench-runs/slurm/beverin-services-<jobid>.out"

# did the agents get their tools?  an agent whose MCP server failed at init never submits,
# burns its whole budget in api_retry, and still exits rc=0 -- so this is not visible in sacct.
# Want one "connected" per agent and no "failed"; a log with neither has not started yet.
grep -ho '"status":"[a-z]*"' <RUN_ROOT>/<jobid>/agents/node-*/*/claude.log | sort | uniq -c

# outcome counts, live, from the per-rank judge shards
python3 - <<'PY'
import glob, sqlite3, collections
counts = collections.Counter()
for shard in glob.glob('<RUN_ROOT>/<jobid>/judge/rank-*/*.db'):
    con = sqlite3.connect(f'file:{shard}?mode=ro', uri=True)
    for route, status, n in con.execute('select route, status, count(*) from calls group by route, status'):
        counts[(route, status)] += n
    con.close()
for key in sorted(counts):
    print(key, counts[key])
PY
```

`RUN_ROOT` is `$SCRATCH/hpcagent-bench-runs` (see the arm's `.env`).

Completion counts matter: an arm cut off by wall clock is `COMPLETED` but did not finish its
full kernel list. Check the counts before treating an arm as done.

After the job, fold the per-rank judge DBs into one and read the balance report:

```bash
python3 merge_results.py  <RUN_ROOT>/<jobid>
python3 monitor_report.py <RUN_ROOT>/<jobid>/monitor
```

## Cancelling

```bash
scancel <jobid> [<jobid> ...]
scancel -u "$USER" --name=<arm>               # by arm name, since every arm is --job-name'd
```

Judge shards written before the cancel survive under `<RUN_ROOT>/<jobid>/judge/`, so a cancelled
arm still carries partial results.

## Known traps

- **A walltime can be lowered but never raised.** `scontrol update jobid=<id> TimeLimit=<t>`
  answers *"Access/permission denied"* when `<t>` is longer than the current limit, so an arm
  submitted too tight has to be cancelled and resubmitted, losing its warmup. Submit with slack.
- **A dependency can only be added while the job is PENDING.** Once it starts,
  `scontrol update jobid=<id> dependency=...` answers *"Job is no longer pending execution"* and
  the two run concurrently.
- **Never edit a file a running arm reads.** Slurm snapshots the BATCH SCRIPT at submit time, so
  editing `beverin.sbatch` does not reach a queued job -- but `run_cluster.sh`, `agent_driver.py`,
  the skills pages, the manifests and the problems lists are all read LIVE from
  `HPCAGENT_BENCH_REPO`, which is the submitting worktree. A `.env.<arm>` is read when the job
  STARTS, not when you submit it, so moving one breaks a pending arm.
- **Arms are only comparable if the serve config is identical.** Changing an `.env.<arm>` file
  mid-campaign splits the A/B.
- **An arm that logs requests but zero `Avg generation throughput` is wedged, not slow.** It will
  burn its whole wall clock. Kill it.
- **An agent whose MCP server failed at init never submits** and still exits rc=0, so the arm
  looks healthy in `sacct`. This has cost 22-25% of an arm's first wave. `AGENT_START_CONCURRENCY`
  staggers the starts; check the connected count rather than assuming.
- **Image patches rewrite the image IN PLACE**, so never let one land while arms are queued
  against it.
- **kimi27code is not a viable family.** At campaign context it needs ~4.1 s per forward pass;
  its envs and probes were removed. Anything reintroducing it needs a decode gate first.

## Infrastructure jobs (images, gates, weights)

Not campaign arms. One to four nodes, and what you submit when the question is "can the campaign
move", not "how did the model score".

```bash
cd containers/cluster/ce-images

# One Dockerfile per role, and IMAGE_DIR must be spelled: without it build.sbatch exits in
# about a second and the job looks like it ran. Each lands as <role>-candidate.sqsh.
#
# The judge image pins compilers by MAJOR version only (gcc 16, LLVM 22) because the PPA serves
# 16.0.1, not a fixed point release; the build records what it resolved to in
# /usr/local/share/toolchain-provenance.
sbatch --partition=mi300 --export=ALL,IMAGE_DIR=$PWD/judge-agent-amd judge-agent-amd/build.sbatch
sbatch --partition=mi300 --export=ALL,IMAGE_DIR=$PWD/sglang         sglang/build.sbatch
sbatch --partition=mi300 --export=ALL,IMAGE_DIR=$PWD/vllm           vllm/build.sbatch
```

`inference/build/` is the multi-phase chain that produced the upstream pulls these Dockerfiles
replaced. It is kept because arms are still running on those images, not because it is how a new
image gets built.

Promotion is verify, then rename. There is ONE version per role, so the rename is what publishes.
`promote_image.sh` does both the rename and the `.digest`/`.sha256` sidecar move, then repoints
the EDFs:

```bash
sbatch --export=ALL,IMAGE=$SCRATCH/ce-images/optarena-ce-amd-mi300-candidate.sqsh,\
PROFILE=judge-agent-amd verify_image.sbatch     # 0 failures, nothing resolving outside
./promote_image.sh judge-agent-amd              # or --all for every role with a candidate
```

A rename is safe while arms are running: a mounted squashfs is held by its inode, so a job that
already started keeps reading the bytes it opened. What is never safe is writing over the file in
place, which is why build.sbatch refuses to when an EDF mounts it.

### Weights: iopsstor and striping (already done -- verify, do not redo)

`run_cluster.sh` puts `HF_HOME` and `VLLM_CACHE_ROOT` on iopsstor (9.45 GB/s at 16 readers vs
capstor's 0.83) and sets a PFL default on the hub dir: narrow below 64 MiB, 16 OSTs at 4 MiB
above. Every large blob of the served models is striped 16. Re-check with:

```bash
lfs getstripe -c <blob> | head -1     # head, NOT tail: getstripe prints a trailing blank line
```

Only if that ever reports a narrow count, and only while NOTHING is reading the model:

```bash
lfs migrate -c 16 -S 4M <blob>
```
