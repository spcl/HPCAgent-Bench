# Submitting a campaign on Beverin

Every command below runs from `containers/cluster/example-script/`.

```bash
cd containers/cluster/example-script
```

The live campaign is `llr8`: the `llr-focus40` tag (40 kernels, one agent each) crossed over two
models and two languages, in two legs -- base prompt, and hints plus the per-language skills
packet. See `containers/cluster/example-script/README.md` for what an arm IS; this page is how to
put one on the machine.

## Node budget

Hard ceiling: **36 nodes in flight**, agreed with the team sharing the machine.

| model | nodes per arm | why |
|---|---|---|
| qwen30b | 6 | 1 inference + 1 agent + 4 judge |
| oss120b | 8 | 1 inference + 1 agent + 6 judge |

`JUDGE_NODES` is sized from the measured grading rate, not picked: judges >= agents x
grades-per-agent-per-hour / 30. One judge node runs `JUDGES_PER_NODE` judges, one per socket.

**Never pass `--nodes` yourself.** `arm_nodes.sh` derives it from the arm's own `.env`, and
`beverin.sbatch` exits 2 before the run starts if the allocation disagrees with
`INFERENCE_NODES + AGENT_NODES + JUDGE_NODES`.

**No `--account` on beverin.** The user default association is `root`, and `root`, `a-g200` and
`a-g34` all carry QOS `normal` with no GrpTRES, MaxJobs, MaxSubmit, MaxTRES or priority set --
measured, and a 2-node accountless job allocated and ran (600940). Omitting the flag costs nothing
in scheduling and stops jobs from silently splitting across two project accounts depending on
which command line was typed.

## Submitting

One script. `MODEL` names the family and therefore the env files
(`.env.llr8-<MODEL>-<lang>[-skills]`). Extra arguments pass through to `sbatch` verbatim.

| knob | values | default |
|---|---|---|
| `MODEL` | `qwen30b` `oss120b` | **required** |
| `LANGS` | `c` `fortran`, space separated | both |
| `LEGS` | `1` (base) `2` (skills) | both |
| `TIME` | wall clock | `08:00:00` |
| `CAMPAIGN` | job-name prefix | `llr8` |

```bash
MODEL=qwen30b ./submit-llr8.sh --partition=mi300     # 4 arms, peaks at 12 nodes
MODEL=oss120b ./submit-llr8.sh --partition=mi300     # 4 arms, peaks at 16 nodes
```

Both models together is 28 nodes at peak, inside the ceiling. Within a leg the languages are
chained `--dependency=afterany`, so at most one language of each leg holds nodes -- the pair costs
wall clock instead of nodes. `afterany`, never `afterok`: an arm that dies still leaves a usable
judge DB and you want the pair either way.

Narrow it to one arm by naming the slice:

```bash
LANGS=c LEGS=1 MODEL=qwen30b ./submit-llr8.sh --partition=mi300
```

A smoke run is the same command with the walltime cut, which answers "does a task reach an agent,
get graded, and come back" before you commit a wave:

```bash
TIME=01:00:00 LANGS=c LEGS=1 MODEL=qwen30b ./submit-llr8.sh --partition=mi300
```

## Before you submit

**The skills packet is FROZEN INTO the problems file.** `make_problems.py` inlines the `SKILL.md`
bodies at generation time; a running arm never re-reads the pages. The submitter refuses a stale
list rather than grading a treatment nobody meant to run, so an edited page shows up as a refused
submit:

```bash
re-run the arm's submit-*.sh
```

The lists are named for the TAG, not the campaign -- `llr8` reuses the `llr6` focus40 lists
unchanged, which is what makes the two campaigns comparable.

Then confirm nothing is already running and the budget has room:

```bash
squeue -u "$USER" -o "%.10i %.30j %.9T %.10M %.5D"
```

## Watching a run

```bash
squeue -u "$USER" -o "%.10i %.30j %.9T %.10M %.5D %R"
sacct -j <jobid> -o JobID,JobName%30,State,Elapsed,ExitCode --parsable2

# the job's own logs
tail -f results/beverin-services-<jobid>.err

# is the engine actually decoding?  zero of these after requests arrive = a wedged engine
grep -c 'Avg generation throughput' results/beverin-services-<jobid>.out

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

Completion counts matter: an arm cut off by wall clock is `COMPLETED` but did not finish all 40
kernels. Check the counts before treating an arm as done.

After the job, fold the per-rank judge DBs into one and read the balance report:

```bash
python3 merge_results.py  <RUN_ROOT>/<jobid>
python3 monitor_report.py <RUN_ROOT>/<jobid>/monitor
```

## Cancelling

```bash
scancel <jobid> [<jobid> ...]
scancel -u "$USER" --name=llr8-qwen30b-c     # by arm name, since every arm is --job-name'd
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
- **Arms are only comparable if the serve config is identical.** Changing an `.env.llr8-*` file
  mid-campaign splits the A/B.
- **An arm that logs requests but zero `Avg generation throughput` is wedged, not slow.** It will
  burn its whole wall clock. Kill it.
- **An agent whose MCP server failed at init never submits** and still exits rc=0, so the arm
  looks healthy in `sacct`. Historically 22-25% of every arm's first wave. `AGENT_START_CONCURRENCY`
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

Promotion is verify, then rename. There is ONE version per role, so the rename is what publishes:

```bash
sbatch --export=ALL,IMAGE=$SCRATCH/ce-images/optarena-ce-amd-mi300-candidate.sqsh,\
PROFILE=judge-agent-amd verify_image.sbatch     # 0 failures, nothing resolving outside
mv $SCRATCH/ce-images/optarena-ce-amd-mi300-candidate.sqsh \
   $SCRATCH/ce-images/optarena-ce-amd-mi300.sqsh    # move the .digest and .sha256 with it
ALLOW_REPOINT=1 ./install_edfs.sh
```

A rename is safe while arms are running: a mounted squashfs is held by its inode, so a job that
already started keeps reading the bytes it opened. What is never safe is writing over the file in
place, which is why build.sbatch refuses to when an EDF mounts it.

### Weights: iopsstor and striping (already done -- verify, do not redo)

`run_cluster.sh` puts `HF_HOME` and `VLLM_CACHE_ROOT` on iopsstor (9.45 GB/s at 16 readers vs
capstor's 0.83) and sets a PFL default on the hub dir: narrow below 64 MiB, 16 OSTs at 4 MiB
above. Verified 2026-08-19 -- every large blob of all five models is striped 16. Re-check with:

```bash
lfs getstripe -c <blob> | head -1     # head, NOT tail: getstripe prints a trailing blank line
```

Only if that ever reports a narrow count, and only while NOTHING is reading the model:

```bash
lfs migrate -c 16 -S 4M <blob>
```
