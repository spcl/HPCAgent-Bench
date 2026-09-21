# Launching jobs on Beverin: commands with examples

Every command below runs from `experiments/` on a login node. Every job reads the LIVE checkout
(hpcagent-bench `main`) when it STARTS, not when it is submitted, so a fix merged while a job
waits in the queue reaches it. Always pass `-A a-g34 --partition=mi300 --no-requeue`. Jobs never
resubmit themselves; a failed job is resubmitted by hand.

Common setup for the Python planners:

```bash
cd $SCRATCH/hpcagent-bench/experiments
R=$(dirname $PWD)
export PYTHONPATH=$R:$R/hpcagent_bench/numpy_translators/src
```

## 1. Owed waves (reruns of kernels an arm still owes)

`submit-owed-wave.sh` plans one fused job per (experiment, model, harness): one inference server
serving every owed kernel of that model across the experiment's arms. By default it only does a
DRY RUN.

```bash
# dry run: print the waves (setups, kernel counts, nodes, walltime), submit nothing
./submit-owed-wave.sh MODEL=qwen38 EXPERIMENTS=llr-focus40,llr-focus40-blind \
    TOKEN_SCALE=2 TIME_SCALE=2

# same command + SUBMIT=1 submits it (HOLD=1 as well submits it held)
./submit-owed-wave.sh MODEL=qwen38 EXPERIMENTS=llr-focus40,llr-focus40-blind \
    TOKEN_SCALE=2 TIME_SCALE=2 SUBMIT=1
```

- `MODEL`: `qwen38`, `oss120b` (single-node inference) or `kimi27sglang` (6-node inference).
  Submit the single-node models first.
- `EXPERIMENTS`: the RECORDED experiment, not the submit script's name. `llr-focus40` covers
  every LLR arm, CPU and GPU: the `cpf-llr-focus40-*` packet arms (cpf, cpfsrc, skills,
  perf-playbook-cpu) and the `gpu-llr-focus40-*` arms. `llr-focus40-blind` is no-score,
  `harness-focus20` the harness study. It defaults to `llr-focus40,llr-focus40-blind`.
- `TOKEN_SCALE=2 TIME_SCALE=2`: the owed rule. A kernel owed for hitting its budget reruns at 2x
  of the arm's base (base since 2026-09-21: 24M tokens; 6 h qwen38/oss120b, 12 h kimi27sglang;
  so 48M and 12 h / 20 h capped). This scales only the budget class; infra-class kernels rerun at 1x.
- Kernels of an arm with a job still PENDING or RUNNING are not planned again (the log says
  `skip <arm>: a job of it is queued or running`), so running the planner twice does not
  double-submit. The flip side: an arm with a queued job shows no owed work until that job ends.
- Old archived run directories print `unreadable job dir, not coverage`. Those runs count through
  frozen observations; the warning is harmless.
- Every other skip is a `note: skip <arm>/<kernel>: <why>` line. Read them: a skipped kernel is
  owed work the plan dropped.
- Mark a kernel owed by hand (a cheat, or a newly wrong kernel) in `rerun-kernels.tsv`, then run
  the planner again.

### End to end: find and run every owed LLR kernel

```bash
cd $SCRATCH/hpcagent-bench/experiments
R=$(dirname $PWD); export PYTHONPATH=$R:$R/hpcagent_bench/numpy_translators/src  # else: No module named hpcagent_bench
EXPS=llr-focus40,llr-focus40-blind   # LLR cpu + gpu, and no-score

# 1. What is already in flight (owed jobs are named owed-<experiment>-<model>-<harness>-w<N>)
squeue -u $USER -o "%i %j %T %D %S" | grep -E "owed-|llr"

# 2. Detect: dry run per model, plan + files under OUT, nothing submitted
mkdir -p /tmp/owed
for m in qwen38 oss120b kimi27sglang; do
  OUT=/tmp/owed/$m ./submit-owed-wave.sh MODEL=$m EXPERIMENTS=$EXPS TOKEN_SCALE=2 TIME_SCALE=2 \
      2>&1 | grep -v "does not match\|unreadable job dir" | tee /tmp/owed/$m.log
done
# per wave:  owed-llr-focus40-qwen38-claude-w1: 11 kernels, 1 setups, 3 nodes, walltime 15:00:00
# per setup: cpf-llr-focus40-qwen38-c-perf-playbook-cpu-clean.budget2x  11 kernels ... class=budget
# nothing:   no owed kernels for <model>

# 3. Count and check: kernels per model, and any owed kernel the plan dropped
grep -h "kernels, .* setups" /tmp/owed/*.log
grep -h "note: skip .*/" /tmp/owed/*.log    # must be empty or explained

# 4. Inspect one wave's problems before submitting (arm, kernel, task text)
head -c 600 /tmp/owed/qwen38/*.jsonl

# 5. Submit: the same command + SUBMIT=1, single-node models first
for m in qwen38 oss120b kimi27sglang; do
  ./submit-owed-wave.sh MODEL=$m EXPERIMENTS=$EXPS TOKEN_SCALE=2 TIME_SCALE=2 SUBMIT=1
done
# -> submitted owed-llr-focus40-qwen38-claude-w1 -> 645755 (3 nodes, --time 15:00:00) env .rendered/...

# 6. After the jobs end: the same dry run must print "no owed kernels" for every model
```

Example, 2026-09-21: 23 perf-playbook-cpu kernels (qwen38 11, oss120b 12) were owed but the plan
skipped them as `no launched problem entry to rerun`: their arm's newest launch dir held only a
2-kernel top-up. Fixed in 43f5eb4a4 (a rendered-track arm renders the missing task fresh);
submitted as 645755 and 645756.

Harness waves and later experiments go behind the LLR waves:

```bash
./submit-owed-wave.sh MODEL=qwen38 EXPERIMENTS=harness-focus20 TOKEN_SCALE=2 TIME_SCALE=2 SUBMIT=1
scontrol update job=<jobid> nice=500
```

## 2. Regrade (re-time recorded submissions under the current policy)

```bash
sbatch -A a-g34 --partition=mi300 --no-requeue --nice=0 --nodes=3 --time=16:00:00 \
    --job-name=regrade-q0 regrade.sbatch <worklist.jsonl> <out-dir> cells 1
```

- The 4th argument is the number `1` (migrate to mwd-final). The word `migrate` there silently
  means NO migration.
- Each node runs 4 graders (one APU and one GPU each). `--nodes=N` splits the worklist into
  `4N` static shards.
- Each shard writes `<out-dir>/regrade-<shard>.db` row by row and skips rows it already holds.
  A job that hits its time limit is resubmitted with the SAME worklist, out-dir and `--nodes`,
  and it resumes where it stopped.
- To split a large worklist, cut it into a few files (for example 4 files, one job each) rather
  than dozens of one-node jobs: the queue start time is the same, and 4 jobs are easier to watch.
- Once it finishes: `extract_llr40.py ... --regrades "<out-dir>/*/regrade-*.db"`.

## 3. Token-record extraction after exit 75

Exit 75 means the agents ran and the judge rows exist; only the token freeze at the end failed.
`run_cluster.sh` writes `EXTRACTION_FAILED` into the run dir when the steps start and removes it only
after a successful extraction, so ANY finished job that still has the file (exit 75, a dead service
step, scancel, the time limit) needs this recovery. Recover it on the login node with the venv:

```bash
D=$SCRATCH/hpcagent-bench-runs/owed-llr-focus40-20260920/644920
$SCRATCH/venv-hpcagent-bench-314/bin/python $R/reproducibility/llr40/extract_llr40.py \
    --runs $D --benchmarks $R/hpcagent_bench/benchmarks \
    --out $D/observations --db $D/observations/observations.sqlite && rm -f $D/EXTRACTION_FAILED
```

## 4. Images

```bash
cd $SCRATCH/hpcagent-bench/containers/cluster/ce-images
DRY_RUN=1 ./promote_image.sh --all     # what would move
./promote_image.sh --all               # candidate -> live name; pending jobs pick it up at start
REGISTRY_USER=<dockerhub-user> REGISTRY_TOKEN=<PAT with write to spcleth/hpcagent-bench> \
    ./push-images-to-dockerhub.sh      # publish agent + judge
```

A running job keeps the image it opened. Jobs resolve `~/.edf/*-latest.toml` when they start, and
those files point at the promoted `.sqsh`.

## 5. Checking jobs

```bash
squeue -u $USER -o "%.8i %.40j %.8T %.4D %.20S %.8Q"   # state, nodes, estimated start, priority
sacct -j <jobid> -o jobid,jobname%30,state,exitcode,elapsed
```

Exit `75`: see section 3. `FAILED 1:0` on a multi-node job whose steps are `Killed` at the end:
the agents finished and the teardown killed the servers, which is normal.
