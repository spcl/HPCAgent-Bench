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
./submit-owed-wave.sh MODEL=qwen38 EXPERIMENTS=llr-focus40,llr-focus40-blind,gpu-llr-focus40 \
    TOKEN_SCALE=2 TIME_SCALE=2

# same command + SUBMIT=1 submits it (HOLD=1 as well submits it held)
./submit-owed-wave.sh MODEL=qwen38 EXPERIMENTS=llr-focus40,llr-focus40-blind,gpu-llr-focus40 \
    TOKEN_SCALE=2 TIME_SCALE=2 SUBMIT=1
```

- `MODEL`: `qwen38`, `oss120b` (single-node inference) or `kimi27sglang` (6-node inference).
  Submit the single-node models first.
- `EXPERIMENTS`: `llr-focus40` (LLR cpu), `gpu-llr-focus40` (LLR gpu), `llr-focus40-blind`
  (no-score), `cpf-llr-focus40`, `harness-focus20`. It defaults to the first and the third.
- `TOKEN_SCALE=2 TIME_SCALE=2`: the owed rule. A kernel owed for hitting its budget reruns at 2x
  (24M tokens, 8 h). This scales only the budget class; infra-class kernels rerun at 1x.
- Kernels in a job that is still PENDING count as in flight and are not planned again, so
  running the planner twice does not double-submit.
- Old archived run directories print `unreadable job dir, not coverage`. Those runs count through
  frozen observations; the warning is harmless.
- Mark a kernel owed by hand (a cheat, or a newly wrong kernel) in `rerun-kernels.tsv`, then run
  the planner again.

Example, 2026-09-21: every owed wave in priority order:

```bash
for m in qwen38 oss120b kimi27sglang; do
  ./submit-owed-wave.sh MODEL=$m EXPERIMENTS=llr-focus40,llr-focus40-blind,gpu-llr-focus40 \
      TOKEN_SCALE=2 TIME_SCALE=2 SUBMIT=1
done
./submit-owed-wave.sh MODEL=qwen38 EXPERIMENTS=harness-focus20 TOKEN_SCALE=2 TIME_SCALE=2 SUBMIT=1
# later experiments go behind: raise their nice
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
Recover it on the login node with the venv:

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
