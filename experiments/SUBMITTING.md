# Submitting a campaign on Beverin

How to size, submit, watch and cancel a campaign arm on Beverin (AMD MI300A, partition `mi300`), and
which serving configurations complete. Commands run from `experiments/` unless a block says
otherwise. Arms and the role split: [README.md](README.md); more commands (regrade, extraction,
canon, ML scaling): [LAUNCH.md](LAUNCH.md); Slurm flags per serving role:
[`docs/serving/`](../docs/serving/README.md).

A node is 4 MI300A APUs (96 Zen 4 cores, 192 hardware threads, 4 GPUs), about 513 GB.

## Binding rules

- **At most 36 nodes in flight**, shared with the other team on the machine.
- **Partition `mi300`.** `beverin.sbatch` defaults to it. Another partition needs `PARTITION=<p>` and
  a `layers/partition-<p>.env` layer (see LAUNCH.md for mi200).
- **Never pass `--nodes` by hand.** `arm_nodes.sh` sums `INFERENCE_NODES + AGENT_NODES +
  JUDGE_NODES` from the arm's `.env`; `beverin.sbatch` exits 2 when the allocation disagrees.
- **Never pass `--account`/`-A`.** Export `SBATCH_ACCOUNT` (or set it in `layers/site.env`);
  `. experiments/env.sh` hands it to `srun`/`salloc` too, so every job of a campaign bills one account.
- **Size judges from the grading rate** ([README.md](README.md#roles-and-nodes)).
- **Edit an env line, never append.** The file is sourced, so a duplicate silently shadows.

## Submit

`submit.sh` stages the arms' `.env` files, renders the problem lists, and submits through
`submit_common.sh`, which hands `beverin.sbatch` a read-only snapshot of the env and problems file.
Its header lists the knobs.

```bash
TAG=llr-focus40 ./submit.sh                   # dry run: prints each arm and node count
TAG=llr-focus40 SUBMIT=1 ./submit.sh          # submit; HOLD=1 submits held, NICE=<n> sets --nice
```

One arm straight from an existing env file:

```bash
. ./arm_nodes.sh
sbatch --nodes="$(arm_nodes .env.<arm>)" --time="$(arm_walltime .env.<arm> 40)" \
    --partition=mi300 --no-requeue --job-name=<arm> \
    --export=ALL,CLUSTER_ENV_FILE="$PWD/.env.<arm>" beverin.sbatch
```

`arm_walltime <env> <kernels>` covers every agent batch plus `STAGING_HOURS` (default 3). A smoke
run is the same command with a short `--time`. Check the queue and budget first:
`squeue -u "$USER" -o "%.10i %.30j %.9T %.10M %.5D %R"`.

## Sizing agents and walltime

`AGENTS_PER_NODE x AGENT_NODES` is a rolling pool. A pass is `ceil(kernels / agents)`, so the floor
is `passes x AGENT_TIMEOUT_SECONDS` plus serving start-up and the judge drain. Give an arm its floor
plus half again.

| Model | `AGENTS_PER_NODE` | `AGENT_TIMEOUT_SECONDS` (LLR base) | 40 kernels |
| --- | --- | --- | --- |
| oss120b | 40 | 21600 | 1 pass |
| qwen38 | 40 | 21600 | 1 pass |
| kimi27sglang | 20 | 43200 | 2 passes |

Agents stop at `AGENT_TIMEOUT_SECONDS`, not at the job's wall clock. One inference server does not
carry 120 qwen38 agents (decode slows until the arm records almost nothing); split a long list with
`KERNELS_FILE` instead.

## Submission modes

| mode | `/score` | `/submit` | keys |
|---|---|---|---|
| Open (`multi`) | unlimited | unlimited, last verified one counts | `AGENT_SINGLE_SUBMISSION=0`, `AGENT_SUBMISSION_POLICY_FILE=submission-multi.md` |
| Single (`single`) | unlimited | once | `AGENT_SINGLE_SUBMISSION=1`, `AGENT_SUBMISSION_POLICY_FILE=submission-single.md` |
| Blind (`blind`) | none | once | Single's keys with `submission-blind.md`, plus `AGENT_SCORE_TOOL=0` and `HPCAGENT_BENCH_SERVICE_SCORE_ENABLED=0` |

`layers/common.env` defaults to Single. An arm that wants Open or Blind pins both keys itself; Blind
also needs the judge-side switch, or an agent's own HTTP call still reaches `/score`. Tool gates:
[`docs/agents_and_tool_access.md`](../docs/agents_and_tool_access.md).

## Serving configurations

| Configuration | Result |
| --- | --- |
| oss120b on vLLM, aiter off | completes reliably |
| Kimi K2.7 on SGLang, `--attention-backend triton`, `SGLANG_USE_AITER=1` | completes |
| qwen38 on SGLang, same attention config | full accuracy up to 51,200-token cases |
| `JUDGE_NODES=1` (4 ranks) for 40 agents | no judge backlog |
| `--language-only` | campaigns are text-only; a vision stack only costs KV cache |
| weights on `iopsstor` | much higher concurrent-read throughput than general scratch |
| aiter on, vLLM path | fails: kernels JIT-build behind a lock and outlive the engine's RPC deadline |
| qwen38 on vLLM | fails: a fraction of SGLang throughput; `mtp`, `fp8kv+mtp`, aiter legs do not serve |
| aiter MLA on gfx942 | fails: `fmha_v3_varlen_fwd invalid argument` |
| `INFERENCE_ENGINE=sglang` with a vLLM `INFERENCE_CE_ENV` | fails: the image has no sglang |

SGLang needs both `--reasoning-parser` and `--tool-call-parser`; with one missing, turn-1 tool calls
are swallowed and the run reports success with no submission. A job that dies mid-aiter-build leaves
its lock and every later server on that cache blocks; before an aiter run, delete stale locks when no
job runs:

```bash
find "${JIT_CACHE_ROOT:-${SCRATCH}/.hpcagentbench-cache}/.aiter" -name 'lock' -o -name 'lock_*'
```

## Effort

Each model's `arms.yaml` entry declares the rungs its server accepts in `EFFORT_LADDER`; `effort.py` sends
`xhigh` if present, else the top rung, else no effort field (`AGENT_EFFORT_POLICY=max`), exported as
`AGENT_EFFORT`.

| Model | `EFFORT_LADDER` | Sent |
| --- | --- | --- |
| oss120b | `low medium high` | `high` |
| qwen38 | `low medium xhigh` | `xhigh` |
| kimi27sglang | empty | no field |

Never delete the line: `agent_driver.py` defaults a missing `AGENT_EFFORT` to `xhigh`. A harness
whose client types fewer rungs gets the top one it can spell; `harness-end.json` records it. qwen38
arms pass `--chat-template ${SCRIPT_DIR}/chat-template-qwen38.jinja` in `SGLANG_EXTRA_ARGS` (the
stock template rejects `max`, which SGLang maps `xhigh` to); re-apply it when the weights change.

## Problem lists

`make_problems.py` generates problems from the registry and drops kernels that lack the requested
language. Lists are gitignored; regenerate after any skill page changes, because a `--skills` list
inlines the packet.

```bash
"$HPCAGENT_BENCH_HOST_PYTHON" make_problems.py --track loop_level_reasoning --language c \
    --tag llr-focus40 > problems-llr-focus40-c.jsonl          # skills leg: add --skills
```

`JUDGE_INPUT_MODE=source` makes the judge accept only `<kernel>.<ext>` in the arm's language.

## Watch

`RUN_ROOT` comes from the arm's `.env` (`$SCRATCH/hpcagent-bench-runs/<experiment>-<stamp>`); the
run directory is `$RUN_ROOT/<jobid>`.

```bash
sacct -j <jobid> -o JobID,JobName%30,State,Elapsed,ExitCode --parsable2
scontrol show job <jobid> | grep -E 'StdOut|StdErr'

# engine decoding? requests but zero lines = wedged engine
grep -c 'Avg generation throughput' <stdout-log>

# agents got their tools? want one "connected" per agent, no "failed"
grep -ho '"status":"[a-z]*"' "$RUN_ROOT/<jobid>"/agents/node-*/*/claude.log | sort | uniq -c

# live outcome counts from the per-rank judge shards
python3 - "$RUN_ROOT/<jobid>" <<'PY'
import collections, glob, sqlite3, sys
counts = collections.Counter()
for shard in glob.glob(f"{sys.argv[1]}/judge/rank-*/*.db"):
    con = sqlite3.connect(f"file:{shard}?mode=ro", uri=True)
    counts.update({(r, s): n for r, s, n in con.execute("select route, status, count(*) from calls group by 1, 2")})
    con.close()
for key in sorted(counts):
    print(key, counts[key])
PY
```

After the job, merge the judge shards and read the balance report:

```bash
python3 merge_results.py  "$RUN_ROOT/<jobid>"
python3 monitor_report.py "$RUN_ROOT/<jobid>/monitor"
```

## Cancel

```bash
scancel <jobid> [<jobid> ...]
scancel -u "$USER" --name=<arm>      # every arm is --job-name'd
```

Judge shards written before the cancel stay under `$RUN_ROOT/<jobid>/judge/`.

## Traps

- **Exit state is not the result.** An arm cut off by its wall clock reads `COMPLETED`; one where
  every agent exits nonzero reads `FAILED` and may still hold many graded submissions. Read the judge
  DBs.
- **An agent whose MCP server failed at init never submits yet exits rc=0.** Check the `connected`
  count; `AGENT_START_CONCURRENCY` (default 8) staggers starts.
- **Requests logged but zero `Avg generation throughput` means a wedged engine.** Cancel it.
- **The image moves with the engine.** Change `INFERENCE_CE_ENV` together with `INFERENCE_ENGINE`.
  The judge logs `Application startup complete` before the model server dies.
- **The code tree is frozen at job START, not at submit** ([README.md](README.md#isolation)); a
  PENDING job picks up every commit that lands before then. Data roots stay on the live tree.
- **The skills packet is frozen into the problems file.** Re-run the submitter after editing a
  `SKILL.md`.
- **Never edit an env file or launcher while its jobs run**; roles re-source them.
- **Never export `CPF_*` in the submitting shell.** `sbatch --export=ALL` would stage the CPF
  drop-in into a control arm; `submit_arm_job` strips `CPF_DROPIN_DIR`, `CPF_FORMS_DIR` and
  `HPCAGENT_BENCH_SERVICE_CANONICAL_PARALLEL_FORM_DIR`.
- **Walltime can be lowered, never raised.** Submit with slack. A dependency can be added only while
  the job is PENDING. `launch failed requeued held` does not restart: `scontrol release <id>`.
- **Arms compare only under identical serve config.** Changing a model layer mid-campaign splits the
  A/B.
- **Long tool arguments look like a dead stream.** SGLang's `qwen3_coder` parser emits a tool
  argument only when fully decoded; `run_cluster.sh` derives `CLAUDE_STREAM_IDLE_TIMEOUT_MS` from
  `CONTEXT_LENGTH` and `AGENTS_PER_NODE` (`stream_idle_timeout.py`).
- **`verdicts:` in the run report** is utilization advice, not scoring.
- **Harness smokes** need about 2 h wall clock.

## Images

Build, verify and promote: [`containers/README.md`](../containers/README.md). Never write over a live
`.sqsh`; promote by rename (`containers/images/promote_image.sh`). A started job keeps the image it
mounted.

## Python

Host-side steps run `$HPCAGENT_BENCH_HOST_PYTHON` (site layer; `scripts/host_python.sh`). The repo is
mounted, not installed: put it on `PYTHONPATH`. Keep caches off `$HOME` (inode quota). Put the venv on
`PATH` for `pre-commit`, or its format hook reports `missing formatter(s): ruff`.
