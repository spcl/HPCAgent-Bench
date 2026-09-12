# Submitting on beverin (AMD MI300A)

How the campaign submission stack fits together (`beverin.sbatch`, `run_cluster.sh`,
`arm_nodes.sh`), how to size and submit an arm, and which serving configurations for the three
campaign models actually complete versus fail.

Node facts: 192 cores, 4 GPU dies, about 513 GB per node, partition `mi300`. The Slurm flags that
matter for a role inside this stack (`--mem=0`, `--cpus-per-task`, never `--account`) are covered
in [`docs/serving/README.md`](../docs/serving/README.md#3-submitting-the-slurm-flags-and-why-each-one)
and [`docs/serving/knobs.md`](../docs/serving/knobs.md); this page does not repeat them. **House
ceiling is 36 nodes in flight.**

## The stack

```
submit-llr8.sh          picks arms, sizes nodes, chains dependencies
  -> beverin.sbatch     one allocation; splits it into inference / agent / judge roles
    -> run_cluster.sh   re-entered INSIDE each role's container; builds the serve command
```

`CLUSTER_ENV_FILE` selects the arm and is the only thing that differs between arms, so several
arms run in parallel from one script. The env file is `. `-sourced, meaning **a duplicate
assignment silently shadows the earlier one** -- edit the line, never append a second.

`beverin.sbatch` refuses an allocation that does not equal `INFERENCE_NODES + AGENT_NODES +
JUDGE_NODES` exactly. `arm_nodes.sh` reads those three from the same file the launcher reads, so
the submitter and the launcher cannot disagree; always size with it rather than a literal.

One judge NODE is four ranks, one per socket (`GRADE_CPUS` = cores-per-socket, from
`run_cluster.sh:88`). One judge node covers 40 agents with headroom, so `JUDGE_NODES=1` is right
for every arm here.

## Submitting one arm

Each campaign family owns a `submit-<family>.sh` script in this directory (`submit-cpf-llr40.sh`,
`submit-gpu-llr40.sh`, `submit-git-scicomp.sh`, `submit-llrblind.sh`, `submit-scicomp-dc.sh`, and
others): it builds or points at a problem list, picks the `.env.<arm>` files for its arms, and
submits each through `beverin.sbatch`. Read the header comment of the script you are running; the
roster, the legs and the languages it covers are named there.

`run_campaign.sh <variant> [sbatch args...]` is the generic single-arm entry point several of them
use: it reads `.env.<variant>`, generates the problem list `PROBLEMS_FILE` names (or requires it to
already exist, for a completion run), sizes the allocation from
`INFERENCE_NODES + AGENT_NODES + JUDGE_NODES`, and submits `beverin.sbatch`. Its own header comment
lists the variant names it recognizes.

To submit one arm directly against an existing `.env.<arm>` file without either wrapper:

```bash
cd experiments
CLUSTER_ENV_FILE=.env.cpf-llr-focus40-oss120b-c sbatch \
    --nodes="$(. ./arm_nodes.sh; arm_nodes .env.cpf-llr-focus40-oss120b-c)" \
    --time=08:00:00 --partition=mi300 --job-name=cpf-llr40-oss120b-c beverin.sbatch
```

### Splitting a roster across waves

A model whose agent count cannot hold a full kernel list in one arm needs the list split into
disjoint halves with their own arm labels, each submitted as a separate job. Verify the halves are
disjoint and their union is the full list before trusting a wave; nothing enforces that for you. A
retry arm holds only the kernels an earlier wave never submitted.

## What worked

| config | result |
|---|---|
| oss120b on vLLM, **aiter off** | completes reliably across many arms |
| Kimi K2.7 on SGLang, `--attention-backend triton` + `SGLANG_USE_AITER=1` | completes, tens of submissions per arm |
| qwen3.8 on SGLang, same attention config | serves with full accuracy up to 51,200-token cases, zero request errors |
| `JUDGE_NODES=1` (4 ranks) for a 40-agent arm | holds hundreds of scoring calls without a judge backlog |
| Text-only serving (`--language-only`) | campaign is always text-only; a vision stack only costs KV |
| Weights on `iopsstor` rather than `capstor` | many times the concurrent-read throughput; see [`docs/serving/knobs.md`](../docs/serving/knobs.md) |

SGLang needs **both** `--reasoning-parser` and `--tool-call-parser` named. With one missing, turn-1
tool calls are swallowed and the run is logged as a success that submitted nothing.

## What did not work

| config | what happens |
|---|---|
| **aiter master switch on, vLLM path** | kernels JIT-build on the first request behind a baton lock; the build outlives the engine's RPC deadline. The engine dies with `step_counter=0`: not one token decoded, zero submissions, the allocation lost |
| qwen3.8 on vLLM, any leg | decodes at a small fraction of the SGLang throughput; `mtp`, `fp8kv+mtp` and every aiter leg fail to serve at all |
| qwen3.8 `fp8kv` on vLLM | serves, then decodes at 0.0 tok/s |
| aiter MLA on gfx942 | `fmha_v3_varlen_fwd invalid argument` |
| `INFERENCE_ENGINE=sglang` with a vLLM `INFERENCE_CE_ENV` | `/opt/venv/bin/python3` in that image has neither sglang nor huggingface_hub; the role dies in seconds resolving the model path |
| qwen3.8 on the stock chat template, at any effort | claude always sends `output_config.effort`; SGLang renames only the top rung, `"max" if oc.effort == "xhigh" else oc.effort`, and Qwen3.8's stock template accepts only `xhigh`/`medium`/`low`. `xhigh` arrives as `max` and the built-in default `high` arrives as `high`; both raise, and the arm submits nothing |
| `AGENTS_PER_NODE=120` | a coding agent spends most of its wall clock in tools or a compile, so the batch never fills |

The aiter failures above share one mechanism, not several unrelated bugs. A job that dies mid-build **leaves its lock**, and
every later server on that cache root then blocks on a baton nobody holds; a 0-byte lock names
nobody at all. Before any run that enables aiter, sweep them:

```bash
find "${SCRATCH}/.jit-cache" -name 'lock' -o -name 'lock_*'   # inspect, then delete if no job is running
```

On SGLang aiter is fine and stays on -- it imports a prebuilt `module_aiter_core` and serves. It is
the vLLM path that builds on first request and dies.

## Traps that cost whole runs

- **The image moves with the engine.** Changing `INFERENCE_ENGINE` without changing
  `INFERENCE_CE_ENV` gets you an interpreter with neither the engine nor its dependencies. The
  judge's uvicorn logs `Application startup complete` before the model server dies, so the log
  looks healthy.
- **A job's exit code is not its result.** An arm is FAILED when *every* agent exits nonzero and
  COMPLETED when one does not. A FAILED arm can still produce many submissions across many
  kernels, and a COMPLETED arm can have most of its agents exit nonzero. Read the judge DBs, not
  sacct.
- **`verdicts:` in the run report is utilization advice, not scoring.** `verdicts: none` means the
  monitor had no sizing complaint.
- **`launch failed requeued held` does not restart.** Slurm holds the job; `scontrol release <id>`.
- **Results live in per-rank SQLite**, `<run>/judge/rank-*/hpcagent_bench*.db`, tables
  `submissions` / `attempts` / `calls`. Join on the bare kernel name -- the problems lists carry a
  full path and the DB stores the basename.
- **Agents are cut off by `AGENT_TIMEOUT_SECONDS`, not job wall clock.** This can end most of an
  arm's agents at their per-agent timeout while kernels remain to grade. A shorter kernel list does
  not buy an agent more time.
- **Never edit an env file or a launcher script while jobs run** -- roles re-source them.

## Agents

`AGENTS_PER_NODE` x `AGENT_NODES` is a ROLLING pool -- one
`ThreadPoolExecutor(max_workers=workers)` with a submit per problem, so that many start and each
finisher launches the next. It is not a barrier.

| model | agents | kernels | passes | `AGENT_TIMEOUT_SECONDS` | worst case | `--time` |
|---|---|---|---|---|---|---|
| oss120b | 40 | 40 | 1 | 14400 (4h) | 4h | 08:00:00 |
| qwen3.8 | 20 | 40 | 2 | 14400 (4h) | 8h | 14:00:00 |
| Kimi K2.7 | 12 | 20 per half | 1 per half | 28800 (8h) | 8h | 14:00:00 |

Kimi cannot hold 40 kernels in one arm at 12 agents, which is why C is split into `-a` / `-b`.

**Size `--time` from passes, not from the per-agent cap.** A pass is `ceil(kernels / agents)`, and
the pool runs them back to back, so the floor is `passes x AGENT_TIMEOUT_SECONDS`, and the arm
still has to bring the server up before it and drain the judge after it. qwen3.8's two passes give
it an 8h floor: submitting it at exactly that limit leaves no room for staging and the judge drain,
and Slurm cancels it partway through. Give every arm at least its floor plus half again, passed to
whichever submit script or `sbatch` invocation you are using.

A limit can be LOWERED with `scontrol` after the fact and never raised, so an arm submitted short
has to be resubmitted -- cheap in the first minutes, an entire wall clock later.

## Effort levels -- per model, not a shared dial

| model | value | why |
|---|---|---|
| oss120b | `high` | ladder is low/medium/high, and the template renders `Reasoning: <v>` VERBATIM with no guard -- a wrong value is pasted into the system prompt rather than refused |
| qwen3.8 | `xhigh` | needs the patched template below -- the ladder is low/medium/xhigh, and `xhigh` only reaches it under the name `max` |
| Kimi K2.7 | *(empty)* | no ladder at all -- `reasoning_effort` is a K3-only field |

Never delete the line. `agent_driver.py` defaults a MISSING `AGENT_EFFORT` to `xhigh`, which is
not what an arm that wants the empty value gets.

**Every request carries an effort, whatever you do.** Setting `AGENT_EFFORT` empty does not send
nothing: `agent_driver.py` pops `CLAUDE_CODE_EFFORT_LEVEL` and claude falls back to its own built-in
default, `high` -- captured off the wire as `output_config: {"effort": "high"}` with an empty config
directory, so this is claude's default and not the submitter's `~/.claude/settings.json`. Kimi has no
ladder and ignores whatever arrives, so its arms are not effort-free either; their results stand, but
do not describe them as such.

Qwen3.8 therefore needs `chat-template-qwen38.jinja`, the stock template plus three lines that
resolve `max` to `xhigh` before the validation below it. The arms pass it with
`--chat-template ${SCRIPT_DIR}/chat-template-qwen38.jinja` in `SGLANG_EXTRA_ARGS`. Rendered output
for `max` is byte-identical to the stock template's for `xhigh`, and every other level -- `high`
included -- still raises exactly as the vendor wrote it, so a misconfigured arm still fails loudly.
Re-copy the file from the model snapshot and re-apply those three lines when the weights change.

## Problem lists

Regenerate whenever a SKILL.md changes -- a `-skills` list INLINES the packet and goes stale
silently. Lists are gitignored generated artifacts; the as-run copies for recorded experiments live
in `ICLR26Reproducibility/paper_artifacts/problems/`.

```bash
V=/capstor/scratch/cscs/ybudanaz/x86_64/venv-optarena-314/bin/python3
cd <repo root>
PYTHONPATH=$PWD $V experiments/make_problems.py \
    --track loop_level_reasoning --language c --tag llr-focus40 \
    > experiments/problems-llr-focus40-c.jsonl
# skills leg: the same command plus --skills

# Kimi halves, disjoint and covering the tag:
head -20 problems-llr-focus40-c.jsonl > problems-llr-focus40-kimi-c-a.jsonl
tail -20 problems-llr-focus40-c.jsonl > problems-llr-focus40-kimi-c-b.jsonl
```

`hints-and-triggers.md` is NOT checked in -- `materialize_shared.sh` builds it at launch from
`containers/agent/hints.md` plus `skill-triggers.md`, so it always tracks the repo.

## Results and watching

`RUN_ROOT` in the .env decides where a run lands; point a new campaign at a new folder rather than
mixing waves. Scores are in `<RUN_ROOT>/<jobid>/judge/rank-*/hpcagent_bench*.db`.

```bash
squeue -u $USER -o "%.10i %.28j %.2t %.10M %.6D %R"
tail -f results/beverin-services-<jobid>.out
scontrol release <jobid>          # for launch failed requeued held
```

## Python

`/capstor/scratch/cscs/ybudanaz/x86_64/venv-optarena-314` (3.14.7, pyenv global). The repo is
MOUNTED, never pip-installed, so put it on `PYTHONPATH`. Rebuild with
`tools/rebuild_venv.sh`. Keep caches off HOME -- that quota is INODES, not bytes.
Note that `pre-commit`'s format hook needs the venv on `PATH` or it reports `missing formatter(s):
ruff` even when ruff is installed.
