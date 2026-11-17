# First-class tools, design (2026-11-17, phase 1: no code)

Six ad-hoc tools of the last three days become HB features. Built on `feat/episode-archives` (`runs pack/unpack/verify/
stage`, `owed --class extend`, `stats.limits.as_extended`). Rules kept: one `experiments.toml`, no new config files, one
way per thing, read-only on run folders, secrets never copied, no schema change unless listed.

Summary of `experiments.toml` changes (all optional, three keys in model tables, one new table kind):
`KV_TOKENS_PER_AGENT` (model), `MOE_CHECKPOINT` (model), `[bench.<name>]` (serve benchmark overlays). Nothing else.

## 1. Serving benchmark and guards

**Bench mode.** `hpcagent-bench serve bench <model> [--config NAME ...] [--concurrency 20,40] [--seconds 900]` submits
one `serve-only.sbatch` job per config (<= 3 queued, rule: slurm hygiene) with `SERVE_BENCH=<name>`. No second sbatch
copy: `serve-only.sbatch` ends, when `SERVE_BENCH` is set, with `python -m hpcagent_bench.cluster.serve_bench` instead of
the "endpoint is live" banner. A config is an overlay of server args in `experiments.toml`:

```toml
[bench.kimi-ep4-triton]
model = "kimi27sglang"
set = { "--ep-size" = "4", "--moe-runner-backend" = "triton" }   # flag = value; "" = bare flag
drop = ["--cuda-graph-max-bs-decode"]
```

`serve_args.apply(base, set, drop)` (new, `hpcagent_bench/cluster/serve_args.py`, plain tokenised-string edit of
`SGLANG_EXTRA_ARGS`) replaces the `*.env` files. The overlay goes through the same checks as a real setup (rules 1b, 1c),
so a bench cannot exercise a combination a launch would refuse; `--force-rules` for deliberate crash hunts.
`serve_bench.py` = `agent_load.py` moved in (httpx, agent-shaped: prefix + turns), and writes `bench-<name>.json` into the
serve job's run dir: config, resolved args, KV pool size (from the server log), TTFT p50/p90, tok/s per agent and total,
cache hit rate (from `--enable-cache-report` usage), API-up seconds, plus the decode-batch log tail. `serve bench report
<dir>...` prints the table that is hand-maintained in `docs/serving/*.md` ("Tuning record"); the record stays prose + the
table pasted from it. `watch.sh` is dropped: the report verb lists `FAIL|Traceback|Killed` lines per job (one function
shared with `validate_run.py`).

**KV guard (1b).** New `hpcagent_bench/cluster/inference_service.py` step after readiness: parse `max_total_num_tokens`
from `server-0.log` (PP0 line; vLLM: `GPU KV cache size`) and compare with `AGENTS_PER_PIPELINE x KV_TOKENS_PER_AGENT`
(model key; kimi27sglang 90000, from the record: pool 1.85M / 20 agents gave 90% cache hit, 0.98M gave 88% at 4.3 tok/s,
0.21M gave 0% and 1.9 tok/s; so 1.6M floor for 20). Below: the job fails before any agent starts (`KV_GUARD=warn` downgrades
to a banner line and a `kv-guard.json` in the run dir; bench mode always warns, it measures starved pools on purpose). Key
absent = no guard (qwen, oss). Test: log fixtures with 0.15M / 1.9M.

**EP rule (1c).** Lives with the other pre-launch arg checks that already run in `setup_nodes.sh` (AGENTS_PER_PIPELINE
check) as a Python rule table `serve_args.RULES`, called from `setup_nodes`, the render (`env_spec`, so `submit` refuses
early) and the bench. Data-driven by one model key, `MOE_CHECKPOINT = 'int4-wna16'` (kimi27sglang only): rule "`--ep-size`
> 1 needs `--moe-runner-backend triton` when `MOE_CHECKPOINT` is int4-wna16" with the one-line reason (dispatcher keeps
global expert ids for the aiter runner; ROCm int4 MoE runs the Triton runner). Not hard-coded to a model name, not in the
sbatch. It never auto-adds the flag: it refuses with the fix (one way, explicit config). Also move the "fraction at or below
0.55 / cuda-graph cap = live agents + slack" rules of `docs/serving/kimi27sglang.md#rules` into the same table.

Replaces: `jobs/kimi-tune-1117/{serve-bench.sbatch,agent_load.py,*.env,watch.sh}`. Tests: `apply` round trip;
rules refuse ep4-without-triton on int4, allow it with; bench report from a canned JSON; KV guard fixtures.

## 2. Submit

**2a. Pipeline split (default for pp models).** A setup whose model has `AGENTS_PER_PIPELINE` and more problems than one
pipeline serves is submitted as N = ceil(P / AGENTS_PER_PIPELINE) jobs, each `INFERENCE_REPLICAS=1`, `AGENT_NODES` for its
share, named `<setup>-p<k>`; the bash block at `submit.sh:265-280` (shrinks replicas for small owed reruns) is replaced
by `hpcagent_bench/cluster/pipelines.py split(problems, per_pipeline) -> list[list[ProblemLine]]` called from submit:
sort by `ProblemLine.level` (KernelBench level; kernels without one are one class), deal to parts in snake order (0..N-1,
N-1..0), so each part has the same level mix and size within 1. Writes `problems-<setup>.p<k>.jsonl`; each part is an
ordinary job of the same setup (same RUN_ROOT, so `owed`/`extract` already pool several jobs per setup; `launch` counts
per (kernel, slot), unaffected). Env stamp `PIPELINE_PART=k/N` (recorded in `inference.json`, no DB column). `INFERENCE_REPLICAS`
stays only for non-pp models (qwen/oss data parallel). Opt out: `SPLIT_PIPELINES=0` (old behaviour). Why split rather than
replicas: 40 agents on one pipeline evicts conversations (46% hit, record row 742706), and a failed pipeline then costs one
part, not the setup. Cost: N serve startups (~1100-1800 s each) run concurrently, no extra node-hours. Open: the 5-replica
setups at `experiments.toml:617` (decide at implementation whether they become 5 parts).

**2b. Reservation-aware time.** `hpcagent_bench/cluster/reservations.py`: `next_blocking(partition, nodes, now)` parses
`scontrol show res -o` (StartTime, EndTime, Nodes, Users/Accounts/Flags; ignores reservations the submitter may use,
those with no overlap with the partition, and the standing year-long ones such as `cscs`, i.e. `Duration > 7 d`), and
`fit(requested_s, minimum_s) -> seconds | Refused(next_start)`. `submit_common.sh` gets one function `fit_walltime
<requested> <min>` (calls the module through the host python) used by:

- `prepare` and `goldens` jobs (`submit_prepare`, `submit_goldens`, today `#SBATCH --time=06:00:00` plus no fit): `--time
  = min(6 h, start - margin)`, minimum 1 h; they resume from caches, a shorter job is useful. Below the minimum: not
  submitted, message names the reservation and its end, `--begin=<end>` is offered via `BEGIN_AFTER_RESERVATION=1`.
- experiment jobs (`job_walltime`): requested = `setup_walltime`. **Never shortened below what the limits need** (the
  existing rule: capped jobs would end before an agent had `AGENT_TIMEOUT_SECONDS`); fitting trims only slack above need;
  if need does not fit, the job is submitted with `--begin=<reservation end>` and the message says so.

Replaces the hand-computed `--time`. `DEADLINE` stays (user wave deadline); the fit takes the earlier. Tests: parser on
saved `scontrol` lines (reservation tomorrow 07:00, standing reservation, user-owned reservation), fit/refuse/begin.

## 3. `hpcagent-bench job retune <jobid> KEY=VALUE ...`

Lives in `hpcagent_bench/cluster/retune.py` (login-node action like `job submit`; forwarded from `cmd_job`). Steps: (1)
refuse unless `squeue -j <id>` says PENDING (a running job has read its env); (2) find the rendered env from the job's
command (the scan `submit_common.sh:151-160` already does; one shared helper, bash and Python both call it); (3) refuse
keys in `env_spec.LIMIT_OVERRIDE_KEYS`, `RESUBMISSION`, secrets, `PROBLEMS_FILE`/`RUN_ROOT`/`SETUP` (identity), and any
key not already in the env unless `--add`; the new value is checked by the render's validators (`env_spec`, the rules of
1b/1c for `SGLANG_EXTRA_ARGS`); (4) copy the env to `<env>.retune-<n>.bak` (0400), write the edited file via temp +
`os.replace` (the original is 0400: the helper chmods u+w for the swap and restores 0400, then re-reads and diffs: only
the requested lines differ); (5) append a ledger line `{ts, jobid, env, key, old, new, reason, user}` to
`experiments/.rendered/retune.jsonl` (`--reason` required). `job retune --show <jobid>` prints the ledger lines. When the job
starts, `run_cluster.sh` already copies the resolved env into the run dir; it also copies the job's ledger lines there as
`retune.jsonl` so the record travels with the run (included in the episode archive set, 6). Replaces `sed -i` on read-only
files. Tests: refuses RUNNING, refuses limit key, diff-only-requested, backup + ledger written, SGLANG value rule applied.

## 4. Runs tooling (`runs inventory`, `runs nodehours`, `clean`)

All three open every DB with `file:...?mode=ro` and write only to `--out` (refused inside a protected root,
`data_guard`). New module `hpcagent_bench/results/inventory.py`, dispatched by the `runs` verb in `artifacts.py` (no new
top-level verb), job enumeration reuses `owed.collect_jobs`/`job_setup`.

- `runs inventory --runs ROOT... --out DIR` writes `run-inventory.csv` (columns as the ad-hoc file: root, job, setup, study,
  model, language, packet, harness, temperature, episodes, ended, kernels, token_limit, time_limit_h, grades, finals;
  fields read from `episodes`/`grades` and the job's `inference.json`/resolved env, not parsed from names) and
  `EXPERIMENTS.generated.md`: the per-root table, limits by `episodes.limits_source`, models, hardware. The hand-written
  prose of `EXPERIMENTS.md` stays hand-written; the generated file is its tables only.
- `runs nodehours --runs ROOT... [--sacct-file F]` sacct (`-X -P -o JobID,JobName,State,ElapsedRaw,NNodes`, ids from
  folder names, batched; `--sacct-file` replays a saved dump), node-hours per job, medians per model and per
  40-kernel-vs-other class (model from the env, not from job-name substrings), COMPLETED/TIMEOUT only. CSV + printed table.
- `hpcagent-bench clean caches [--root R ...] [--apply]` (`hpcagent_bench/cluster/clean_caches.py`): dry run by default
  (counts inodes per class, `df -i`). **Delete list** (allowlist, not blocklist): `__pycache__`, `.ruff_cache`,
  `.pytest_cache`, `.mypy_cache`, `dacecache*`, `.hypothesis`, `build/` of HB worktrees only. **Never** (checked first,
  by resolved path, overrides the allowlist): any `.venv`, `uv-cache`, spack `buildcache`, image-build caches, protected
  roots (runs, archives), `aarch64`-named trees, a tree an active job (`squeue`) names. No wrapper around `uv cache clean`;
  `clean` refuses the argument (a message citing the 11-17 incident). Replaces `jobs/archive-1117/clean_caches.sh`.

Tests: inventory over a synthetic two-job root, nodehours with a recorded sacct dump, clean refuses each protected
class and removes only allowlisted names in a tmp tree.

## 5. Owed arms: one path, mode explicit

`owed run --mode {fresh,resume}` (required for `--submit`, no silent default). Today classes picked the behaviour (`infra`
and `task` relaunch fresh; `extend` resumes). Table, enforced in `owed.py` (one `MODES` dict):

| class | fresh | resume | resume needs |
|---|---|---|---|
| `infra` (repair) | default | allowed when the episode has an archive with a session (job died, node failure) | `runs stage`; limits counted from the record |
| `task` (task changed) | only | refused (the session saw the old task) | n/a |
| `extend` (timed out, longer wall clock) | refused (second budget) | only | `--time-limit-s T` |

`--class extend` implies `--mode resume` and refuses `fresh`, as today. A resumed episode keeps its token counters and wall
clock (`AGENT_TIMEOUT_SPENT_SECONDS`), a fresh one restarts them; both are recorded: `launch_kind` already holds
`infra|task|extend`; add `episodes.launch_mode` (`fresh|resume`, schema bump, NULL before, column `episode_launch_mode`).
Alternative with no bump: derive it from `fresh_relaunches`/`resumes`; I recommend the column, it is the user-visible
promise. `owed collect` lists both modes' candidates with the mode each would use; `owed run` prints it. The compare
rule (`stats.limits.as_extended`) is unchanged and applies to `extend` only. Tests: the mode x class table (refusals),
stage of an `infra` resume from a v2 archive, `launch_mode` recorded.

## 6. Structured log storage (priority)

### Layout (revises the branch's one opaque `<label>.tar.zst`)

One archive hides what is inside and forces a full decompress to read a transcript. Proposed: **one directory per
episode, a few named members, each zstd-compressed alone**, indexed once per job:

```
<job>/episodes/index.jsonl                      one line per episode (below); the only index
<job>/episodes/<setup>/<label>/                 label = <experiment>-<model>-<lang>.n<node>.p<problem>.w<worker> (episode_id)
    episode.json              what restore/resume read: tokens.json + attempts + limits + identity + agent-folders + retune lines + member list with sha256/bytes + inventory schema version (v2)
    prompt.txt.zst            the task prompt as sent
    session/<session>.jsonl.zst   the agent's own conversation store (full tool results; needed to resume)
    transcript.jsonl.zst      the extract (results/transcripts.py): reasoning, text, every tool call whole, tool results as id+len+sha, usage per call, attempt boundaries
    attempts.jsonl.zst        attempt ledger, kept verbatim (also inside episode.json)
    workspace.tar.zst         shared/agent-K/ folders (+ worker files that are not logs), at run-relative paths
    stream.log.zst            OPTIONAL (--keep-stream): claude.log whole, for audit
```

`index.jsonl` line: `{label, setup, schema:2, result, grade summary ids, bytes, members:{name:sha256}}`; `runs verify`
checks every member's sha256 and size, `runs unpack` restores in place. `episode.json` is plain (not zstd) because it is
the one thing read without decompressing.

### What an episode and a run produce today, classified
(from `rev-scicomp40-20261115/742706`, read-only; sizes are that job's, 40 episodes)

| file | size | class | why / what rebuilds it |
|---|---|---|---|
| `agents/node-N/problem-P-worker-W/claude.log` (stream JSON) | 13 MB one episode | **derivable, drop after extract verifies** | 68% `content_block_delta`, 20% `thinking_tokens` estimates; its content blocks are in the session, its usage and reasoning in `transcript.jsonl`; `transcripts.restore` rebuilds a log `token_cost` accepts. Opt-in `--keep-stream` |
| `claude.attempt*.log` | per crash | same as above (one extract each) | |
| `home/.claude/projects/*/<id>.jsonl` | 0.4 MB | **keep** (`session/`) | the conversation store: full tool results, thinking blocks; `claude --resume` needs it. Drop its `api-request*`, `file-history-snapshot`, `atis-latch`, `queue-operation` lines (8% of bytes, request metadata) only in the optional lean mode; default verbatim |
| `home/.claude/` rest (config, telemetry, history, snapshots, `.claude.json`) | small | drop (secrets/regenerable) | resume writes a fresh config |
| `prompt.txt` | 14 KB | keep | exact prompt; not derivable |
| `tokens.json` | 1.5 KB | keep (in `episode.json`, and loose) | token accounting: fresh/cached/output/lost/void, limits, identity, resources; the DB episode row is built from it |
| `attempts.jsonl` | 0.2 KB | keep | attempt ledger (cause, returncode, resumed, ms) |
| `.submission-spent` | 70 B | keep (in `episode.json`) | submission counter for `--multi`/resume |
| `cancelled`, `earlier-leg.json` | tiny | keep | owed / extension read them |
| `mcp.json`, `litellm.yaml`, `episode-secret`, `.claude.json` | | **drop (secret)** | never archived; regenerated at launch |
| `harness-checkpoint.json`, `openhands-state/`, `*.traj.json`, `usage*.jsonl` (other harnesses) | | keep the extract; `usage.jsonl` is derivable from it and from `tokens.json` -> drop; mini/OpenHands state kept only if `resume` of that harness needs it (open) | |
| `workspaces.tar` (job) and loose `shared/agent-K/` | 8 MB, 4 MB | keep per episode as `workspace.tar.zst`; **job-level tar derivable, drop after all episodes packed** | each folder belongs to one episode (`agent-folders.jsonl`) |
| `shared/litellm.yaml`, `shared/` task copies | | drop (secret / copied from the repo at the recorded commit) | |
| `agent-folders.jsonl` | 3 KB | derivable (folder -> episode now in each `episode.json`) | keep in the job until migration done; then drop |
| `results.db`, `judge/rank-N/*.db` | 8.6 + 9 MB | **keep as is** | grades, submissions, graded sources; not logs; queried live; sealed record |
| `judge/upstream-N.log` | 43 KB | keep (zstd, `job-logs.tar.zst`) | judge upstream errors |
| `monitor/*.csv` | 5.6 MB | keep, zstd per file | utilisation, feeds `runs nodehours`-style plots |
| `vllm/nccl.*.log` | 230 MB | **drop** | NCCL debug dumps (15 MB x16); the fabric line ("Using network AWS Libfabric") is copied into `inference.json` by `serve` |
| `inference-server/<job>/server-N.log` (outside run) | | keep zstd as `server-logs.tar.zst` in the job | KV pool, decode logs, startup time; bench/guard read them |
| `aggregate-throughput-node0.json` | 1.8 MB | keep, zstd | throughput evidence; `throughput-node0.json` kept whole |
| `inference.json`, `edf/*.toml`, `prepare_job.sh`, `.agent-launch/`, `*.resolved`, `retune.jsonl` | small | keep (`job-meta.tar.zst`, plain `inference.json`+`*.resolved`) | provenance: exact env, container defs, commit; cheap, not derivable after the repo moves |
| `episode-secret` | 64 B | drop (secret) | |

Estimated: this episode 13 MB loose log -> ~0.3 MB extract + 0.1 MB session zst; the job's 1.5 GB agents/ and 230 MB
vllm/ to roughly 60-100 MB (harness20 measured 10.0 GB -> 0.19 GB by the existing extract rules).

**Not stored in the DB**: no log, transcript, or prompt goes into `results.db`; `index.jsonl` stores only ids and hashes.

### Restoring and resuming need

- **Restore** (`runs unpack`): `episode.json`, `workspace.tar.zst` (at its run-relative paths), `session/`, `prompt.txt`,
  tokens/attempts; `transcript.jsonl` is read in place, never unpacked.
- **Resume** (item 5): the above plus, rebuilt from the extract, a minimal `claude.log` if the driver's token counter reads
  it (**open: confirm in `agent_driver`**; if it reads only `tokens.json` no log is rebuilt), and `earlier-leg.json` from `stage`.
- **Audit/regrade**: `workspace.tar.zst` + `results.db`/`judge/`; collect and `extract` read `transcript.jsonl` as today.

### Verification before anything is deleted

`pack` (default) writes `episodes/` only, removes nothing. `--remove-loose` (existing flag, still explicit, user decision
per job) additionally requires: round-trip (unpack to a temp dir, byte-compare every kept file), `transcripts` totals of
the extract equal `tokens.json` (`token_cost` from restore == from log), index written, and job not in squeue. The sealed
record (`data_guard`) is unsealed/resealed through `data_guard.unsealed`. No tool deletes a run folder itself.

### Migration of existing run folders

- Branch v1 archives (`<label>.tar.zst`, none exist outside tests/dev jobs): `runs repack` converts v1 -> v2 (read tar,
  write members, verify, remove the v1 file only after verify).
- Loose job dirs (every `hpcagent-bench-runs/*/*`): `runs pack <job>` as is. Ordered plan: pack read-only first for all
  jobs of `rev-*` roots (writes only `episodes/`, sealed dirs via `unsealed`), `runs verify`, then the user decides on
  `--remove-loose` per root. Nothing in this phase touches a run folder.
- Old `collect copy` output keeps working: it reads the transcript extract from either loose logs or `transcript.jsonl.zst`
  (one function, `transcripts.sources(worker_or_episode_dir)`).

## Tests (new/changed), summary

`tests/test_serve_args.py`, `test_kv_guard.py`, `test_pipeline_split.py` (snake deal, level balance, sizes within 1),
`test_reservations.py`, `test_retune.py`, `test_runs_inventory.py`, `test_clean_caches.py`, `test_owed.py` (mode table),
`test_artifacts.py` (v2 layout, round trip, extract==log totals, secrets absent, v1 repack). CI-safe: no Slurm, `scontrol`/
`sacct`/`squeue` replayed from fixtures via a `SLURM_BIN` seam already used by `job_running`.

## Open questions

1. Reservation vs experiments: trim slack only and defer with `--begin` (proposed), or refuse? Standing-reservation cutoff (`Duration > 7 d`) right?
2. `launch_mode` DB column (schema bump) or derive from resumes/fresh_relaunches?
3. `infra` + `resume` allowed (proposed) or only `extend` resumes? A resumed infra episode shares counters with its first leg.
4. KV floor: 90 000 tokens per agent for kimi (pool 1.6M for 20); fail (proposed) or warn by default?
5. Drop `claude.log` by default after the extract verifies, or keep a `stream.log.zst` for the first N jobs (~13 MB/episode raw, a few MB zstd)?
6. Does `agent_driver` resume read the token total from `claude.log`, or from `tokens.json`? Decides whether `stage` rebuilds a log.
7. mini-SWE-agent / OpenHands resume state (`harness-checkpoint.json`, `openhands-state/`): keep verbatim in `session/` (proposed, small) or extract only?
8. Pipeline split for the 5-replica setups at `experiments.toml:617`; and job naming `<setup>-p<k>` vs a suffix in the job comment.
9. `episode.json` plain JSON per episode and also `index.jsonl`: keep both (proposed; index is rebuildable by `runs verify --reindex`), or index only?

## Implemented (phase 2): where the code differs from the design above

One commit per item on this branch. The decisions of the review: `claude.log` is kept (compressed), pack/cleanup is an optional
post-experiment step, `infra` resumes by default, reservations defer experiments and trim prepare/goldens, the KV guard fails by
default at 80000 tokens per Kimi agent, the mini-SWE/OpenHands resume state is kept verbatim, split parts keep the setup identity.

1. **Serve.** The verb is `hpcagent-bench serve-bench submit|report` (`serve` is the judge service). `[bench.<name>]` has `model`,
   `set`, `drop` and `env`. Rules are refused at submit (`env_spec check-staged`), in `serve-only.sbatch` and in the bench
   (`BENCH_FORCE_RULES=1` skips). `RULES` holds the int4 expert-parallel rule only; the fraction and cuda-graph advice of the Kimi
   page stays prose (the fraction ceiling differs per model). The KV guard reads the pool from the server logs: `serve-only.sbatch`
   checks once after the API is up, an experiment job runs `serve_guard watch` from the batch shell (disowned, so `wait -n` is
   unaffected) and cancels the job below the floor.
2. **Submit.** `pipelines.py` splits every pp setup (also a single pipeline's rerun, pinning one pipeline); the stamp is the env key
   `PIPELINE_PART=k/N` (in the job's `.agent-launch/<job>/.env`; not in `inference.json`). The jobs of a setup are named
   `<setup>-p<k>`; env files `.env.<setup>.p<k>`. `reservations.py` ignores reservations over 7 days and ones the submitter
   may use; an experiment job that does not fit begins after the reservation (`--begin`), prepare and goldens are trimmed to
   at least 1 h, else deferred the same way. `tests/conftest.py` sets `RESERVATION_FIT=0` for every stub-sbatch test.
3. **Retune.** The ledger is `retune.jsonl` beside the snapshot, copied into the run directory at the job's start.
4. **Runs tooling.** `runs inventory` adds `hardware` and `limits_source` columns; the page is `EXPERIMENTS.generated.md`. The
   cleaner is the top-level verb `clean caches` (not under `runs`), and also never touches a build folder under the working
   directory of a running job.
5. **Owed.** `--mode` is optional: its default is the class's (`infra` resume, `task` fresh, `extend` resume), and the mode used is
   printed. A resumed repair states `spent_seconds` per problem (the driver prefers it to `AGENT_TIMEOUT_SPENT_SECONDS`, which
   stays the job's value for an extension and is the least of them for a repair, so the job's time covers the longest left).
   `launch_mode` is schema 33, `NOT NULL DEFAULT ''` as `launch_kind`; the extractor derives it for an older record from
   `launch_kind` (extend resume, infra and task fresh) rather than from `resumes`/`fresh_relaunches`, which count the ends the
   driver continued or restarted inside a job, not how the launch began.
6. **Log storage.** The folder holds `logs.tar.zst` (every file the transcript extract is read from: stream logs, sessions,
   checkpoints), `worker.tar.zst` (the rest of the worker directory), `workspace.tar.zst`, `transcript.extract.jsonl.zst` and
   `episode.json`, not the `session/` and `stream.log.zst` of the table above: with the stream log kept, restoring and resuming
   need no rebuild, and the whole tree stays one rule (a path at its place in the job). `runs repack` was not written: no v1 archive
   exists outside the development branch. `collect copy` still reads the worker directories (`runs unpack` first).

Tests: `tests/test_{serve_args,serve_guard,serve_bench,pipelines,reservations,submit_pipelines,retune,runs_inventory,clean_caches,
artifacts,owed,matched_limits}.py`, each file ending with explicit calls. 11 tests fail the same way on the base commit in the
Slurm environment (`test_recording`, `test_workspace`, `test_agent_driver_claude_bare[openai|anthropic]`: compiler and sealed-home
setup), none in a file this branch changes but the last, whose `earlier` keyword the extension work left out of a stub.
