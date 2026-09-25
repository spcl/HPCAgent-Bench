# Adding an inference engine or an LLM

Two changes: (A) serving a new model on an engine the cluster path already runs (SGLang or vLLM),
and (B) adding a third engine. Measured values (memory fraction, pool size, ready timeout) live in
the comments of `experiments/layers/model-<tag>.env`, `experiments/arms.yaml` and
`docs/serving/<tag>.md`. Copy them from there.

## A. A new model on SGLang or vLLM

| What you touch | Why |
| --- | --- |
| `experiments/layers/model-<tag>.env` | the serving recipe; every campaign renders it as `<campaign>:<tag>` |
| `experiments/arms.yaml` `models.<tag>` | only where one campaign must differ for this model (optional) |
| `hpcagent_bench/envs/registry.yaml` `models:` | display name, and the checkpoint the arms must serve |
| `docs/serving/<tag>.md` | the measurements behind the recipe |

**1. Fetch the weights.** `<tag>` is the model token in arm names (`llr-focus40-<tag>-c`). The job
downloads inside the `hpcagent-bench-sglang-mi300-latest` EDF into `${HF_HOME}` (default
`${FAST_SCRATCH}/.hpcagentbench-cache/hf`; `FAST_SCRATCH` defaults to the iopsstor scratch, see
`scripts/cache_env.sh`),
then restripes every blob over 1 GiB on the host; `AUDIT_ONLY=1` only checks the layout.
```bash
MODELS="org/Name" sbatch containers/cluster/ce-images/inference/fetch_weights.sbatch
```

**2. Write `layers/model-<tag>.env`.** Extend the family layers with the same engine and node
shape (`replicas.env` for one node, `pp.env` for a 4-node pipeline, `sglang.env` for sglang,
`service.env` for a hosted API) and set only what is this model's own. `layers/model-qwen38.env`:

```bash
# extends: replicas.env
# extends: sglang.env
INFERENCE_CE_ENV=hpcagent-bench-sglang-mi300-latest
VLLM_MODEL=Qwen/Qwen3.8-27B-FP8
HPCAGENT_BENCH_OPTIMIZER=Qwen/Qwen3.8-27B-FP8
```

The llr40 campaign then needs its window and ladder, in `arms.yaml` under `campaign.models.<tag>`
(`EFFORT_LADDER`, `CONTEXT_LENGTH`, `SGLANG_EXTRA_ARGS` with `--mem-fraction-static <measured>` and
both parsers). Check the result with `experiments/env_spec.py render campaign:<tag>`.

| Key | Read by | Meaning |
| --- | --- | --- |
| `INFERENCE_ENGINE` | `run_cluster.sh` `run_vllm_node` | `sglang` runs `sglang.launch_server`; anything else, or unset, runs `vllm serve` |
| `INFERENCE_CE_ENV` | `run_cluster.sh` `role_srun` | EDF name in `~/.edf`; its image must carry that engine |
| `VLLM_MODEL`, `VLLM_SERVED_MODEL` | `run_vllm_node`, `run_agent_node` | HF repo id (both engines), and the name agents request |
| `GPUS_PER_NODE`, `INFERENCE_NODES`, `INFERENCE_MODE` | `run_vllm_node` | TP size; `pp` splits one model over the nodes, `replicas` runs one server per node |
| `SGLANG_EXTRA_ARGS`, `VLLM_EXTRA_ARGS` | `run_vllm_node` (`read -r -a`) | split on whitespace, no quoting inside; name both parsers |
| `SGLANG_ATTENTION_BACKEND` | `run_vllm_node` | absent appends `--attention-backend aiter`; assigned empty omits it |
| `HPCAGENT_BENCH_OPTIMIZER` | `tests/test_display_names.py` | the checkpoint id; must equal the registry `serves:` |
| `EFFORT_LADDER` | `effort.py`, from `run_cluster.sh` and `harnesses.py` | the rungs THIS server accepts, lowest first; empty for a model with no ladder. The launcher resolves `AGENT_EFFORT` from it (`AGENT_EFFORT_POLICY=max`: xhigh where the ladder has it, else its top rung, else no field) and a client that types fewer rungs gets the top one it can spell |

Compaction needs no key of its own: `agent_driver.claude_context_env` reads `CONTEXT_LENGTH` /
`--context-length` / `--max-model-len` off the same env (`agent_driver.served_context`) and computes
the trigger itself, capped at 262144. See [`docs/token_accounting.md`](../token_accounting.md#context-compaction).

Model files such as a chat template sit in `experiments/`, named through `${SCRIPT_DIR}`, which
`run_cluster.sh` sets before sourcing the env and mounts into the inference container. A
counterfactual of an existing model is its own layer extending the same family layers (see
`experiments/README.md` "Arm envs").

**3. Serve it alone.** From `experiments/`, `SUBMIT=0 MODEL=<tag> ./serve-only.sbatch` prints the
plan and `MODEL=<tag> ./serve-only.sbatch` runs the campaign's own `--vllm-node` role with the base
env plus `serve-only.env`, sized from `INFERENCE_NODES`, and prints a working `curl`. For the
tool-call, reasoning and long-context accuracy gates on SGLang, submit the smoke from its own
directory, where its log path and verifier resolve:

```bash
cd containers/cluster/ce-images/inference
EDF=$HOME/.edf/<edf>.toml MODEL_REPO=org/Name SERVED_MODEL=<tag> \
TOOL_PARSER=<parser> REASONING_PARSER=<parser> LANGUAGE_ONLY=<0|1> \
MEM_FRACTION=<measured> CONTEXT_LEN=<context> SGLANG_EXTRA_ARGS="<model flags>" \
    sbatch --nodes=<INFERENCE_NODES> smoke-kimi-sglang.sbatch
```

The smoke hard-codes some flags of its own, so a pass shows the image serves the model and
`serve-only.sbatch` shows the recipe does. `submit-glm53-sglang.sh` wraps it for one model; for
vLLM, `smoke-kimi-eager-pg.sbatch` takes `MODEL_REPO`, `TOOL_PARSER`, `REASONING_PARSER`, `EXTRA_SERVE_ARGS`.

**4. Register the tag** as `<tag>: {name: <Display Name>, serves: org/Name}` at the END of `models:` in
`registry.yaml` (key order is marker order; `tests/test_palette.py` pins it); aliases go under `aliases.models`.

**5. Submit.** Every launcher renders `<campaign>:${model}`, so `MODELS=<tag>` is enough.

**In-process and API models.** The Python harness ignores these env files. An OpenAI-shaped endpoint
is one `ModelSpec` entry in `MODELS` (`hpcagent_bench/harness/baselines.py`): `backend="openai"`,
`model`, `base_url`, `api_key_env`, `context_tokens`, and `accepts_sampling`/`max_tokens_field` for
an endpoint that rejects sampling or renames the reply cap (see `kimi`). `OpenAIAgent` falls back to
`OPENAI_BASE_URL`, `VLLM_BASE_URL`, then `localhost:8000/v1`. A new wire protocol is an `Agent`
subclass beside `ClaudeAgent` and `OllamaAgent` in `harness/agent.py`, added to `BACKENDS`
(baselines.py), `_agent_registry` (cli.py) and `ModelSpec.agent`; see [writing_an_agent.md](../writing_an_agent.md).

Checklist A:
- [ ] `fetch_weights.sbatch` prints `WEIGHTS READY`; `INFERENCE_CE_ENV` carries `INFERENCE_ENGINE`
- [ ] both parsers named; `HPCAGENT_BENCH_OPTIMIZER` = `VLLM_MODEL` = registry `serves:`
- [ ] `serve-only.sbatch` reaches `endpoint is live`; the smoke ends without `SMOKE FAILED`
- [ ] `pytest tests/test_display_names.py tests/test_palette.py tests/test_model_of.py --maxfail=10`

## B. A new engine

| What you touch | Why |
| --- | --- |
| `containers/cluster/ce-images/<engine>/` | `Dockerfile`, `build.sh`, `build.sbatch`, `edf.toml.example` |
| `containers/cluster/ce-images/images.env` | one row: role, `INFERENCE_<ENGINE>` prefix, platform, dir, partition, profile, candidate, squashfs, EDF, template, tag |
| `containers/cluster/ce-images/verify_image.py`, `verify_image.sbatch` | the engine's verify profile |
| `experiments/run_cluster.sh` `run_vllm_node` | interpreter and launch command |

**1. Image directory.** Copy `sglang/`. `build.sh` pins the base by digest and builds from the repo
root so the Dockerfile can COPY `inference/moe-configs/`; `build.sbatch` refuses to overwrite a
mounted `.sqsh`. `edf.toml.example` keeps the `PLACEHOLDER.sqsh` image line, a multi-line
`mounts = [` block (`derived_edf` exits 2 on a one-line block), absolute `PATH` and `LD_LIBRARY_PATH`
in `[env]` (the CE drops the image's own ENV) and the three fabric hook annotations.

**2. Build, verify, promote.** With the `images.env` row beside the sglang one, run
`IMAGE_DIR=$PWD/<engine> sbatch build_and_verify.sbatch`, then `./promote_image.sh <engine>`.
`build_and_verify.sbatch`, `install_edfs.sh`, `promote_image.sh`, `pull_image.sh`, `pull_images.sbatch` and
`push_images.sbatch` read the row; `experiments/smoke-new-images.sh` names the engine itself.

**3. Launch it in `run_vllm_node`.** `INFERENCE_ENGINE` is tested in three places: `engine_python`
(the interpreter that resolves the snapshot), the non-SGLang default `VLLM_ROCM_USE_AITER=0`, and
the `command=(...)` branch. Add a branch beside `sglang` that serves `${model_path}` as
`${VLLM_SERVED_MODEL}` on `0.0.0.0:${VLLM_PORT}` with TP `GPUS_PER_NODE`; under `pp` it takes PP
size, rank and rendezvous from `INFERENCE_NODES`, `SLURM_PROCID` and `VLLM_MASTER_HOST:VLLM_MASTER_PORT`,
and only rank 0 binds the port. Append `<ENGINE>_EXTRA_ARGS` via `read -r -a`. The job expects
`GET /v1/models` (readiness), `POST /v1/chat/completions` (throughput probe) and, in the default
`AGENT_LLM_MODE=direct`, Anthropic `POST /v1/messages` at the server root (`AGENT_LLM_MODE=litellm`
fronts the engine with a proxy). The aggregate probe in `agent_driver.py` sums only `vllm:*` metrics.

**4. Smoke and tests.**
```bash
experiments/smoke-new-images.sh <engine>
cd experiments && MODEL=<tag> ./serve-only.sbatch    # a base env with INFERENCE_ENGINE=<engine>
pytest tests/test_vllm_pp_serve_args.py tests/test_derived_edf.py --maxfail=10
```
Checklist B:
- [ ] candidate carries `.verified`; `promote_image.sh <engine>` re-renders `<engine>-latest`
- [ ] every script in step 2 names the engine; the branch handles `pp` and `replicas`
- [ ] the endpoint answers the three routes in step 3, or the arm sets `AGENT_LLM_MODE=litellm`
