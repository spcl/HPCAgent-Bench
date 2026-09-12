# Adding an inference engine or an LLM

Two changes: (A) serving a new model on an engine the cluster path already runs (SGLang or vLLM),
and (B) adding a third engine. Measured values (memory fraction, pool size, ready timeout) live in
the comments of `experiments/.env.base-<tag>` and in `docs/serving/<tag>.md`. Copy them from there.

## A. A new model on SGLang or vLLM

| What you touch | Why |
| --- | --- |
| `experiments/.env.base-<tag>` | serving recipe plus agent knobs; campaign launchers copy it |
| `hpcagent_bench/envs/registry.yaml` `models:` | display name, and the checkpoint the arms must serve |
| model maps in `experiments/submit-*.sh` | launchers that do not read `.env.base-${model}` directly |
| `docs/serving/<tag>.md` | the measurements behind the recipe |

**1. Fetch the weights.** `<tag>` is the model token in arm names (`llr-focus40-<tag>-c`). The job
downloads inside the `sglang-latest` EDF into `${HF_HOME}` (default `/iopsstor/scratch/cscs/$USER/hf`),
then restripes every blob over 1 GiB on the host; `AUDIT_ONLY=1` only checks the layout.
```bash
MODELS="org/Name" sbatch containers/cluster/ce-images/inference/fetch_weights.sbatch
```

**2. Write `.env.base-<tag>`.** Copy the base with the same engine and node shape (`qwen38` and
`oss120b` use one node, `kimi27sglang` and `glm53` four in `pp` mode) and edit the serving keys.
Trimmed from `.env.base-qwen38`:

```bash
INFERENCE_NODES=1
GPUS_PER_NODE=4
INFERENCE_MODE=replicas
INFERENCE_CE_ENV=sglang-latest
AGENT_EFFORT=xhigh
VLLM_MODEL=Qwen/Qwen3.8-27B-FP8
VLLM_SERVED_MODEL=optarena-vllm
OPTARENA_OPTIMIZER=Qwen/Qwen3.8-27B-FP8
CLAUDE_AUTOCOMPACT=200144
INFERENCE_ENGINE=sglang
SGLANG_EXTRA_ARGS="--chat-template ${SCRIPT_DIR}/chat-template-qwen38.jinja --trust-remote-code --context-length 262144 --mem-fraction-static <measured> --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-metrics"
```

| Key | Read by | Meaning |
| --- | --- | --- |
| `INFERENCE_ENGINE` | `run_cluster.sh` `run_vllm_node` | `sglang` runs `sglang.launch_server`; anything else, or unset, runs `vllm serve` |
| `INFERENCE_CE_ENV` | `run_cluster.sh` `role_srun` | EDF name in `~/.edf`; its image must carry that engine |
| `VLLM_MODEL`, `VLLM_SERVED_MODEL` | `run_vllm_node`, `run_agent_node` | HF repo id (both engines), and the name agents request |
| `GPUS_PER_NODE`, `INFERENCE_NODES`, `INFERENCE_MODE` | `run_vllm_node` | TP size; `pp` splits one model over the nodes, `replicas` runs one server per node |
| `SGLANG_EXTRA_ARGS`, `VLLM_EXTRA_ARGS` | `run_vllm_node` (`read -r -a`) | split on whitespace, no quoting inside; name both parsers |
| `SGLANG_ATTENTION_BACKEND` | `run_vllm_node` | absent appends `--attention-backend aiter`; assigned empty omits it |
| `OPTARENA_OPTIMIZER` | `tests/test_display_names.py` | the checkpoint id; must equal the registry `serves:` |
| `AGENT_EFFORT` | `agent_driver.py` | empty sends no effort level; absent means `xhigh` |
| `CLAUDE_AUTOCOMPACT` | `arm_nodes.sh` `check_context_budget` | at most context - 32000 - 30000 |

Model files such as a chat template sit in `experiments/`, named through `${SCRIPT_DIR}`, which
`run_cluster.sh` sets before sourcing the env and mounts into the inference container. For a
counterfactual of an existing model, generate the envs so only the serving block differs:
`make_glm53_envs.py` (kimi to GLM-5.3), `make_model_arm.py --to-model <tag>` (existing arm files;
add a `MODELS` entry), `make_llrbase_lang_envs.py` (fortran and skills siblings; add to `MODELS`).
Fix a value in the base and its generator together; the next run overwrites a generated file.

**3. Serve it alone.** From `experiments/`, `SUBMIT=0 MODEL=<tag> ./serve-only.sbatch` prints the
plan and `MODEL=<tag> ./serve-only.sbatch` runs the campaign's own `--vllm-node` role with the base
env plus `.env.serve-only`, sized from `INFERENCE_NODES`, and prints a working `curl`. For the
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

**5. Name it in the launchers.** `submit-cpf-llr40.sh` reads `.env.base-${model}`, so `MODELS=<tag>`
is enough. `submit-gpu-llr40.sh`, `submit-scicomp-dc.sh`, `submit-git-scicomp.sh` (`BASE_ENV`) and
`submit-llrblind.sh` (`MAX_TOKENS_BY_MODEL`, `.env.llrbase-<tag>-<lang>`) keep their own per-model
maps. Each runs `check_context_budget`, which refuses an arm before it is submitted.

**In-process and API models.** The Python harness ignores these env files. An OpenAI-shaped endpoint
is one `ModelSpec` entry in `MODELS` (`hpcagent_bench/harness/baselines.py`): `backend="openai"`,
`model`, `base_url`, `api_key_env`, `context_tokens`, and `accepts_sampling`/`max_tokens_field` for
an endpoint that rejects sampling or renames the reply cap (see `kimi`). `OpenAIAgent` falls back to
`OPENAI_BASE_URL`, `VLLM_BASE_URL`, then `localhost:8000/v1`. A new wire protocol is an `Agent`
subclass beside `ClaudeAgent` and `OllamaAgent` in `harness/agent.py`, added to `BACKENDS`
(baselines.py), `_agent_registry` (cli.py) and `ModelSpec.agent`; see [writing_an_agent.md](../writing_an_agent.md).

Checklist A:
- [ ] `fetch_weights.sbatch` prints `WEIGHTS READY`; `INFERENCE_CE_ENV` carries `INFERENCE_ENGINE`
- [ ] both parsers named; `OPTARENA_OPTIMIZER` = `VLLM_MODEL` = registry `serves:`
- [ ] `serve-only.sbatch` reaches `endpoint is live`; the smoke ends without `SMOKE FAILED`
- [ ] `pytest tests/test_display_names.py tests/test_palette.py tests/test_model_of.py --maxfail=10`

## B. A new engine

| What you touch | Why |
| --- | --- |
| `containers/cluster/ce-images/<engine>/` | `Dockerfile`, `build.sh`, `build.sbatch`, `edf.toml.example` |
| `containers/cluster/ce-images/images.env` | `INFERENCE_<ENGINE>_SQSH`, `_EDF_LATEST`, `_TEMPLATE`, `_REPO`, `_TAG` |
| `containers/cluster/ce-images/install_edfs.sh` | `try_render` for `INFERENCE_<ENGINE>_*` |
| `experiments/run_cluster.sh` `run_vllm_node` | interpreter and launch command |

**1. Image directory.** Copy `sglang/`. `build.sh` pins the base by digest and builds from the repo
root so the Dockerfile can COPY `inference/moe-configs/`; `build.sbatch` refuses to overwrite a
mounted `.sqsh`. `edf.toml.example` keeps the `PLACEHOLDER.sqsh` image line, a multi-line
`mounts = [` block (`derived_edf` exits 2 on a one-line block), absolute `PATH` and `LD_LIBRARY_PATH`
in `[env]` (the CE drops the image's own ENV) and the three fabric hook annotations.

**2. Build, verify, promote.** After the `images.env` and `install_edfs.sh` lines beside the sglang ones, run
`IMAGE_DIR=$PWD/<engine> sbatch build_and_verify.sbatch`, then `./promote_image.sh <engine>`. Beyond the table,
the engine name must appear in `build_and_verify.sbatch` (`BUILD_TARGETS_OF`, `TARGET_SQSH`), `verify_image.py`
and `verify_image.sbatch` (profile), the role lists of `promote_image.sh`, `pull_image.sh`, `pull_images.sbatch`,
`push_images.sbatch` and `push_candidates.sbatch`, and `experiments/smoke-new-images.sh`.

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
