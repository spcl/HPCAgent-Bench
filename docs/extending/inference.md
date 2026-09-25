# Adding a model or an inference engine

(A) serves a new model on SGLang or vLLM, which the cluster path already runs. (B) adds a third
engine. Measured serving values (memory fraction, pool size) live in the comments of the env files
and in `docs/serving/<tag>.md`; copy them from there.

## A. A new model

| File | Change |
|---|---|
| `experiments/layers/model-<tag>.env` | serving block (extends `layers/common.env`) |
| `experiments/.env.base-<tag>` | `# extends: layers/model-<tag>.env`, plus effort, context and engine args |
| `hpcagent_bench/envs/registry.yaml` `models:` | `<tag>: {name: <Display Name>, serves: org/Name}`, appended at the end |
| `docs/serving/<tag>.md` | the measurements behind the recipe |

`<tag>` is the model token in arm names (`llr-focus40-<tag>-c`). Env layering is described in
`experiments/README.md` ("Env layers").

**1. Fetch weights** into `${HF_HOME}` (see `scripts/cache_env.sh`); `AUDIT_ONLY=1` only checks the
layout. Success prints `WEIGHTS READY`.

```bash
MODELS="org/Name" sbatch containers/inference/fetch_weights.sbatch
```

**2. Write the env files.** Copy the pair with the same engine and node shape (`qwen38`, `oss120b`:
one node; `kimi27sglang`, `glm53`: four nodes in `pp` mode). From `layers/model-qwen38.env` and
`.env.base-qwen38`, trimmed:

```bash
# layers/model-qwen38.env
INFERENCE_NODES=1
INFERENCE_MODE=replicas
INFERENCE_CE_ENV=hpcagent-bench-sglang-mi300-latest
INFERENCE_ENGINE=sglang
VLLM_MODEL=Qwen/Qwen3.8-27B-FP8
VLLM_SERVED_MODEL=hpcagent-bench-vllm
HPCAGENT_BENCH_OPTIMIZER=Qwen/Qwen3.8-27B-FP8
# .env.base-qwen38
# extends: layers/model-qwen38.env
EFFORT_LADDER="low medium xhigh"
CONTEXT_LENGTH=262144
SGLANG_EXTRA_ARGS="--chat-template ${SCRIPT_DIR}/chat-template-qwen38.jinja --context-length 262144 --mem-fraction-static 0.306 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-metrics"
```

| Key | Meaning |
|---|---|
| `INFERENCE_ENGINE` | `sglang` runs `sglang.launch_server`; unset or anything else runs `vllm serve` |
| `INFERENCE_CE_ENV` | EDF name in `~/.edf`; its image must carry that engine |
| `VLLM_MODEL`, `VLLM_SERVED_MODEL` | HF repo id (both engines), and the name agents request |
| `GPUS_PER_NODE`, `INFERENCE_NODES`, `INFERENCE_MODE` | TP size; `pp` splits one model over the nodes, `replicas` runs one server per node |
| `SGLANG_EXTRA_ARGS`, `VLLM_EXTRA_ARGS` | split on whitespace (`read -r -a`), no quoting; name both parsers |
| `SGLANG_ATTENTION_BACKEND` | unset appends `--attention-backend aiter`; set empty omits it |
| `HPCAGENT_BENCH_OPTIMIZER` | checkpoint id; must equal `VLLM_MODEL` and the registry `serves:` |
| `EFFORT_LADDER` | rungs this server accepts, lowest first (`experiments/effort.py`); empty for no ladder |
| `CONTEXT_LENGTH` | served window; harnesses derive compaction from it ([token_accounting.md](../token_accounting.md#context-compaction)) |

Files such as a chat template sit in `experiments/` and are named through `${SCRIPT_DIR}`, which
`run_cluster.sh` mounts into the inference container.

**3. Serve it alone.** From `experiments/`:

```bash
SUBMIT=0 MODEL=<tag> ./serve-only.sbatch   # print the plan
MODEL=<tag> ./serve-only.sbatch            # serve; the log reaches "endpoint is live" and prints a curl
```

For tool-call, reasoning and long-context accuracy gates, run the smokes in
`containers/inference/` from that directory: `smoke-kimi-sglang.sbatch` takes
`MODEL_REPO`, `SERVED_MODEL`, `TOOL_PARSER`, `REASONING_PARSER`, `MEM_FRACTION`, `CONTEXT_LEN`;
`smoke-kimi-eager-pg.sbatch` (vLLM) takes `MODEL_REPO`, `TOOL_PARSER`, `REASONING_PARSER`,
`EXTRA_SERVE_ARGS`. A failure prints `SMOKE FAILED`.

**4. Name it in launchers.** `submit-cpf-llr40.sh` and `submit-gpu-llr40.sh` read
`.env.base-${model}`, so `MODELS=<tag>` suffices. `submit-scicomp-dc.sh`, `submit-git-scicomp.sh`
(`BASE_ENV`) and `submit-llrblind.sh` (`MAX_TOKENS_BY_MODEL`) keep per-model maps. `make_model_arm.py --to-model <tag>`
re-targets a rendered arm file (needs a `MODELS` entry).

**In-process models.** The Python harness ignores env files. An OpenAI-shaped endpoint is one
`ModelSpec` in `MODELS` (`hpcagent_bench/harness/baselines.py`): `backend="openai"`, `model`,
`base_url`, `api_key_env`, `context_tokens`, plus `accepts_sampling`/`max_tokens_field` for endpoints
that reject sampling or rename the reply cap. A new wire protocol is an `Agent` subclass in
`harness/agent.py` added to `BACKENDS`; see [writing_an_agent.md](../writing_an_agent.md).

```bash
python -m pytest --maxfail=10 tests/test_display_names.py tests/test_palette.py tests/test_model_of.py
```

## B. A new engine

| File | Change |
|---|---|
| `containers/images/<engine>/` | `Dockerfile`, `build.sh`, `build.sbatch`, `edf.toml.example` (copy `sglang/`) |
| `containers/images/images.env` | `INFERENCE_<ENGINE>_SQSH`, `_EDF_LATEST`, `_TEMPLATE`, `_REPO`, `_TAG` |
| `containers/images/install_edfs.sh` | render the new EDF beside the sglang one |
| `experiments/run_cluster.sh` `run_vllm_node` | interpreter (`engine_python`) and a `command=(...)` branch |

`edf.toml.example` keeps the `PLACEHOLDER.sqsh` image line, a multi-line `mounts = [` block, absolute
`PATH` and `LD_LIBRARY_PATH` under `[env]` (the CE drops the image's ENV) and the fabric hook
annotations. The engine name also goes in the profiles of `verify_image.py` and the role lists
of `promote_image.sh`, `pull_image.sh` and `experiments/smoke-new-images.sh` (`SMOKE`).

The `run_vllm_node` branch serves `${model_path}` as `${VLLM_SERVED_MODEL}` on
`0.0.0.0:${VLLM_PORT}` with TP `GPUS_PER_NODE`. Under `pp` it takes size, rank and rendezvous from
`INFERENCE_NODES`, `SLURM_PROCID` and `VLLM_MASTER_HOST:VLLM_MASTER_PORT`; only rank 0 binds the port.
The endpoint must answer `GET /v1/models`, `POST /v1/chat/completions` and, in the default
`AGENT_LLM_MODE=direct`, Anthropic `POST /v1/messages` (`AGENT_LLM_MODE=litellm` fronts it with a proxy).

```bash
REPO=$PWD IMAGE_DIR=containers/images/<engine> sbatch containers/images/build_and_verify.sbatch
containers/images/promote_image.sh <engine>   # after the verify job passes
experiments/smoke-new-images.sh <engine>
python -m pytest --maxfail=10 tests/test_vllm_pp_serve_args.py tests/test_derived_edf.py
```
