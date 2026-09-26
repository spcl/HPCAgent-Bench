# Serving gpt-oss-120b on MI300A

`openai/gpt-oss-120b` on **vLLM 0.23.0**, one node, `tp=4`: the only model here served by vLLM.
Source of truth: `experiments/layers/model-oss120b.env`, plus the `hpcagent-bench-vllm-mi300-latest`
EDF, which owns `VLLM_PLUGINS`. Background: [`knobs.md`](knobs.md).

```bash
cd experiments && MODEL=oss120b ./serve-only.sbatch
```

## Configuration

`run_cluster.sh` adds `--tensor-parallel-size 4 --host 0.0.0.0 --port 8000
--served-model-name hpcagent-bench-vllm`. `VLLM_EXTRA_ARGS` adds:

```
--dtype bfloat16 --load-format safetensors --safetensors-load-strategy prefetch
--generation-config auto
--enable-auto-tool-choice --tool-call-parser openai --reasoning-parser openai_gptoss
--max-model-len 131072 --gpu-memory-utilization 0.70 --max-num-seqs 128
```

EDF environment: `VLLM_PLUGINS=lora_filesystem_resolver,lora_hf_hub_resolver`.

| Argument | Role |
|---|---|
| `--dtype bfloat16` | compute dtype; the checkpoint is pre-quantized mxfp4 |
| `--generation-config auto` | keeps the model's own `generation_config.json` sampling |
| `--max-model-len 131072` | served window; a longer request is an error, not a truncation |
| `--gpu-memory-utilization 0.70` | node-wide on the APU, like SGLang's `--mem-fraction-static` |
| `--max-num-seqs 128` | scheduler concurrency cap; reserves no memory |

The KV pool is in `grep -aE "KV cache|Available KV cache memory" server-0.log`. This model's pool
has not been measured against a working set; size it above conversations x largest prompt and
confirm a per-request prefix-cache hit rate of 0.98 or better.

## DO

- **Use vLLM 0.23.0.** On one pinned node with the same probe and parsers, 0.23.0 serves 3013 tok/s
  against 2405 for 0.27.1 (about 25% slower, all in decode: 3187 against 2540 steady state; prefill
  within 0.3%). The image pins 0.23.0 (`containers/images/vllm/Dockerfile`).
- **Keep `VLLM_PLUGINS` an allowlist, set only in the EDF.** A value exported by an env file would
  override the EDF silently; no `experiments/` env sets it.
- **Pass all three tool flags**: `--enable-auto-tool-choice`, `--tool-call-parser openai`,
  `--reasoning-parser openai_gptoss`. SGLang has no `--enable-auto-tool-choice`, so a line ported
  from an SGLang model fails at the first tool call.
- **Keep `--safetensors-load-strategy prefetch` with `HF_HOME` on `iopsstor`.**
- **Run one server per node**; extra nodes are independent replicas the client balances.
- **Re-measure after an image rebuild.** The EDF names an unversioned image.

## DO NOT

- **Do not leave `VLLM_PLUGINS` unset.** vLLM then loads every plugin, and `quark_online_quant`
  closes an import cycle in the model-registry subprocess:
  ```
  ImportError: cannot import name 'SamplingParams' from 'vllm' (unknown location)
  ```
  A bare smoke may start fine; the cycle closes once the full deployment's aiter settings change the
  import graph.
- **Do not pass `--generation-config vllm`.** It discards the model's sampling defaults silently.
- **Do not pipeline across nodes.** It fits one node; `pp=4` costs about 42% of engine time to stalls.
- **Do not raise `--gpu-memory-utilization` by discrete-GPU analogy.** It is node-wide here.
- **Do not run the server step without `--cpus-per-task`** ([README](README.md#3-slurm-shape)).
