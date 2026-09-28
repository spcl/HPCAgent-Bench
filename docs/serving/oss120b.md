# Serving gpt-oss-120b on MI300A

`openai/gpt-oss-120b` on **vLLM 0.28.0**, one node, `tp=4`: the only model here served by vLLM.
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

| Setting | Why |
|---|---|
| vLLM 0.28.0, the official ROCm image pinned by digest (`containers/images/vllm/Dockerfile`) | one vLLM release on every platform; 16 concurrent 4096-token requests run at about 1100 tok/s aggregate on one node |
| `VLLM_PLUGINS` allowlist, EDF only | unset, `quark_online_quant` closes an import cycle (`cannot import name 'SamplingParams'`) once aiter settings change the import graph; an env file would override the EDF silently |
| `--dtype bfloat16` | compute dtype; the checkpoint is pre-quantized mxfp4 |
| `--generation-config auto` | keeps the model's sampling defaults (`vllm` discards them silently) |
| all three tool flags | SGLang has no `--enable-auto-tool-choice`, so a line ported from SGLang fails at the first tool call |
| `--max-model-len 131072` | a longer request is an error, not a truncation |
| `--gpu-memory-utilization 0.70` | node-wide on the APU; do not raise it by discrete-GPU analogy |
| `--safetensors-load-strategy prefetch` | with `HF_HOME` on `iopsstor` |
| one server per node, no `pp` | extra nodes are replicas the client balances; `pp=4` loses about 42% of engine time to stalls |

The KV pool is in `grep -aE "KV cache|Available KV cache memory" server-0.log`; it has not been
measured against a working set, so size it above conversations x largest prompt and confirm a
prefix-cache hit rate of 0.98 or better. Re-measure after an image rebuild (the EDF names an
unversioned image), and run the server step with `--cpus-per-task` ([README](README.md#3-slurm-shape)).
