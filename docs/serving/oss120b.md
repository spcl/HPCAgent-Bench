# Serving gpt-oss-120b on MI300A

`openai/gpt-oss-120b`. One node, four GPUs. The only model here served by **vLLM** rather than
SGLang.

Authoritative source: `experiments/.env.base-oss120b` plus the `vllm-latest` EDF, which owns
`VLLM_PLUGINS`. If this page and those disagree, they are right. Cross-model background is in
[`knobs.md`](knobs.md).

---

## Best known configuration (2026-09-11)

| | |
|---|---|
| Engine | **vLLM 0.23.0** |
| EDF | `vllm-latest` |
| Nodes | **1**, `tp=4`, no pipeline stage |
| Port | 8000, served name `optarena-vllm` |

```
--tensor-parallel-size 4 --host 0.0.0.0 --port 8000
--dtype bfloat16
--load-format safetensors --safetensors-load-strategy prefetch
--generation-config auto
--max-model-len 131072
--gpu-memory-utilization 0.70
--max-num-seqs 128
--enable-auto-tool-choice
--tool-call-parser openai
--reasoning-parser openai_gptoss
```

Environment, set by the EDF: `VLLM_PLUGINS=lora_filesystem_resolver,lora_hf_hub_resolver`.

```bash
cd experiments
MODEL=oss120b ./serve-only.sbatch
```

### Where this configuration sits against the KV pool threshold

[`knobs.md`](knobs.md) explains the threshold: above a pool-to-working-set ratio of about **1.15**
every configuration reaches a prefix-cache hit rate of 0.984-0.988; below it, every configuration
thrashes.

**This model's ratio has not been measured.** The threshold was established on Qwen3.8 under SGLang;
the mechanism is a property of prefix caching rather than of an engine, but neither the number nor
the crossing point has been confirmed on vLLM. Measure rather than assume. vLLM reports the pool
differently from SGLang:

```bash
grep -aE "KV cache|GPU KV cache size|Available KV cache memory" server-0.log
```

Estimate your working set as concurrent conversations times their largest prompt, and aim the pool
at about 1.3x it.

---

## DO

- **Use vLLM 0.23.0, not 0.27.1.** On one pinned node, the same probe and the same parsers:
  **3013 tok/s against 2405** -- 0.27.1 is about **25% slower**, entirely in decode (steady state
  3187 against 2540; prefill matched to 0.3%). Same dtype, quantization, MoE and attention backends,
  same torch and triton (measured 2026-09-08). The `vllm-latest` EDF points at the 0.23.0 image;
  0.27.1 was retired the same day and lives on a parked branch.
- **Set `VLLM_PLUGINS` to an allowlist.** Unset means **load everything**, and one auto-loaded
  plugin kills the server at startup -- see the DON'T list for the failure. The value in use is
  `lora_filesystem_resolver,lora_hf_hub_resolver`: naming what you want excludes the offender while
  keeping the LoRA resolvers (measured 2026-09-09).
- **Keep `VLLM_PLUGINS` in exactly one place, the EDF.** Two owners that can disagree is worse than
  one: a shell-exported value overrides the EDF and wins silently. One stale empty `VLLM_PLUGINS=`
  still exists in a derived configuration under `experiments/`; it is a leftover, not a second
  opinion.
- **Pass `--generation-config auto`.** It honours the model's own `generation_config.json`.
- **Pass all three tool and reasoning flags**: `--enable-auto-tool-choice`,
  `--tool-call-parser openai`, `--reasoning-parser openai_gptoss`. SGLang has no
  `--enable-auto-tool-choice`, so a launch line ported from an SGLang model arrives missing it and
  fails at the first tool call rather than at startup.
- **Keep `--safetensors-load-strategy prefetch` with `HF_HOME` on `iopsstor`.** Checkpoint loading
  is many concurrent large reads: 9.45 GB/s against 0.83 at 16 readers (measured 2026-08-26).
- **Run one server per node and let the client spread load.** Several nodes give independent
  replicas, each binding port 8000 on its own hostname with its own KV cache.
- **Re-measure after any image rebuild.** The EDF names an unversioned image, so a rebuild changes
  what you get.

## DO NOT

- **Do not leave `VLLM_PLUGINS` unset.** vLLM then auto-loads every registered general plugin, and
  `quark_online_quant` closes an import cycle inside the model-registry subprocess, where
  `vllm/__init__` is still executing:
  ```
  ImportError: cannot import name 'SamplingParams' from 'vllm' (unknown location)
  ```
  The server dies at startup. This checkpoint is pre-quantized mxfp4 and never needs online
  quantization (measured 2026-09-09).
- **Do not conclude from a bare smoke that the plugin allowlist is unnecessary.** A minimal server
  starts fine with plugins on; the cycle closes only once the surrounding environment (the aiter
  settings a full deployment carries) changes the import graph. "It worked when I tried it" is not
  evidence here.
- **Do not pass `--generation-config vllm`.** That value **discards** the model's own
  `generation_config.json` and silently changes sampling. It changed sampling in twenty
  configuration files once before anyone noticed, and nothing in the log announces it.
- **Do not pipeline this model across nodes.** It fits in one. `pp=4` cost about **42%** of engine
  time to stalls on this fleet while single-node servers lost none (measured 2026-08-30).
- **Do not pin a versioned image to freeze behaviour.** Pinning also freezes every later fix to that
  image. Prefer the unversioned name and re-measure after a rebuild.
- **Do not raise `--gpu-memory-utilization` by analogy with a discrete GPU.** On this part it is
  accounted against node-wide memory, the same as SGLang's `--mem-fraction-static`. 0.70 is a
  measured value, not a fraction of a card's VRAM.
- **Do not let the serving step run without `--cpus-per-task`.** A step that does not ask gets one
  core of 192, and the server degrades with load rather than failing. See [`README.md`](README.md).

## Open questions

- **Where this model's KV pool sits against the 1.15 threshold**, and whether the threshold holds at
  the same crossing point under vLLM's prefix cache as under SGLang's. Not measured.
- **Whether a newer vLLM has recovered the 25% decode loss.** The comparison is from 2026-09-08 and
  applies to 0.27.1 specifically. A later release is untested here.

---

## The data behind the instructions

### Engine version, one pinned node, same probe and parsers (2026-09-08)

| Version | Aggregate | Steady-state decode | Prefill |
|---|---|---|---|
| **0.23.0 (in use)** | **3013 tok/s** | **3187 tok/s** | matched |
| 0.27.1 | 2405 tok/s | 2540 tok/s | matched to 0.3% |

Same dtype, quantization, MoE and attention backends, same torch and triton. The entire difference
is in decode.

### Arguments and what each buys

| Argument | What it buys |
|---|---|
| `--dtype bfloat16` | The checkpoint is pre-quantized mxfp4; this is the compute dtype. |
| `--safetensors-load-strategy prefetch` | Overlaps checkpoint reads with load. |
| `--generation-config auto` | Honours the model's own sampling defaults. |
| `--max-model-len 131072` | The served window. A request over it is an error, not a truncation. |
| `--gpu-memory-utilization 0.70` | vLLM's analogue of `--mem-fraction-static`; node-wide in effect here. |
| `--max-num-seqs 128` | Scheduler-side concurrency cap. Reserves no memory. |
| `--enable-auto-tool-choice` + two parsers | All three, or tool calls silently do not parse. |
