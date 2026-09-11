# Serving GLM-5.3 on MI300A

`zai-org/GLM-5.3`. Four nodes, sixteen GPUs.

> **This page ages fastest, and parts of it are provisional.** GLM-5.3 is being retuned as this is
> written (**state as of 2026-09-11**), and two of its configuration keys changed on the day this
> page was drafted. The **aiter position, the attention backend and the `--mem-fraction-static`
> pairing under aiter's 0.85 derate are all OPEN** and have their own section below; nothing in the
> DO or DO NOT lists covers them. Read `experiments/.env.base-glm53` before trusting anything here.
> That file is authoritative and this page is a snapshot of it.

Cross-model background is in [`knobs.md`](knobs.md).

---

## Best known configuration (2026-09-11)

| | |
|---|---|
| Engine | SGLang |
| EDF | **`sglang-candidate`** -- provisional, changed 2026-09-11 |
| Nodes | **4**, `tp=4` inside each node, `pp=4` across them |
| Port | 8000 on **rank 0 only**, served name `optarena-vllm` |
| Weights | fp8, about 755 GB total; roughly 15 minutes of weight load plus memory pool and graph capture |
| KV pool at `--mem-fraction-static 0.55` | 2,583,744 tokens, 28.56 GB per rank |

```
--tp-size 4 --pp-size 4 --nnodes 4 --node-rank <rank> --dist-init-addr <rank0>:29500
--host 0.0.0.0 --port 8000
--trust-remote-code
--watchdog-timeout 1800
--kv-cache-dtype fp8_e4m3
--page-size 64
--context-length 131072
--mem-fraction-static 0.50
--cuda-graph-max-bs-decode 64
--enable-metrics
--pre-warm-nccl
--reasoning-parser glm45
--tool-call-parser glm47
--dsa-prefill-backend tilelang
--dsa-decode-backend tilelang
--enable-cache-report
```

Environment: `SGLANG_USE_AITER=1`, `SGLANG_ROCM_FUSED_DECODE_MLA=0`, `SGLANG_SET_CPU_AFFINITY=0`,
`NCCL_NET_GDR_LEVEL=0`, `AITER_LOG_TUNED_CONFIG=1`.

```bash
cd experiments
MODEL=glm53 ./serve-only.sbatch
```

### Where this configuration sits against the KV pool threshold

[`knobs.md`](knobs.md) explains the threshold: above a pool-to-working-set ratio of about **1.15**
every configuration reaches a prefix-cache hit rate of 0.984-0.988; below it, every configuration
thrashes.

**This model's ratio has not been measured.** The pool is known -- 2,583,744 tokens at
`--mem-fraction-static 0.55` -- but no realistic working set has been put beside it. Measure:

```bash
grep -a "max_total_num_tokens\|KV Cache is allocated" server-0.log
```

Then estimate concurrent conversations times their largest prompt, and aim the pool at about 1.3x
it. On this model the pool responds steeply to the fraction: `KV(f) = 443.4 * (f - 0.486)` GB per
rank, so a small change in `f` is a large change in the ratio.

---

## DO

- **Serve this on SGLang.** There is no vLLM recipe for this model here. Its configuration was
  derived from the Kimi K2.7 one; what differs is the image and a set of deliberate deletions.
- **Use the EDF that `experiments/.env.base-glm53` names.** As of 2026-09-11 that is
  `sglang-candidate`. The model does not load at all without two things, and the image is how it
  gets them:
  - a guard keeping `torch.Tensor.format_ue8m0` false, or the ROCm `fnuz` branch of the DeepSeek
    weight loader returns a fresh tensor without the attribute and the loader dies;
  - `HIPCC_COMPILE_FLAGS_APPEND=-U__HIP_NO_HALF_CONVERSIONS__ -U__HIP_NO_HALF_OPERATORS__`, without
    which one of the fused kernels will not compile.

  `sglang-candidate` carries the guard **in the package at build time** and the HIPCC flags as a
  global image environment variable. The other SGLang EDFs reach the same guard through a
  `PYTHONPATH` under `/capstor`, which the campaign launcher's mount narrowing drops for the
  inference role, so the loader dies on `format_ue8m0` before the model is up.
- **Allocate four nodes.** The weights are about 755 GB. At `pp=2` each stage holds roughly 378 GB
  of the about 412 GB a node has free before load: a 0.917 floor that leaves no KV pool at all.
- **Size `--mem-fraction-static` against the heaviest pipeline stage.** The stages are **uneven**:
  172.4 / 197.2 / 203.8 / 206.1 GB (measured 2026-09-11). Read `avail mem=` on every rank, not just
  rank 0.
- **Pass both parsers**, `--reasoning-parser glm45` and `--tool-call-parser glm47`. The mismatched
  version numbers are correct.
- **Keep `--cuda-graph-max-bs-decode 64`.** Graph capture happens after the KV cache is sized and
  takes its memory from what is left.
- **Give the accuracy gate a real token budget.** This model reasons before it answers, so a gate
  asking over `/v1/chat/completions` with a small `max_tokens` scores a truncated deliberation as a
  wrong answer. 2048 covers it at maximum effort; below about 512 it answers nothing at all and the
  gate reports corruption that is really truncation.
- **Set `NCCL_NET_GDR_LEVEL=0` and confirm the fabric.** `grep -a "Using network" server-0.log`
  should say `AWS Libfabric`; RCCL's TCP fallback is correct and several times slower with no error.
- **Check what you actually got, every time.** Two of this model's settings are in flux:
  ```bash
  grep -aiE "attention.backend|Use dsa attention" server-0.log
  grep -a "max_total_num_tokens\|Mamba\|KV Cache is allocated" server-0.log
  ```

## DO NOT

- **Do not pass `--language-only`.** It selects the vision-encoder-disaggregation *receiver* role,
  and this architecture is off its allowlist: the server refuses to start. Two launches died on
  exactly this (2026-09-08). It is in the Kimi K2.7 and Qwen3.8 recipes, where those architectures
  accept it -- see [`kimi27sglang.md`](kimi27sglang.md). The flag is a per-model answer, not a
  general one.
- **Do not template that flag with `${VAR:-default}`.** `:-` substitutes on *empty* as well as
  unset, so a variable deliberately set to the empty string silently reverts to `--language-only`.
  Two launches died that way *after* the first fix. Use `${VAR-default}` or a `0/1` flag. See the
  README's mechanism section.
- **Do not set `--mem-fraction-static` to 0.62 or above.** The host OOM killer takes the heaviest
  pipeline stage (measured 2026-09-11). Below 0.486 SGLang refuses outright, because the weights
  alone exceed the budget.
- **Do not expect lowering the fraction to free host memory.** It is a **ceiling on weights and KV
  together**, not a KV reservation: `KV(f) = 443.4 * (f - 0.486)` GB per rank, so lowering it
  shrinks KV toward zero instead (measured 2026-09-11, within 1.5% at 0.55).
- **Do not carry `AITER_USE_FLYDSL_MOE_SORTING` over from Kimi K2.7.** Kimi's weights are
  pack-quantized int4; these are fp8, and the configuration that was proven did not set it.
- **Do not enable HiCache (`--enable-hierarchical-cache`, `--hicache-ratio`).** On an APU the host
  tier is the same physical memory: it allocates a second copy of the KV cache rather than
  offloading anything, and the server dies to the host OOM killer with no traceback (measured
  2026-09-03). Untested for this checkpoint, and the proven configuration does not set it.
- **Do not try to serve GLM-5.3-Flash here.** `index_kpool` in its configuration forces
  `IndexerKPool`, which raises "kpool indexer is only supported on CUDA". Plain 5.3 uses the
  ROCm-capable DSA Indexer.
- **Do not change `--kv-cache-dtype` on the strength of a short accuracy check.** An fp8 checkpoint
  ships no calibrated KV scales, so the engine quantizes at runtime against scale 1.0, and a corrupt
  attention path answers short prompts correctly. Gate any change on long context.
- **Do not edit `experiments/.env.base-glm53` or `experiments/make_glm53_envs.py`.** Another agent
  owns both files while the retune is in flight.

---

## OPEN: the aiter position and everything that pairs with it

**Not settled. Do not read the DO list as covering this, and do not copy a setting from this section
into a launch line without measuring.** The owner's direction is that this model should use **as
much aiter as it tolerates**, which points the opposite way from the current suppression logic
described below.

Where it stands as of 2026-09-11:

- **The launch line pins no `--attention-backend`.** The stated reason is that
  `GlmMoeDsaForCausalLM` selects the DSA backend from its own configuration -- the serving log says
  `Use dsa attention backend for DeepSeek with DSA` -- and pinning any value overrides the only
  backend its indexer supports.
- **But the shared launcher supplies one anyway.** `run_cluster.sh` reads
  `${SGLANG_ATTENTION_BACKEND-aiter}`, and **no file in the repository assigns that variable**. An
  absent key takes the default, so a server started through the launcher gets
  `--attention-backend aiter` appended. Only an **assigned, empty** `SGLANG_ATTENTION_BACKEND=`
  suppresses the flag. The README's mechanism section explains the distinction.
- **The configuration that was validated is therefore not the configuration the launcher produces.**
  The smoke that proved the working setup does not go through the launcher, and it served on `dsa`.
- **`SGLANG_USE_AITER=1` is set**, which switches aiter **ops** regardless of the attention backend.
  So "aiter is off for this model" was never true even under the suppression reading.
- **The 0.85 derate question rides on the answer.** If this model ends up on the aiter attention
  backend deliberately, `--mem-fraction-static 0.50` becomes an effective 0.425, which is below the
  0.486 floor measured for it -- so the fraction and the backend have to be re-derived together, not
  one at a time. **The current 0.50 was measured without that pairing in mind.**

**What to do meanwhile:** read the backend out of the log rather than inferring it from the
configuration, and treat every memory number on this page as conditional on which backend you got.
The question will be settled by a long-context accuracy gate plus a concurrency sweep, not by
copying a value from here.

---

## A live example of the generated-configuration trap

`experiments/make_glm53_envs.py` generates this model's per-run configurations from a base file. On
2026-09-11 the base said `INFERENCE_CE_ENV=sglang-glm-halfconv` while the generator said
`sglang-candidate`. Both were plausible, both were checked in, and which image a server actually got
depended on whether it was launched from the base or from a generated file.

That is the shape of the trap: **a generated configuration freezes a decision at generation time**,
and a fix belongs in the base *and* the generator together, never in one generated file. The two
have since been converged on `sglang-candidate`. `sglang-glm-halfconv` -- the same SquashFS as
`sglang-latest` plus those two keys in its `[env]` block -- is the previous route to the same thing
and is still registered. Prefer whatever the base file currently names.

---

## The data behind the instructions

### Memory law at `tp=4` x `pp=4` (measured 2026-09-11)

```
KV(f) = 443.4 * (f - 0.486) GB per rank
```

| `--mem-fraction-static` | Outcome |
|---|---|
| below 0.486 | refuses to start: the weights alone exceed the budget |
| **0.50 (shipped)** | serves |
| 0.55 | serves; within 1.5% of the law, pool 2,583,744 tokens at 28.56 GB per rank, about 137 GB free after the memory pool and before graph capture |
| 0.62 | host OOM killer takes the heaviest pipeline stage |

Pipeline stage sizes, uneven: 172.4 / 197.2 / 203.8 / 206.1 GB. Size against the maximum.

### Arguments and what each buys

| Argument | What it buys |
|---|---|
| `--kv-cache-dtype fp8_e4m3` | Halves KV cost per token. Not free; see the DON'T list. |
| `--page-size 64` | Part of the `gfx942` recipe. Measured null to slightly negative on Qwen3.8; a per-model answer. |
| `--context-length 131072` | Shorter than Kimi's window, because these weights leave less room. |
| `--mem-fraction-static 0.50` | See the memory law above, and the OPEN section. |
| `--cuda-graph-max-bs-decode 64` | Stops graph-capture residual eating the KV pool. |
| `--dsa-prefill-backend tilelang`, `--dsa-decode-backend tilelang` | The DSA indexer's kernels. |
| `--reasoning-parser glm45` + `--tool-call-parser glm47` | Both, or tool calls silently do not parse. |
| `AITER_LOG_TUNED_CONFIG=1` | Logs which tuned MoE configuration was picked. Diagnostic only. |

### Deliberate deletions from the Kimi K2.7 line

| Removed | Why |
|---|---|
| `--language-only` | Selects the vision-encoder receiver role; this architecture is off its allowlist and the server refuses to start (2026-09-08). |
| `--attention-backend` | See the OPEN section. The launcher supplies `aiter` anyway. |
| `--enable-hierarchical-cache` | Host tier is the same physical memory on an APU; untested for this checkpoint. |
| `AITER_USE_FLYDSL_MOE_SORTING` | Kimi's int4 weights need it; these fp8 weights were never measured with it. |

### Reasoning effort

This model maps every unrecognised effort value to `max` in its own chat template, so naming `max`
is a statement of intent rather than a switch. Leaving the field out entirely is what Kimi K2.7
wants and is not the same thing.
