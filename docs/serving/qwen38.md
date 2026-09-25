# Serving Qwen3.8 on MI300A

`Qwen/Qwen3.8-27B-FP8` on SGLang, one node, `tp=4`. The cheapest useful endpoint here. Source of
truth: `experiments/.env.base-qwen38` (layer `layers/model-qwen38.env`). Background:
[`knobs.md`](knobs.md).

```bash
cd experiments && ./serve-only.sbatch     # qwen38 is the default MODEL
```

## Configuration

EDF `hpcagent-bench-sglang-mi300-latest`; env `SGLANG_USE_AITER=1`, `SGLANG_SET_CPU_AFFINITY=0`.
`run_cluster.sh` adds `--tp-size 4 --host 0.0.0.0 --port 8000 --served-model-name hpcagent-bench-vllm`.

```
--chat-template experiments/chat-template-qwen38.jinja --trust-remote-code
--attention-backend aiter --language-only --watchdog-timeout 1800
--context-length 262144 --mem-fraction-static 0.306 --mamba-full-memory-ratio 0.25
--max-running-requests 128 --enable-metrics
--reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-cache-report
```

| Mamba ratio | KV pool | Mamba slots | Captured decode batch | Used by |
|---|---|---|---|---|
| **0.25** | **4.03 M tokens** | 427 | up to 85 | `.env.base-qwen38`, `serve-only.sbatch` |
| 0.5 | 3.32-3.36 M tokens | 704 | full 128 | `.env.llrbase-qwen38-*`, private `mi300` preset |

The 0.25 ratio trades state slots for pool: at 40 concurrent streams the 3.36 M pool did not stay
above the prefix-cache threshold and 4.03 M does. Live mamba slot peak is 315-322, so 427 still
covers it. Choose 0.5 if you need decode batches captured up to 128 and your working set fits 3.3 M.

## Sizing: keep the pool above the working set

Working set = concurrent conversations x largest prompt. Crossing the threshold is worth about 8x;
moving inside either regime is worth nothing. At 40 conversations, with a 3.32 M pool:

| Prompt each | Working set | Pool / working set | Hit rate | Aggregate |
|---|---|---|---|---|
| 33k | 1.93 M | 1.72 | 0.974 | 319 tok/s |
| 60k | 3.18 M | 1.04 | 0.984 | 254 tok/s |

A 2.42 M pool at 40 x 60k: hit 0.35-0.45, 30-34 tok/s.

```bash
grep -a "Mamba Cache is allocated\|max_total_num_tokens" server-0.log
```

## DO

- **Serve on SGLang.** About 19x the throughput of vLLM for this hybrid backbone.
- **Read slot count and pool from the log**, never infer them from the ratio.
- **Move `--mem-fraction-static` and `--mamba-full-memory-ratio` together.** With ratio `r`, SGLang
  splits one budget `R`: mamba `R*r/(1+r)`, KV `R/(1+r)`.
- **Move `--mem-fraction-static` with `--attention-backend aiter`.** aiter derates by 0.85, so 0.306
  is an effective 0.26.
- **Name `--attention-backend aiter`.** SGLang's ROCm default is triton; aiter is about +5% at 40
  streams.
- **Pass the chat template.** `chat-template-qwen38.jinja` is the stock template plus the lines tool
  round-trips need. The env writes it as `${SCRIPT_DIR}/...`, expanded when sourced.
- **Pass both parsers**, `qwen3` and `qwen3_coder`.
- **Measure with 40 long-lived streams** at your p50 and p90 prompt sizes, back to back on one node,
  with the baseline run first and last as a drift control.

## DO NOT

- **Do not drop `--mamba-full-memory-ratio`.** The engine default 0.9 spends 47.4% of the budget on
  state that never fills; the pool falls to 1.91 M (hit 0.20, 25 tok/s at 40 x 60k).
- **Do not raise `--mem-fraction-static` past 0.306.** It leaves about 77 GB of 513 GB free; above
  that, the OOM killer takes the process without a traceback.
- **Do not pass `--page-size 64`** (null to slightly negative, smaller pool) or
  `--chunked-prefill-size 16384` (null). `--page-size 64` belongs to Kimi K2.7.
- **Do not enable HiCache** ([`knobs.md`](knobs.md#hicache-never)).
- **Do not read `--language-only` as "text only".** It selects SGLang's vision-encoder receiver
  role; this architecture accepts it, GLM-5.3 does not.
- **Do not trust a cold smoke** that reports a memory knob as null ([`knobs.md`](knobs.md#the-kv-pool-threshold)).
