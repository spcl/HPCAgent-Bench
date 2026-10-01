# Serving Qwen3.8 on MI300A

`Qwen/Qwen3.8-27B-FP8` on SGLang, one node, `tp=4`. The cheapest useful endpoint here. Source of
truth: `experiment:qwen38` in `experiments/arms.yaml` over `layers/model-qwen38.env`. Background:
[`knobs.md`](knobs.md).

```bash
hpcagent_bench/cluster/serve-only.sbatch     # qwen38 is the default MODEL
```

## Configuration

EDF `hpcagent-bench-sglang-mi300-latest`; env `SGLANG_USE_AITER=1`, `SGLANG_SET_CPU_AFFINITY=0`.
`run_cluster.sh` adds `--tp-size 4 --host 0.0.0.0 --port 8000 --served-model-name hpcagent-bench-vllm`.

```
--chat-template containers/inference/chat-template-qwen38.jinja --trust-remote-code
--attention-backend aiter --language-only --watchdog-timeout 1800
--context-length 262144 --mem-fraction-static 0.306 --mamba-full-memory-ratio 0.25
--max-running-requests 128 --enable-metrics
--reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-cache-report
```

| Mamba ratio | KV pool | Mamba slots | Captured decode batch | Used by |
|---|---|---|---|---|
| **0.25** | **4.03 M tokens** | 427 | up to 85 | `experiment:qwen38`, `serve-only.sbatch` |
| 0.5 | 3.32-3.36 M tokens | 704 | full 128 | `llrbase-c:qwen38`, private `mi300` preset |
| 0.9 (engine default) | 1.91 M tokens | | | never: hit rate 0.20, 25 tok/s at 40 x 60k |

Keep the pool above the working set (concurrent conversations x largest prompt): crossing it is worth
about 8x (40 x 60k on 3.32 M: hit 0.984, 254 tok/s; on 2.42 M: hit 0.35-0.45, 30-34 tok/s). Live
Mamba slot peak is 315-322, so 427 covers 40 streams. Read slots and pool from the log
(`grep -a "Mamba Cache is allocated\|max_total_num_tokens" server-0.log`), never infer them.

## Rules

- Serve on SGLang: about 19x vLLM's throughput for this hybrid backbone.
- Move `--mem-fraction-static` together with `--mamba-full-memory-ratio` (with ratio `r` the budget
  `R` splits mamba `R*r/(1+r)`, KV `R/(1+r)`) and with `--attention-backend aiter` (aiter derates by
  0.85, so 0.306 is an effective 0.26). Never above 0.306: the OOM killer takes the process without a
  traceback.
- Name `--attention-backend aiter` (ROCm default is triton; aiter is about +5%). Pass the chat template
  and both parsers.
- `--page-size 64`, `--chunked-prefill-size 16384` and HiCache do not help
  ([`knobs.md`](knobs.md#hicache-never)). `--language-only` selects SGLang's vision-encoder receiver
  role, not "text only".
- Measure with 40 long-lived streams at your p50 and p90 prompt sizes, back to back on one node; never
  trust a cold smoke ([`knobs.md`](knobs.md#the-kv-pool-threshold)).
