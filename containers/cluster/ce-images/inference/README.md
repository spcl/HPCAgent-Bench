# Serving-side smokes and probes

vLLM and SGLang are each built from a single Dockerfile -- `../vllm/Dockerfile` and
`../sglang/Dockerfile` -- and promoted under one unversioned name per role, which `../images.env`
records and `../install_edfs.sh` registers as `hpcagent-bench-vllm-mi300-latest` and `hpcagent-bench-sglang-mi300-latest`.
Build and promotion are documented in `../../../../experiments/SUBMITTING.md`.

vLLM 0.27.1 is parked on the `parked/vllm-0271` branch. On one pinned node, same probe and parsers,
it served oss120b at 2405 tok/s against 0.23.0's 3013 -- 25% slower, entirely in decode
(steady-state 2540 vs 3187; prefill matched to 0.3%). Same dtype, quantization, MoE and attention
backends, same torch and triton.

## Files

| File | What it does |
| --- | --- |
| `agentlike-probe.py` | Serving throughput under a campaign-shaped load. Node-to-node spread is ~30%, so pin an A/B to one node. |
| `accuracy-gate.py` | Correctness gate a serving change must pass before it is believed. |
| `smoke-kimi-sglang.sbatch` | SGLang serving smoke. |
| `serve-private.sbatch` | Private Qwen3.8-27B on one node, key via `--config`: `PRESET=mi300` (FP8, campaign flags) or `mi200` (BF16). `ACCESS=tunnel` binds 127.0.0.1 for an ssh tunnel; `ACCESS=alps` binds hsn0 for your own jobs on other Alps clusters. `MODE=smoke` runs one leg per tp/mem-fraction; `MODE=serve` holds one server. Guide: `docs/serving/private-endpoint.md`; extending: `docs/serving/extending-private-inference.md`. |
| `serve-daint.sbatch` | vLLM serving on Daint GH200 from the `vllm-cuda` image: `MODEL` = `qwen38`, `oss120b` or `kimi`, the beverin served name, window and parsers; kimi is PP across 4 nodes (`SERVE_NODES=2` allowed). `MODE=smoke` gates tools, reasoning, long-context accuracy and the multi-node NCCL transport; `MODE=serve` binds hsn0, requires a key and writes `endpoint.json` for `alps-endpoint.sh`. `../README.md`, Daint section. |
| `alps-endpoint.sh` | Sourced in your Daint job: checks an `ACCESS=alps` endpoint (key file, `/v1/models`, one chat) and exports `VLLM_BASE_URL`, `VLLM_API_KEY`, `VLLM_MODEL`. |
| `sglang_kernel_launch_check.py` | Launches sgl_kernel silu_and_mul and triton causal_conv1d against torch; verify_image.sbatch runs it for sglang-mi200. |
| `smoke-kimi-replicas.sbatch` | Multi-replica serving smoke. |
| `smoke-kimi-eager-pg.sbatch` | Serving smoke with the eager process-group patch loaded. |
| `prebuild-aiter-jit.sbatch` | Warms the aiter JIT cache. `@compile_ops` is lazy: importing an op never builds it, so this has to CALL each one. Gate on what was built, never on a module name -- the names differ across engine versions. |
| `tune-moe-int4-mi300a.sbatch`, `merge_moe_configs.py`, `moe-configs/` | MoE autotuning sweep and its promoted results. `../sglang/Dockerfile` COPYs `moe-configs/`, so it is build input, not scratch output. |
| `external-eager-pg-patch/` | `sitecustomize.py` loaded into the serving venv. Its effect is unfalsified: both arms of the last A/B reported zero unbatched P2P warnings with it loaded. |
