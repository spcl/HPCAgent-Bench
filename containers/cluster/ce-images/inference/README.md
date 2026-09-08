# Serving-side smokes and probes

The images this directory used to build are gone. vLLM and SGLang are now each
built from a single Dockerfile -- `../vllm/Dockerfile` and `../sglang/Dockerfile`
-- and promoted under one unversioned name per role, which `../images.env`
records and `../install_edfs.sh` registers as `vllm-latest` and `sglang-latest`.
Build and promotion are documented in `../../../../SUBMITTING.md`.

What was here until 2026-09-08 was the earlier path: a multi-phase chain
(`build/`, plus a `beverin-rocm723-host-ofi-phase1/` base) that assembled an
image by layering sbatch jobs onto an upstream pull. It produced
`rocm723-vllm-0.23.0-pytorch211-ofi.sqsh` and
`sglang-rocm-v0.5.18-rocm720-mi30x.sqsh`, and it is why the sglang arms carried
`PYTHONPATH=${SCRATCH}/pyprefix` for cupy and flydsl: what the Dockerfiles built
had never been promoted into service, so the served image did not have them.
Those pulls also carried a squashfs sha256 and no `.digest`, so publishing one
published something the repo could not rebuild. Recover the chain from git
history if a phase of it is ever needed again; do not restore it wholesale.

vLLM 0.27.1 was retired the same day and lives on the `parked/vllm-0271`
branch. On one pinned node, same probe and parsers, it served oss120b at 2405
tok/s against 0.23.0's 3013 -- 25% slower, entirely in decode (steady-state 2540
vs 3187; prefill matched to 0.3%). Same dtype, quantization, MoE and attention
backends, same torch and triton.

## What remains here

| File | What it does |
| --- | --- |
| `agentlike-probe.py` | Serving throughput under a campaign-shaped load. Node-to-node spread is ~30%, so pin an A/B to one node. |
| `accuracy-gate.py` | Correctness gate a serving change must pass before it is believed. |
| `smoke-kimi-sglang.sbatch` | SGLang serving smoke. |
| `smoke-kimi-replicas.sbatch` | Multi-replica serving smoke. |
| `smoke-kimi-eager-pg.sbatch` | Serving smoke with the eager process-group patch loaded. |
| `prebuild-aiter-jit.sbatch` | Warms the aiter JIT cache. `@compile_ops` is lazy: importing an op never builds it, so this has to CALL each one. Gate on what was built, never on a module name -- the names differ across engine versions. |
| `tune-moe-int4-mi300a.sbatch`, `merge_moe_configs.py`, `moe-configs/` | MoE autotuning sweep and its promoted results. `../sglang/Dockerfile` COPYs `moe-configs/`, so it is build input, not scratch output. |
| `external-eager-pg-patch/` | `sitecustomize.py` loaded into the serving venv. Its effect is unfalsified: both arms of the last A/B reported zero unbatched P2P warnings with it loaded. |
