# Inference on one partition, agents and judge on another

A setup whose model one partition cannot serve (Kimi K2.7: two pp=4 pipelines on 8 MI300A nodes) runs its
agents and judge on another (mi200) against a keyed endpoint a serving job on mi300 publishes. Two jobs, no
heterogeneous allocation:

| Job | Partition | What it runs | Writes / reads |
|---|---|---|---|
| `serve-only.sbatch ACCESS=alps` | mi300 | the experiment's own serving (`run_cluster.sh --vllm-node`), keyed | writes `$SCRATCH/inference-server/<jobid>/endpoint.json` |
| the setup (`submit.sh`, `INFERENCE_SOURCE=endpoint`) | mi200 | agents and judge, no inference node | reads that `endpoint.json` |

Why not a heterogeneous Slurm job: every launcher here (`services.sbatch`, `run_cluster.sh`, `setup_nodes.sh`, the
job options) assumes one partition, one `--gpus-per-node` and one node list; a het job would touch each of them and
would need both partitions free at once. The two-job form reuses the experiment's serving unchanged and the
`endpoint.json` contract `alps-endpoint.sh` already reads.

## 1. Security

The serving job keeps every property of [`private-endpoint.md`](private-endpoint.md), section 1:

- The key comes from `KEY_FILE` (default `~/.config/hpcagent-bench/<MODEL>-endpoint.key`), refused unless mode 600,
  yours, at least 32 characters. It reaches SGLang only through `--config <run dir>/sglang-auth.yaml`, never argv.
- SGLang only (vLLM serves `/metrics` and its non-`/v1` routes without the key). The experiment's `--enable-metrics`
  is dropped; `--host`, `--port`, `--api*`, `--admin*`, `--config`, `--tokenizer-worker*` in its server arguments are
  refused. `PREWARM=1` is refused.
- Each node binds the address its own name resolves to (on Beverin its `hsn0` address, the one clients dial), never
  the wildcard; the server refuses to start when the name resolves to a loopback or no address.
- The run directory is mode 700 with scratch's default ACLs removed (the server log prints the key).
- `endpoint.json` (mode 600) is written only after a request without the key answered 401 and one with it 200 on
  every pipeline. On exit the job deletes `endpoint.json` first, then stops the server and deletes the secrets.

The setup side refuses an `endpoint.json` or key file that is not yours and mode 600, a URL that is not plain
`http://<host>:<port>/v1` on a named host, `INFERENCE_NODES` other than 0, and an endpoint that serves another
checkpoint than the setup's `VLLM_MODEL`. The key is read in the launcher's shell from the key file the endpoint
names and reaches the agents and the judge as `VLLM_API_KEY`, as a hosted service's key does. Sharing rules:
[`private-endpoint.md`](private-endpoint.md), section 8.

## 2. Run it

Once, on Beverin:

```bash
umask 077; mkdir -p ~/.config/hpcagent-bench
openssl rand -hex 32 > ~/.config/hpcagent-bench/kimi27sglang-endpoint.key
```

The serving job, on mi300, with the setup's own serving keys (`mlgpu40:kimi27sglang`: 8 nodes, 2 pipelines) and a
time limit that covers the mi200 job's:

```bash
S=hpcagent_bench/cluster/serve-only.sbatch
ACCESS=alps MODEL=kimi27sglang SERVE_BASE_ENV=mlgpu40:kimi27sglang SERVE_TIME=23:00:00 SUBMIT=0 $S   # dry run
ACCESS=alps MODEL=kimi27sglang SERVE_BASE_ENV=mlgpu40:kimi27sglang SERVE_TIME=23:00:00 $S
```

It submits to the site's system (`beverin`: partition `mi300`), as `serve-only.sbatch` always does.

Once it prints `===== keyed endpoint is live =====` (or as soon as it is RUNNING: the setup waits up to its
`AGENT_READY_TIMEOUT_SECONDS` for the file), the setup on mi200:

```bash
E=$SCRATCH/inference-server/<serving jobid>/endpoint.json
cd experiments
BASE=mlgpu40 TAG=ml40 MODELS=kimi27sglang LANGUAGES=hip RECORD_STUDY=cpf-ml40-mi200 \
    EXTRA_ENV_KV="INFERENCE_SOURCE=endpoint INFERENCE_NODES=0 INFERENCE_ENDPOINT_FILE=$E" \
    SUBMIT=1 ../hpcagent_bench/cluster/submit.sh --system beverin-mi200 --account <project>
```

`INFERENCE_SOURCE=endpoint` takes no serving layer, so `[hardware.mi200.models.kimi27sglang]` is not needed. The row
records the judge's hardware (`HPCAGENT_BENCH_HARDWARE=mi200`, study `*-mi200`), and `inference.json` records
`source: endpoint` with the serving job's engine, EDF, checkpoint, job id, nodes and pipeline URLs, read from
`endpoint.json`. The agents are striped over the pipelines as in one job (`VLLM_REPLICA_URLS`). The driver's aggregate
throughput probe reads `/metrics`, which a keyed endpoint does not serve, so such a run has no aggregate series.

An mi200 node reaching an mi300 node's server port (both on Beverin's 172.28.0.0/16) is not yet exercised by a full
run: the first launch of a study should be a short one (`--time 02:00:00`, a few kernels) whose agents log turns.

## 3. Stop

`scancel` the setup first, then the serving job. A serving job that ends first leaves the agents without a model;
give it the longer time limit.
