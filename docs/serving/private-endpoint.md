# Private Qwen3.8 endpoint on beverin

This page starts an OpenAI-compatible Qwen3.8 server on one beverin compute node and connects to it from
your laptop or from your own jobs on another Alps cluster. Only you can use the server.

| Client | `ACCESS` | Server binds | Connection |
|---|---|---|---|
| Your laptop | `tunnel` (default) | `127.0.0.1` | ssh tunnel through ela (section 4) |
| Your own job on Daint or another Alps cluster | `alps` | the node's `hsn0` address | `alps-endpoint.sh` and the key (section 5) |

These are the only supported connections. From outside Alps the ela tunnel is the only way in, and it
needs a CSCS-signed ssh key (section 8).

| File | Purpose |
|---|---|
| `containers/inference/serve-private.sbatch` | Slurm launcher, submitted on beverin |
| `containers/inference/alps-endpoint.sh` | client check, sourced in your Daint job |
| `tests/test_serve_private.py`, `tests/test_alps_endpoint.py` | tests for the properties in section 1 |
| [`extending-private-inference.md`](extending-private-inference.md) | contributor guide: contract, new presets, access paths, engines |

`experiments/serve-only.sbatch` serves on every interface without a key, so any Alps user can call it
([`README.md`](README.md)).

## 1. Design

- Each job starts one SGLang server on one node. `PRESET` selects the image, the weights and the flags.
- The API key is read from a file you own with mode 600. The launcher passes it to SGLang in a
  `--config` YAML file, so the key does not appear in the process argument list, which `ps` shows to
  every user on the node. The launcher does not print the key.
- SGLang (`sglang/srt/utils/auth.py` in the image) serves paths that start with `/health` or `/metrics`
  without the key, and it checks the key only when the server runs a single tokenizer worker. The
  launcher accounts for both:
  - With `ACCESS=alps` it omits `--enable-metrics`, so the server has no `/metrics` endpoint.
  - `/health` and `/health_generate` stay open. Each returns an empty response with status 200 (ready)
    or 503 (starting, unhealthy or shutting down). `/health_generate`, and `/health` when
    `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` is set, generates one token internally and discards it.
  - `EXTRA_ARGS` may not contain `--host`, `--port`, `--api*`, `--admin*`, `--config`,
    `--tokenizer-worker*` or `--enable-metrics*`.
- `--host` is the last argument on the server command line: `127.0.0.1`, or the node's `hsn0` address.
  The launcher has no wildcard bind.
- Once `/health` returns 200, the launcher sends one chat request without the key and one with it. It
  prints connection instructions only if the first returns 401 and the second 200.
- SGLang writes its arguments, key included, to `server.log`. The run directory
  `$SCRATCH/inference-server/<preset>-private/<jobid>` is therefore mode 700 with the scratch default
  ACLs removed. When the job exits, the launcher deletes `endpoint.json` first, then `sglang-auth.yaml`,
  `api.key` and `auth.header`. `server.log` remains inside the mode-700 directory.

### Presets

| `PRESET` | GPUs per node | Image (EDF) | Weights | Attention | Default `LEGS` | Leg to serve |
|---|---|---|---|---|---|---|
| `mi300` | 4x MI300A (APU) | `hpcagent-bench-sglang-mi300-latest` | `Qwen/Qwen3.8-27B-FP8` | aiter | `tp4:0.306` | `tp4:0.306` |
| `mi200` | 8x MI250X (64 GiB each) | `hpcagent-bench-sglang-mi200-latest` | `Qwen/Qwen3.8-27B` (BF16) | triton | `tp4:0.80 tp4:0.88 tp8:0.80` | `tp8:0.80` |

- Both presets pass `--context-length 262144 --max-running-requests 128 --mamba-full-memory-ratio 0.5
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder` with the chat template
  `experiments/chat-template-qwen38.jinja`, and serve the model as `hpcagent-bench-vllm`.
- `mi300` uses the qwen38 campaign flags from `SGLANG_EXTRA_ARGS` of `llrbase-c:qwen38` (`experiments/arms.yaml`);
  `tests/test_serve_private.py` fails if the two diverge. `--mem-fraction-static 0.306` is a fraction of
  the whole APU node's memory, and the aiter backend multiplies it by 0.85, which gives 0.26. Change it
  only together with `--attention-backend` and `--mamba-full-memory-ratio` ([`qwen38.md`](qwen38.md)).
- `mi200`: the MI250X supports neither FP8 nor the aiter kernels, so this preset serves BF16 weights with
  triton attention, `--disable-custom-all-reduce` and `SGLANG_USE_AITER=0`. Its memory fraction applies
  to each GPU's 64 GiB, so mi300 fractions do not transfer. The image is built as described in
  `containers/images/sglang-mi200/README.md`.

Measurements on one node. Every leg passed the 401/200 check, the tool-call and reasoning gate, and the
load probe, which sends 16 concurrent requests that each generate exactly 256 tokens (`ignore_eos`).

| Job | Preset, leg | KV pool (tokens) | Mamba slots | Max running requests | Load probe |
|---|---|---|---|---|---|
| 637107 | mi300 `tp4:0.306` | 3,362,173 | 714 | not recorded | 4,096 tokens in 48.9 s (84 tok/s); the job's only leg, so warm-up is included |
| 637198 | mi200 `tp4:0.80` | 1.64M | not recorded | 69 | 44.7 tok/s; Triton JIT compilation overlapped the leg, not comparable |
| 637198 | mi200 `tp4:0.88` | 1.86M | not recorded | 78 | 315 tok/s |
| 637198 | mi200 `tp8:0.80` | 1.91M | not recorded | 128 | 322 tok/s |

[`qwen38.md`](qwen38.md) reports 3,318,498 tokens and 704 Mamba slots for the same mi300 flags in the
campaigns. Each leg copies SGLang's KV and Mamba allocation lines to `<leg dir>/memory.txt`. If a leg
allocates a clearly different pool, compare the `argv:` line in the job output with section 3 first.

## 2. One-time setup

On beverin:

```bash
umask 077
mkdir -p ~/.config/hpcagent-bench
openssl rand -hex 32 > ~/.config/hpcagent-bench/mi200-endpoint.key     # for mi300: mi300-endpoint.key
```

- `KEY_FILE` defaults to `~/.config/hpcagent-bench/<preset>-endpoint.key`. The launcher rejects a key file
  that is not mode 600, is not owned by you, is shorter than 32 characters, or contains characters
  outside `[A-Za-z0-9._~+/=-]`.
- Slurm writes the job output to `serve-private-<jobid>.out` in the directory you `sbatch` from
  (`#SBATCH --output=%x-%j.out`; `#SBATCH` directives cannot expand `$SCRATCH`).
- The weights must already be in `$HF_HOME/hub` (default `HF_HOME=${FAST_SCRATCH}/.hpcagentbench-cache/hf`;
  see `scripts/cache_env.sh`). The
  server runs with `HF_HUB_OFFLINE=1`, and the launcher refuses to start if the model directory is
  missing.
- To rotate the key, overwrite the file and restart the job. A running server keeps the key it read at
  start.

On your laptop (`ACCESS=tunnel` only), add to `~/.ssh/config`:

```text
Host ela
    HostName ela.cscs.ch
    User <you>
    IdentityFile ~/.ssh/cscs-key
    IdentitiesOnly yes

Host beverin
    HostName beverin.alps.cscs.ch
    ProxyJump ela
    User <you>
    IdentityFile ~/.ssh/cscs-key
    IdentitiesOnly yes

Host nid*
    User <you>
    IdentityFile ~/.ssh/cscs-key
    IdentitiesOnly yes
```

- `~/.ssh/cscs-key` is a key signed by the CSCS SSH key service. A signature is valid for one day, and
  you can sign at most five keys per day.
- The `Host nid*` entry makes the last hop, to the compute node, use the same key.
- `beverin.alps.cscs.ch` follows the CSCS `<cluster>.alps.cscs.ch` naming but has not been tested from
  this repository. `ssh beverin hostname` should print the name of a beverin login node.

## 3. Start the server on beverin

Pass `--partition` and `--gpus-per-node` on every submission. The script sets neither, the default
partition on beverin is `mi200`, and each preset refuses to run on the other partition.

```bash
cd <hpcagent-bench checkout>
L=containers/inference/serve-private.sbatch

# Smoke test: every LEGS entry is started, checked and stopped.
PRESET=mi200 sbatch --partition=mi200 --gpus-per-node=8 "$L"

# Serve for your laptop.
PRESET=mi200 MODE=serve LEGS=tp8:0.80 sbatch --partition=mi200 --gpus-per-node=8 --time=08:00:00 "$L"

# Serve for your Daint jobs.
PRESET=mi200 MODE=serve LEGS=tp8:0.80 ACCESS=alps sbatch --partition=mi200 --gpus-per-node=8 --time=08:00:00 "$L"

# mi300 with the campaign configuration.
PRESET=mi300 MODE=serve sbatch --gpus-per-node=4 --time=08:00:00 "$L"

# Run every check and print the server command line without starting anything.
PRESET=mi200 MODE=serve DRY_RUN=1 bash "$L"
```

| Variable | Default | Meaning |
|---|---|---|
| `PRESET` | required | `mi300` or `mi200` |
| `MODE` | `smoke` | `smoke` starts each `LEGS` entry in turn, runs the checks (401/200, tool-call and reasoning gate, load probe) and stops it. `serve` starts the first entry and keeps it running until the job ends |
| `ACCESS` | `tunnel` | `tunnel` or `alps` |
| `LEGS` | per preset | space-separated `tp<N>:<mem-fraction-static>` entries, N = 1, 2, 4 or 8; entry i listens on `API_PORT + i` |
| `API_PORT` | `30000` | port of the first entry |
| `KEY_FILE` | `~/.config/hpcagent-bench/<preset>-endpoint.key` | API key file |
| `SERVED_MODEL` | `hpcagent-bench-vllm` | model name that clients send |
| `EDF`, `MODEL_REPO`, `HF_HOME` | per preset | image, weights repository, weights cache |
| `EXTRA_ARGS` | empty | additional SGLang flags, placed before `--host` |
| `READY_TIMEOUT` | `3600` | seconds to wait for `/health` to return 200 |
| `LOAD_CONCURRENCY` | `16` | concurrent requests in the smoke load probe |
| `--time` | `04:00:00` | Slurm walltime from the script header; override it on the `sbatch` line |

- Job output: `serve-private-<jobid>.out`, in the directory you `sbatch` from.
- Node name: `squeue -u "$USER" -n serve-private -o '%i %T %N'`.
- In `serve` mode the job prints its connection block after the lines `unauthenticated POST: 401` and
  `authenticated POST: 200`. Copy the block from your job output. It contains the key file's path, not
  the key.

## 4. Connect from your laptop (`ACCESS=tunnel`)

The job prints these commands with the node, port and key path filled in. For `mi200`:

```bash
umask 077; mkdir -p ~/.config/hpcagent-bench
scp beverin:<KEY_FILE> ~/.config/hpcagent-bench/                                   # once per key
printf 'Authorization: Bearer %s\n' "$(cat ~/.config/hpcagent-bench/mi200-endpoint.key)" \
  > ~/.config/hpcagent-bench/mi200-endpoint.header
ssh -N -J ela,beverin -L 127.0.0.1:30000:127.0.0.1:30000 <you>@<node>        # leave it running
curl -s http://127.0.0.1:30000/v1/models -H "@$HOME/.config/hpcagent-bench/mi200-endpoint.header"
```

- Copy the key with `scp` only. `printf` is a shell builtin and `curl -H @file` reads the header from a
  file, so the key does not appear in any process argument list on the laptop.
- The tunnel listens on the laptop's `127.0.0.1`, so other machines on your network cannot use it.
- If local port 30000 is taken, forward another one, for example `-L 127.0.0.1:31000:127.0.0.1:30000`,
  and use 31000 in the URL.

With the Python `openai` client:

```python
import pathlib

from openai import OpenAI

key = (pathlib.Path.home() / ".config/hpcagent-bench/mi200-endpoint.key").read_text().strip()
client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key=key)
reply = client.chat.completions.create(
    model="hpcagent-bench-vllm", max_tokens=128, messages=[{"role": "user", "content": "Say hi."}]
)
print(reply.choices[0].message.content)
```

## 5. Connect from your Daint job (`ACCESS=alps`)

1. On beverin, start the server with `ACCESS=alps` (section 3). After the 401/200 check the job prints:

   ```text
   ===== alps endpoint is live: http://172.28.9.16:30000/v1 on <node>, job 123456 =====
   endpoint.json: {"url": "http://172.28.9.16:30000/v1", "served_model": "hpcagent-bench-vllm", "key_file": "...", ...}
   In your job on Daint (or another Alps cluster), while this job runs (docs/serving/private-endpoint.md):
     source <checkout>/containers/inference/alps-endpoint.sh <run dir>/endpoint.json
   ```

2. In your Daint job script, run that `source` line before the agent starts:

   ```bash
   source <checkout>/containers/inference/alps-endpoint.sh <run dir>/endpoint.json || exit 1
   ```

`alps-endpoint.sh` runs these steps in order and stops at the first failure:

1. Read `url`, `served_model` and `key_file` from `endpoint.json`. The file holds the key file's path,
   not the key.
2. Check that the key file is owned by you and has mode 600.
3. Check that `GET /v1/models` returns 200 and lists the served model.
4. Check that one `POST /v1/chat/completions` returns 200 with at least one choice.
5. Export `VLLM_BASE_URL`, `VLLM_API_KEY` and `VLLM_MODEL`.

- On failure the script exports nothing and prints the HTTP status it received (`000` when no connection
  was made). It returns 2 for a problem with `endpoint.json` or the key file and 1 for a failed server
  check.
- The script passes the key to curl on standard input (`-H @-`), so the key does not appear in any
  process argument list on the Daint node.
- When the serving job ends, the launcher deletes `endpoint.json` before it stops the server. Requests
  from a running Daint job then fail to connect, and a later `source` fails with `cannot read`.

**Not yet tested from a Daint compute node:**

- a TCP connection to a beverin compute node's `hsn0` address on the server port. Beverin compute nodes
  and Daint login nodes are both on 172.28.0.0/16, and beverin's login node reaches Daint's login nodes
  on port 22; Daint compute nodes were not checked.
- reading the key file and `endpoint.json` at their beverin paths under `/users` and `$SCRATCH`.

Before a campaign, run step 2 once in a short Daint job. Exit status 0 confirms both.

## 6. Stop

- On the laptop, stop the tunnel with Ctrl-C. On beverin, run `scancel <jobid>`; the job also ends when
  its `--time` expires.
- If the node crashes, the cleanup does not run. Delete any remaining `endpoint.json`,
  `sglang-auth.yaml`, `api.key` and `auth.header` from the run directory by hand.

## 7. Troubleshooting

| Message or symptom | Cause | Fix |
|---|---|---|
| `serve-private: refusing: ...` in the job output | a check before launch failed; the message names it | correct that input: key file mode, partition, missing weights, `LEGS` or `EXTRA_ARGS` |
| `SERVE FAILED` | the server exited, or the 401/200 check failed | read `server.log` in the leg directory; common causes are in [`README.md`](README.md) section 6 |
| `401` on the laptop | wrong or outdated key, or missing header file | copy the key again and rebuild the header file; after a key rotation, restart the job |
| `curl: (7) ... Connection refused` on the laptop | the tunnel is not running, or it forwards a different local port | start the tunnel; use its local port in the URL |
| `channel N: open failed: connect failed` from ssh | wrong node or port, or the server is not ready | take the node from `squeue`; wait for the `private endpoint is live` block |
| `Permission denied (publickey)` | the `cscs-key` signature has expired | sign the key again |
| `bind [127.0.0.1]:30000: Address already in use` | local port 30000 is taken | forward another port: `-L 127.0.0.1:31000:127.0.0.1:30000` |
| `alps-endpoint: cannot read '...'` | the serving job has ended, or the path is wrong | check `squeue` on beverin; copy the `source` line from the job output |
| `alps-endpoint: ... is not an endpoint file` | the path points to a different file | use the `endpoint.json` path from the job output |
| `alps-endpoint: key file ... must ... be mode 600` | the key file's mode or owner is wrong | `chmod 600` the key file on beverin |
| `alps-endpoint: GET .../models answered 000` | the server is not ready, the job has ended, or the Daint node cannot reach beverin | wait for the `alps endpoint is live` block; if the job is running and the error persists, the Daint-to-beverin connection is failing (section 5) |
| `alps-endpoint: ... answered 401` | the key file changed after the job started | restart the serving job |

## 8. Rules

Sources:

- CSCS SSH documentation, section "SSH tunnel to a service on Alps compute nodes via Ela"
  (https://docs.cscs.ch/access/ssh/): bind the server to localhost and forward a local port through
  ela.
- CSCS User Regulations (https://www.cscs.ch/services/user-regulations): an account is for the applicant
  only, and the applicant may not give any other person, project member or otherwise, access to CSCS
  facilities, explicitly or through negligence.
- ETH Zurich BOT (RSETHZ 203.21): Art. 9(2) access credentials are personal; Art. 11(2) security
  measures may not be circumvented; Art. 14(2c) port scans are prohibited.

You may use the two connections above for yourself and your own jobs. The following are not allowed:

- giving anyone else the key, the header file, `endpoint.json`, the URL or your tunnel. This includes
  project members and students; they start their own job with their own key.
- a request folder that other people write to and that your job serves.
- binding the server to the wildcard address, or using `ssh -g` or `-L 0.0.0.0:...` on the laptop.
- reverse tunnels or relays out of Alps: `ssh -R`, ngrok, cloudflared, frp.
- running the server on a login node.
- putting the key on a command line, or in chat, email, a ticket, a repository or a shared directory.
- scanning ports to find a server; take the node name from `squeue`.
