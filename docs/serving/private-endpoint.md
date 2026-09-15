# Private Qwen3.8 endpoint on beverin

An OpenAI-compatible Qwen3.8 server on one beverin compute node that only you can use.

| Client | `ACCESS` | Server binds | How the client reaches it |
|---|---|---|---|
| Your laptop | `tunnel` (default) | `127.0.0.1` | ssh tunnel through ela (section 4) |
| Your own job on Daint or another Alps cluster | `alps` | the node's `hsn0` address | `alps-endpoint.sh` with the key (section 5) |

There is no other path. Without a signed CSCS ssh key nothing outside Alps reaches the server
(section 8).

| File | Role |
|---|---|
| `containers/cluster/ce-images/inference/serve-private.sbatch` | launcher, runs on beverin |
| `containers/cluster/ce-images/inference/alps-endpoint.sh` | client check, sourced in your Daint job |
| `tests/test_serve_private.py`, `tests/test_alps_endpoint.py` | the properties below |

`experiments/serve-only.sbatch` is a different launcher: every interface, no key, callable by any Alps
user ([`README.md`](README.md)).

## 1. Design

- One job, one node, one SGLang server. `PRESET` fixes image, weights and flags.
- The key lives in a file you own, mode 600. It reaches SGLang only through `--config <yaml>`, never
  argv (`ps` shows argv to every user on the node), and is never printed.
- SGLang checks the key on every path except `/health*` and `/metrics*`, and only with one tokenizer
  worker (`sglang/srt/utils/auth.py`). Therefore:
  - `ACCESS=alps` drops `--enable-metrics`. `/health` and `/health_generate` stay open; they return a
    status code, never model output.
  - `EXTRA_ARGS` may not name `--host`, `--port`, `--api*`, `--admin*`, `--config`,
    `--tokenizer-worker*` or `--enable-metrics*`.
- `--host` is always last on the command line and is never the wildcard address.
- Before printing any connection line, every start checks `401` without the key and `200` with it.
- SGLang logs `server_args`, key included. The run dir `$SCRATCH/inference-server/<preset>-private/<jobid>`
  is mode 700 with scratch default ACLs stripped. On exit the launcher deletes `endpoint.json` first,
  then `sglang-auth.yaml`, `api.key` and `auth.header`. `server.log` stays, inside the 700 dir.

### Presets

| `PRESET` | GPUs per node | Image (EDF) | Weights | Attention | Default `LEGS` | Serve |
|---|---|---|---|---|---|---|
| `mi300` | 4x MI300A (APU) | `sglang-latest` | `Qwen/Qwen3.8-27B-FP8` | aiter | `tp4:0.306` | `tp4:0.306` |
| `mi200` | 8x MI250X (64 GiB) | `sglang-mi200-latest` | `Qwen/Qwen3.8-27B` (BF16) | triton | `tp4:0.80 tp4:0.88 tp8:0.80` | `tp8:0.80` |

- Both: `--context-length 262144 --max-running-requests 128 --mamba-full-memory-ratio 0.5
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder`, chat template
  `experiments/chat-template-qwen38.jinja`, served name `optarena-vllm`.
- `mi300` uses the qwen38 campaign flags (`SGLANG_EXTRA_ARGS` in `experiments/.env.llrbase-qwen38-c`;
  the test pins the match). 0.306 is node-wide on the APU and aiter derates it to about 0.26. Change it
  only together with `--attention-backend` and `--mamba-full-memory-ratio` ([`qwen38.md`](qwen38.md)).
- `mi200`: MI250X has no FP8 and no aiter kernels, hence BF16, triton, `--disable-custom-all-reduce`,
  `SGLANG_USE_AITER=0`. Fractions are per GPU, not node-wide: mi300 numbers do not carry over. Image
  build: `containers/cluster/ce-images/sglang-mi200/README.md`.

Measured, one node, every check passed (load probe: 16 concurrent requests x 256 tokens):

| Job | Leg | KV pool (tokens) | Max running requests | Load probe |
|---|---|---|---|---|
| 637107 | mi300 `tp4:0.306` | 3.36M | -- | 84 tok/s (only leg; warm-up not separated) |
| 637198 | mi200 `tp4:0.80` | 1.64M | 69 | cold Triton JIT, ignore |
| 637198 | mi200 `tp4:0.88` | 1.86M | 78 | 315 tok/s |
| 637198 | mi200 `tp8:0.80` | 1.91M | 128 | 322 tok/s |

Each leg writes its allocator lines to `<leg dir>/memory.txt`. A KV pool far from this table means a
different configuration.

## 2. Setup, once

On beverin:

```bash
umask 077
mkdir -p ~/.config/optarena /capstor/scratch/cscs/$USER/x86_64/ce-images/logs
openssl rand -hex 32 > ~/.config/optarena/mi200-endpoint.key     # mi300: mi300-endpoint.key
```

- `KEY_FILE` defaults to `~/.config/optarena/<preset>-endpoint.key`. The launcher refuses it unless it
  is mode 600, owned by you, at least 32 characters from `[A-Za-z0-9._~+/=-]`.
- Slurm writes the job output to that `logs` directory and does not create it.
- Weights: `$HF_HOME/hub` (default `HF_HOME=/iopsstor/scratch/cscs/$USER/hf`). The server runs offline;
  the launcher refuses missing weights.
- Rotating the key: overwrite the file, restart the job. A running server keeps the key it read.

On the laptop (`ACCESS=tunnel` only), `~/.ssh/config`:

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

- `cscs-key` is signed by the CSCS SSH key service: valid 1 day, at most 5 per day.
- `beverin.alps.cscs.ch` follows the CSCS `<cluster>.alps.cscs.ch` pattern but is not verified here.
  Check with `ssh beverin hostname`.

## 3. Start the server (beverin)

Always pass `--partition` and `--gpus-per-node`. The script names neither, beverin's default partition
is `mi200`, and a preset refuses the other partition.

```bash
cd <optarena checkout>
L=containers/cluster/ce-images/inference/serve-private.sbatch

# Smoke: every LEGS spec, checked, then stopped.
PRESET=mi200 sbatch --partition=mi200 --gpus-per-node=8 "$L"

# Serve for the laptop.
PRESET=mi200 MODE=serve LEGS=tp8:0.80 sbatch --partition=mi200 --gpus-per-node=8 --time=08:00:00 "$L"

# Serve for your Daint jobs.
PRESET=mi200 MODE=serve LEGS=tp8:0.80 ACCESS=alps sbatch --partition=mi200 --gpus-per-node=8 --time=08:00:00 "$L"

# mi300, campaign configuration.
PRESET=mi300 MODE=serve sbatch --partition=mi300 --gpus-per-node=4 --time=08:00:00 "$L"

# All checks and the server argv, no launch.
PRESET=mi200 MODE=serve DRY_RUN=1 bash "$L"
```

| Variable | Default | Meaning |
|---|---|---|
| `PRESET` | required | `mi300` or `mi200` |
| `MODE` | `smoke` | `smoke`: one server per `LEGS` spec, each checked (auth, tool-call and reasoning gate, load probe), then stopped. `serve`: first spec only, held until the job ends |
| `ACCESS` | `tunnel` | `tunnel` or `alps` |
| `LEGS` | per preset | `tp<N>:<mem-fraction-static>`, N in 1, 2, 4, 8; leg i uses port `API_PORT + i` |
| `API_PORT` | `30000` | first port |
| `KEY_FILE` | `~/.config/optarena/<preset>-endpoint.key` | key file |
| `SERVED_MODEL` | `optarena-vllm` | model name clients send |
| `EDF`, `MODEL_REPO`, `HF_HOME` | per preset | image, weights |
| `EXTRA_ARGS` | empty | extra SGLang flags, placed before `--host` |
| `READY_TIMEOUT` | `3600` | seconds to wait for `/health` |
| `--time` | `04:00:00` | walltime |

- Job output: `/capstor/scratch/cscs/$USER/x86_64/ce-images/logs/serve-private-<jobid>.out`.
- Node: `squeue -u "$USER" -n serve-private -o '%i %T %N'`.
- `MODE=serve` prints its connection block after `unauthenticated POST: 401` and
  `authenticated POST: 200`. Copy that block; it never contains the key.

## 4. Connect from your laptop (`ACCESS=tunnel`)

The block the job prints, for `mi200`:

```bash
umask 077; mkdir -p ~/.config/optarena
scp beverin:<KEY_FILE> ~/.config/optarena/                                   # once per key
printf 'Authorization: Bearer %s\n' "$(cat ~/.config/optarena/mi200-endpoint.key)" \
  > ~/.config/optarena/mi200-endpoint.header
ssh -N -J ela,beverin -L 127.0.0.1:30000:127.0.0.1:30000 <you>@<node>        # keep it running
curl -s http://127.0.0.1:30000/v1/models -H "@$HOME/.config/optarena/mi200-endpoint.header"
```

- The key moves by `scp` only. `printf` is a shell builtin and `-H @file` reads a file, so the key
  stays off argv on the laptop too.
- The laptop end binds `127.0.0.1`: nobody else on your network can use the tunnel.
- Local port taken: `-L 127.0.0.1:31000:127.0.0.1:30000`, then use port 31000.

Python:

```python
import pathlib

from openai import OpenAI

key = (pathlib.Path.home() / ".config/optarena/mi200-endpoint.key").read_text().strip()
client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key=key)
reply = client.chat.completions.create(
    model="optarena-vllm", max_tokens=128, messages=[{"role": "user", "content": "Say hi."}]
)
print(reply.choices[0].message.content)
```

## 5. Connect from your Daint job (`ACCESS=alps`)

1. On beverin, serve with `ACCESS=alps` (section 3). The job prints:

   ```text
   ===== alps endpoint is live: http://172.28.9.16:30000/v1 on nid002536, job 123456 =====
   endpoint.json: {"url": "http://172.28.9.16:30000/v1", "served_model": "optarena-vllm", "key_file": "...", ...}
   In your job on Daint (or another Alps cluster), while this job runs (docs/serving/private-endpoint.md):
     source <checkout>/containers/cluster/ce-images/inference/alps-endpoint.sh <run dir>/endpoint.json
   ```

2. In your Daint job script, before the agent starts:

   ```bash
   source <checkout>/containers/cluster/ce-images/inference/alps-endpoint.sh <run dir>/endpoint.json || exit 1
   ```

`alps-endpoint.sh`, in order:

1. Reads `url`, `served_model` and `key_file` from `endpoint.json`. The file names the key file, never
   the key.
2. Requires the key file to be owned by you and mode 600.
3. Requires `GET /v1/models` to list the served model.
4. Requires one `POST /v1/chat/completions` to return a choice.
5. Exports `VLLM_BASE_URL`, `VLLM_API_KEY` and `VLLM_MODEL`.

- Any failed step exports nothing and returns non-zero (2: file or key, 1: endpoint).
- The key reaches curl on stdin (`-H @-`), never argv.
- `/users` and `/capstor` are shared across Alps, so the beverin paths in `endpoint.json` are read as is.
- When the serving job ends, `endpoint.json` is deleted first and the Daint job's next request fails.

**Not verified yet.** A Daint compute node connecting to a beverin compute node on the port (both sit
on 172.28.0.0/16), and Daint reading the beverin-written paths. Run step 2 in a short Daint job before
a campaign.

## 6. Stop

- Laptop: Ctrl-C the tunnel. Beverin: `scancel <jobid>`; `--time` also ends the job.
- After a node crash, delete any remaining `endpoint.json`, `sglang-auth.yaml`, `api.key` and
  `auth.header` from the run dir.

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `serve-private: refusing: ...` in the job output | a check before launch failed; the message names it | fix it: key mode, partition, missing weights, `LEGS`, `EXTRA_ARGS` |
| `SERVE FAILED` | server died or the 401/200 check failed | `server.log` in the leg dir; [`README.md`](README.md) section 6 |
| laptop `401` | wrong or stale key, header file missing | re-`scp` the key, rebuild the header; a rotated key needs a job restart |
| laptop `curl: (7) ... Connection refused` | tunnel not running, or a different local port | start the tunnel; use the same port in the URL |
| tunnel `channel N: open failed: connect failed` | wrong node or port, or server not ready | node from `squeue`; wait for the `private endpoint is live` block |
| `Permission denied (publickey)` | `cscs-key` signature expired | sign the key again |
| `bind [127.0.0.1]:30000: Address already in use` | local port taken | `-L 127.0.0.1:31000:127.0.0.1:30000` |
| `alps-endpoint: cannot read '...'` | serving job ended, or wrong path | `squeue` on beverin; copy the `source` line from the job output |
| `alps-endpoint: key file ... mode 600` | key file mode or owner | `chmod 600` the key file on beverin |
| `alps-endpoint: GET .../models answered 000` | server not ready, job ended, or Daint cannot reach beverin | wait for the `alps endpoint is live` block; if it persists while the job runs, the network path is the problem (section 5) |
| `alps-endpoint: ... answered 401` | key file changed after the job started | restart the serving job |

## 8. Rules

Sources:

- CSCS SSH documentation, "SSH tunnel to a service on Alps compute nodes via Ela"
  (https://docs.cscs.ch/access/ssh/): server bound to localhost, local port forward through ela.
- CSCS User Regulations (https://www.cscs.ch/services/user-regulations): an account is for the applicant
  only; no access for any other person, project member or otherwise, explicitly or through negligence.
- ETH Zurich BOT (RSETHZ 203.21): Art. 9(2) credentials are personal; Art. 11(2) no circumventing
  security measures; Art. 14(2c) no port scans.

Allowed: the two paths above, for your own use and your own jobs.

Not allowed:

- giving anyone else the key, the header file, `endpoint.json`, the URL or your tunnel. Project members
  and students included: they run this launcher in their own job with their own key.
- a request folder that other people write to and your job serves.
- binding the wildcard address on the node; `ssh -g` or `-L 0.0.0.0:...` on the laptop.
- reverse tunnels or relays out of Alps: `ssh -R`, ngrok, cloudflared, frp.
- running the server on a login node.
- the key on a command line, in chat, email, a ticket, a repository or a shared directory.
- port scans to find a server; take the node from `squeue`.
