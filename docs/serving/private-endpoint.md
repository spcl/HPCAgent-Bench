# Private Qwen3.8 endpoint on Beverin

`containers/cluster/ce-images/inference/serve-private.sbatch` starts a keyed, OpenAI-compatible
Qwen3.8 SGLang server on one Beverin node that only you can use. `experiments/serve-only.sbatch`, by
contrast, serves every interface without a key ([`README.md`](README.md)). Contributors:
[`extending-private-inference.md`](extending-private-inference.md).

| Client | `ACCESS` | Server binds | Connection |
|---|---|---|---|
| your laptop | `tunnel` (default) | `127.0.0.1` | ssh tunnel through ela (section 4) |
| your own job on Daint or another Alps cluster | `alps` | the node's `hsn0` IPv4 address | `alps-endpoint.sh` + key (section 5) |

## 1. Security design

- `PRESET` selects image, weights and flags; one server per job, one node.
- The key comes from a file you own, mode 600. It reaches SGLang only through a `--config` YAML, so it
  never appears in the process argument list. The launcher never prints it.
- SGLang serves `/health*` and `/metrics*` without the key, and checks the key only with a single
  tokenizer worker. So `ACCESS=alps` drops `--enable-metrics`, and `EXTRA_ARGS` may not contain
  `--host`, `--port`, `--api*`, `--admin*`, `--config`, `--tokenizer-worker*` or `--enable-metrics*`.
  `/health` and `/health_generate` stay open (empty body, 200 or 503).
- `--host` is the last server argument; there is no wildcard bind.
- Connection instructions print only after a request without the key returns 401 and one with it
  returns 200.
- SGLang logs its arguments, key included, to `server.log`. The run directory
  `$SCRATCH/inference-server/<preset>-private/<jobid>` is mode 700 with scratch default ACLs removed.
  On exit the launcher deletes `endpoint.json` first, then `sglang-auth.yaml`, `api.key`,
  `auth.header`.

### Presets

| `PRESET` | Hardware | EDF | Weights | Attention | Default `LEGS` | Serve with |
|---|---|---|---|---|---|---|
| `mi300` | 4x MI300A | `hpcagent-bench-sglang-mi300-latest` | `Qwen/Qwen3.8-27B-FP8` | aiter | `tp4:0.306` | `tp4:0.306` |
| `mi200` | 8x MI250X, 64 GiB each | `hpcagent-bench-sglang-mi200-latest` | `Qwen/Qwen3.8-27B` (BF16) | triton | `tp4:0.80 tp4:0.88 tp8:0.80` | `tp8:0.80` |

Both pass `--context-length 262144 --max-running-requests 128 --mamba-full-memory-ratio 0.5
--reasoning-parser qwen3 --tool-call-parser qwen3_coder`, the chat template
`experiments/chat-template-qwen38.jinja`, and serve as `hpcagent-bench-vllm`.

- `mi300` matches `SGLANG_EXTRA_ARGS` in `experiments/.env.llrbase-qwen38-c`
  (`tests/test_serve_private.py` fails if they diverge). 0.306 is node-wide on the APU and derated
  to 0.26 by aiter; move it only with the backend and mamba ratio ([`qwen38.md`](qwen38.md)).
- `mi200`: MI250X has neither FP8 nor aiter kernels, so BF16, triton attention,
  `--disable-custom-all-reduce`, `SGLANG_USE_AITER=0`. Its fraction is per 64 GiB GPU; mi300
  values do not transfer. Image: `containers/cluster/ce-images/sglang-mi200/README.md`.

Measured per leg (401/200 check, tool and reasoning gate, 16 concurrent 256-token requests):

| Preset, leg | KV pool | Max running | Load probe |
|---|---|---|---|
| mi300 `tp4:0.306` | 3.36 M tokens, 714 Mamba slots | not recorded | 84 tok/s, cold (first leg, warm-up included) |
| mi200 `tp4:0.88` | 1.86 M | 78 | 315 tok/s |
| mi200 `tp8:0.80` | 1.91 M | 128 | 322 tok/s |

Each leg writes SGLang's KV and Mamba allocation lines to `<leg dir>/memory.txt`. A clearly different
pool: compare the job's `argv:` line with section 3.

## 2. One-time setup

On Beverin:

```bash
umask 077
mkdir -p ~/.config/hpcagent-bench
openssl rand -hex 32 > ~/.config/hpcagent-bench/mi300-endpoint.key     # or mi200-endpoint.key
```

- `KEY_FILE` defaults to `~/.config/hpcagent-bench/<preset>-endpoint.key`. Refused unless mode 600,
  owned by you, at least 32 characters, and only `[A-Za-z0-9._~+/=-]`.
- Weights must already be in `$HF_HOME/hub` (default from `scripts/cache_env.sh`); the server runs
  with `HF_HUB_OFFLINE=1`. Fetch with `containers/cluster/ce-images/inference/fetch_weights.sbatch`.
- To rotate the key, overwrite the file and restart the job.

On your laptop (`ACCESS=tunnel`), `~/.ssh/config`:

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

`~/.ssh/cscs-key` is signed by the CSCS SSH key service (valid one day, at most five signatures per
day). `ssh beverin hostname` should print a Beverin login node.

## 3. Start the server

The script sets neither partition nor GPU count, and each preset refuses the other partition, so pass
both on every submission:

```bash
cd "$REPO"
L=containers/cluster/ce-images/inference/serve-private.sbatch

PRESET=mi300 MODE=serve sbatch --partition=mi300 --gpus-per-node=4 --time=08:00:00 "$L"               # laptop
PRESET=mi300 MODE=serve ACCESS=alps sbatch --partition=mi300 --gpus-per-node=4 --time=08:00:00 "$L"   # Daint jobs
PRESET=mi200 MODE=serve LEGS=tp8:0.80 sbatch --partition=mi200 --gpus-per-node=8 --time=08:00:00 "$L"
PRESET=mi200 sbatch --partition=mi200 --gpus-per-node=8 "$L"                  # smoke every LEGS entry
PRESET=mi300 MODE=serve DRY_RUN=1 bash "$L"                                   # checks + argv, starts nothing
```

| Variable | Default | Meaning |
|---|---|---|
| `PRESET` | required | `mi300` or `mi200` |
| `MODE` | `smoke` | `smoke`: start, check (401/200, tool gate, load probe) and stop each leg. `serve`: first leg only, held until the job ends |
| `ACCESS` | `tunnel` | `tunnel` or `alps` |
| `LEGS` | per preset | `tp<N>:<mem-fraction-static>` entries, N in 1, 2, 4, 8; entry i listens on `API_PORT + i` |
| `API_PORT` | `30000` | first entry's port |
| `KEY_FILE` | `~/.config/hpcagent-bench/<preset>-endpoint.key` | key file |
| `SERVED_MODEL` | `hpcagent-bench-vllm` | name clients send |
| `EDF`, `MODEL_REPO`, `HF_HOME` | per preset | image, weights repo, weights cache |
| `EXTRA_ARGS` | empty | extra SGLang flags, placed before `--host` |
| `READY_TIMEOUT` | `3600` | seconds to wait for `/health` 200 |
| `LOAD_CONCURRENCY` | `16` | concurrent requests in the smoke load probe |
| `--time` | `04:00:00` | Slurm walltime |

Job output: `serve-private-<jobid>.out` in the submit directory. Node:
`squeue -u "$USER" -n serve-private -o '%i %T %N'`. In `serve` mode the connection block follows
`unauthenticated POST: 401` and `authenticated POST: 200`; it names the key file, never the key.

## 4. Connect from your laptop (`ACCESS=tunnel`)

The job prints these lines with node, port and key path filled in:

```bash
umask 077; mkdir -p ~/.config/hpcagent-bench
scp beverin:<KEY_FILE> ~/.config/hpcagent-bench/                                  # once per key
printf 'Authorization: Bearer %s\n' "$(cat ~/.config/hpcagent-bench/mi300-endpoint.key)" \
  > ~/.config/hpcagent-bench/mi300-endpoint.header
ssh -N -J ela,beverin -L 127.0.0.1:30000:127.0.0.1:30000 <you>@<node>             # leave running
curl -s http://127.0.0.1:30000/v1/models -H "@$HOME/.config/hpcagent-bench/mi300-endpoint.header"
```

`printf` is a builtin and `curl -H @file` reads from a file, so the key never lands in an argv. The
tunnel listens on the laptop's loopback only. If local port 30000 is taken, forward another
(`-L 127.0.0.1:31000:127.0.0.1:30000`) and use it in the URL.

```python
import pathlib

from openai import OpenAI

key = (pathlib.Path.home() / ".config/hpcagent-bench/mi300-endpoint.key").read_text().strip()
client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key=key)
reply = client.chat.completions.create(
    model="hpcagent-bench-vllm", max_tokens=128, messages=[{"role": "user", "content": "Say hi."}]
)
print(reply.choices[0].message.content)
```

## 5. Connect from your Daint job (`ACCESS=alps`)

With `ACCESS=alps` the job prints, after the 401/200 check:

```text
===== alps endpoint is live: http://172.28.9.16:30000/v1 on nid002536, job 123456 =====
endpoint.json: {"url": "http://172.28.9.16:30000/v1", "served_model": "hpcagent-bench-vllm", "key_file": "...", ...}
```

In your Daint job, before the client starts:

```bash
source "$REPO"/containers/cluster/ce-images/inference/alps-endpoint.sh <run dir>/endpoint.json || exit 1
```

`alps-endpoint.sh` reads `url`, `served_model` and `key_file` from `endpoint.json` (a path, not the
key); checks the key file is yours and mode 600; checks `GET /v1/models` lists the model and one chat
returns a choice; then exports `VLLM_BASE_URL`, `VLLM_API_KEY`, `VLLM_MODEL`. On failure it exports
nothing and prints the HTTP status (`000` = no connection); it returns 2 for a file problem, 1 for a
failed server check. The key goes to curl on stdin (`-H @-`). When the serving job ends,
`endpoint.json` is deleted before the server stops.

Daint compute node to Beverin compute node on the server port is untested (both sit on
172.28.0.0/16), as is reading the key file and `endpoint.json` from a Daint node. Run the `source`
line once in a short Daint job before relying on it; exit 0 confirms both.

## 6. Stop

Ctrl-C the tunnel; `scancel <jobid>` on Beverin (the job also ends at `--time`). If the node crashes,
cleanup does not run: delete leftover `endpoint.json`, `sglang-auth.yaml`, `api.key`, `auth.header`
from the run directory by hand.

## 7. Troubleshooting

| Message or symptom | Cause | Fix |
|---|---|---|
| `serve-private: refusing: ...` | a pre-launch check failed; the message names it | fix key mode, partition, weights, `LEGS` or `EXTRA_ARGS` |
| `SERVE FAILED` | server exited, or 401/200 check failed | read `server.log` in the leg dir; [`README.md`](README.md#5-healthy-or-sick) |
| `401` on the laptop | wrong or stale key, or missing header file | recopy the key, rebuild the header; restart the job after rotation |
| `curl: (7) ... Connection refused` | tunnel down or on another local port | start the tunnel; use its local port |
| `channel N: open failed` from ssh | wrong node or port, or server not ready | node from `squeue`; wait for `private endpoint is live` |
| `Permission denied (publickey)` | `cscs-key` signature expired | sign again |
| `bind [127.0.0.1]:30000: Address already in use` | local port taken | `-L 127.0.0.1:31000:127.0.0.1:30000` |
| `alps-endpoint: cannot read '...'` | serving job ended, or wrong path | `squeue`; copy the line from the job output |
| `alps-endpoint: ... is not an endpoint file` | wrong file | use the printed `endpoint.json` path |
| `alps-endpoint: key file ... must ... be mode 600` | key mode or owner | `chmod 600` on Beverin |
| `alps-endpoint: GET .../models answered 000` | not ready, job ended, or Daint cannot reach Beverin | wait for `alps endpoint is live`; if it persists, see section 5 |
| `alps-endpoint: ... answered 401` | key file changed after the job started | restart the serving job |

## 8. Rules

Sources: CSCS SSH docs, "SSH tunnel to a service on Alps compute nodes via Ela"
(https://docs.cscs.ch/access/ssh/); CSCS User Regulations
(https://www.cscs.ch/services/user-regulations): an account is for the applicant only; ETH Zurich BOT
(RSETHZ 203.21): Art. 9(2) credentials are personal, Art. 11(2) no circumventing security measures,
Art. 14(2c) no port scans.

Use the two connections above for yourself and your own jobs only. Not allowed:

- sharing the key, header file, `endpoint.json`, URL or tunnel with anyone, project members and
  students included (they start their own job with their own key);
- a request folder other people write to and your job serves;
- binding the wildcard address, or `ssh -g` / `-L 0.0.0.0:...` on the laptop;
- reverse tunnels or relays out of Alps: `ssh -R`, ngrok, cloudflared, frp;
- running the server on a login node;
- putting the key on a command line, or in chat, email, a ticket, a repository or a shared directory;
- scanning ports to find a server (take the node from `squeue`).
