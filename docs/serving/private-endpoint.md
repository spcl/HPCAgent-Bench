# Private Qwen3.8 endpoint on beverin

Your own OpenAI-compatible Qwen3.8 server on one beverin compute node. Only you reach it, two ways:

- `ACCESS=tunnel` (default): from your laptop, through an ssh tunnel via ela (section 5).
- `ACCESS=alps`: from your own jobs on Daint or another Alps cluster, with the key (section 5b).

There is no third way. Without a working CSCS ssh key nothing outside Alps reaches it (section 8).

One launcher, two presets:

| `PRESET` | Partition | GPUs | EDF | Weights | Attention | State |
|---|---|---|---|---|---|---|
| `mi300` | `mi300` | 4x MI300A (APU, gfx942) | `sglang-latest` | `Qwen/Qwen3.8-27B-FP8` | aiter | qwen38 campaign config |
| `mi200` | `mi200` | 8x MI250X (64 GiB, gfx90a) | `sglang-mi200-latest` | `Qwen/Qwen3.8-27B` (BF16) | triton | smoke 637198 passed; serve `tp8:0.80` |

Launcher: `containers/cluster/ce-images/inference/serve-private.sbatch`.

- `MODE=smoke` (default): one fresh server per `LEGS` spec (`tp<N>:<mem-fraction-static>`), each
  checked (401/200 auth, tool-call + reasoning gate, load probe), then stopped.
- `MODE=serve`: first `LEGS` spec only. Waits for readiness, checks 401/200, prints your tunnel line,
  then holds the server until the job ends or you `scancel` it.
- `DRY_RUN=1 bash <launcher>`: all checks + the server argv, no launch.

Not `experiments/serve-only.sbatch`: that one binds every interface with no key, so any beverin user
can call it (see [`README.md`](README.md)).

## 1. Why it is private

- `ACCESS=tunnel`: server binds `127.0.0.1` on the compute node. Nothing off that node connects.
- `ACCESS=alps`: server binds the node's `hsn0` address, never `0.0.0.0`. Any Alps node can open a
  connection (all clusters share that network), so the key is what keeps it yours.
- SGLang checks the key on every path except `/health*` and `/metrics*` (`sglang/srt/utils/auth.py`),
  and only with one tokenizer worker. `ACCESS=alps` drops `--enable-metrics`; `/health` and
  `/health_generate` stay open and return a status, never model output. `EXTRA_ARGS` naming
  `--tokenizer-worker-num` or `--enable-metrics` is refused.
- SGLang has no environment variable for the key, only `--api-key` or `--config <yaml>`. The launcher
  uses the yaml, so the key never reaches argv (`ps` shows argv to every user on the node).
- SGLang logs `server_args`, `api_key` included. Run dir is mode 700, scratch default ACLs stripped.
- Key file refused unless mode 600, owned by you, >= 32 characters.
- `--host <address> --port N` is last on argv; `EXTRA_ARGS` naming host, port, key or config is refused.
- Key never printed; yaml, header and in-container key files deleted when the job exits.
- Laptop reaches the port with `ssh -L` through ela and the beverin login node, the CSCS-documented
  path (section 8).

## 2. Laptop, once: `~/.ssh/config`

    Host ela
        HostName ela.cscs.ch
        User ybudanaz
        IdentityFile ~/.ssh/cscs-key
        IdentitiesOnly yes

    Host beverin
        HostName beverin.alps.cscs.ch
        ProxyJump ela
        User ybudanaz
        IdentityFile ~/.ssh/cscs-key
        IdentitiesOnly yes

    Host nid*
        User ybudanaz
        IdentityFile ~/.ssh/cscs-key
        IdentitiesOnly yes

- `cscs-key` is the key signed by the CSCS SSH key service (https://docs.cscs.ch/access/ssh/). The
  signature expires; sign again each day you connect.
- `beverin.alps.cscs.ch` is not verified from this repo. If `ssh beverin` already works for you, keep
  your own entry.
- `Host nid*` makes the final hop (the compute node) use the same key.

Check: `ssh beverin hostname` prints a login node name.

## 3. Key, once (on beverin)

    umask 077; mkdir -p ~/.config/optarena
    openssl rand -hex 32 > ~/.config/optarena/mi300-endpoint.key
    openssl rand -hex 32 > ~/.config/optarena/mi200-endpoint.key     # exists already

- Default `KEY_FILE` = `~/.config/optarena/<preset>-endpoint.key`. Override with `KEY_FILE=`.
- Rotate: write a new key, restart the job, re-copy to the laptop (section 5). A running job keeps the
  key it read at start.

## 4a. mi300: Qwen3.8 FP8, campaign config

Same flags as the qwen38 campaigns (`SGLANG_EXTRA_ARGS` in `experiments/.env.llrbase-qwen38-c`;
`tests/test_serve_private.py` pins the match). Why each flag: [`qwen38.md`](qwen38.md).

    --tp-size 4 --mem-fraction-static 0.306 --attention-backend aiter
    --chat-template experiments/chat-template-qwen38.jinja --trust-remote-code --language-only
    --watchdog-timeout 1800 --context-length 262144 --mamba-full-memory-ratio 0.5
    --max-running-requests 128 --enable-metrics --reasoning-parser qwen3
    --tool-call-parser qwen3_coder --enable-cache-report
    --served-model-name optarena-vllm --host 127.0.0.1 --port 30000

- Env: `SGLANG_USE_AITER=1 SGLANG_SET_CPU_AFFINITY=0`. Custom all-reduce on.
- Default `LEGS=tp4:0.306`. 0.306 is node-wide on the APU, and aiter derates it to an effective
  0.26: move it only together with `--attention-backend` and `--mamba-full-memory-ratio`.
- Weights: `$HF_HOME/hub/models--Qwen--Qwen3.8-27B-FP8` (`HF_HOME` default
  `/iopsstor/scratch/cscs/$USER/hf`). Server runs `HF_HUB_OFFLINE=1`; launcher refuses if missing.

Serve:

    cd <optarena checkout>
    PRESET=mi300 MODE=serve sbatch --partition=mi300 --gpus-per-node=4 --time=08:00:00 \
      containers/cluster/ce-images/inference/serve-private.sbatch

Smoke first (optional, same flags, stops at the end):

    PRESET=mi300 sbatch --partition=mi300 --gpus-per-node=4 \
      containers/cluster/ce-images/inference/serve-private.sbatch

Expect:

- Readiness: no published start-up time for this model. The campaigns allow 2400 s
  (`VLLM_READY_TIMEOUT_SECONDS`); the launcher waits up to `READY_TIMEOUT=3600`.
- KV pool 3,318,498 tokens, 704 mamba slots (from [`qwen38.md`](qwen38.md)). The launcher copies the
  allocator lines to `memory.txt` in the leg dir. Numbers far off = different config; stop and check.

## 4b. mi200: Qwen3.8 BF16, new image

- Image `sglang-mi200` is new: MI250X needs sgl_kernel rebuilt for gfx90a, aiter has no gfx90a
  kernels, MI250X has no FP8, hence BF16 weights and triton attention.
- Build, verify and promote: `containers/cluster/ce-images/sglang-mi200/README.md`. Build 637177 was
  promoted; `sglang-mi200-latest` is installed.
- Smoke 637198, one node, every leg PASS (401/200, tool-call + reasoning gate, 16/16 load):

  | Leg | KV pool (tokens) | Max running requests | Load probe |
  |---|---|---|---|
  | `tp4:0.80` | 1.64M | 69 | cold Triton JIT, ignore |
  | `tp4:0.88` | 1.86M | 78 | 315 tok/s |
  | `tp8:0.80` | 1.91M | 128 | 322 tok/s |

  Serve `tp8:0.80`. Fractions here are per device VRAM, not node-wide like mi300: do not carry
  numbers across.
- Flags: as mi300 minus aiter, plus `--attention-backend triton --disable-custom-all-reduce`;
  env `SGLANG_USE_AITER=0 SGLANG_SET_CPU_AFFINITY=0`.
- Weights: `$HF_HOME/hub/models--Qwen--Qwen3.8-27B`.

Smoke (run this before serving):

    PRESET=mi200 sbatch --partition=mi200 --gpus-per-node=8 \
      containers/cluster/ce-images/inference/serve-private.sbatch

Serve (first leg unless `LEGS` given):

    PRESET=mi200 MODE=serve LEGS=tp8:0.80 sbatch --partition=mi200 --gpus-per-node=8 --time=08:00:00 \
      containers/cluster/ce-images/inference/serve-private.sbatch

## 5. Connect

**Always pass `--partition`**: the job script names none, the beverin default is `mi200`, and a preset
refuses a job on the other partition.

1. Node + job output:

        squeue -u "$USER" -n serve-private -o '%i %T %N'
        less /capstor/scratch/cscs/$USER/x86_64/ce-images/logs/serve-private-<jobid>.out

   `MODE=serve` prints, once `unauthenticated POST: 401` and `authenticated POST: 200` pass:

        ===== private endpoint is live: 127.0.0.1:30000 on nid002968, job 123456 =====
        On your laptop, while this job runs (docs/serving/private-endpoint.md):
          umask 077; mkdir -p ~/.config/optarena
          scp beverin:/users/ybudanaz/x86_64/.config/optarena/mi300-endpoint.key ~/.config/optarena/
          printf 'Authorization: Bearer %s\n' "$(cat ~/.config/optarena/mi300-endpoint.key)" \
            > ~/.config/optarena/mi300-endpoint.header
          ssh -N -J ela,beverin -L 127.0.0.1:30000:127.0.0.1:30000 ybudanaz@nid002968
          curl -s http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' \
            -H "@$HOME/.config/optarena/mi300-endpoint.header" \
            -d '{"model": "optarena-vllm", "max_tokens": 128, "messages": [{"role": "user", "content": "Say hi."}]}'
        Stop: Ctrl-C the tunnel, then scancel 123456.
        Do not share the key, the header file or this endpoint with anyone, project members included.

   Node, job and paths above are an example; copy the block your job prints. It never contains the key.

2. Key to laptop, once per key: the `scp` + `printf` lines. ssh only; never chat, email, tickets or a
   repo. `printf` is a shell builtin, so the key does not reach argv on the laptop either.

3. Tunnel: the `ssh -N -J ela,beverin -L 127.0.0.1:PORT:127.0.0.1:PORT <you>@<node>` line. Leave it
   running. The laptop end binds 127.0.0.1 too: nobody on your network uses it.

4. Call it. curl, key from a header file (`-H @file`, never `-H "Authorization: ..."` on argv):

        curl -s http://127.0.0.1:30000/v1/models -H "@$HOME/.config/optarena/mi300-endpoint.header"

   Python `openai` client, key read from the file:

        import pathlib

        from openai import OpenAI

        key = (pathlib.Path.home() / ".config/optarena/mi300-endpoint.key").read_text().strip()
        client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key=key)
        reply = client.chat.completions.create(
            model="optarena-vllm", max_tokens=128, messages=[{"role": "user", "content": "Say hi."}]
        )
        print(reply.choices[0].message.content)

   Served model name is `optarena-vllm` (`SERVED_MODEL=`). Port is `API_PORT` (default 30000). Tool
   calls and reasoning work: both parsers are on.

## 5b. Connect from your Daint jobs: `ACCESS=alps`

For your own agent jobs on Daint or another Alps cluster. No ssh from the job, no watcher, no shared
request folder: you submit the server on beverin, your Daint job reads one file and checks the endpoint.

1. Beverin:

        cd <optarena checkout>
        PRESET=mi200 ACCESS=alps MODE=serve LEGS=tp8:0.80 sbatch --partition=mi200 --gpus-per-node=8 \
          --time=08:00:00 containers/cluster/ce-images/inference/serve-private.sbatch

   Once `401` / `200` pass, the job output prints:

        ===== alps endpoint is live: http://172.28.9.16:30000/v1 on nid002536, job 123456 =====
        endpoint.json: {"url": "http://172.28.9.16:30000/v1", "served_model": "optarena-vllm", ...}
        In your job on Daint (or another Alps cluster), while this job runs (docs/serving/private-endpoint.md):
          source <checkout>/containers/cluster/ce-images/inference/alps-endpoint.sh <run dir>/endpoint.json

   - The URL is the node's `hsn0` address, the one its Slurm node name resolves to.
   - `endpoint.json` sits in the mode-700 run dir. It names the key file, never the key, and is
     deleted when the job ends.

2. Daint, in your job script, before the agent starts, copy the `source` line from step 1:

        source <checkout>/containers/cluster/ce-images/inference/alps-endpoint.sh <run dir>/endpoint.json || exit 1

   - Checks the key file is yours and mode 600, `GET /v1/models` lists the served model, one chat
     answers.
   - Then exports `VLLM_BASE_URL`, `VLLM_API_KEY`, `VLLM_MODEL`. Key goes to curl on stdin, never argv.
   - Any failed check exports nothing and returns non-zero.

3. Stop: `scancel <jobid>` on beverin. `endpoint.json` goes first; the Daint job's next request fails.

- **Not verified yet**, run step 2 once in a short Daint job before a campaign:
  - a Daint compute node opening a connection to a beverin compute node on port 30000 (both on
    172.28.0.0/16; beverin's login node reaches Daint's);
  - Daint reading the key file and run dir at their beverin paths (`/users`, `/capstor` are shared
    Alps file systems).
- Someone else's job, a student's included, is not yours: they run this launcher themselves, with
  their own key.

## 6. Stop

- Ctrl-C the tunnel.
- `scancel <jobid>`. Job also ends at `--time`.
- On exit the launcher deletes `sglang-auth.yaml`, `api.key`, `auth.header` from the run dir
  (`$SCRATCH/inference-server/<preset>-private/<jobid>`). After a node crash check that dir and
  remove any that remain; `server.log` there still holds the key, dir stays 700.

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `401` | wrong or stale key, header file missing or mistyped | re-`scp` the key, rebuild the header file; rotated key needs a job restart |
| `curl: (7) ... Connection refused` on the laptop | tunnel not running, or local port differs | start the tunnel; use the same port in the URL |
| tunnel prints `channel N: open failed: connect failed: Connection refused` | wrong node or port, or server not ready | `squeue` for the node; wait for the `private endpoint is live` block; port from that block |
| `Permission denied (publickey)` | `cscs-key` not signed today | sign it again (CSCS SSH key service), retry |
| ssh to the node refused after a while | job ended, node released | `squeue`; resubmit |
| `bind [127.0.0.1]:30000: Address already in use` | laptop port taken | `-L 127.0.0.1:31000:127.0.0.1:30000`, then use 31000 in the URL |
| job output `serve-private: refusing: ...` | a check failed before launch; message names it | fix that (key mode, partition, missing weights, bad `LEGS`) |
| `SERVE FAILED` | server died or auth check failed | `server.log` in the leg dir; common causes in [`README.md`](README.md) section 6 |
| `alps-endpoint: cannot read '...'` | serving job ended (file deleted), or wrong path | `squeue` on beverin; copy the `source` line from the job output |
| `alps-endpoint: GET .../models answered 000` | server not ready, job ended, or Daint cannot reach beverin | wait for the `live` block; if it persists while the job runs, report it (section 5b, not verified) |
| `alps-endpoint: ... answered 401` | key file changed after the job started | restart the serving job |
| `alps-endpoint: key file ... mode 600` | key file mode or owner wrong | `chmod 600` it on beverin |

## 8. CSCS rules

Sources:

- https://docs.cscs.ch/access/ssh/, section "SSH tunnel to a service on Alps compute nodes via Ela":
  local port forward through ela while your job holds the node. This page follows it.
- CSCS User Regulations: no access to CSCS resources for any other person, project member or
  otherwise.
- ETH Zurich BOT (RSETHZ 203.21): Art. 9(2) access credentials are personal; Art. 11(2) no
  circumventing security measures; Art. 14(2c) no port scans.

Do not:

- share the key, the header file, the endpoint or your tunnel with anyone, project members included.
  Someone else wants Qwen3.8: they run their own job with their own key from this page.
- bind `0.0.0.0` on the node, or an `hsn` address other than through `ACCESS=alps` (key, no metrics);
  on the laptop no `ssh -g`, no `-L 0.0.0.0:...`.
- open reverse tunnels or relays to outside hosts: no `ssh -R`, ngrok, cloudflared, frp. Without a
  working CSCS ssh key there is no path from outside Alps; sign a key first.
- serve other people's jobs from yours. `ACCESS=alps` is for your own jobs; a request folder that
  others write to is access for another person.
- run the server on a login node.
- put the key on a command line, in chat, email, a ticket, a repo or a shared directory.
- scan ports on compute nodes to find a server; take the node from `squeue`.
