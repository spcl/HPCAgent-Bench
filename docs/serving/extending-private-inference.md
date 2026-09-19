# Extending private inference

Contributor guide for `containers/cluster/ce-images/inference/serve-private.sbatch` and
`containers/cluster/ce-images/inference/alps-endpoint.sh`. How to use them:
[`private-endpoint.md`](private-endpoint.md).

## 1. Contract

A change to either script keeps every property below. The last column names the test that fails when a
property breaks; `tests/test_serve_private.py` covers the launcher, `tests/test_alps_endpoint.py` the
client.

| Property | Code | Test |
|---|---|---|
| The server binds `127.0.0.1` (`tunnel`) or the node's `hsn0` IPv4 address (`alps`), as the last argument. The launcher file contains no `0.0.0.0` literal, comments included. | `use_access`, `server_argv` | `test_every_leg_of_each_preset_binds_loopback_last_on_the_command_line`, `test_alps_access_binds_every_leg_to_this_nodes_hsn0_address_last` |
| The key file is mode 600, owned by the user, at least 32 characters. The key reaches SGLang only through the `--config` YAML. `API_KEY` is not exported, so `srun --export=ALL` does not pass it on. | `read_key_file`, `write_secret` | `test_each_preset_passes_the_key_only_through_a_config_file_in_a_mode_700_run_dir` |
| SGLang serves `/health*` and `/metrics*` without the key and checks the key only with one tokenizer worker. `EXTRA_ARGS` therefore may not set the host, port, key, config, tokenizer workers or metrics, and metrics run only on loopback. | `validate_extra_args`, `use_access` | `test_the_launcher_refuses_extra_args_that_name_the_host_port_key_config_tokenizer_workers_or_metrics`, `test_alps_access_serves_no_metrics` |
| Every refusal happens before a file is written or weights load. | order of the calls at the end of the launcher | `assert_untouched` in each refusal test |
| The run directory is mode 700 with ACLs removed. Cleanup deletes `endpoint.json` first, then the three secret files. | `make_private_dir`, `cleanup` | `test_alps_serve_mode_deletes_endpoint_json_with_the_secrets_when_the_job_exits` |
| No connection line is printed before a request without the key returns 401 and one with it returns 200. | `auth_checks`, `serve_leg` | none: `DRY_RUN` skips it; a `MODE=smoke` job is the test |
| The client sends the key on curl's stdin and exports nothing until `/v1/models` lists the model and one chat returns a choice. | `alps_endpoint` | `test_the_key_reaches_neither_curls_argv_nor_any_output`, the `exports_nothing` tests |

## 2. Launcher flow

1. `use_preset` sets the `PRESET_*` values. The key file, the Triton cache and the run root are named
   after the preset.
2. Checks, in order: `read_key_file`, `validate_extra_args`, `use_access`, `LEGS` parsing, weights present
   under `$HF_HOME/hub`, `gpu_arch_check.sh` (skipped under `DRY_RUN`).
3. `make_private_dir`, then `write_secret` for `sglang-auth.yaml`, `api.key` and `auth.header`.
4. `MODE=smoke`: `run_leg` per `LEGS` entry, which runs `start_leg` (one `srun --environment=<EDF>` step),
   `wait_ready`, `record_memory`, `auth_checks`, `tools_gate`, `load_probe` and `stop_server`.
5. `MODE=serve`: `serve_leg` runs the first entry through `auth_checks`, then prints `tunnel_help`, or
   `write_endpoint_file` and `alps_help`, and waits on the server.

## 3. Adding a preset

A preset is one partition, image and weights combination.

1. Add a case to `use_preset`: `PRESET_PARTITION`, `PRESET_GPUS`, `PRESET_EDF`, `PRESET_MODEL`,
   `PRESET_LEGS`, `PRESET_AITER`, `PRESET_FLAGS`.
2. `server_argv` hard-codes Qwen3.8 flags for every preset: the chat template
   `experiments/chat-template-qwen38.jinja`, `--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder`,
   `--mamba-full-memory-ratio 0.5`, `--context-length 262144` and `--language-only`. A different model
   family needs these moved into `PRESET_FLAGS`. The mi300 test compares against the campaign's
   `SGLANG_EXTRA_ARGS` and must still pass.
3. A new partition needs a `GPU_ARCH_<partition>` line in `containers/cluster/ce-images/gpu_arch.env`.
   Build its image on that partition: SGLang's `setup_rocm.py` and cupy compile for the GPU visible at
   build time ([`sglang-mi200/README.md`](../../containers/cluster/ce-images/sglang-mi200/README.md)).
4. Download weights with `containers/cluster/ce-images/inference/fetch_weights.sbatch` (`MODELS=...`) and
   verify the Lustre striping it reports. New downloads do not reliably inherit the directory layout.
5. Tests: add the preset to `PRESETS`, `OTHER_PARTITION` and `DEFAULT_LEG_COUNT`, add its weights
   repository to the loop in `launch()`, and add a flags test modelled on the mi200 one.
6. Run `MODE=smoke` with several `LEGS` on one node. Add KV pool, max running requests and load-probe
   throughput per leg to the table in `private-endpoint.md`. `--mem-fraction-static` is a fraction of
   node memory on the MI300A APU and of each GPU's memory on discrete GPUs, so values do not carry across
   partitions.

## 4. Adding an access path

A new path must leave the server usable by its owner only (`private-endpoint.md` section 8). Checklist:

- Bind one specific address.
- Take the list of unauthenticated routes from the engine's auth middleware in the image, and decide
  each one.
- Keep the key off argv on every host it passes through.
- Publish connection details, if any, in a mode-700 directory, without the key, and delete them before
  the server stops.
- Give the client a check that proves authentication, model name and one completion before it exports
  anything.
- Test refusal before side effects, the command line, key absence from all output, and cleanup.

Designs considered and rejected:

| Design | Reason |
|---|---|
| A request folder other users write to, served by your job | CSCS User Regulations: access for another person |
| ngrok, cloudflared, frp, `ssh -R` | circumvents security measures (ETH BOT Art. 11(2)) |
| Binding `0.0.0.0` without a key | every node on 172.28.0.0/16 reaches it, across Alps clusters; `experiments/serve-only.sbatch` behaves this way |
| An ssh tunnel opened from a Daint job | needs a CSCS-signed personal key on Alps storage; a signature is valid for one day |
| A server on a login node | not allowed on login nodes |

Open questions on the `alps` path: a Daint compute node reaching a beverin compute node on the server
port has not been tested, and neither has Daint resolving beverin node names. The published URL uses the
IP address for that reason.

## 5. Adding an engine: vLLM

Facts from `hpcagent-bench-vllm.sqsh` (vLLM 0.23.0):

- The key comes from `--api-key` or the environment variable `VLLM_API_KEY`
  (`vllm/entrypoints/openai/api_server.py`). Set the variable inside the server step, for example
  `bash -c 'VLLM_API_KEY="$(cat <file>)" exec vllm serve ...'`, so the key is expanded inside the process.
  Putting `VLLM_API_KEY=<key>` on the `srun ... env` line would place it on argv.
- Authentication covers only paths that start with `/v1`, `/v2` or `/inference` (`GUARDED_PREFIX` in
  `vllm/entrypoints/serve/utils/server_utils.py`). Every other route, `/health` for example, is served
  without the key. A vLLM preset stays `tunnel`-only until the routes of that image have been listed
  and judged.

Engine-specific parts of the launcher: `server_argv` (module and flags), the key writer (the YAML
`--config` is SGLang's), `record_memory` (greps SGLang log lines) and the `SGLANG_*` environment in
`start_leg`. `wait_ready`, `auth_checks`, `tools_gate` (`verify-tools-reasoning.py`) and `load_probe`
use the OpenAI API and apply to both engines.

## 6. Serving from another partition in one job

A heterogeneous Slurm job was tested on beverin (Slurm 25.05) with one mi300 and one mi200 component:

- `sbatch --partition=mi300 ... : --partition=mi200 ...` is accepted; `--time` on the command line
  applies to both components.
- The batch environment sets `SLURM_HET_SIZE`, `SLURM_JOB_NODELIST_HET_GROUP_<n>`,
  `SLURM_JOB_PARTITION_HET_GROUP_<n>` and `SLURM_JOB_ID_HET_GROUP_<n>`. Component 1 has its own job id,
  but inside an `srun --het-group=1` step `SLURM_JOB_ID` is the leader's.
- `srun --het-group=1 --exclusive` sees all 8 MI250X devices, and HTTP from the mi300 node to a server on
  the mi200 node over `hsn` works.

## 7. Testing

```bash
scripts/run_tests.sh -q -W error tests/test_serve_private.py tests/test_alps_endpoint.py
```

- Launcher tests run the script with `DRY_RUN=1`, an empty environment and stub `srun`, `sbatch` and
  `curl` that record any call. A stub `ip` prints `HSN0_LINE`. Nothing starts.
- Client tests run the real curl, wrapped to log its argv, against a local `ThreadingHTTPServer` that
  answers like SGLang with `--api-key`.
- A job is the only test for the 401/200 check, the tool-call and reasoning gate, KV pool size and
  `hsn` reachability. Run `MODE=smoke` before `MODE=serve`.

## 8. Known limits

- `#SBATCH --output`/`--error` are both `%x-%j.out`, relative to the submission directory
  (`#SBATCH` directives cannot expand `$SCRATCH`).
- MI250X (gfx90a) has no FP8 kernels in hipBLASLt and no aiter kernels, so mi200 serves BF16 with triton
  attention.
- `beverin.alps.cscs.ch` in the laptop ssh configuration has not been tested.
