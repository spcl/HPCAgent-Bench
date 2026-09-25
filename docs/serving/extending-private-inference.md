# Extending private inference

Contributor guide for `containers/inference/serve-private.sbatch` (launcher) and
`containers/inference/alps-endpoint.sh` (client). Usage:
[`private-endpoint.md`](private-endpoint.md).

## 1. Contract

Every change keeps these properties. `tests/test_serve_private.py` covers the launcher,
`tests/test_alps_endpoint.py` the client.

| Property | Code | Test |
|---|---|---|
| Binds `127.0.0.1` (`tunnel`) or the node's `hsn0` IPv4 (`alps`) as the last argument; the file has no `0.0.0.0` literal, comments included | `use_access`, `server_argv` | `test_every_leg_of_each_preset_binds_loopback_last_on_the_command_line`, `test_alps_access_binds_every_leg_to_this_nodes_hsn0_address_last` |
| Key file mode 600, owned by the user, 32+ characters; key reaches SGLang only via the `--config` YAML; `API_KEY` is never exported, so `srun --export=ALL` cannot carry it | `read_key_file`, `write_secret` | `test_each_preset_passes_the_key_only_through_a_config_file_in_a_mode_700_run_dir` |
| SGLang leaves `/health*` and `/metrics*` open and checks the key only with one tokenizer worker, so `EXTRA_ARGS` may not set host, port, key, config, tokenizer workers or metrics, and metrics run only on loopback | `validate_extra_args`, `use_access` | `test_the_launcher_refuses_extra_args_that_name_the_host_port_key_config_tokenizer_workers_or_metrics`, `test_alps_access_serves_no_metrics` |
| Every refusal happens before a file is written or weights load | call order at the end of the launcher | `assert_untouched` in each refusal test |
| Run dir mode 700, ACLs removed; cleanup deletes `endpoint.json` first, then the three secret files | `make_private_dir`, `cleanup` | `test_alps_serve_mode_deletes_endpoint_json_with_the_secrets_when_the_job_exits` |
| No connection line before an unkeyed request returns 401 and a keyed one 200 | `auth_checks`, `serve_leg` | none (`DRY_RUN` skips it): a `MODE=smoke` job is the test |
| The client sends the key on curl's stdin and exports nothing until `/v1/models` lists the model and one chat returns a choice | `alps_endpoint` | `test_the_key_reaches_neither_curls_argv_nor_any_output`, the `*_exports_nothing*` tests |

## 2. Launcher flow

1. `use_preset` sets `PRESET_*`; key file, Triton cache and run root are named after the preset.
2. Checks in order: `read_key_file`, `validate_extra_args`, `use_access`, `LEGS` parsing, weights under
   `$HF_HOME/hub`, `gpu_arch_check.sh` (skipped under `DRY_RUN`).
3. `make_private_dir`, then `write_secret` for `sglang-auth.yaml`, `api.key`, `auth.header`.
4. `MODE=smoke`: `run_leg` per entry: `start_leg` (one `srun --environment=<EDF>` step), `wait_ready`,
   `record_memory`, `auth_checks`, `tools_gate`, `load_probe`, `stop_server`.
5. `MODE=serve`: `serve_leg` runs the first entry through `auth_checks`, then prints `tunnel_help`, or
   `write_endpoint_file` + `alps_help`, and waits on the server.

## 3. Adding a preset

A preset is one partition + image + weights combination.

1. Add a `use_preset` case: `PRESET_PARTITION`, `PRESET_GPUS`, `PRESET_EDF`, `PRESET_MODEL`,
   `PRESET_LEGS`, `PRESET_AITER`, `PRESET_FLAGS`.
2. `server_argv` hard-codes Qwen3.8 flags for every preset (chat template, `--reasoning-parser qwen3`,
   `--tool-call-parser qwen3_coder`, `--mamba-full-memory-ratio 0.5`, `--context-length 262144`,
   `--language-only`). Another model family moves these into `PRESET_FLAGS`; the mi300 test against
   `.env.llrbase-qwen38-c` must still pass.
3. A new partition needs `GPU_ARCH_<partition>` in `containers/images/gpu_arch.env`, and
   its image built on that partition (SGLang's `setup_rocm.py` and cupy compile for the visible GPU;
   [`sglang-mi200/README.md`](../../containers/images/sglang-mi200/README.md)).
4. Fetch weights with `MODELS=<repo> sbatch containers/inference/fetch_weights.sbatch`
   and check the Lustre striping it reports.
5. Tests: add the preset to `PRESETS`, `OTHER_PARTITION`, `DEFAULT_LEG_COUNT`, its weights repo to the
   loop in `launch()`, and a flags test modeled on the mi200 one.
6. Run `MODE=smoke` with several `LEGS` on one node and add KV pool, max running requests and load
   throughput per leg to the table in `private-endpoint.md`. The fraction is node-wide on MI300A and
   per GPU on discrete GPUs, so values never carry across partitions.

## 4. Adding an access path

A new path keeps the server usable by its owner only ([`private-endpoint.md`](private-endpoint.md#8-rules)):

- bind one specific address;
- list the engine's unauthenticated routes from its auth middleware in the image, and decide each;
- keep the key off argv on every host it crosses;
- publish connection details only in a mode-700 directory, without the key, deleted before the server stops;
- give the client a check of auth, model name and one completion before it exports anything;
- test refusal before side effects, the command line, key absence from all output, and cleanup.

Rejected designs:

| Design | Reason |
|---|---|
| request folder other users write to | CSCS User Regulations: access for another person |
| ngrok, cloudflared, frp, `ssh -R` | circumvents security measures (ETH BOT Art. 11(2)) |
| `0.0.0.0` without a key | reachable from every node on 172.28.0.0/16 across Alps (what `serve-only.sbatch` does) |
| ssh tunnel opened from a Daint job | needs a CSCS-signed personal key on Alps storage, valid one day |
| server on a login node | not allowed |

The `alps` URL uses the IP address because Daint resolving Beverin node names is untested.

## 5. Adding an engine: vLLM

Facts from the vLLM 0.23.0 image:

- The key comes from `--api-key` or `VLLM_API_KEY` (`vllm/entrypoints/openai/api_server.py`). Set the
  variable inside the server step, e.g. `bash -c 'VLLM_API_KEY="$(cat <file>)" exec vllm serve ...'`;
  `srun ... env VLLM_API_KEY=<key>` would put it on argv.
- Auth covers only paths under `/v1`, `/v2`, `/inference` (`GUARDED_PREFIX` in
  `vllm/entrypoints/serve/utils/server_utils.py`); `/health` and every other route are open. A vLLM
  preset stays `tunnel`-only until those routes are listed and judged.

Engine-specific launcher parts: `server_argv`, the key writer (the YAML `--config` is SGLang's),
`record_memory` (greps SGLang log lines), and the `SGLANG_*` env in `start_leg`. `wait_ready`,
`auth_checks`, `tools_gate` (`verify-tools-reasoning.py`) and `load_probe` use the OpenAI API and work
for both.

## 6. Another partition in one job

Beverin (Slurm 25.05) accepts a heterogeneous job, e.g. `sbatch --partition=mi300 ... : --partition=mi200 ...`
(command-line `--time` applies to both components). The batch env sets `SLURM_HET_SIZE` and
`SLURM_JOB_{NODELIST,PARTITION,ID}_HET_GROUP_<n>`; inside `srun --het-group=1`, `SLURM_JOB_ID` is the
leader's. `srun --het-group=1 --exclusive` sees all 8 MI250X devices, and HTTP from the mi300 node to
a server on the mi200 node over `hsn` works.

## 7. Testing

```bash
scripts/run_tests.sh -W error tests/test_serve_private.py tests/test_alps_endpoint.py
```

- Launcher tests run the script with `DRY_RUN=1`, an empty environment, and stub `srun`, `sbatch`,
  `curl` that record calls; a stub `ip` prints `HSN0_LINE`. Nothing starts.
- Client tests run the real curl, wrapped to log its argv, against a local `ThreadingHTTPServer` that
  answers like SGLang with `--api-key`.
- Only a job tests the 401/200 check, the tool and reasoning gate, KV pool size and `hsn`
  reachability. Run `MODE=smoke` before `MODE=serve`.

## 8. Known limits

- `#SBATCH --output`/`--error` are `%x-%j.out` in the submit directory (`#SBATCH` cannot expand `$SCRATCH`).
- MI250X (gfx90a) has no FP8 kernels in hipBLASLt and no aiter kernels: mi200 serves BF16 with triton.
