# Choosing a container image

Every role runs from an image: the agent, the judge and the inference server. Under the CSCS Container Engine
an image is named by its EDF (`~/.edf/<name>.toml`); under Apptainer, Podman or Docker by an image reference.
Building, verifying and publishing the images: [containers/README.md](../containers/README.md). How each runtime
starts a step: [runtime.md](runtime.md#cluster-launcher-one-container-seam).

## The images

One registry tag per image, in `docker.io/spcleth/hpcagent-bench` (`containers/images/images.env`):

| tag | build directory | CPU / GPU | toolkit | EDF |
|---|---|---|---|---|
| `agent-amd-latest`, `judge-amd-latest` | `judge-agent-amd` | x86-64-v3; gfx90a, gfx942, gfx950 | ROCm | `hpcagent-bench-{agent,judge}-{mi300,mi200}-latest` |
| `agent-nvidia-latest`, `judge-nvidia-latest` | `judge-agent-cuda` | aarch64 (GH200); sm_80 to sm_120 | CUDA 13.4.1 (NGC PyTorch 26.09) | `hpcagent-bench-{agent,judge}-gh200-latest` |
| `agent-nvidia-x86_64-latest`, `judge-nvidia-x86_64-latest` | `judge-agent-cuda` built on x86_64 | x86-64-v3; sm_75 to sm_120 | CUDA 13.4.1 | `hpcagent-bench-{agent,judge}-nvidia-x86_64-latest` |
| `agent-cpu-<arch>-latest`, `judge-cpu-<arch>-latest` | `judge-agent-cpu` | x86-64-v3 or armv8.2-a | none | `hpcagent-bench-{agent,judge}-cpu-<arch>-latest` |
| `verifier-{amd,nvidia,nvidia-x86_64,cpu-<arch>}-latest` | `verifier` | the judge's | the judge's | none: Harbor pulls it ([why](hf_dataset_and_harbor.md#harbor-adapter)) |
| `sglang-mi300-latest` | `sglang` | gfx942, gfx90a | ROCm | `hpcagent-bench-sglang-mi300-latest` |
| `vllm-amd-latest` | `vllm` | gfx90a, gfx942, gfx950 | ROCm | `hpcagent-bench-vllm-{mi300,mi200}-latest` |
| `vllm-gh200-latest` | `vllm-cuda` | aarch64 only | CUDA 12.9 (vLLM v0.28.0) | `hpcagent-bench-vllm-gh200-latest` |

Which of them are published is the registry's listing, not this page:
`curl -s 'https://hub.docker.com/v2/repositories/spcleth/hpcagent-bench/tags?page_size=100'`. A row whose tag is
not listed is built locally (`containers/images/build_and_verify.sbatch <dir>`) before it is used. With
`CE_IMAGE_FLAVOR=native` the agent and judge EDFs are the `-native` builds instead of `-latest`
([containers/README.md](../containers/README.md#getting-the-images-download-default-or-build-natively)).

Get an image and its EDF (the EDFs mount `$SCRATCH`, so run this on the cluster that will use them):

```bash
sbatch containers/images/registry.sbatch pull judge-agent-amd        # a role of images.env
containers/images/install_edfs.sh                                     # the node's platform; CE_PLATFORM=amd | gh200 | nvidia-x86 | cpu names one
ls ~/.edf/
```

## Switching the image of a run

| what runs | Container Engine | other runtimes (`CONTAINER_RUNTIME=apptainer \| podman \| docker`) |
|---|---|---|
| an experiment's agent | `AMD_CE_ENV` (or `AGENT_CE_ENV` for the agent step alone) | `AGENT_IMAGE` (the agent target; default `BENCH_IMAGE`) |
| an experiment's judge | `JUDGE_CE_ENV` | `BENCH_IMAGE` |
| an experiment's inference server | `INFERENCE_CE_ENV` | `INFERENCE_IMAGE` |
| a helper job (`grade-under`, `prepare`) | `JUDGE_EDF=~/.edf/<name>.toml` | (CE only) |
| the Harbor adapter | -- | `images:` in the configuration (`HPCAGENT_BENCH_IMAGES_<HW>_AGENT` / `_VERIFIER`) |
| the local container launch (`helpers/scripts/run_agent_in_container.sh`) | -- | `HPCAGENT_BENCH_DOCKER_IMAGE` or `HPCAGENT_BENCH_SIF`, else `container_backends.txt` (`hpcagent_bench:<hw>`, or a `.sif` in the checkout) |

The `*_CE_ENV` values are EDF names (no path, no `.toml`). `[common]` sets the MI300 pair,
`AMD_CE_ENV=hpcagent-bench-agent-mi300-latest` and `JUDGE_CE_ENV=hpcagent-bench-judge-mi300-latest`; a setup's
`.env` or the shell overrides them. `submit.sh --hardware <hardware>` renames every `*_CE_ENV` from `-mi300-` to
`-<hardware>-` and pins `[hardware.<hardware>]` ([configuration.md](configuration.md#hardware-is-not-a-site-value)),
so a hardware with a layer needs no EDF edit:

```bash
TAG=llr40 hpcagent_bench/cluster/submit.sh --hardware mi200                         # the -mi200- EDFs
TAG=llr40 JUDGE_CE_ENV=my-judge-candidate hpcagent_bench/cluster/submit.sh         # one role, any EDF in ~/.edf
JUDGE_EDF=~/.edf/hpcagent-bench-judge-cpu-x86_64-latest.toml \
  hpcagent-bench job submit --nodes 1 hpcagent_bench/cluster/prepare.sbatch --tag llr40 --language c
```

Experiments (`submit.sh`) run on the hardware that has layers, `mi300` (the base), `mi200` and `gh200`; under the
Container Engine `submit.sh` refuses a run with no hardware or with a hardware that has no
`[hardware.<hardware>]`. x86 NVIDIA nodes run the helper jobs and the CLI with their
own EDFs; [containers/README.md](../containers/README.md#nvidia-gh200-daint) shows a Daint job using a Beverin
inference endpoint.

## NCCL and RCCL over Slingshot

The GPU judge and agent EDFs (`containers/images/judge-agent-{amd,cuda}/*.edf.toml.in`) set CSCS's
[NCCL settings](https://docs.cscs.ch/software/communication/nccl/) in `[env]`. CSCS has no RCCL page; RCCL reads
the same variables through the same `aws-ofi-nccl` plugin. The CE netstack hook (the `com.hooks.netstack.*`
annotations, [resource hooks](https://docs.cscs.ch/software/container-engine/resource-hook/)) sets its own values
when a container starts. An EDF's `[env]` wins over the hook, and the hook wins over the step environment (`srun
--export`). Job 754679 measured this: one `srun` with `NCCL_CROSS_NIC=7 NCCL_PROTO=Simple` in its environment saw
the hook's `0` and `^LL128` under the old judge EDF and `1` and `^LL128` under the new one. A graded launch's ranks
start in the same EDF through the gang relay, so they get the same values. Every rank of a P=8 grade on two `mi300`
nodes (job 754679) recorded the values in the last column, and RCCL logged `NCCL_CROSS_NIC set by environment to 1`
and `Using network AWS Libfabric`.

| variable | CSCS NCCL page | netstack hook (AMD judge) | EDF `[env]` | a rank sees |
|---|---|---|---|---|
| `NCCL_NET` | `AWS Libfabric` | `AWS Libfabric` | `AWS Libfabric` | `AWS Libfabric` |
| `NCCL_NET_GDR_LEVEL` | `PHB` | `PHB` | `PHB` | `PHB` |
| `NCCL_CROSS_NIC` | `1` | `0` | `1` | `1` |
| `NCCL_PROTO` | `^LL128` | `^LL128` | `^LL128` | `^LL128` |
| `FI_CXI_DEFAULT_CQ_SIZE` | `131072` | `131072` | `131072` | `131072` |
| `FI_CXI_DEFAULT_TX_SIZE` | `16384` | `16384` | `16384` | `16384` |
| `FI_CXI_DISABLE_HOST_REGISTER` | `1` | `1` | `1` | `1` |
| `FI_MR_CACHE_MONITOR` | `userfaultfd` | unset | `disabled` (AMD), `userfaultfd` (GH200) | `disabled` |
| `FI_PROVIDER` | unset | unset | unset | unset |

The hook also sets values no EDF touches: `NCCL_NET_PLUGIN=ofi`, `NCCL_SOCKET_IFNAME=hsn`,
`NCCL_NCHANNELS_PER_NET_PEER=4`, `NCCL_PXN_DISABLE=1`, `NCCL_NET_GDR_READ=1`, the GH200 values
`NCCL_P2P_LEVEL=NVL` and `NCCL_NET_GDR_C2C=1`, `FI_CXI_RDZV_*`, `FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD` and
`FI_MR_CACHE_MAX_*`.

Three settings differ from CSCS's on purpose:

- `FI_PROVIDER` stays unset. The image runs MPICH, which inherits it, and `MPI_Init` aborts
  (`tests/test_edf_contract.py::test_mpich_does_not_inherit_a_libfabric_provider`). CSCS's
  [Beverin vLLM tutorial](https://docs.cscs.ch/tutorials/ml/vllm-sglang-amdgpus/) sets `FI_PROVIDER=cxi` for a
  container without MPI.
- `FI_MR_CACHE_MONITOR=disabled` on the AMD EDFs. `userfaultfd` is not shown to work on Beverin, and the Beverin
  tutorial disables the cache as well.
- `com.hooks.aws_ofi_nccl.variant="rocm6"` stays. The hook page says the variant is ignored when netstack artifacts
  are used, and that the artifacts are "fully tested and supported only on GH200 vClusters".

An agent role's EDF switches the comm hooks off at run time and drops `NCCL_NET`
(`container_runtime.sh`); the other lines are harmless there. The serving EDFs are separate
([serving/known-issues.md](serving/known-issues.md)): they keep `FI_CXI_DEFAULT_TX_SIZE=32768`, and the server
command sets `NCCL_NET_GDR_LEVEL=LOC`, because a value in the step environment loses to the hook.
