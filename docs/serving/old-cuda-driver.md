# Serving on a node whose driver is older than the image's CUDA

The vLLM images and wheels the bench uses are built for CUDA 13 (`vllm/vllm-openai`, torch `cu130`). A node
whose NVIDIA driver predates CUDA 13 (driver 570, CUDA 12.8) cannot run them as they are: torch reports

```
RuntimeError: The NVIDIA driver on your system is too old (found version 12080).
```

and `torch.cuda.is_available()` is `False`. Data-center GPUs (A100, H100, GH200, L40S and the like) can still run
them through CUDA forward compatibility: the image ships a newer user-mode driver (`libcuda.so`) in
`/usr/local/cuda/compat`, and the old kernel driver accepts it. Consumer GPUs (GeForce, RTX workstation cards)
do not support forward compatibility and need a newer driver.

## Check the node

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader   # e.g. NVIDIA A100-SXM4-40GB, 570.86.15
```

A driver below 580 with a CUDA 13 image needs the steps below.

## Run the image with the compat library first

Put `/usr/local/cuda/compat` at the front of `LD_LIBRARY_PATH` inside the container, so the image's
`libcuda.so` loads before the host driver's copy that `--nv` (or `--gpus`) mounts in.

Apptainer or Singularity:

```bash
singularity pull vllm.sif docker://vllm/vllm-openai:latest
COMPAT=(--env LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/nvidia/lib64:/usr/local/cuda/lib64)
singularity exec --nv "${COMPAT[@]}" vllm.sif python3 -c \
  "import torch; print(torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
singularity exec --nv "${COMPAT[@]}" -B /scratch vllm.sif vllm serve openai/gpt-oss-120b \
  --tensor-parallel-size 4 --enable-prefix-caching ...
```

Docker:

```bash
docker run --gpus all -e LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
  vllm/vllm-openai:latest --model openai/gpt-oss-120b --tensor-parallel-size 4 --enable-prefix-caching ...
```

The serving arguments are the ones of the model's page ([`oss120b.md`](oss120b.md), [`qwen38.md`](qwen38.md)).

## Measured

On an A100-SXM4-40GB node with driver 570.86.15 (CUDA 12.8) and `vllm/vllm-openai:latest` (vLLM 0.34.0, torch
2.14.0+cu130):

| `LD_LIBRARY_PATH` | `torch.cuda.is_available()` | a kernel on the GPU |
|---|---|---|
| image default | `False` (driver too old) | fails |
| `/usr/local/cuda/compat` first | `True`, 4 devices | runs |

With the compat library first, `vllm serve openai/gpt-oss-120b --tensor-parallel-size 4 --enable-prefix-caching`
(arguments of [`oss120b.md`](oss120b.md)) started on the four GPUs and answered tool calls with reasoning.

## Pitfalls

- A pip install of vLLM into a host virtual environment pulls the CUDA 13 torch as well; it fails the same way and
  has no compat library to load. Use the image, or install a torch built for the driver's CUDA and a vLLM wheel
  built against it.
- Setting `LD_LIBRARY_PATH` on the host does not reach the container: pass it with `--env` (Apptainer) or `-e`
  (Docker).
