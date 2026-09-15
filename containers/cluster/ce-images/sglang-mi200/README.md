# sglang-mi200

SGLang image for partition `mi200`: 8x MI250X (gfx90a, 64 GiB), EPYC 7A53, 4 NUMA, hsn0-3.
Serves `Qwen/Qwen3.8-27B` BF16. `../sglang` (MI300A, gfx942) stays untouched.

## Differs from ../sglang

| What | sglang | sglang-mi200 | Why |
| --- | --- | --- | --- |
| base | pinned digest | same digest | one ROCm/torch/sglang, test pins it |
| sgl_kernel common_ops | base copy, gfx942 only | rebuilt `--amdgpu-target=gfx90a` | device code is per arch; hot path (SiluAndMul) |
| `-DENABLE_FP8` | on | dropped | no ROCm source reads it; MI250X has no FP8 |
| cupy | gfx942 | gfx90a | AOT device code per arch |
| aiter JIT prebuild | shipped | none, `SGLANG_USE_AITER=0` | aiter has no gfx90a kernels |
| flydsl pin | 0.3.2 | not pinned | only reached through aiter |
| tuned MoE configs | baked | not baked | Qwen3.8-27B is dense |
| ue8m0 loader patch | applied | not applied | DeepSeek fp8 loader only |
| `HIPCC_COMPILE_FLAGS_APPEND` | set | unset | aiter JIT only |
| fabric gate, EDF hooks | yes | same | same netstack artifact |

## Autodetect trap

setup_rocm.py and cupy take the GPU visible at build time. A build on mi300 silently yields gfx942.
Guards:
- `ROCM_ARCH` comes from `../gpu_arch.env` for the build job's partition (no Dockerfile default); setup_rocm.py edited so `AMDGPU_TARGET` wins (asserted edit).
- Gate: `device_arch_gate.sh --exact` on installed common_ops and on cupy: device targets (`amdgcn-*-gfx*`) must be exactly `ROCM_ARCH`. rocprim's bare arch-name table does not count.
- `/opt/gpu-arch` stamps the arch; `gpu_arch_check.sh` compares it with the partition table and rocminfo at launch.
- build.sbatch and verify_image.sbatch refuse any partition but mi200.

## Build + verify (one job, mi200)

    REPO=<checkout> IMAGE_DIR=containers/cluster/ce-images/sglang-mi200 \
      sbatch --partition=mi200 --cpus-per-task=64 --gpus-per-node=8 \
      <checkout>/containers/cluster/ce-images/build_and_verify.sbatch

Writes `optarena-sglang-mi200-candidate.sqsh` + `.verified`. Verify includes
`inference/sglang_kernel_launch_check.py`: sgl_kernel silu_and_mul and triton causal_conv1d vs torch.

Verify alone:

    REPO=<checkout> IMAGE=$SCRATCH/ce-images/optarena-sglang-mi200-candidate.sqsh PROFILE=sglang-mi200 \
      sbatch --partition=mi200 --gpus-per-node=8 <checkout>/containers/cluster/ce-images/verify_image.sbatch

## Serve smoke (private endpoint)

    umask 077; mkdir -p ~/.config/optarena; openssl rand -hex 32 > ~/.config/optarena/mi200-endpoint.key
    PRESET=mi200 EDF=<candidate edf.toml> sbatch --partition=mi200 --gpus-per-node=8 \
      containers/cluster/ce-images/inference/serve-private.sbatch

127.0.0.1 only, key via `--config` yaml in a mode-700 run dir. Legs default `tp4:0.80 tp4:0.88 tp8:0.80`.
Laptop access (`MODE=serve`, ssh tunnel): `docs/serving/private-endpoint.md`.

## Promote

    ./promote_image.sh sglang-mi200        # DRY_RUN=1 first

Renames to `optarena-sglang-mi200.sqsh`, renders `sglang-mi200-latest` via install_edfs.sh.
