# Build the known-good vLLM Container Engine image on Beverin

This directory records the **working** Beverin build path extracted from the
former `vvlm-mi300-3-main.zip`. The resulting
Container Engine (CE) image contains:

- Ubuntu 24.04 and ROCm 7.2.3 from the Phase 1 base image;
- the AWS OFI RCCL network plugin built against Beverin's host
  Slingshot/CXI ABI;
- Python 3.12, PyTorch `2.11.0+rocm7.2`, torchvision
  `0.26.0+rocm7.2`, and torchaudio `2.11.0+rocm7.2`; and
- vLLM `0.23.0`, compiled for the MI300A `gfx942` target.

The final, known-good artifact name is
`containers/rocm723-vllm-0.23.0-pytorch211-ofi.sqsh`. Do not select the older
`rocm723-ofi-vllm-0.23.0.sqsh`/`/opt/vllm-venv` path: that belongs to an earlier
build line. The final image uses `/opt/pytorch211`.

This is a build and setup guide, not a test procedure. The build scripts do
perform fail-fast checks while assembling the image, but no separate test jobs
are required by this guide.

## 1. Requirements

Run these steps on Beverin with:

- access to the `mi300` Slurm partition and a four-GPU MI300A node;
- Podman for the initial OCI build;
- Enroot and Slurm for the `.sqsh` image builds;
- outbound access to GitHub, Ubuntu package repositories, PyPI, and the
  PyTorch ROCm 7.2 wheel index; and
- a checkout on the shared `/iopsstor` filesystem, visible at the same path
  from login and compute nodes.

The promoted build scripts derive their root from this directory. Override it
with `VLLM_BUILD_ROOT` if the build inputs are copied elsewhere. Slurm output
paths are relative, so submit from `$ROOT` (or use `sbatch --chdir="$ROOT"`).
Create the output directories:

```bash
export ROOT=/iopsstor/scratch/cscs/$USER/vllm-mi300-3
export VLLM_BUILD_ROOT="$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/containers" "$ROOT/phase1-passed"
cd "$ROOT"
```

Keep the repository mounted at `$ROOT` during all Enroot builds. The vLLM
inner build writes its detailed build log back to `$ROOT/logs`.

## 2. Build the Beverin host-OFI base image

This phase is important: the compute-node host hook supplies `libfabric` and
`libcxi` at runtime, so the RCCL network plugin must be compiled against the
same host ABI. The relevant files are under
`beverin-rocm723-host-ofi-phase1/`:

- `pack-beverin-host-sdk.sh` captures the host headers and builder-only
  dependency closure;
- `Containerfile.rocm723-ofi-host-diag` starts from ROCm 7.2.3 and builds
  AWS OFI NCCL `v1.20.0`;
- `host-loader-gate.sh` and `torch-dist-allreduce-mi300.py` are copied into the
  image as diagnostics; and
- `rocm723-ofi-host-diag.toml` records the required CE host-network hooks.

On Beverin, create the SDK and OCI image:

```bash
cd "$ROOT/beverin-rocm723-host-ofi-phase1"
./pack-beverin-host-sdk.sh beverin-host-sdk.tar.gz

podman build \
  --file Containerfile.rocm723-ofi-host-diag \
  --tag rocm723-ofi-host-diag:phase1 \
  .
```

Import that local OCI image with the site's normal Enroot workflow and place
the resulting squashfs file at the path expected by the next build:

```text
$ROOT/phase1-passed/rocm723-ofi-host-diag-phase1.sqsh
```

For example, when the installed Enroot supports its Podman URI importer:

```bash
enroot import --output \
  "$ROOT/phase1-passed/rocm723-ofi-host-diag-phase1.sqsh" \
  podman://localhost/rocm723-ofi-host-diag:phase1
```

Use Beverin's site-provided OCI-to-Enroot command instead if its Enroot build
does not enable the Podman importer. The only contract for the following step
is the final `.sqsh` pathname above.

## 3. Add the qualified PyTorch 2.11 ROCm environment

Return to the repository root and submit:

```bash
cd "$ROOT"
sbatch build/build-pytorch211-phase1.sbatch
```

The Slurm wrapper `build/build-pytorch211-phase1.sbatch` creates a writable
Enroot container from the Phase 1 image, runs
`build/build-pytorch211-phase1-inner.sh`, and exports:

```text
$ROOT/containers/rocm723-pytorch211-ofi-phase1-candidate.sqsh
```

The inner script deliberately installs PyTorch into the separate
`/opt/pytorch211` virtual environment. It does not replace the base image's
`/opt/venv`; this separation is part of the working recipe.

### Required NumPy patch

The successful sequence added NumPy to that intermediate image before the
vLLM build. After the PyTorch job completes successfully, submit:

```bash
sbatch build/add-numpy-pytorch211.sbatch
```

This updates the same candidate image atomically and keeps the original as
`rocm723-pytorch211-ofi-phase1-candidate.before-numpy.sqsh`. Do not skip this
job: `build/build-vllm023-pt211-inner.sh` imports NumPy during its base-image
qualification.

## 4. Build vLLM 0.23.0 for MI300A

After the NumPy job completes successfully, submit:

```bash
sbatch build/build-vllm023-pt211.sbatch
```

The wrapper `build/build-vllm023-pt211.sbatch` uses the patched PyTorch image,
runs `build/build-vllm023-pt211-inner.sh`, and exports the final CE image. The
inner script:

1. keeps the qualified ROCm PyTorch packages instead of allowing vLLM's
   dependencies to replace them;
2. installs the matching torchvision and torchaudio ROCm wheels;
3. clones the exact `v0.23.0` vLLM tag;
4. constrains the complete GPU package stack;
5. removes the CUDA-only `torch-c-dlpack-ext` optional package; and
6. builds the ROCm extensions with `VLLM_TARGET_DEVICE=rocm` and
   `PYTORCH_ROCM_ARCH=gfx942`.

The outputs are:

```text
$ROOT/containers/rocm723-vllm-0.23.0-pytorch211-ofi.sqsh
$ROOT/containers/rocm723-vllm-0.23.0-pytorch211-ofi.sqsh.sha256
```

The recorded checksum from the successful build is retained inside
`archive/vvlm-mi300-3-main.zip` at
`vvlm-mi300-3-main/containers/rocm723-vllm-0.23.0-pytorch211-ofi.sqsh.sha256`.
Its absolute path is historical; compare the digest (the first field), not the
recorded filename.

## 5. Register the final image with Container Engine

Copy the final EDF template and replace its image and work-directory paths:

```bash
mkdir -p "$HOME/.edf"
cp rocm723-vllm-0.23.0-pytorch211-ofi.toml \
  "$HOME/.edf/rocm723-vllm-0.23.0-pytorch211-ofi.toml"

sed -i \
  -e "s|@ROOT@|$ROOT|g" \
  -e "s|@WORKDIR@|$(dirname "$ROOT")|g" \
  "$HOME/.edf/rocm723-vllm-0.23.0-pytorch211-ofi.toml"
```

The EDF enables Beverin's CXI and host netstack hooks, selects the OFI network
plugin, disables DMA-BUF for the current Beverin kernel, and puts
`/opt/pytorch211/bin` first on `PATH`. Use the environment name
`rocm723-vllm-0.23.0-pytorch211-ofi` with the site's CE command.

## File selection summary

Use only this build chain:

```text
beverin-rocm723-host-ofi-phase1/pack-beverin-host-sdk.sh
beverin-rocm723-host-ofi-phase1/Containerfile.rocm723-ofi-host-diag
    -> phase1-passed/rocm723-ofi-host-diag-phase1.sqsh

build/build-pytorch211-phase1.sbatch
build/build-pytorch211-phase1-inner.sh
    -> containers/rocm723-pytorch211-ofi-phase1-candidate.sqsh

build/add-numpy-pytorch211.sbatch
    -> patches the PyTorch candidate in place

build/build-vllm023-pt211.sbatch
build/build-vllm023-pt211-inner.sh
    -> containers/rocm723-vllm-0.23.0-pytorch211-ofi.sqsh

rocm723-vllm-0.23.0-pytorch211-ofi.toml
    -> final CE runtime environment
```

Files whose names contain `.before-` or `.failed-` are retained history, not
build inputs. `phase2-vllm-passed/rocm723-ofi-vllm-023.toml` describes the
older `/opt/vllm-venv` image and must not be used for this PyTorch 2.11 build.

## Archived material

Everything from the ZIP that is not part of the promoted build chain above is
retained in `archive/vvlm-mi300-3-main.zip`. This includes historical logs,
failed or superseded scripts, checksums, discovery output, and runtime
experiments. Keeping the untouched ZIP in the archive avoids adding hundreds of
generated files and bypassing the repository's 500 KiB new-file guard; the
necessary build inputs remain directly available in this directory.
