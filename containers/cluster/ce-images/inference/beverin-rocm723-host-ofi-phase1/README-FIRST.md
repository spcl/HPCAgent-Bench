# Beverin ROCm 7.2.3 host-OFI — Phase 1

This bundle validates the host Slingshot/CXI ABI before rebuilding vLLM.

## Captured contract

- Host GLIBC requirement: 2.38
- Host libfabric package: 2.3.1
- Host libfabric SONAME target: libfabric.so.1.29.1
- Highest host symbol version: FABRIC_1.8
- Host libcxi SONAME target: libcxi.so.1.5.0
- Highest libcxi symbol family observed: LIBCXI_1.9
- Runtime stack: supplied by `com.hooks.netstack.source = "host"`

## 1. Create the compile-only host SDK

Run on Beverin where `/opt/cray/libfabric/host` and `/usr/lib64/libcxi.so.1`
are visible:

```bash
./pack-beverin-host-sdk.sh beverin-host-sdk.tar.gz
```

Keep the generated tarball in the same build context as the Containerfile.
It contains host headers and a builder-only dependency closure. The final image
copies only the compiled AWS-OFI plugin and the `fi_info` diagnostic executable.

## 2. Build the diagnostic OCI image

```bash
podman build \
  --file Containerfile.rocm723-ofi-host-diag \
  --tag rocm723-ofi-host-diag:phase1 \
  .
```

After the first successful build, inspect and record the base-image digest and
the AWS-OFI commit written to `/opt/aws-ofi-nccl/BUILD-MANIFEST.txt`, then pin
both instead of relying on a mutable tag.

Convert the OCI image to `.sqsh` using the same import workflow already used
for the existing vLLM images, and update `rocm723-ofi-host-diag.toml`.

## 3. Register the EDF

```bash
mkdir -p "$HOME/.edf"
cp rocm723-ofi-host-diag.toml "$HOME/.edf/"
```

Use the EDF name expected by your Container Engine installation; for example,
rename it to `$HOME/.edf/rocm723-ofi-host-diag.toml` if environments are selected
by basename.

## 4. Run the mandatory gate and one-rank test

```bash
ENVIRONMENT=rocm723-ofi-host-diag ./run-phase1-one-rank.sh
```

The run is successful only if:

1. `ldd -r` is clean.
2. `libfabric.so.1` and `libcxi.so.1` resolve from host-hook paths.
3. `fi_info-host -p cxi` enumerates `cxi0` through `cxi3`.
4. `ctypes.CDLL(..., RTLD_NOW)` succeeds.
5. The Python program prints `process group initialized` and `correct=True`.

Do not proceed to two or four ranks until this one-rank test passes.
