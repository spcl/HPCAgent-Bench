# syntax=docker/dockerfile:1
#
# SGLang inference on AMD MI300A (gfx942). This is the DOMINANT engine, not the niche one: it
# serves qwen38 and kimi, which is 12 of the 18 llr40v9 arms. Only oss120b stays on vLLM.
#
# Build context is the REPOSITORY ROOT, because the tuned fused_moe configs are baked in from
# there:
#   podman build -f containers/cluster/ce-images/sglang/Dockerfile -t optarena-sglang .
# build.sh does that with the right podman flags; build.sbatch runs build.sh on a compute node.
#
# NOT YET BUILT. The deployed artifact was a straight PULL and never had a Dockerfile:
# ~/.edf/sglang-rocm-mi30x.toml names the base below, pulled as job 604787 (52.3 GB). Everything
# here is derived from that toml, from run_cluster.sh's sglang branch, and from the
# SGLANG_EXTRA_ARGS of the live .env.llr40v9-{kimi27sglang,qwen38}-* files.
#
# The base is ROCm 7.2.0, which is now the global pin -- it is the vendor's own tag for this
# SGLang release and it is what every measured kimi and qwen38 number was produced on. Pinned by
# DIGEST because the date-stamped tag is still a tag.
#
# Read back from the deployed image, and load-bearing below: Ubuntu 22.04, glibc 2.35, python
# 3.10 in /opt/venv, torch 2.9.1+rocm7.2.0, triton 3.7.0+amd.rocm7.2.0, sglang
# 0.5.18.dev20260822 at /sgl-workspace/sglang/python, amd-aiter 0.1.19.post3.dev139 at
# /sgl-workspace/aiter, flydsl 0.3.1.
ARG BASE_IMAGE=docker.io/lmsysorg/sglang-rocm:v0.5.18-rocm720-mi30x-20260822\
@sha256:4af1a96f988c523fd2848138b035eb2bd7654454d4f371e8ab0a379d50ee3feb

# ------------------------------------------------------------------------------------------------
# Stage 1: a SELF-CONTAINED Slingshot/CXI stack -- cassini headers, libcxi, libfabric with the cxi
# provider, and the AWS OFI plugin -- all compiled against THIS image's glibc 2.35.
#
# Everything about this stage follows from one measured fact. The host's own libraries cannot be
# used: /opt/cray/libfabric/2.3.1/lib64/libfabric.so.1.29.1 and /usr/lib64/libcxi.so.1.5.0 both
# reference GLIBC_2.38, and this image's loader is 2.35. That is 595217 -- grafting the host
# netstack kills the image's loader -- and it is why the EDF sets neither com.hooks.cxi.enabled nor
# com.hooks.netstack.source, leaving cross-node RCCL on TCP over hsn*.
#
# Note this is the OPPOSITE of the vLLM image, whose Ubuntu 24.04 base is glibc 2.39: 2.39 > 2.38,
# so there the host graft works and the plugin is built against host headers with the runtime
# coming from the hook. Same cluster, same plugin, two different correct answers, decided purely by
# which side of 2.38 the base sits on.
#
# The versions are the host's, read off the login node rather than guessed:
#   libfabric 2.3.1 (api 2.3) -> libfabric.so.1.29.1
#   cray-libcxi-1.0.2, built from SHS 13.1.0 -> libcxi.so.1.5.0
# libcxi is a userspace shim over the /dev/cxi ioctl uAPI, so its SHS release must match the host
# kernel driver; SHS_REF is an ARG so that pin can be corrected without touching this file.
# ------------------------------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS ofi-builder
ARG BASE_IMAGE
ARG SHS_REF=release/shs-13.1.0
ARG LIBFABRIC_REF=v2.3.1
ARG AWS_OFI_NCCL_REF=v1.20.0
USER root

# Every network git call goes through this, and where a mirror is mounted none of them
# reaches GitHub at all. The `insteadOf` rewrite is what covers the callers the wrapper
# cannot: spack clones its package repo in-process and vLLM's CMake FetchContent clones
# triton itself, and both died on the same rate limiting with no retry of their own.
# Conditional, so a build with no mirror mounted behaves exactly as it did before.
# `protocol.file.allow` goes with it: git has refused local-path submodule clones since the
# CVE-2022-39253 fix, and the rewrite turns every submodule URL into exactly that, so
# `submodule update` aborts with `transport 'file' not allowed` before it looks for the repo.
# Every network git call goes through this. An unauthenticated clone or fetch from a busy egress
# IP gets a 401, which git reports as `could not read Username for https://github.com`: it looks
# like a credentials failure and is a transient one. GIT_TERMINAL_PROMPT=0 makes it fail instead
# of blocking on a prompt no build has a terminal for; the backoff is what actually fixes it.
RUN set -eux; \
    printf '%s\n' \
      '#!/bin/sh' \
      'set -eu' \
      'export GIT_TERMINAL_PROMPT=0' \
      'n=1' \
      'while :; do' \
      '  git "$@" && exit 0' \
      '  [ "${n}" -ge 10 ] && exit 1' \
      '  d=$((n * n * 10)); if [ "${d}" -gt 300 ]; then d=300; fi' \
      '  sleep "${d}"' \
      '  n=$((n + 1))' \
      'done' \
      > /usr/local/bin/gitretry; \
    chmod 0755 /usr/local/bin/gitretry; \
    if [ -d /git-mirrors ]; then \
      git config --global "url./git-mirrors/.insteadOf" "https://github.com/"; \
      git config --global --add safe.directory '*'; \
      git config --global protocol.file.allow always; \
    fi

# The second list is libcxi's own configure.ac, transcribed: libconfig>=1.5, libuv>=1.18,
# fuse>=2.9.7, yaml-0.1, libnl-3.0, numa>=2.0 and sensors/sensors.h are each a hard
# AC_MSG_ERROR, so a missing one fails ./configure a quarter of an hour into the build.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      autoconf automake binutils build-essential ca-certificates git \
      libcurl4-openssl-dev libhwloc-dev libjson-c-dev libnl-3-dev libnl-genl-3-dev \
      libnuma-dev libtool libyaml-dev make patchelf pkg-config uuid-dev \
      libconfig-dev libfuse-dev libuv1-dev libsensors-dev \
 && rm -rf /var/lib/apt/lists/*

# The uAPI headers libcxi compiles against, installed headers-only, from TWO repos.
# cassini-headers carries the hardware definitions; `uapi/misc/cxi.h`, which libcxi.h
# includes on its first line, is the KERNEL DRIVER's and lives in shs-cxi-driver. Without
# the second the configure passes and every .lo fails with `uapi/misc/cxi.h: No such file`,
# a quarter of an hour in, so the test names the miss at the copy instead.
# csr_defs.json is data, not a header: libcxi generates utils/cxi_dump_csrs.h from it during
# make, and its loader only looks at /usr/share/cassini-headers or a sibling install tree.
WORKDIR /tmp/build
RUN set -eux; \
    gitretry clone --branch "${SHS_REF}" --depth 1 \
      https://github.com/HewlettPackard/shs-cassini-headers.git cassini-headers; \
    git -C cassini-headers rev-parse HEAD > /tmp/cassini-headers.commit; \
    mkdir -p /opt/ofi/include; \
    cp -a cassini-headers/include/. /opt/ofi/include/; \
    gitretry clone --branch "${SHS_REF}" --depth 1 \
      https://github.com/HewlettPackard/shs-cxi-driver.git cxi-driver; \
    git -C cxi-driver rev-parse HEAD > /tmp/cxi-driver.commit; \
    cp -a cxi-driver/include/uapi /opt/ofi/include/; \
    test -f /opt/ofi/include/uapi/misc/cxi.h; \
    mkdir -p /usr/share/cassini-headers; \
    cp -a cassini-headers/share/cassini-headers/csr_defs.json \
      /usr/share/cassini-headers/

RUN set -eux; \
    gitretry clone --branch "${SHS_REF}" --depth 1 \
      https://github.com/HewlettPackard/shs-libcxi.git libcxi; \
    cd libcxi; \
    git rev-parse HEAD > /tmp/libcxi.commit; \
    ./autogen.sh; \
    CPPFLAGS='-I/opt/ofi/include' ./configure --prefix=/opt/ofi \
      --disable-static; \
    make -j"$(nproc)"; \
    make install; \
    test -e /opt/ofi/lib/libcxi.so.1

# libfabric with the cxi provider compiled IN. --disable-verbs and friends keep everything else
# out: this build exists to reach Slingshot, and a provider the cluster does not have is a
# dependency the image does not need.
RUN set -eux; \
    gitretry clone --branch "${LIBFABRIC_REF}" --depth 1 https://github.com/ofiwg/libfabric.git libfabric; \
    cd libfabric; \
    git rev-parse HEAD > /tmp/libfabric.commit; \
    ./autogen.sh; \
    PKG_CONFIG_PATH=/opt/ofi/lib/pkgconfig \
    CPPFLAGS='-I/opt/ofi/include' \
    LDFLAGS='-L/opt/ofi/lib -Wl,-rpath,/opt/ofi/lib' \
      ./configure --prefix=/opt/ofi --enable-cxi=/opt/ofi \
        --disable-verbs --disable-psm3 --disable-efa --disable-usnic --disable-opx \
        --enable-sockets --enable-tcp; \
    make -j"$(nproc)"; \
    make install; \
    test -e /opt/ofi/lib/libfabric.so.1; \
    LD_LIBRARY_PATH=/opt/ofi/lib /opt/ofi/bin/fi_info -l | grep -q cxi

RUN set -eux; \
    gitretry clone --branch "${AWS_OFI_NCCL_REF}" --depth 1 https://github.com/aws/aws-ofi-nccl.git plugin; \
    cd plugin; \
    git rev-parse HEAD > /tmp/aws-ofi-nccl.commit; \
    ./autogen.sh; \
    env \
      PKG_CONFIG_PATH=/opt/ofi/lib/pkgconfig \
      LD_LIBRARY_PATH=/opt/ofi/lib:/opt/rocm/lib \
      CPPFLAGS='-I/opt/ofi/include' \
      LDFLAGS='-L/opt/ofi/lib -Wl,-rpath,/opt/ofi/lib' \
      ./configure --prefix=/opt/ofi --with-libfabric=/opt/ofi --with-rocm=/opt/rocm \
        --disable-tests; \
    make -j"$(nproc)"; \
    make install

# RCCL loads libnccl-net.so; the plugin installs as librccl-net.so. Unlike the vLLM image the rpath
# is KEPT: here /opt/ofi is the runtime location, not a builder staging dir, and the plugin must
# find the libfabric and libcxi that ship beside it rather than any host copy. The last check is
# what proves the stack is self-contained -- nothing may resolve outside the image.
RUN set -eux; \
    test -e /opt/ofi/lib/librccl-net.so; \
    ln -sfn librccl-net.so /opt/ofi/lib/libnccl-net.so; \
    LD_LIBRARY_PATH=/opt/ofi/lib:/opt/rocm/lib ldd -r /opt/ofi/lib/librccl-net.so > /tmp/plugin-ldd.txt; \
    cat /tmp/plugin-ldd.txt; \
    ! grep -E 'not found|undefined symbol' /tmp/plugin-ldd.txt; \
    ! grep -E '/opt/cray|/usr/lib64/libcxi' /tmp/plugin-ldd.txt

RUN { \
      echo "base_image=${BASE_IMAGE}"; \
      echo "shs_ref=${SHS_REF}"; \
      printf 'cassini_headers_commit='; cat /tmp/cassini-headers.commit; \
      printf 'libcxi_commit='; cat /tmp/libcxi.commit; \
      echo "libfabric_ref=${LIBFABRIC_REF}"; \
      printf 'libfabric_commit='; cat /tmp/libfabric.commit; \
      echo "aws_ofi_nccl_ref=${AWS_OFI_NCCL_REF}"; \
      printf 'aws_ofi_nccl_commit='; cat /tmp/aws-ofi-nccl.commit; \
      echo 'host_contract=libfabric_2.3.1_libcxi_1.0.2_SHS13.1.0_glibc_2.38'; \
      echo 'image_glibc=2.35_so_the_host_stack_cannot_be_mounted'; \
    } > /opt/ofi/BUILD-MANIFEST.txt

# ------------------------------------------------------------------------------------------------
# Stage 2: the served image.
# ------------------------------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS final
USER root

# flydsl, PINNED and INSIDE the image.
#
# The base already carries 0.3.1; what the arms actually load is 0.3.2, installed by
# install-flydsl.sbatch into ${SCRATCH}/pyprefix/sglang-rocm-mi30x and reached through PYTHONPATH
# in .env.llr40v9-kimi27sglang-*. So the out-of-image prefix was an UPGRADE shadowing the image's
# copy, not a missing package -- and an upgrade reached that way is invisible to the image digest,
# which is exactly the attribution problem this consolidation exists to kill.
#
# It matters because kimi's int4 MoE dispatches through it: AMD measure fused_moe at 87.8-89.7% of
# GPU time on this shape, and their W4A16 FlyDSL kernel at +162% throughput / -47% TTFT. aiter
# reaches it when AITER_USE_FLYDSL_MOE_SORTING=1 and the package imports.
#
# The wheel, not a source build: FlyDSL ships its own LLVM/MLIR and embeds the MLIR python
# bindings, and pointing MLIR_PATH at a foreign one is a documented build failure. There is a
# cp310 manylinux_2_27 wheel, which is this image's interpreter.
ARG FLYDSL_VERSION=0.3.2
RUN set -eux; \
    /opt/venv/bin/python3 -m pip install --no-cache-dir "flydsl==${FLYDSL_VERSION}"; \
    /opt/venv/bin/python3 -c 'import flydsl; assert flydsl.__version__ == "'"${FLYDSL_VERSION}"'", flydsl.__version__'

COPY --from=ofi-builder /opt/ofi /opt/ofi

# Tuned fused_moe Triton configs. Losing these once voided a whole set of throughput numbers, so
# they are baked rather than dropped: without them kimi served on the default MoE config at ~90
# tok/s aggregate, ~11x off the reference for its shape (595040/595049).
#
# TWO destinations, deliberately. /opt/moe-configs is the canonical copy, in the same place the
# vLLM image keeps it, so one path finds them in either image. The second is SGLang's OWN lookup:
# get_moe_configs reads ${SGLANG_MOE_CONFIG_DIR:-dirname(__file__)}/configs/triton_<ver>/<name>.json
# with this image's triton being 3.7.0. Copying INTO the vendor tree rather than setting
# SGLANG_MOE_CONFIG_DIR is the point -- that variable REPLACES the search root, so setting it would
# silently discard every config the vendor tuned.
#
# Unverified: the filenames are vLLM's benchmark_moe convention, and whether SGLang's own
# get_config_file_name asks for the same string has not been checked on a live model. kimi's int4
# path goes through aiter/flydsl, not SGLang's triton fused_moe, so this may never be consulted.
# Baked anyway -- a config that exists nowhere is a config that is gone.
COPY containers/cluster/ce-images/inference/moe-configs/ /opt/moe-configs/
RUN set -eux; \
    d=/sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs; \
    test -d "${d}"; \
    mkdir -p "${d}/triton_3_7_0"; \
    cp /opt/moe-configs/E=*.json "${d}/triton_3_7_0/"; \
    ls "${d}/triton_3_7_0/"

# aiter's JIT, built HERE instead of on the first request.
#
# SGLANG_USE_AITER=1 is set by every live kimi and qwen38 arm, so unlike the vLLM image this engine
# runs aiter ON -- and aiter ships no prebuilt .so. It JIT-builds module_aiter_core on import and
# each op module on FIRST USE, behind a baton lock that outlives the deadline: 610164 blew the
# 300 s watchdog that way, and smoke-kimi-sglang.sbatch works around it by pointing AITER_JIT_DIR
# at persistent scratch. The wedge is a property of the image, so it is fixed in the image.
#
# A GPU IS needed, since aiter 0.1.19: `import aiter` runs utility/dtypes.py at module level, which
# calls get_gfx_runtime(), which shells out to rocminfo and ignores GPU_ARCHS by design (its own
# docstring says so). v0.1.13 used get_gfx() and read the env, which is why this step used to run
# device-less. build.sh hands podman /dev/kfd; without it this step dies on `rocminfo returned
# non-zero exit status 1`. A launch is still not issued here, so this proves the modules built,
# not that they run.
#
# AITER_JIT_DIR points INSIDE the image so a hit is a read off the squashfs. A miss still writes,
# into the per-job CE overlay, so an op this prebuild did not cover degrades to today's cold-cache
# behaviour rather than failing. run_cluster.sh reads it as ${AITER_JIT_DIR:-...}, so this value
# wins over its scratch default without a launcher change.
ENV AITER_JIT_DIR=/opt/aiter-jit \
    GPU_ARCHS=gfx942
RUN mkdir -p /opt/aiter-jit /tmp/aiter-home
RUN HOME=/tmp/aiter-home /opt/venv/bin/python3 <<'PY'
import aiter

print("aiter core built at", aiter.__file__, flush=True)
# Each import is guarded because coverage varies by aiter build and a family this release does not
# carry must not abort the image build.
for name in ("aiter.ops.rmsnorm", "aiter.ops.norm", "aiter.ops.gemm_op_a8w8", "aiter.ops.moe_op",
             "aiter.ops.mha", "aiter.ops.attention", "aiter.ops.rope", "aiter.ops.quant",
             "aiter.ops.shuffle", "aiter.fused_moe"):
    try:
        __import__(name)
        print("built:", name, flush=True)
    except Exception as exc:
        print(f"skip {name}: {type(exc).__name__}: {exc}", flush=True)
PY
# A prebuild that produced nothing is the wedge with extra steps, so it fails the build.
RUN set -eux; \
    find /opt/aiter-jit -name '*.so' | wc -l | xargs printf 'aiter prebuilt modules: %s\n'; \
    test -n "$(find /opt/aiter-jit -name '*.so' -print -quit)"; \
    rm -rf /tmp/aiter-home

# The EDF's env, restated in the image so it is under the digest rather than only in ~/.edf.
#
# LD_LIBRARY_PATH deliberately does NOT include /opt/ofi/lib, and NCCL_NET_PLUGIN is deliberately
# unset: every measured kimi and qwen38 number ran with cross-node RCCL on TCP over hsn*, and
# turning the plugin on is a transport change that has to be measured, not a default. To enable it,
# an EDF adds the same stanza the vLLM one carries:
#   LD_LIBRARY_PATH = "/opt/ofi/lib:/opt/rocm/lib:/opt/rocm/lib64"
#   NCCL_NET_PLUGIN = "ofi"   NCCL_NET = "AWS Libfabric"   FI_PROVIDER = "cxi"
#
# PATH keeps the image's own ordering, which omits /opt/venv/bin. That omission is why
# run_cluster.sh names SGLANG_PYTHON=/opt/venv/bin/python3 explicitly: a bare python3 here has
# neither sglang nor huggingface_hub, and the snapshot resolution dies with model_path empty AFTER
# the whole allocation is up. Left as the vendor set it rather than "fixed", because the launcher
# already depends on naming the interpreter and a silently different PATH is a worse surprise.
ENV HSA_ENABLE_IPC_MODE_LEGACY=1 \
    SAFETENSORS_FAST_GPU=1 \
    HIP_FORCE_DEV_KERNARG=1 \
    TOKENIZERS_PARALLELISM=false \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64 \
    PATH=/usr/local/bin:/opt/rocm/bin:/usr/sbin:/usr/bin:/sbin:/bin

# TWO SERVING FACTS that no ENV can carry, because they are per-model CLI arguments. They live in
# SGLANG_EXTRA_ARGS in the env files and must survive any edit of those:
#
#   * BOTH --reasoning-parser AND --tool-call-parser are required. With only one, turn 1 comes back
#     400 and the launcher logs it as SUCCESS. kimi uses `kimi_k2` for both; qwen38 uses `qwen3`
#     and `qwen3_coder`. There is no --enable-auto-tool-choice on SGLang -- that is a vLLM flag.
#   * --mem-fraction-static runs BACKWARDS on an APU. It sizes KV against NODE-WIDE RAM per rank,
#     so the value is LOWERED, not raised: 0.42 for kimi, 0.18 for qwen38.

RUN /opt/venv/bin/python3 <<'PY'
import json
import pathlib
import subprocess
import sys

import flydsl
import huggingface_hub
import sglang
import torch
import triton

print("torch", torch.__version__, "triton", triton.__version__, file=sys.stderr)
print("flydsl", flydsl.__version__, "hf_hub", huggingface_hub.__version__, file=sys.stderr)
assert torch.version.hip is not None, "not a ROCm torch"
# __path__, not __file__: sglang imports as a namespace/editable package here, so __file__ is None
# and the old check died on NoneType rather than on a wrong location (job 619990). __path__ answers
# the question that was actually being asked -- which tree this import resolves to -- for a regular
# package, a namespace package and a PEP 660 editable install alike. Printed as well as asserted,
# so a future move shows the path instead of only the failure.
sglang_paths = [str(p) for p in sglang.__path__]
print("sglang", sglang_paths, file=sys.stderr)
assert any(p.startswith("/sgl-workspace/sglang/") for p in sglang_paths), sglang_paths
subprocess.run([sys.executable, "-m", "sglang.launch_server", "--help"], check=True,
               stdout=subprocess.DEVNULL)

# The baked-in inputs, asserted rather than assumed: a COPY that silently landed nothing is how a
# tuned-MoE folder goes missing again.
configs = sorted(pathlib.Path("/opt/moe-configs").glob("E=*.json"))
assert len(configs) >= 2, f"tuned MoE configs missing: {configs}"
for c in configs:
    json.loads(c.read_text())
vendor = pathlib.Path("/sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils"
                      "/configs")
assert sorted(p.name for p in (vendor / "triton_3_7_0").glob("E=*.json")) == \
    sorted(c.name for c in configs), "tuned configs did not reach SGLang's lookup path"
assert (vendor / "triton_3_2_0").is_dir(), "the vendor config set was displaced"

for lib in ("libnccl-net.so", "librccl-net.so", "libfabric.so.1", "libcxi.so.1"):
    assert (pathlib.Path("/opt/ofi/lib") / lib).exists(), f"OFI stack incomplete: {lib}"
PY

WORKDIR /workspace

# LIMITATION -- read before trusting this file.
#
# A podman build has no GPUs and no Slingshot NIC, so nothing below the import level is proven:
#
#   * that the self-contained CXI stack can open a real /dev/cxi. libcxi is a shim over the kernel
#     driver's ioctl uAPI, so SHS_REF must match the host driver. It is pinned to release/shs-13.1.0
#     because the login node reports cray-libcxi-1.0.2-SHS13.1.0; a mismatch shows up as an ioctl
#     failure at fi_getinfo, which only a two-node RCCL run will reveal. This is the highest-risk
#     stage in the file and it has never been built.
#   * that the aiter modules prebuilt above LAUNCH.
#   * that SGLang serves. Use inference/smoke-kimi-sglang.sbatch before promoting.
#
# Until it has been built AND smoke-tested, the pulled image named in ~/.edf/sglang-rocm-mi30x.toml
# remains the authority and this file is a candidate, not a replacement.
