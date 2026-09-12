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
# UBUNTU 24.04, and that is why this tag rather than the newest 7.2.0 one.
#
# glibc decides everything here. The CE cxi hook injects the host Slingshot stack -- libfabric
# 2.3.1, libcxi 1.0.2, SHS 13.1.0 -- built against glibc 2.38. Ubuntu 24.04 is glibc 2.39, so
# that stack loads and this image takes libfabric and libcxi from the host like the other two.
# On the previous 22.04 base (glibc 2.35) it could not, and every workaround failed: building a
# libfabric of our own linked libcurl, which resolved to the host's and needed 2.38 anyway
# (629822, "Failed to initialize any NET plugin").
#
# THE COST: this line is ROCm 7.2.4 / torch 2.10.0 against the 7.2.0 / 2.9.1 the judge and vllm
# images pin -- lmsysorg publishes exactly one 24.04 line and it is 7.2.4. Inference runs in its
# own container and shares no library with the judge, so a newer ROCm is contained. But every
# recorded kimi and qwen38 serving number was taken on 7.2.0 / 2.9.1 / aiter 0.1.19, and this
# moves all three at once. Re-measure before comparing against any of them.
#
# Pinned by DIGEST because a date-stamped tag is still a tag.
ARG BASE_IMAGE=docker.io/lmsysorg/sglang-rocm:v0.5.19-rocm724-mi30x-20260908\
@sha256:0405baaf36945fa8164c57d4f1b6b178bae5804fae606ff2db3816a6cab6dafc

# ------------------------------------------------------------------------------------------------
# One stage. This image builds NO fabric software at all -- no libfabric, no libcxi, no RCCL net
# plugin. CSCS ships all three as one internally consistent artifact, matched to the host driver,
# and the enroot hooks install it when the EDF sets com.hooks.netstack / .cxi / .aws_ofi_nccl.
# Measured on an unmodified image in job 629967:
#
#     plugin:    /opt/cscs/netstack/librccl-net.so
#     transport: NET/OFI, "Using network AWS Libfabric"
#     NET/AWS Libfabric : GPU Direct RDMA Enabled for HCA 0 'cxi0' .. 'cxi2'
#
# Nothing here links against libfabric either -- sglang reaches the fabric through RCCL, not MPI,
# so unlike judge-agent-amd there is not even a compile-time link target to produce. A gate below
# asserts the shipped image carries none of the three, because any copy would shadow the artifact.
# ------------------------------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS final
USER root

# OCI metadata, because these are published as OCI artifacts and a registry shows nothing about
# where an image came from unless the image says so. base.name records what FROM resolved to
# WITHOUT unpacking the image -- and since it is pinned by digest above, that is the whole
# reproducibility claim in one label. BASE_IMAGE is re-declared here because an ARG from before
# the first FROM is not in scope inside a stage.
ARG BASE_IMAGE
# The registry reference, not whatever FROM resolved to: a cache hit rewrites BASE_IMAGE to a
# local dir: path, and that must not become the published base.name.
ARG BASE_IMAGE_REF="${BASE_IMAGE}"
LABEL org.opencontainers.image.title="optarena-sglang" \
      org.opencontainers.image.description="SGLang inference for MI300A: ROCm 7.2.4, CSCS netstack fabric, pinned flydsl" \
      org.opencontainers.image.version="0.5.19" \
      org.opencontainers.image.source="https://github.com/spcl/HPCAgent-Bench" \
      org.opencontainers.image.licenses="GPL-3.0-or-later" \
      org.opencontainers.image.base.name="${BASE_IMAGE_REF}"

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

# cupy, IN the image, for the same reason flydsl is.
#
# It was reaching the sglang arms through PYTHONPATH=${SCRATCH}/pyprefix/sglang-rocm-mi30x, a host
# directory supplying cupy and flydsl 0.3.2. Measured: with that prefix injected both resolve
# under /capstor, so what an arm actually ran was invisible to the image digest -- the attribution
# hole this image exists to close. It survived because the base ships flydsl 0.3.1 and no cupy at
# all, so the prefix was genuinely load-bearing. With both baked it has nothing left to add.
#
# From SOURCE: there is no ROCm wheel, the cupy-rocm-* names are dead and PyPI carries only the
# sdist. Same recipe as judge-agent-amd, which builds cupy 14.2.0 this way.
#
# numpy is CONSTRAINED to what is installed and the build fails if it moved: the sdist would
# otherwise pull its own, and numpy is what served results are compared against.
ARG CUPY_VERSION=14.2.0
ARG ROCM_ARCH=gfx942
RUN set -eux; \
    numpy_before="$(/opt/venv/bin/python3 -c 'import numpy; print(numpy.__version__)')"; \
    printf 'numpy==%s\n' "${numpy_before}" > /tmp/constraint.txt; \
    CUPY_INSTALL_USE_HIP=1 ROCM_HOME=/opt/rocm HCC_AMDGPU_TARGET="${ROCM_ARCH}" \
    CUPY_NUM_BUILD_JOBS="$(nproc)" PIP_CONSTRAINT=/tmp/constraint.txt \
      /opt/venv/bin/python3 -m pip install --no-cache-dir "cupy==${CUPY_VERSION}"; \
    rm -f /tmp/constraint.txt; \
    test "$(/opt/venv/bin/python3 -c 'import numpy; print(numpy.__version__)')" = "${numpy_before}" \
      || { echo "cupy moved numpy off ${numpy_before}" >&2; exit 2; }; \
    /opt/venv/bin/python3 -c "import cupy; print('cupy', cupy.__version__)"

# Gate: both must resolve INSIDE the image. This is what lets the PYTHONPATH be deleted from the
# campaign .env files rather than quietly remaining load-bearing.
RUN /opt/venv/bin/python3 -c "import cupy, flydsl; \
    assert cupy.__file__.startswith('/opt/'), cupy.__file__; \
    assert flydsl.__file__.startswith('/opt/'), flydsl.__file__; \
    print('self-contained: cupy', cupy.__version__, '| flydsl', flydsl.__version__)"

# No RCCL plugin is shipped. CSCS provides one through com.hooks.aws_ofi_nccl.enabled and job
# 629967 measured it working: /opt/cscs/netstack/librccl-net.so, "Using network AWS Libfabric",
# GPU Direct RDMA on cxi0-2. This image ships no libfabric, libcxi or plugin -- all three come
# from the pinned netstack artifact, which also carries its own libc and libcurl and is therefore
# internally consistent, unlike the host graft that broke this image before (629822).
# Ship no fabric library OF OUR OWN. The distinction matters and job 629985 is why: the gate used
# to refuse ANY libfabric anywhere on the filesystem, and it killed the build on
# /usr/lib/x86_64-linux-gnu/libfabric.so.1.20.0 -- which is not this image's file at all. That path
# is where the CSCS hook BIND-MOUNTS the artifact's libfabric at run time, verified by mounting the
# image with no annotations: the file that appears there is dated Aug 18 15:39 nobody:nogroup, the
# artifact's own stamp, and lsof/mountinfo show it as a lustre bind from
# /capstor/store/cscs/.../netstack. Refusing it forbade the hook its own mount point.
#
# What must NOT ship is a fabric library in a prefix WE control, because those are what RPATH can
# bind to and RPATH is searched before everything else -- the 629966 failure, where MPICH held a
# providerless libfabric and neither the hook nor LD_LIBRARY_PATH could displace it.
RUN set -eux; \
    rm -f /opt/ofi-sdk/lib/libfabric.so*; \
    found="$(find /opt/ofi-sdk /opt/ofi /opt/aws-ofi-nccl \
                  \( -name 'libfabric.so*' -o -name 'libcxi.so*' -o -name 'librccl-net.so*' \) \
                  2>/dev/null || true)"; \
    [ -z "${found}" ] \
      || { echo "this image ships a fabric library in a prefix it controls:" >&2; \
           echo "${found}" >&2; \
           echo "RPATH would bind to it and shadow the CSCS netstack artifact" >&2; exit 1; }; \
    echo "no fabric library in /opt/ofi-sdk, /opt/ofi or /opt/aws-ofi-nccl -- the artifact supplies them"

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
# non-zero exit status 1`. The op calls below DO issue launches, which is what builds the op
# modules at all -- /dev/dri and the seccomp/keep-groups flags in build_common.sh are what
# make that work, and the gate below fails the build if they ever stop working.
#
# AITER_JIT_DIR points INSIDE the image so a hit is a read off the squashfs. A miss still writes,
# into the per-job CE overlay, so an op this prebuild did not cover degrades to today's cold-cache
# behaviour rather than failing. run_cluster.sh reads it as ${AITER_JIT_DIR:-...}, so this value
# wins over its scratch default without a launcher change.
ENV AITER_JIT_DIR=/opt/aiter-jit \
    AITER_ROOT_DIR=/opt/aiter-root \
    GPU_ARCHS=gfx942
RUN mkdir -p /opt/aiter-jit /opt/aiter-root /tmp/aiter-home
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
        print("imported (a build still needs a call):", name, flush=True)
    except Exception as exc:
        print(f"skip {name}: {type(exc).__name__}: {exc}", flush=True)

# Importing an op family does NOT build its kernel module. @compile_ops is lazy and builds on the
# first CALL, so the loop above only ever produced module_aiter_core.so -- read back off the
# deployed image, which carried exactly that one file. Every other module was still built at serve
# time: module_rmsnorm_quant cost 74 s in 628158 and the sampling template ops another 77 s.
# Measured in 628179, these calls build them here in 78 s total, and the sampling artefact hashes
# match the ones the server asks for, so a serve is a cache hit rather than a near-miss.
import os
import pathlib

import torch

_dev = torch.device("cuda")
_jit = pathlib.Path(os.environ["AITER_JIT_DIR"])


def _prebuild(label, fn):
    # Report the module each call produced. 628185 built four .so and still failed a gate that
    # asked for module_rmsnorm_quant.so by name: the three images here carry different aiter
    # builds and the names do not match across them.
    before = {p.name for p in _jit.rglob("*.so")}
    try:
        fn()
    except Exception as exc:
        print(f"skip {label}: {type(exc).__name__}: {exc}", flush=True)
        return
    print("prebuilt:", label, "->", sorted({p.name for p in _jit.rglob("*.so")} - before), flush=True)


def _rms_norm():
    # 2-D, bf16, hidden <= 8192 is the case that dispatches to the hand-tuned HIP
    # module_rmsnorm_quant; anything else takes the generic opus path and builds nothing.
    x = torch.randn(4, 7168, dtype=torch.bfloat16, device=_dev)
    w = torch.ones(7168, dtype=torch.bfloat16, device=_dev)
    aiter.rms_norm(x, w, 1e-6)


def _probs():
    return torch.softmax(torch.randn(4, 1024, dtype=torch.float32, device=_dev), dim=-1).contiguous()


def _top_k_top_p():
    import aiter.ops.sampling  # noqa: F401
    k = torch.full((4,), 50, dtype=torch.int32, device=_dev)
    p = torch.full((4,), 0.9, dtype=torch.float32, device=_dev)
    torch.ops.aiter.top_k_top_p_sampling_from_probs(_probs(), None, k, 0, p, 0.0, deterministic=True)


def _top_p():
    p = torch.full((4,), 0.9, dtype=torch.float32, device=_dev)
    torch.ops.aiter.top_p_sampling_from_probs(_probs(), None, p, 0.0, deterministic=True)


def _top_k_renorm():
    k = torch.full((4,), 50, dtype=torch.int32, device=_dev)
    torch.ops.aiter.top_k_renorm_probs(_probs(), k, 0)


# Signatures and deterministic=True come from the engine call site, so what is prebuilt is what
# gets asked for. Each is guarded: an op a release does not carry must degrade to a runtime build.
_prebuild("rms_norm", _rms_norm)
_prebuild("top_k_top_p_sampling_from_probs", _top_k_top_p)
_prebuild("top_p_sampling_from_probs", _top_p)
_prebuild("top_k_renorm_probs", _top_k_renorm)
# The serving path calls far more than sampling. These eight were read off what GLM-5.3 and kimi
# actually JIT-built at serve time (628603 built module_custom_all_reduce during startup), so the
# image ships them instead of paying for them behind a baton lock inside the RPC deadline.
_bf16, _fp8 = torch.bfloat16, torch.float8_e4m3fnuz


def _layernorm2d():
    n = 7168
    x = torch.randn(4, n, dtype=_bf16, device=_dev)
    aiter.layernorm2d_fwd_with_add(torch.empty_like(x), x, x.clone(), torch.empty_like(x),
                                   torch.ones(n, dtype=_bf16, device=_dev),
                                   torch.zeros(n, dtype=_bf16, device=_dev), 1e-6)


def _dynamic_quant():
    x = torch.randn(4, 7168, dtype=_bf16, device=_dev)
    aiter.dynamic_per_tensor_quant(torch.empty(x.shape, dtype=_fp8, device=_dev), x,
                                   torch.ones(1, dtype=torch.float32, device=_dev))


def _silu_and_mul():
    x = torch.randn(4, 2 * 7168, dtype=_bf16, device=_dev)
    aiter.silu_and_mul(torch.empty(4, 7168, dtype=_bf16, device=_dev), x)


def _reshape_and_cache():
    k = torch.randn(4, 8, 128, dtype=_bf16, device=_dev)
    kc = torch.zeros(64, 8, 128, 16, dtype=_bf16, device=_dev)
    aiter.reshape_and_cache(k, k.clone(), kc, kc.clone(),
                            torch.arange(4, dtype=torch.int64, device=_dev), "auto")


def _moe_sum():
    # topk MUST be one of 2/4/5. Any other value takes a python-side torch.sum branch that never
    # reaches the kernel: it would build nothing while _prebuild reported success.
    x = torch.randn(8, 4, 7168, dtype=_bf16, device=_dev)
    aiter.moe_sum(x, torch.empty(8, 7168, dtype=_bf16, device=_dev))


def _greedy_sample():
    logits = torch.randn(4, 1024, dtype=torch.float32, device=_dev)
    aiter.greedy_sample(torch.empty(4, dtype=torch.int64, device=_dev), logits)


_prebuild("layernorm2d_fwd_with_add", _layernorm2d)
_prebuild("dynamic_per_tensor_quant", _dynamic_quant)
_prebuild("silu_and_mul", _silu_and_mul)
_prebuild("reshape_and_cache", _reshape_and_cache)
_prebuild("moe_sum", _moe_sum)
_prebuild("greedy_sample", _greedy_sample)
_prebuild("get_padded_m", lambda: aiter.get_padded_m(128, 7168, 7168, 1))
_prebuild("moe_sorting_opus_get_workspace_size",
          lambda: aiter.moe_sorting_opus_get_workspace_size(128, 256, 8))

# The defect this step exists to catch, stated as the build's own assertion: an import-only
# prebuild left module_aiter_core.so and nothing else, and the shell gate below cannot tell that
# apart from a healthy image without naming a module.
_ops = {p.name for p in _jit.rglob("*.so")} - {"module_aiter_core.so"}
if not _ops:
    raise SystemExit("aiter prebuild produced no op module beyond module_aiter_core")
print("aiter op modules prebuilt:", sorted(_ops), flush=True)

PY
# The prebuild step above already fails the build if it produced no op module. What is left for
# the shell is the AITER_ROOT_DIR pin: aiter's template ops default to $HOME/.aiter, so if the pin
# does not hold, HOME grows a .aiter and the artefacts are outside the image.
RUN set -eux; \
    find /opt/aiter-jit -name '*.so' | wc -l | xargs printf 'aiter prebuilt modules: %s\n'; \
    find /opt/aiter-root -mindepth 1 -maxdepth 2 | head -20; \
    test ! -e /tmp/aiter-home/.aiter; \
    rm -rf /tmp/aiter-home

# GLM-5.3 and every other GlmMoeDsa/DeepSeek fp8 checkpoint, fixed IN THE PACKAGE.
#
# sglang's DeepSeek weight loader reads `weight_scale.format_ue8m0` as a plain attribute. fp8.py
# stamps that attribute onto a scale tensor, but on gfx942 the fp8-fnuz branch runs
# normalize_e4m3fn_to_e4m3fnuz first, which returns a FRESH tensor carrying no such attribute, so
# the loader raises AttributeError before the model is up. The branch is AMD-only, which is why it
# ships broken; 0.5.19 and the 20260909 nightly both still carry it.
#
# The fix is the guard the SAME TREE already uses in fp8_utils.py, and False is correct rather than
# merely safe: the branch it guards leads to should_deepgemm_weight_requant_ue8m0, and deepgemm is
# NVIDIA-only, so the block is dead on this hardware.
#
# In the source, not monkeypatched onto torch.Tensor by a host sitecustomize on PYTHONPATH: a host
# path is invisible to the image digest, and the per-role mount policy does not carry it into the
# inference container, so the patch silently did not apply. The build FAILS when the pattern is
# absent, so a release that fixes this upstream is reported here instead of being patched twice.
RUN /opt/venv/bin/python3 <<'PY'
import pathlib
import re
import sys

# A READ of .format_ue8m0. An assignment target (a single = that is not ==) is left alone: fp8.py
# stamps the attribute all over and rewriting those would not parse.
# The receiver may be a dotted CHAIN (layer.weight_scale.format_ue8m0), not just a bare name. A
# pattern that excludes a leading dot misses those, and because the assertion below reuses it, such
# a read would slip through while the build still reported success.
PATTERN = re.compile(r"(?<![\w.])((?:\w+\.)*\w+)\.format_ue8m0\b(?!\s*=(?!=))")
GUARD = r'getattr(\1, "format_ue8m0", False)'

tree = pathlib.Path("/sgl-workspace/sglang/python/sglang")
patched = []
for path in sorted(tree.rglob("*.py")):
    text = path.read_text()
    fixed = PATTERN.sub(GUARD, text)
    if fixed != text:
        path.write_text(fixed)
        patched.append(f"{path.relative_to(tree)}: {len(PATTERN.findall(text))}")
if not patched:
    sys.exit("no unguarded .format_ue8m0 read found; upstream may have fixed it -- drop this step")
print("format_ue8m0 guarded in:", *patched, sep="\n  ", flush=True)
for path in tree.rglob("*.py"):
    assert not PATTERN.search(path.read_text()), path
PY

# The aiter kernel fused_qk_norm_rope_cache_quant.cu does static_cast<float>(__half) while the
# PyTorch extension build defines -D__HIP_NO_HALF_CONVERSIONS__ / -D__HIP_NO_HALF_OPERATORS__,
# which remove __half's conversion and arithmetic operators. hipcc APPENDS this and a later -U
# beats an earlier -D, so the JIT build gets them back without patching the kernel. In the image
# rather than in a second EDF: every checkpoint that JITs that kernel needs it.
ENV HIPCC_COMPILE_FLAGS_APPEND="-U__HIP_NO_HALF_CONVERSIONS__ -U__HIP_NO_HALF_OPERATORS__"

# The EDF's env, restated in the image so it is under the digest rather than only in ~/.edf.
#
# TRANSPORT, named in the image so it is under the digest and not only in somebody's ~/.edf.
# NO transport variables here. libfabric, libcxi and the RCCL plugin all arrive from the CSCS
# netstack artifact via the EDF's hooks, and the plugin the hook installs already knows where its
# libfabric is -- naming NCCL_NET_PLUGIN or FI_PROVIDER in the image would only let a stale value
# outlive the hook. FI_PROVIDER in particular is not free: set globally it reaches MPI too, and
# job 629966 aborted MPI_Init with "OFI call getinfo failed" that way.
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
SGLANG_TREE = "/sgl-workspace/sglang"
assert any(p == SGLANG_TREE or p.startswith(SGLANG_TREE + "/") for p in sglang_paths), sglang_paths
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

# The plugin only. libfabric and libcxi are deliberately NOT here -- they are the two things this
# image takes from the host, through com.hooks.cxi.enabled, because they must match the kernel
# driver. Asserting their presence would assert the bug this image was just moved off.
# No fabric libraries here on purpose: the netstack artifact supplies libfabric, libcxi and the
# RCCL plugin at run time. Asserting ours exist would assert the design this image moved off.
for stray in ("libfabric.so.1", "libcxi.so.1", "librccl-net.so"):
    assert not list(pathlib.Path("/opt").rglob(stray)), f"{stray} ships and would shadow the artifact"
PY

WORKDIR /workspace

# LIMITATION -- read before trusting this file.
#
# The build DOES have GPUs: build_common.sh:ce_gpu_args passes /dev/kfd, /dev/dri, keep-groups and
# an unconfined seccomp profile, so the aiter prebuild above launches kernels and the gate fails
# the build if that ever stops working. What the build has no access to is a Slingshot NIC:
#
#   * the plugin has not yet been shown to select OFI on this base. /dev/cxi0-3 do reach the
#     container and the image loads under the hook (629822), but the transport that a cross-node
#     collective actually chooses is only visible in an NCCL_DEBUG=INFO log from a real two-node
#     run: NET/OFI is the goal, NET/Socket is the silent fallback.
#     is the highest-risk stage in the file.
#   * whether SGLang SERVES is a separate question from whether it imports. Run
#     inference/smoke-kimi-sglang.sbatch before promoting a rebuild.
