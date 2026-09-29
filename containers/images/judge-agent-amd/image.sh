# Build inputs of the AMD judge and agent images, sourced by ../build.sh. One portable image for every
# AMD partition: device code for gpu_arch.env's AMD targets and cpu_target.env's baseline CPU, whichever
# partition builds it. `judge` is `FROM agent` plus the hpcagent_bench install, so both targets cost
# the agent build plus a pip layer. The build context is the repository root: the Dockerfile COPYs
# pyproject.toml and the harness build inputs from agent/harness; tool scripts are bound at launch.

# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
TARGET_ORDER="agent judge"
TARGET_ROLE=([agent]=judge-agent-amd [judge]=judge)
# Pinned by DIGEST: the bare tag would quietly unpin the build and the base.name label would record a
# mutable reference.
BASE_IMAGE="${BASE_IMAGE:-docker.io/rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1@sha256:a3b65813621095e3389269417e963725b59310184588c9d2490d44e6e83fa01c}"

ce_image_args() {
    ce_amd_targets
    ce_spack_target
    ce_dace_commit
    ce_libfabric_commit
    IMAGE_VERSION="${IMAGE_VERSION:-dev}"
    ce_build_args IMAGE_VERSION DACE_COMMIT LIBFABRIC_REF LIBFABRIC_COMMIT ROCM_ARCH ROCM_ARCH_CSV SPACK_TARGET
}

# The 30 GB rocm/pytorch base comes from the scratch cache; spack's binary buildcache (gcc 16 and
# llvm 22 are 60-80 minutes) and the pip wheel cache survive between jobs outside the image. The pip
# cache is keyed by the GPU target list: pip keys a built wheel by its sdist, not HCC_AMDGPU_TARGET.
ce_image_inputs() {
    ce_mirror_args
    ce_require_mirror_commit "spcl/dace.git" "${DACE_COMMIT}"
    ce_require_mirror_commit "ofiwg/libfabric.git" "${LIBFABRIC_COMMIT}"
    ce_cache_base_image
    ce_cache_args spack-buildcache "pip-cache/${ROCM_ARCH//;/-}"
    INPUT_ARGS=("${MIRROR_ARGS[@]}" "${CACHE_ARGS[@]}")
}
