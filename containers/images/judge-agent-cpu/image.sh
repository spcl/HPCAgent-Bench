# Build inputs of the CPU-only judge and agent images, sourced by ../build.sh, for the CPU family of the
# node that builds it (x86_64 or aarch64, the portable cpu_target.env baseline unless
# CE_IMAGE_FLAVOR=native). `judge` is `FROM agent` plus the hpcagent_bench install.

# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
TARGET_ORDER="agent judge"
TARGET_ROLE=([agent]=judge-agent-cpu [judge]=judge-cpu)
BASE_IMAGE="${BASE_IMAGE:-docker.io/library/ubuntu:24.04@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3}"

ce_image_args() {
    ce_spack_target
    ce_march
    ce_dace_commit
    IMAGE_VERSION="${IMAGE_VERSION:-dev}"
    ce_build_args IMAGE_VERSION SPACK_TARGET MARCH
}

# Base cache, spack buildcache and uv cache per CPU family: an x86_64 layer is no use to aarch64.
ce_image_inputs() {
    ce_require_kernelbench
    ce_mirror_args
    ce_require_mirror_commit "spcl/dace.git" "${DACE_COMMIT}"
    BASE_CACHE="${BASE_CACHE:-${SCRATCH:?}/base-images-$(uname -m)}"
    ce_cache_base_image
    ce_cache_args "spack-buildcache-$(uname -m)" uv-cache
    INPUT_ARGS=("${MIRROR_ARGS[@]}" "${CACHE_ARGS[@]}")
}
