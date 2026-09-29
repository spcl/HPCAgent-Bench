# Build inputs of the AMD SGLang serving image, sourced by ../build.sh: built for the GPU arch of the
# partition that builds it (gpu_arch.env), always the latest flavor.

# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
TARGET_ORDER="-"
TARGET_ROLE=([-]=sglang)
BASE_IMAGE="${BASE_IMAGE:-$(ce_dockerfile_base "${IMAGE_DIR}/Dockerfile")}"

ce_image_args() {
    ce_gpu_arch
    ce_build_args ROCM_ARCH
}

ce_image_inputs() {
    ce_mirror_args
    ce_gpu_args
    ce_cache_base_image
    INPUT_ARGS=("${MIRROR_ARGS[@]}" "${GPU_ARGS[@]}")
}
