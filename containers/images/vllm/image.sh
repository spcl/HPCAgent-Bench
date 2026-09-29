# Build inputs of the AMD vLLM serving image, sourced by ../build.sh: a prebuilt engine with device code
# for every gpu_arch.env AMD target, so always the latest flavor.

# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
TARGET_ORDER="-"
TARGET_ROLE=([-]=vllm)
BASE_IMAGE="${BASE_IMAGE:-$(ce_dockerfile_base "${IMAGE_DIR}/Dockerfile")}"

ce_image_args() {
    ce_amd_targets
    ce_build_args ROCM_ARCH ROCM_ARCH_CSV
}

ce_image_inputs() {
    ce_mirror_args
    ce_gpu_args
    ce_cache_base_image
    INPUT_ARGS=("${MIRROR_ARGS[@]}" "${GPU_ARGS[@]}")
}
