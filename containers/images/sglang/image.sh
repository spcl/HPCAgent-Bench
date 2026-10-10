# Build inputs of the AMD SGLang serving image, sourced by ../build.sh: device code for the GPU arch of
# EVERY partition gpu_arch.env names (MI300A gfx942, MI250X gfx90a), one image serving both. Not the
# portable AMD_GPU_TARGETS list: sgl_kernel picks one FP8 type for the whole binary (FNUZ, gfx942's),
# which would be wrong on gfx950. Built on mi300: the aiter prebuild launches on the build node's GPU.

# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
TARGET_ORDER="-"
TARGET_ROLE=([-]=sglang)
BASE_IMAGE="${BASE_IMAGE:-$(ce_dockerfile_base "${IMAGE_DIR}/Dockerfile")}"

ce_image_args() {
    ce_partition_targets
    ce_build_args ROCM_ARCH ROCM_ARCH_CSV
}

ce_image_inputs() {
    ce_mirror_args
    ce_gpu_args
    ce_cache_base_image
    INPUT_ARGS=("${MIRROR_ARGS[@]}" "${GPU_ARGS[@]}")
}
