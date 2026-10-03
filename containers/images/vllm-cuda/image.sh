# Build inputs of the GH200 vLLM serving image, sourced by ../build.sh: aarch64 only, always latest.

# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
ce_require_arch aarch64
TARGET_ORDER="-"
TARGET_ROLE=([-]=vllm-cuda)
BASE_IMAGE="${BASE_IMAGE:-docker.io/vllm/vllm-openai:v0.28.0-aarch64-cu129@sha256:60fa2715937e604931086a790fff2978c09995eff93439261ba09a79f02e9e68}"

ce_image_args() {
    IMAGE_VERSION="${IMAGE_VERSION:-dev}"
    ce_build_args IMAGE_VERSION
}

ce_image_inputs() {
    ce_mirror_args
    ce_cache_base_image
    INPUT_ARGS=("${MIRROR_ARGS[@]}")
}
