"""Derive GLM-5.3 arm envs from the kimi sglang arms they are the counterfactual of.

Only the serving block differs: same nodes, same agents, same problems, same packet. Anything else
that changed would make an arm comparison a comparison of two things at once.

The GLM launch line is the one smoke 628603 proved, and every deletion from the kimi line is
load-bearing:
  --attention-backend triton   the kimi line pins triton; no backend key is set here at all, so
                               run_cluster.sh's aiter default applies. It reads the key as
                               ${VAR-default}, so only an ASSIGNED EMPTY value omits the flag.
  --language-only              selects the VLM encoder-disaggregation RECEIVER role in this
                               SGLang, and GlmMoeDsaForCausalLM is off its allowlist -- 628586 and
                               628589 both died at launch on exactly this.
  --enable-hierarchical-cache  offloads KV to host memory, which on MI300A is the SAME pool the
                               weights live in. Untested for this checkpoint; the proven config
                               does not set it.
  AITER_USE_FLYDSL_MOE_SORTING kimi's weights are pack-quantized int4; these are fp8, and the
                               smoke that produced the numbers did not set it.
"""

import pathlib
import sys

#: The served window, and what a request has to leave room for: the completion reservation plus
#: one turn, which a compiler log or an asm dump can fill on its own.
GLM_CONTEXT = 131072
COMPLETION_RESERVE = 32000
TURN_HEADROOM = 30000

GLM_ARGS = (
    '"--trust-remote-code --watchdog-timeout 1800 --kv-cache-dtype fp8_e4m3 --page-size 64 '
    "--context-length 131072 --mem-fraction-static 0.588 --cuda-graph-max-bs-decode 64 "
    "--enable-metrics --pre-warm-nccl --reasoning-parser glm45 --tool-call-parser glm47 "
    '--dsa-prefill-backend tilelang --dsa-decode-backend tilelang --enable-cache-report"'
)

# mem-fraction-static is a CEILING on weights+KV together on MI300A, not a KV reservation:
# KV(f) = 443.4 * (f - 0.486) GB/rank at tp=4 pp=4, in EFFECTIVE fraction. aiter attention
# multiplies the flag by 0.85, so the flag carries 0.588 to reach the 0.50 effective the serving
# numbers were taken at. The usable band in flag terms is 0.572 (effective 0.486, below which
# sglang refuses) to 0.729 (effective 0.62, which let the heaviest PP stage at 206.1 GB reach the
# host OOM killer -- the stages are UNEVEN). Never move the flag without the backend, or the other
# way round.
CE_ENV = """
# sglang-candidate is the only sglang EDF whose image can load GLM-5.3: the DeepSeek weight
# loader's format_ue8m0 reads are guarded in the package at build time and
# HIPCC_COMPILE_FLAGS_APPEND is a global image ENV. The other sglang EDFs reach the same patch
# through a PYTHONPATH under /capstor, which role_mounts drops for the inference role, so the
# loader dies on format_ue8m0 before the model is up.
INFERENCE_CE_ENV=sglang-candidate
"""

HEADER = """
# --- GLM-5.3 deviations from the kimi arm this env was derived from ------------------
# mem-fraction-static is a CEILING on weights+KV together on MI300A, not a KV reservation:
# KV(f) = 443.4 * (f - 0.486) GB/rank at tp4 x pp4 in EFFECTIVE fraction, so lowering it shrinks KV
# toward zero rather than freeing host memory. aiter attention multiplies the flag by 0.85: the
# flag reads 0.588 for an effective 0.50, and the band is flag 0.572 to 0.729. At the top of it the
# heaviest PP stage OOM-kills; the stages are UNEVEN (172.4/197.2/203.8/206.1 GB), so size against
# the max.
# No --language-only: it selects the VLM encoder-disaggregation receiver role and this
# architecture is off its allowlist, so the server refuses to start.
# GLM-5.3-Flash cannot run here at all: index_kpool in its config forces IndexerKPool, which
# raises "kpool indexer is only supported on CUDA". Plain 5.3 uses the ROCm-capable DSA Indexer.
"""


def derive(src: pathlib.Path, dst: pathlib.Path) -> None:
    out, replaced = [], set()
    for line in src.read_text().splitlines():
        if line.startswith("VLLM_MODEL="):
            line = "VLLM_MODEL=zai-org/GLM-5.3"
            replaced.add("model")
        elif line.startswith("OPTARENA_OPTIMIZER="):
            line = "OPTARENA_OPTIMIZER=zai-org/GLM-5.3"
            replaced.add("optimizer")
        elif line.startswith("AITER_USE_FLYDSL_MOE_SORTING="):
            continue
        elif line.startswith("INFERENCE_CE_ENV="):
            # The kimi source says sglang-latest. Without this branch a regeneration silently
            # dropped GLM back to the shared EDF, which cannot load GLM-5.3 at all -- and the
            # arm would look correct right up until the loader died on format_ue8m0.
            out.extend(CE_ENV.strip("\n").splitlines())
            replaced.add("ce_env")
            continue
        elif line.startswith("CLAUDE_AUTOCOMPACT="):
            # GLM serves HALF the kimi context, so the inherited threshold would sit above the
            # window entirely and every agent would 400 before it could ever compact.
            line = f"CLAUDE_AUTOCOMPACT={GLM_CONTEXT - COMPLETION_RESERVE - TURN_HEADROOM}"
            replaced.add("autocompact")
        elif line.startswith("SGLANG_EXTRA_ARGS="):
            out.extend(HEADER.strip("\n").splitlines())
            line = f"SGLANG_EXTRA_ARGS={GLM_ARGS}"
            replaced.add("args")
        out.append(line)
    missing = {"model", "optimizer", "args", "ce_env", "autocompact"} - replaced
    if missing:
        raise SystemExit(f"{src.name}: never matched {sorted(missing)}")
    dst.write_text("\n".join(out) + "\n")
    print(f"{dst.name}")


def main() -> int:
    here = pathlib.Path(__file__).parent
    for suffix in ("", "-skills"):
        src = here / f".env.llrbase-kimi27sglang-c{suffix}"
        if not src.is_file():
            raise SystemExit(f"missing base {src.name}")
        derive(src, here / f".env.llrbase-glm53-c{suffix}")
    # The GPU and CPF launchers sed .env.base-<model>, a different family from the llrbase one
    # above, so glm53 needs its counterpart there too or it cannot be a model those campaigns name.
    # Same derivation, so the serving block cannot drift between the two families.
    src = here / ".env.base-kimi27sglang"
    if not src.is_file():
        raise SystemExit(f"missing base {src.name}")
    derive(src, here / ".env.base-glm53")
    return 0


if __name__ == "__main__":
    sys.exit(main())
