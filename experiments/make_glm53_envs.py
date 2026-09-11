"""Derive GLM-5.3 arm envs from the kimi sglang arms they are the counterfactual of.

Only the serving block differs: same nodes, same agents, same problems, same packet. Anything else
that changed would make an arm comparison a comparison of two things at once.

The GLM launch line is the one smoke 628603 proved, and every deletion from the kimi line is
load-bearing:
  --attention-backend triton   GLM-5.3 selects the DSA backend from its own config ("Use dsa
                               attention backend for DeepSeek with DSA"); pinning triton overrides
                               the only backend its indexer has.
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
    "--context-length 131072 --mem-fraction-static 0.50 --cuda-graph-max-bs-decode 64 "
    "--enable-metrics --pre-warm-nccl --reasoning-parser glm45 --tool-call-parser glm47 "
    '--dsa-prefill-backend tilelang --dsa-decode-backend tilelang --enable-cache-report"'
)

# mem-fraction-static is a CEILING on weights+KV together on MI300A, not a KV reservation:
# KV(f) = 443.4 * (f - 0.486) GB/rank at tp=4 pp=4. Below 0.486 sglang refuses; 0.62 let the
# heaviest PP stage (node3 at 206.1 GB -- the stages are UNEVEN) reach the host OOM killer.
CE_ENV = """
# SAME CONTAINER AS EVERY OTHER SGLANG ARM. sglang-glm-halfconv and sglang-latest both resolve to
# optarena-sglang.sqsh -- the identical squashfs. GLM is NOT a patched image and there is no GLM
# build. The EDF differs from the shared one by exactly two [env] keys and nothing else:
#   1. PYTHONPATH, reaching a sitecustomize that sets torch.Tensor.format_ue8m0 = False as a CLASS
#      default, or the ROCm fnuz branch returns a fresh tensor without the attribute and the loader
#      dies -- GLM-5.3 cannot load at all without it;
#   2. HIPCC_COMPILE_FLAGS_APPEND=-U__HIP_NO_HALF_CONVERSIONS__ -U__HIP_NO_HALF_OPERATORS__,
#      without which module_fused_qk_norm_rope_cache_quant_shuffle will not compile.
# Do NOT move those into sglang-latest: kimi and qwen build their aiter kernels fine, and those
# macros exist to stop ambiguous __half overloads.
# Both are ordinary process env vars and role_srun passes --export=ALL, so they could live in the
# .env instead and retire the second EDF. Worth doing, but only behind a serving smoke: a second
# sitecustomize.py exists in the tree (external-eager-pg-patch, baked into the vLLM image), Python
# imports the FIRST one on sys.path and stops, and the two collide silently if an arm ever sets
# VLLM_EAGER_PG_PATCH=1 alongside this. GLM serves correctly today.
INFERENCE_CE_ENV=sglang-glm-halfconv
"""

HEADER = """
# --- GLM-5.3 deviations from the kimi arm this env was derived from ------------------
# Serving config proven by smoke 628603: 4 nodes, tp=4 x pp=4, fp8 weights and fp8_e4m3 KV.
# mem-fraction-static is a CEILING on weights+KV together on MI300A, not a KV reservation:
# KV(f) = 443.4 * (f - 0.486) GB/rank here, so lowering it shrinks KV toward zero rather than
# freeing host memory. 0.55 measured within 1.5% of that law; 0.62 OOM-killed the heaviest PP
# stage. The PP stages are UNEVEN (172.4/197.2/203.8/206.1 GB) -- size against the max.
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
