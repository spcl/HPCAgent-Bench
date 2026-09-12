"""Derive GLM-5.3 arm envs from the kimi sglang arms they are the counterfactual of.

Only the serving block differs: same nodes, same agents, same problems, same packet. Anything else
that changed would make an arm comparison a comparison of two things at once.

Every deviation from the kimi line is load-bearing:
  SGLANG_ATTENTION_BACKEND=    ASSIGNED EMPTY, not absent. run_cluster.sh reads it as
                               ${VAR-aiter}, so an absent key appends --attention-backend aiter
                               and only an assigned empty value omits the flag. Omitted is what
                               this model needs -- see BACKEND_ENV below.
  --language-only              selects the VLM encoder-disaggregation RECEIVER role in this
                               SGLang, and GlmMoeDsaForCausalLM is off its allowlist, so the
                               server refuses to start.
  --enable-hierarchical-cache  offloads KV to host memory, which on MI300A is the SAME pool the
                               weights live in, so it allocates a second KV cache instead.
  AITER_USE_FLYDSL_MOE_SORTING kimi's weights are pack-quantized int4; these are fp8.
"""

import pathlib
import sys

#: The served window, and what a request has to leave room for: the completion reservation plus
#: one turn, which a compiler log or an asm dump can fill on its own.
GLM_CONTEXT = 131072
COMPLETION_RESERVE = 32000
TURN_HEADROOM = 30000

#: Slowest pipeline stage (1521 s to over 5400 s measured) plus KV allocation plus graph capture.
#: The inherited 7200 expires while the server is still loading and the arm abandons it.
READY_TIMEOUT = 10800

GLM_ARGS = (
    '"--trust-remote-code --watchdog-timeout 1800 --kv-cache-dtype fp8_e4m3 --page-size 64 '
    "--context-length 131072 --mem-fraction-static 0.55 --cuda-graph-max-bs-decode 64 "
    "--enable-metrics --pre-warm-nccl --reasoning-parser glm45 --tool-call-parser glm47 "
    '--dsa-prefill-backend tilelang --dsa-decode-backend tilelang --enable-cache-report"'
)

# The memory law and the pool the fraction has to reach are in HEADER, which ships with the env
# the server is launched from. Keep the two in step: 0.55 is the only tuned number here.
CE_ENV = """
# sglang-candidate is the only sglang EDF whose image can load GLM-5.3: the DeepSeek weight
# loader's format_ue8m0 reads are guarded in the package at build time and
# HIPCC_COMPILE_FLAGS_APPEND is a global image ENV. The other sglang EDFs reach the same patch
# through a PYTHONPATH under /capstor, which role_mounts drops for the inference role, so the
# loader dies on format_ue8m0 before the model is up.
INFERENCE_CE_ENV=sglang-candidate
"""

#: ASSIGNED EMPTY so run_cluster.sh's ${SGLANG_ATTENTION_BACKEND-aiter} omits the flag and
#: GlmMoeDsaForCausalLM picks dsa, the only backend this checkpoint is known to serve on.
BACKEND_ENV = "SGLANG_ATTENTION_BACKEND="

HEADER = """
# --- GLM-5.3 deviations from the kimi arm this env was derived from ------------------
# mem-fraction-static is a CEILING on weights+KV together, not a KV reservation, so lowering it
# shrinks the KV pool toward zero: pool(f) = 39.0M * (f - 0.4838) tokens at tp4 x pp4. Below 0.486
# sglang refuses and at 0.62 the host OOM killer takes the heaviest pipeline stage, whose 206.1 GB
# is the one to size against -- the stages are UNEVEN (172.4/197.2/203.8/206.1 GB). 0.55 gives a
# 2.58M-token pool against the 1.38M an arm's 20 agents hold, a pool/working-set ratio of 1.87
# where the prefix cache holds.
# SGLANG_ATTENTION_BACKEND is assigned EMPTY so no --attention-backend reaches the server and
# GlmMoeDsaForCausalLM selects dsa from its own config. An explicit aiter suppresses that and also
# scales mem-fraction-static by 0.85, so the flag would no longer be the effective fraction.
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
        elif line.startswith(("AITER_USE_FLYDSL_MOE_SORTING=", "SGLANG_ATTENTION_BACKEND=")):
            # Re-emitted with the args, so there is exactly one assignment in the result.
            continue
        elif line.startswith("VLLM_READY_TIMEOUT_SECONDS="):
            # Shadowed: agent_driver falls back to it only when AGENT_READY is unset, and it
            # reads as if it bounded the server wait.
            replaced.add("dead_ready")
            continue
        elif line.startswith("VLLM_ENGINE_READY_TIMEOUT_S="):
            line = f"VLLM_ENGINE_READY_TIMEOUT_S={READY_TIMEOUT}"
            replaced.add("engine_ready")
        elif line.startswith("AGENT_READY_TIMEOUT_SECONDS="):
            line = f"AGENT_READY_TIMEOUT_SECONDS={READY_TIMEOUT}"
            replaced.add("agent_ready")
        elif line.startswith("INFERENCE_CE_ENV="):
            # Without this branch a regeneration drops GLM back to the shared EDF, which cannot
            # load GLM-5.3 at all, and the arm looks correct until the loader dies on format_ue8m0.
            out.extend(CE_ENV.strip("\n").splitlines())
            replaced.add("ce_env")
            continue
        elif line.startswith("CLAUDE_AUTOCOMPACT="):
            # GLM serves HALF the kimi context, so the inherited threshold would sit above the
            # window entirely and every agent would 400 before it could ever compact.
            line = f"CLAUDE_AUTOCOMPACT={GLM_CONTEXT - COMPLETION_RESERVE - TURN_HEADROOM}"
            replaced.add("autocompact")
        elif line.startswith("SGLANG_EXTRA_ARGS="):
            # The block above states the SOURCE model's pairing; HEADER states this one's.
            while out and out[-1].startswith("#"):
                out.pop()
            out.extend(HEADER.strip("\n").splitlines())
            out.append(BACKEND_ENV)
            line = f"SGLANG_EXTRA_ARGS={GLM_ARGS}"
            replaced.add("args")
        out.append(line)
    missing = {
        "model",
        "optimizer",
        "args",
        "ce_env",
        "autocompact",
        "dead_ready",
        "engine_ready",
        "agent_ready",
    } - replaced
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
