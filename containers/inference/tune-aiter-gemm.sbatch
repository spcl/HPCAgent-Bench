#!/usr/bin/env bash
# Tune aiter's bf16 GEMM dispatch for the shapes our served models run, on one MI300A node, inside the
# SGLang image. aiter ships bf16_tuned_gemm.csv rows for gfx950 and gfx1250 only, so on gfx942 every
# bf16 GEMM without a row logs "not found tuned config ... using torch solution" and leaves aiter.
# Output: aiter-configs/bf16_tuned_gemm_gfx942.csv beside this script, which the SGLang image installs as
# one of aiter's model_configs files (containers/images/sglang/Dockerfile); rebuild the image after a
# re-run. Re-run after an aiter bump or for a new model's shapes.
#   containers/images/submit.sh containers/inference/tune-aiter-gemm.sbatch   # the system's options (beverin: mi300)
#   sbatch -A <account> -p mi300 --gpus-per-node=4 containers/inference/tune-aiter-gemm.sbatch
# Candidates are aiter's own kernels (--libtype: asm, flydsl, skinny, triton, opus), never torch or hipBLASLt:
# a row exists to keep the GEMM in aiter (torch, allowed, wins only prefill shapes with M >= 7424, and
# hipBLASLt's 1363 solutions per shape outrun a 4 h job).
# SHAPES: "N,K" pairs, from the "shape is M:.., N:.., K:.." lines of a server log (AITER_LOG_TUNED_CONFIG=1).
#SBATCH --job-name=tune-aiter-gemm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=%x-%j.out
set -euo pipefail
ulimit -c 0
here="${HPCAGENT_BENCH_REPO:?set HPCAGENT_BENCH_REPO to the checkout}/containers/inference"
# qwen38: 24,5120 (Gated DeltaNet ba projection, tp4). glm53: 256,6144 (MoE router), 32,6144 (DSA indexer).
SHAPES="${SHAPES:-24,5120 256,6144 32,6144}"
work="${SCRATCH:?}/.tmp/tune-aiter-gemm-${SLURM_JOB_ID}"
mkdir -p "${work}" "${here}/aiter-configs"
trap 'rm -rf "${work}"' EXIT
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_ON_NODE}" --environment=hpcagent-bench-sglang-mi300-latest \
    env SHAPES="${SHAPES}" WORK="${work}" OUT="${here}/aiter-configs/bf16_tuned_gemm_gfx942.csv" \
    bash -c '
set -euo pipefail
cd /sgl-workspace/aiter/csrc/gemm_a16w16
# Every M aiter can look up for a shape is a padded bucket (tuned_gemm.py tries get_padded_m with gl 0
# and 1), so tune those: M up to the 16384-token prefill chunk.
/opt/venv/bin/python3 - <<PY
import aiter
rows = ["M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle"]
for nk in "${SHAPES}".split():
    n, k = map(int, nk.split(","))
    ms = sorted({aiter.get_padded_m(m, n, k, gl) for m in range(1, 16385) for gl in (0, 1)})
    rows += [f"{m},{n},{k},False,torch.bfloat16,torch.bfloat16,False,False" for m in ms]
    print(n, k, len(ms), "buckets", flush=True)
open("${WORK}/untuned.csv", "w").write("\n".join(rows) + "\n")
PY
/opt/venv/bin/python3 gemm_tuner.py -i "${WORK}/untuned.csv" -o "${OUT}" --mp 4 --sort True --libtype asm,flydsl,skinny,triton,opus
'
echo "tuned rows: $(($(wc -l < "${here}/aiter-configs/bf16_tuned_gemm_gfx942.csv") - 1))"
