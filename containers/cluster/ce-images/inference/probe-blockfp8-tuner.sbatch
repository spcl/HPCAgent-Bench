#!/usr/bin/env bash
# What does the block-FP8 GEMM tuner want, and which shapes does qwen38 actually ask for?
#
# Serving qwen38 on 0.27.1 warns "Using default W8A8 Block FP8 kernel config. Performance might be
# sub-optimal! Config not found: N=5120,K=4352,device_name=AMD_Instinct_MI300A,dtype=fp8_w8a8,
# block_shape=[128,128].json" -- and none of the 219 configs the image ships is for MI300A. This
# reads the tuner's CLI and the model's real linear shapes so the tuning job can be written
# against measured names rather than guessed ones. It only matters if aiter's LINEAR path does
# NOT take that GEMM, which probe-aiter-ab.sbatch is what answers.
# ANSWERED 610156: the shipped tuner takes only --tp-size/--block-n/--block-k/--batch-size and
# tunes a HARDCODED DeepSeek-V3/R1 weight-shape list -- there is no --model, so it cannot emit
# qwen38's shapes without editing that list. Device key is AMD_Instinct_MI300A. qwen38 at TP=4
# wants N=2048/K=5120, N=5120/K=1536, N=8704/K=5120 and N=5120/K=4352 -- the last is the one
# the server names in its warning -- all at block [128,128].
#SBATCH --job-name=probe-blockfp8
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=logs/probe-blockfp8-%j.out
#SBATCH --error=logs/probe-blockfp8-%j.err

set -Eeuo pipefail
ulimit -c 0
FAST_SCRATCH="/iopsstor/scratch/cscs/${USER}"
[[ -d "${FAST_SCRATCH}" ]] || FAST_SCRATCH="${SCRATCH:-/capstor/scratch/cscs/ybudanaz/x86_64}"
export HF_HOME="${HF_HOME:-${FAST_SCRATCH}/hf}"
export MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.8-27B-FP8}"
INFERENCE_EDF="${INFERENCE_EDF:-rocm723-vllm-0.27.1-pytorch211-ofi}"

srun --nodes=1 --ntasks=1 --exclusive --export=ALL --environment="${INFERENCE_EDF}" bash -c '
set -Eeuo pipefail
BENCH="$(python3 -c "import vllm,pathlib;print(pathlib.Path(vllm.__file__).parent.parent)")/benchmarks/kernels"
[[ -d "${BENCH}" ]] || BENCH=/opt/vllm-src/benchmarks/kernels
echo "=== tuner CLI ==="
python3 "${BENCH}/benchmark_w8a8_block_fp8.py" --help 2>&1 | head -40

echo
echo "=== where vllm looks the config up ==="
grep -rn "block_shape\|device_name\|_get_config_file_name\|configs/" \
    /opt/vllm-src/vllm/model_executor/layers/quantization/utils/fp8_utils.py 2>/dev/null | head -20

echo
echo "=== device name vllm will key on ==="
python3 -c "import torch;print(torch.cuda.get_device_name(0).replace(\" \", \"_\"))"

echo
echo "=== the linear shapes this model asks for (TP=4) ==="
python3 - <<PYEOF
import json
import os
import pathlib

from huggingface_hub import snapshot_download

# Qwen3.8 is a heterogeneous config: hidden_size and friends live under text_config, and reading
# them off the transformers object raises AttributeError. Read the json instead.
root = pathlib.Path(snapshot_download(repo_id=os.environ["MODEL_REPO"], local_files_only=True))
cfg = json.loads((root / "config.json").read_text())
text = cfg.get("text_config", cfg)
tp = 4
hidden = text["hidden_size"]
inter = text["intermediate_size"]
heads = text["num_attention_heads"]
kv = text["num_key_value_heads"]
head_dim = text.get("head_dim", hidden // heads)
print("block", cfg.get("quantization_config", {}).get("weight_block_size"),
      "hidden", hidden, "intermediate", inter, "heads", heads, "kv_heads", kv, "head_dim", head_dim)
for label, n, k in (
    ("qkv_proj", (heads + 2 * kv) * head_dim // tp, hidden),
    ("o_proj", hidden, heads * head_dim // tp),
    ("gate_up_proj", 2 * inter // tp, hidden),
    ("down_proj", hidden, inter // tp),
):
    print(f"  {label:14s} N={n} K={k}")
PYEOF
'
