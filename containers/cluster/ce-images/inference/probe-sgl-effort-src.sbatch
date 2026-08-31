#!/usr/bin/env bash
# Read the three regions that decide whether the xhigh -> max collapse can be undone by config.
# 610209 located it: anthropic/serving.py:624-632 collapses it deliberately, and server_args.py:1438
# is the default_chat_template_kwargs help, whose wording about per-request precedence is what
# says whether a server-side default can win.
#SBATCH --job-name=probe-sgl-src
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=logs/probe-sgl-src-%j.out
#SBATCH --error=logs/probe-sgl-src-%j.err
set -Eeuo pipefail
srun --environment="${INFERENCE_EDF:-sglang-rocm-mi30x}" bash -c '
S=/sgl-workspace/sglang/python/sglang/srt
echo "########## anthropic/serving.py 600-660 (the collapse) ##########"
sed -n "600,660p" "${S}/entrypoints/anthropic/serving.py"
echo
echo "########## anthropic/protocol.py 320-350 (the documented mapping) ##########"
sed -n "320,350p" "${S}/entrypoints/anthropic/protocol.py"
echo
echo "########## server_args.py 1420-1455 (default_chat_template_kwargs help) ##########"
sed -n "1420,1455p" "${S}/server_args.py"
echo
echo "########## where chat_template_kwargs is merged at request time ##########"
grep -rn "chat_template_kwargs" "${S}" | head -20'
