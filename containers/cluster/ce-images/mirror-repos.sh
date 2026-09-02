#!/bin/bash
# Mirror every repo the container builds clone, from the LOGIN node, which reaches GitHub
# reliably. The builds then clone from here instead of from GitHub.
set -u
MIRROR="${SCRATCH:?}/git-mirrors"
REPOS="
spack/spack
spack/spack-packages
ofiwg/libfabric
aws/aws-ofi-nccl
HewlettPackard/shs-cassini-headers
HewlettPackard/shs-libcxi
vllm-project/vllm
ROCm/aiter
triton-lang/triton
ROCm/triton
HewlettPackard/shs-cxi-driver
bondhugula/pluto
icl-utk-edu/papi
spcl/dace
"
export GIT_TERMINAL_PROMPT=0
for r in $REPOS; do
  dest="$MIRROR/$r.git"
  mkdir -p "$(dirname "$dest")"
  if [ -d "$dest" ]; then
    printf '%-40s update ... ' "$r"
    git -C "$dest" remote update --prune >/dev/null 2>&1 && echo OK || echo FAIL
  else
    printf '%-40s clone  ... ' "$r"
    git clone --mirror -q "https://github.com/$r.git" "$dest" >/dev/null 2>&1 && echo OK || echo FAIL
  fi
done
du -sh "$MIRROR"
