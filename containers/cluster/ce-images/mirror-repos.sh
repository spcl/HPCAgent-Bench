#!/bin/bash
# Mirror every repo the container builds clone, from the LOGIN node, which reaches GitHub
# reliably. The builds then clone from here instead of from GitHub.
#
# The seed list is what the Dockerfiles name directly. Submodules are discovered, not listed:
# `submodule update --init --recursive` clones the .gitmodules URLs, those go through the same
# `insteadOf` rewrite, and a missing mirror there fails the build exactly like a missing seed.
# So the loop below is a BFS -- mirror a repo, read its .gitmodules, enqueue what it names.
set -u
MIRROR="${SCRATCH:?}/git-mirrors"
QUEUE="
spack/spack
spack/spack-packages
ofiwg/libfabric
aws/aws-ofi-nccl
HewlettPackard/shs-cassini-headers
HewlettPackard/shs-libcxi
HewlettPackard/shs-cxi-driver
vllm-project/vllm
ROCm/aiter
triton-lang/triton
ROCm/triton
bondhugula/pluto
icl-utk-edu/papi
spcl/dace
"
export GIT_TERMINAL_PROMPT=0
seen=""

mirror_one() {
  local r="$1" dest="$MIRROR/$1.git"
  mkdir -p "$(dirname "${dest}")"
  if [ -d "${dest}" ]; then
    printf '%-44s update ... ' "${r}"
    git -C "${dest}" remote update --prune >/dev/null 2>&1 && echo OK || { echo FAIL; return 1; }
  else
    printf '%-44s clone  ... ' "${r}"
    git clone --mirror -q "https://github.com/${r}.git" "${dest}" >/dev/null 2>&1 \
      && echo OK || { echo FAIL; return 1; }
  fi
}

# The submodule URLs a mirror names at HEAD, as org/repo, github only.
submodules_of() {
  git -C "$MIRROR/$1.git" show HEAD:.gitmodules 2>/dev/null \
    | sed -n 's#^[[:space:]]*url[[:space:]]*=[[:space:]]*##p' \
    | sed -e 's#^git@github\.com:#https://github.com/#' -e 's#\.git$##' -e 's#/$##' \
    | sed -n 's#^https://github\.com/##p'
}

while [ -n "$(echo "${QUEUE}" | tr -d '[:space:]')" ]; do
  next=""
  for r in ${QUEUE}; do
    case " ${seen} " in *" ${r} "*) continue ;; esac
    seen="${seen} ${r}"
    mirror_one "${r}" || continue
    next="${next} $(submodules_of "${r}" | tr '\n' ' ')"
  done
  QUEUE="${next}"
done

du -sh "${MIRROR}"
