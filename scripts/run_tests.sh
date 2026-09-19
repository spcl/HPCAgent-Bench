#!/usr/bin/env bash
# Run the pytest suite in the environment the suite actually needs.
#
# THIS EXISTS BECAUSE THE SUITE LIES WITHOUT IT. A full run launched as
# "${VENV}/bin/python -m pytest tests/" reported 90 failures, of which 84 were the environment and
# not the tree: 48 pythran e2e cases (the oracle spawns the "pythran" CONSOLE SCRIPT, and running
# the venv's interpreter by path never puts the venv's bin on PATH), 31 native-build cases across
# blas_link_order / default_datatype_tolerance / compiler_family / baseline_model / api /
# cli_subcommands / agent_* (cblas.h, no PKG_CONFIG_PATH), and 5 dace agreement kernels (cblas.h
# again, from dace's OWN build, which does not read pkg-config -- hence CPATH). Every one of them
# fails as a red assertion rather than a skip, so the cost is three hours and a false verdict.
#
# Everything is DERIVED. A spack prefix carries a content hash that changes on every reinstall, so
# a pasted path is a setting that silently stops existing -- the glob spelling is the one already
# used by experiments/smoke-gpu-models.sbatch and experiments/mpi/smoke-mpi-judge.sbatch.
#
# A FULL run still belongs in an sbatch on a compute node -- this only fixes what the run sees,
# not where it belongs; the login node is for a targeted selection.
#
# NEITHER of the above is where the translator suite's C23 cases actually need to run. Both the
# login node and a bare compute node are the cluster's own SLES image, whose gcc has no -std=c23 --
# about 720 translator cases fail there as a red compile error instead of the translator bug they
# exist to catch. `--container` runs the SAME command inside the judge image
# (containers/cluster/ce-images/judge-agent-amd/Dockerfile), which ships gcc 16 and is the
# toolchain graded runs already use, so a green run here means what it says.
#
# Usage: scripts/run_tests.sh [pytest args...]              (default: -q --maxfail=20 tests/)
#        scripts/run_tests.sh --container [pytest args...]  (submits scripts/run_tests_container.sbatch,
#                                                           1 mi300 node, and waits for it)
set -Eeuo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == --container ]]; then
    shift
    : "${SCRATCH:?run_tests.sh --container needs SCRATCH set (sbatch propagates it to the job)}"
    # The Slurm account is resolved once, here, and handed to sbatch through its own SBATCH_ACCOUNT
    # env var -- never a literal -A: the account differs per person (scripts/cscs/account_env.sh).
    . "${REPO}/scripts/cscs/account_env.sh"
    echo "run_tests.sh: submitting the suite inside the judge EDF (mi300, 1 node)..." >&2
    exec sbatch --wait --partition=mi300 --job-name=run-tests-container \
        "${REPO}/scripts/run_tests_container.sbatch" "$@"
fi
# PATH, PYTHONPATH, PYTHONHASHSEED and ulimit -c 0 all come from the one file that already derives
# them, so this script cannot drift from what a campaign runs under.
. "${REPO}/experiments/env.sh"

# Only APPEND, and only when nothing already answers: a container carrying its own OpenBLAS must
# keep it rather than being sent to a host build from another toolchain.
if ! pkg-config --exists openblas 2>/dev/null; then
    for prefix in "${SCRATCH}"/spack/opt/spack/*/openblas-*; do
        [[ -d "${prefix}/lib/pkgconfig" ]] || continue
        export PKG_CONFIG_PATH="${PKG_CONFIG_PATH:+${PKG_CONFIG_PATH}:}${prefix}/lib/pkgconfig"
        # dace generates C++ that includes <cblas.h> and builds it through its own CMake, which
        # never consults pkg-config. Without these five kernels fail as compile_fail and read as a
        # numeric regression.
        export CPATH="${CPATH:+${CPATH}:}${prefix}/include"
        export LIBRARY_PATH="${LIBRARY_PATH:+${LIBRARY_PATH}:}${prefix}/lib"
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${prefix}/lib"
    done
fi

# Same reason, same shape, for FFTW: the C/C++/Fortran translator lowers a whole-array 1-D
# np.fft.fft/ifft to FFT_LIBRARY_MARKER, an fftw_plan_dft_1d/fftwf_... call (languages.py's
# FFT_LINKED_LANGS), so every native-build test needs <fftw3.h> on CPATH and libfftw3 on the
# link line without PKG_CONFIG_PATH pointing at it.
if ! pkg-config --exists fftw3 2>/dev/null; then
    for prefix in "${SCRATCH}"/spack/opt/spack/*/fftw-*; do
        [[ -d "${prefix}/lib/pkgconfig" ]] || continue
        export PKG_CONFIG_PATH="${PKG_CONFIG_PATH:+${PKG_CONFIG_PATH}:}${prefix}/lib/pkgconfig"
        export CPATH="${CPATH:+${CPATH}:}${prefix}/include"
        export LIBRARY_PATH="${LIBRARY_PATH:+${LIBRARY_PATH}:}${prefix}/lib"
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${prefix}/lib"
    done
fi

# mpi4py imports at collection time in the MPI tests; these are the same values every dace command
# in this repo runs under, and without them the run hangs instead of skipping.
#
# OMP_NUM_THREADS is DELIBERATELY ABSENT from that list. It is part of the dace MPI prefix, but on a
# suite run exporting 1 serializes every threaded and timed test and disarms the race checks -- a
# dropped WCR or a parallelised reducing axis is invisible at one thread. The suite's own default
# has to win.
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp PMIX_MCA_gds=hash
export UCX_VFS_ENABLE=n HWLOC_COMPONENTS=-gl MPI4PY_RC_INITIALIZE=0

command -v pythran >/dev/null || echo "WARNING: no pythran on PATH -- every pythran e2e case will FAIL, not skip" >&2
pkg-config --exists openblas || echo "WARNING: no openblas found -- every native build case will FAIL on cblas.h" >&2
pkg-config --exists fftw3 || echo "WARNING: no fftw3 found -- every FFT-library-lowering case will FAIL on fftw3.h" >&2

cd -- "${REPO}"
[[ $# -gt 0 ]] || set -- -q --maxfail=20 tests/
exec "${PY}" -m pytest "$@"
