# syntax=docker/dockerfile:1
#
# Generic AMD MI300A CE base for CSCS Alps.
# Build context must be the repository root:
#   podman build -f containers/cluster/ce-images/amd/Dockerfile -t optarena-ce:amd-mi300 .
#
# The public CSCS docs do not currently list an Alps Extended ROCm/PyTorch image
# equivalent to the NVIDIA NGC extended images. Start from AMD's ROCm PyTorch image
# and keep the MI300A target explicit: MI300A is gfx942.
ARG BASE_IMAGE=docker.io/rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.9.1
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG ROCM_ARCH=gfx942
ARG PYTHON_REQUIREMENTS=requirements/amd.txt

ENV LC_ALL=C \
    PLAYWRIGHT_BROWSERS_PATH=/opt/playwright \
    ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    PYTORCH_ROCM_ARCH=${ROCM_ARCH} \
    HCC_AMDGPU_TARGET=${ROCM_ARCH} \
    PATH=/opt/rocm/bin:/usr/local/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH}

# linux-perf and linux-cpupower are DEBIAN package names and both bases are Ubuntu, so they are
# installed through a fallback chain onto Ubuntu's linux-tools instead of in the main list, where a
# missing name fails the whole layer. The /usr/bin/perf shipped by linux-tools-common is a wrapper
# that resolves its real binary through the HOST kernel's `uname -r`, which never matches inside a
# container, so the versioned binaries are symlinked onto PATH directly. cpupower comes from the
# same linux-tools directory.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv python3-dev \
      build-essential gcc g++ gfortran clang lld \
      gdb valgrind util-linux hwloc msr-tools numactl \
      libgoogle-perftools-dev heaptrack likwid papi-tools libpapi-dev strace ltrace binutils \
      ca-certificates git curl wget openssh-client gnupg ripgrep fd-find jq less tree htop \
      unzip vim nano file \
      automake autoconf libtool bison flex \
      libopenblas-dev liblapack-dev liblapacke-dev libfftw3-dev libfftw3-mpi-dev \
      libhdf5-dev libnetcdf-dev libgsl-dev libeigen3-dev libsuitesparse-dev libmetis-dev \
      libsuperlu-dev libmumps-seq-dev libhypre-dev libsundials-dev libboost-all-dev \
      libblis-dev libucx-dev libomp-dev libtbb-dev libsleef-dev libxsimd-dev libhwy-dev \
      libnuma-dev libhwloc-dev libarmadillo-dev libfabric-dev libfabric-bin \
      mpich libmpich-dev libscalapack-mpich-dev libpetsc-real-dev \
      make cmake ninja-build pkg-config; \
    if ! apt-get install -y --no-install-recommends flang; then \
      apt-get install -y --no-install-recommends flang-18 || true; \
      if command -v flang-new-18 >/dev/null 2>&1; then ln -sf "$(command -v flang-new-18)" /usr/local/bin/flang; fi; \
      if command -v flang-18 >/dev/null 2>&1; then ln -sf "$(command -v flang-18)" /usr/local/bin/flang; fi; \
    fi; \
    if ! apt-get install -y --no-install-recommends linux-perf; then \
      apt-get install -y --no-install-recommends linux-tools-common linux-tools-generic || true; \
    fi; \
    for tool_bin in /usr/lib/linux-tools*/*/perf /usr/lib/linux-tools*/perf; do \
      if [ -x "${tool_bin}" ]; then ln -sf "${tool_bin}" /usr/local/bin/perf; fi; \
    done; \
    for tool_bin in /usr/lib/linux-tools*/*/cpupower /usr/lib/linux-tools*/cpupower; do \
      if [ -x "${tool_bin}" ]; then ln -sf "${tool_bin}" /usr/local/bin/cpupower; fi; \
    done; \
    if apt-cache show rocm-dev >/dev/null 2>&1; then apt-get install -y --no-install-recommends rocm-dev; fi; \
    if apt-cache show rocprofiler-sdk >/dev/null 2>&1; then apt-get install -y --no-install-recommends rocprofiler-sdk; fi; \
    rm -rf /var/lib/apt/lists/*

# The NEWEST LLVM apt.llvm.org publishes for this codename, plus the newest distro gcc (operator
# decision 2026-08-07: latest available, not a pin). The harness probes the BARE driver names, so
# the unversioned symlinks in /usr/local/bin are load-bearing rather than cosmetic. The distro
# clang and the distro-flang fallback installed above stay as the LAST resort, for a base whose
# codename apt.llvm.org does not publish -- hence the block is best-effort and the drivers are
# only relinked once a versioned one exists.
RUN set -eux; \
    codename="$(. /etc/os-release; echo "${VERSION_CODENAME:-}")"; \
    for attempt in 1 2 3; do \
      if wget -qO /tmp/apt.llvm.org.asc https://apt.llvm.org/llvm-snapshot.gpg.key; then break; fi; \
      echo "apt.llvm.org key fetch attempt ${attempt} failed; retrying in $((attempt * 5))s" >&2; \
      sleep $((attempt * 5)); \
    done; \
    llvm_major=""; \
    if [ -n "${codename}" ] && [ -s /tmp/apt.llvm.org.asc ]; then \
      mv /tmp/apt.llvm.org.asc /etc/apt/trusted.gpg.d/apt.llvm.org.asc; \
      for candidate in 23 22 21 20; do \
        echo "deb http://apt.llvm.org/${codename}/ llvm-toolchain-${codename}-${candidate} main" \
          > /etc/apt/sources.list.d/llvm.list; \
        if apt-get update >/dev/null 2>&1 \
           && apt-cache show "clang-${candidate}" >/dev/null 2>&1; then \
          llvm_major="${candidate}"; \
          break; \
        fi; \
      done; \
    fi; \
    if [ -n "${llvm_major}" ]; then \
      apt-get install -y --no-install-recommends \
        "clang-${llvm_major}" "flang-${llvm_major}" "libomp-${llvm_major}-dev" \
      || apt-get install -y --no-install-recommends \
        "clang-${llvm_major}" "libomp-${llvm_major}-dev"; \
      ln -sf "$(command -v clang-${llvm_major})" /usr/local/bin/clang; \
      ln -sf "$(command -v clang++-${llvm_major})" /usr/local/bin/clang++; \
      if command -v "flang-${llvm_major}" >/dev/null 2>&1; then \
        ln -sf "$(command -v flang-${llvm_major})" /usr/local/bin/flang; \
      elif command -v "flang-new-${llvm_major}" >/dev/null 2>&1; then \
        ln -sf "$(command -v flang-new-${llvm_major})" /usr/local/bin/flang; \
      fi; \
      echo "${llvm_major}" > /usr/local/share/llvm-major; \
    fi; \
    newest_gcc=""; \
    for candidate in 16 15 14 13; do \
      if apt-cache show "gcc-${candidate}" >/dev/null 2>&1; then newest_gcc="${candidate}"; break; fi; \
    done; \
    if [ -n "${newest_gcc}" ]; then \
      apt-get install -y --no-install-recommends \
        "gcc-${newest_gcc}" "g++-${newest_gcc}" "gfortran-${newest_gcc}"; \
      ln -sf "$(command -v gcc-${newest_gcc})" /usr/local/bin/gcc; \
      ln -sf "$(command -v g++-${newest_gcc})" /usr/local/bin/g++; \
      ln -sf "$(command -v gfortran-${newest_gcc})" /usr/local/bin/gfortran; \
    fi; \
    rm -f /tmp/apt.llvm.org.asc; \
    rm -rf /var/lib/apt/lists/*; \
    clang --version | head -1; \
    flang --version | head -1; \
    gcc --version | head -1

COPY containers/build-hptt.sh /build-hptt.sh
RUN sh /build-hptt.sh

COPY requirements/ /requirements/
RUN set -eux; \
    PIP_BREAK_SYSTEM_PACKAGES=1 MPICC=mpicc.mpich python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel; \
    PIP_BREAK_SYSTEM_PACKAGES=1 MPICC=mpicc.mpich python3 -m pip install --no-cache-dir --no-binary=mpi4py -r "/${PYTHON_REQUIREMENTS}"

RUN PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --no-cache-dir \
      "litellm[proxy]==1.94.0" fastapi uvicorn httpx pydantic orjson tenacity rich typer

COPY containers/judge/requirements.txt /opt/optarena-judge/requirements.txt
RUN set -eux; \
    PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --no-cache-dir \
      -r /opt/optarena-judge/requirements.txt; \
    playwright install --with-deps chromium; \
    chmod -R a+rX "${PLAYWRIGHT_BROWSERS_PATH}"

RUN set -eux; \
    git clone --depth 1 --recurse-submodules --shallow-submodules \
      --branch extended https://github.com/spcl/dace.git /opt/dace; \
    PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --no-cache-dir -e /opt/dace

# Node 20 from nodesource, NOT the distro nodejs/npm: the Ubuntu 22.04 base ships node 12, and
# @anthropic-ai/claude-code below needs node >= 18. The distro nodejs and npm are therefore gone
# from the apt list above -- the nodesource package provides both and conflicts with them.
RUN set -eux; \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -; \
    apt-get install -y --no-install-recommends nodejs; \
    rm -rf /var/lib/apt/lists/*; \
    node --version; \
    npm --version

RUN npm install -g @anthropic-ai/claude-code

COPY containers/agent /opt/optarena-agent
COPY containers/judge /opt/optarena-judge
RUN chmod +x /opt/optarena-agent/start_agents.sh /opt/optarena-agent/start_run.sh

RUN set -eux; \
    command -v hipcc; \
    command -v amdclang++ || true; \
    command -v rocprofv3 || command -v rocprof || true; \
    command -v claude; \
    command -v litellm; \
    node_major="$(node --version | sed 's/^v//; s/\..*//')"; \
    if [ "${node_major}" -lt 18 ]; then \
      echo "node >= 18 required by @anthropic-ai/claude-code, found $(node --version)" >&2; \
      exit 1; \
    fi; \
    # The harness compiles submissions with the BARE names, and PATH puts /opt/rocm/bin ahead of \
    # /usr/local/bin -- so if the ROCm base ships a bare clang/flang, it silently shadows the \
    # apt.llvm.org symlinks and every graded arm compiles with the wrong toolchain. Fail the \
    # build instead of grading a campaign with it. \
    llvm_major="$(cat /usr/local/share/llvm-major 2>/dev/null || echo)"; \
    if [ -n "${llvm_major}" ]; then \
      clang --version | head -1 | grep -q "clang version ${llvm_major}" \
        || { echo "bare clang is not LLVM ${llvm_major}: $(command -v clang)" >&2; exit 1; }; \
      flang --version | head -1 | grep -q " ${llvm_major}" \
        || { echo "bare flang is not LLVM ${llvm_major}: $(command -v flang)" >&2; exit 1; }; \
    fi; \
    python3 -c "import importlib; [importlib.import_module(name) for name in ('numpy', 'scipy', 'numba', 'pythran', 'torch', 'dace', 'mpi4py')]"

WORKDIR /work
