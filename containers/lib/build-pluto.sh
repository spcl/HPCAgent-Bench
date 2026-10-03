#!/bin/sh
# Build Pluto (polycc, the polyhedral source-to-source tool the `pluto` framework shells out to)
# against a distro clang and install it into /usr/local. Run as root: the judge-agent Dockerfiles
# COPY and run it (and drop the apt lists after it), CI runs it under sudo.
#
#   https://github.com/bondhugula/pluto
#
# Not a pip package: it needs isl, clan, candl and pet. pet tracks clang's C++ API closely, so it is
# built against clang 17, not the image's newest LLVM. polycc is a standalone transformer whose
# output the graded toolchain compiles afterwards, so which clang parsed the scop is not a property
# of any graded number.
#
# Env: PLUTO_COMMIT (pin), PLUTO_CLANG_MAJOR (17), PLUTO_SRC (/opt/pluto), PLUTO_GCC_INSTALL_DIR
# (optional: the gcc whose libstdc++ clang uses; a clang this old cannot parse gcc 15+ headers).
set -eu

# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
# bondhugula/pluto @ 0.12.0-33-gdc46216
PLUTO_COMMIT="${PLUTO_COMMIT:-dc462163c8b4fc97d378a4d245d1a64741cb4111}"
PLUTO_CLANG_MAJOR="${PLUTO_CLANG_MAJOR:-17}"
PLUTO_SRC="${PLUTO_SRC:-/opt/pluto}"
llvm="/usr/lib/llvm-${PLUTO_CLANG_MAJOR}"
# gitretry (git_mirror.sh setup) inside an image build, plain git elsewhere.
git_net="$(command -v gitretry || command -v git)"

# pkg-config: pet's configure uses PKG_CHECK_MODULES(ISL, isl), unexpanded without pkg.m4.
# libyaml-dev: pet. texinfo: candl's configure hard-requires makeinfo.
apt-get update
apt-get install -y --no-install-recommends \
    "clang-${PLUTO_CLANG_MAJOR}" "llvm-${PLUTO_CLANG_MAJOR}" "llvm-${PLUTO_CLANG_MAJOR}-dev" \
    "libclang-${PLUTO_CLANG_MAJOR}-dev" libltdl-dev \
    autoconf automake libtool pkg-config libgmp-dev libyaml-dev flex bison texinfo

# pet prefers the monolithic -lclang-cpp and Ubuntu ships only the versioned .so.NN; without this
# link configure falls back to enumerating -lclang* and misses -lclangASTMatchers.
ln -sf "${llvm}/lib/libclang-cpp.so.${PLUTO_CLANG_MAJOR}" "${llvm}/lib/libclang-cpp.so"

# cloog builds its PDF manual with texi2dvi, which needs a full TeX engine: a no-op that creates
# the -o target instead.
printf '%s\n' '#!/bin/sh' \
    'out=; while [ $# -gt 0 ]; do [ "$1" = "-o" ] && { out=$2; shift; }; shift; done' \
    '[ -n "$out" ] && : > "$out"; exit 0' > /usr/local/bin/texi2dvi
chmod +x /usr/local/bin/texi2dvi

gcc_dir=""
if [ -n "${PLUTO_GCC_INSTALL_DIR:-}" ]; then
    test -d "${PLUTO_GCC_INSTALL_DIR}" || { echo "build-pluto.sh: no gcc at ${PLUTO_GCC_INSTALL_DIR}" >&2; exit 1; }
    gcc_dir=" --gcc-install-dir=${PLUTO_GCC_INSTALL_DIR}"
fi

"${git_net}" clone --recursive https://github.com/bondhugula/pluto.git "${PLUTO_SRC}"
git -C "${PLUTO_SRC}" checkout "${PLUTO_COMMIT}"
"${git_net}" -C "${PLUTO_SRC}" submodule update --init --recursive
cd "${PLUTO_SRC}"
./autogen.sh
# The gcc pin rides on CC/CXX because pet's configure overwrites CXXFLAGS.
./configure --with-clang-prefix="${llvm}" \
    CC="clang-${PLUTO_CLANG_MAJOR}${gcc_dir}" CXX="clang++-${PLUTO_CLANG_MAJOR}${gcc_dir}" \
    CXXFLAGS='-std=c++17 -include cstdint' \
    || { grep -B5 -A15 -iE '\berror\b' pet/config.log 2>/dev/null | tail -150 >&2; exit 1; }
make -j"$(nproc)"
make install
ldconfig
# The build tree stays: polycc bakes absolute build-tree paths (its pluto binary, inscop and
# getversion's .git) that `make install` does not relocate.

# polycc must transform a scop, not only install.
smoke="$(mktemp -d)"
printf '%s\n' \
    '#include <stdint.h>' \
    'void mm(const int64_t N, double (*restrict A)[N], double (*restrict B)[N], double (*restrict C)[N]) {' \
    '#pragma scop' \
    '  for (int64_t i=0;i<N;i++) for (int64_t j=0;j<N;j++) for (int64_t k=0;k<N;k++) C[i][j]+=A[i][k]*B[k][j];' \
    '#pragma endscop' \
    '}' > "${smoke}/smoke.c"
(cd "${smoke}" && polycc --pet smoke.c -o smoke_out.c && test -s smoke_out.c)
rm -rf "${smoke}"
echo "polycc OK: $(command -v polycc)"
