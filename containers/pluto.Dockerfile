# Pluto (polyhedral source-to-source parallelizer) built against LLVM/clang 17.
#
# Provides `polycc` for the `pluto` backend in the e2e numerical sweep
# (hpcagent_bench/numerical_oracle.py::_run_pluto). Building Pluto from source is slow and
# apt-heavy; baking it into this image keeps that cost OUT of every CI run and makes
# the build reproducible + testable locally:
#
#   docker build -f containers/pluto.Dockerfile -t hpcagent_bench-pluto .
#   docker run --rm hpcagent_bench-pluto polycc --version
#
# The commit is pinned (bump deliberately) and matches the unit-tests / e2e job's
# PLUTO_COMMIT in .github/workflows/tests.yml.
#
# pet<->clang link: pet prefers the monolithic `-lclang-cpp`, but Ubuntu ships only
# the versioned `libclang-cpp.so.17` (no unversioned `.so`), so pet's configure falls
# back to enumerating individual `-lclang*` static libs and misses `-lclangASTMatchers`
# -> `undefined reference to clang::ast_matchers::*` at link. Fix: symlink
# `libclang-cpp.so -> .so.17` (the ln -s step below) so pet's FIRST choice links the one
# monolithic lib -- every clang symbol, ast_matchers included. Verified end-to-end: the
# final RUN builds polycc, so a broken recipe fails THIS image build (not silently in CI).
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
# bondhugula/pluto @ 0.12.0-33-gdc46216
ARG PLUTO_COMMIT=dc462163c8b4fc97d378a4d245d1a64741cb4111

# clang-17/llvm-17-dev feed pet via --with-clang-prefix; the rest bootstrap the
# pluto dev tree (autogen) + isl/cloog/clan. build-essential/gcc for the runtime
# compile of the transformed C; git to clone.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git build-essential \
        clang-17 llvm-17 llvm-17-dev libclang-17-dev \
        autoconf automake libtool pkg-config libgmp-dev libyaml-dev \
        flex bison texinfo libltdl-dev \
    && rm -rf /var/lib/apt/lists/*
# Non-obvious build deps (each found by actually building this image):
#   * pkg-config -- pet/configure uses PKG_CHECK_MODULES(ISL, isl); without its
#     pkg.m4, autoreconf leaves the macro unexpanded and configure dies with
#     `syntax error near unexpected token 'ISL,'`.
#   * libyaml-dev -- pet's YAML dependency.
#   * texinfo     -- candl/configure hard-requires it (`Please install texinfo`),
#     for `makeinfo`.

# The unversioned dev symlink `-lclang-cpp` needs (see header): Ubuntu ships only
# libclang-cpp.so.17. With it present, pet's configure links the monolithic clang lib.
RUN ln -sf /usr/lib/llvm-17/lib/libclang-cpp.so.17 /usr/lib/llvm-17/lib/libclang-cpp.so

# cloog-isl unconditionally builds doc/cloog.pdf via `texi2dvi --pdf` (which needs a
# full TeX engine we do not want to pull in for a doc we never use). Shadow texi2dvi
# with a no-op that just creates the -o target, so the PDF "builds" and install works.
RUN printf '%s\n' '#!/bin/sh' \
      'out=; while [ $# -gt 0 ]; do [ "$1" = "-o" ] && { out=$2; shift; }; shift; done' \
      '[ -n "$out" ] && : > "$out"; exit 0' > /usr/local/bin/texi2dvi \
    && chmod +x /usr/local/bin/texi2dvi

RUN git clone --recursive https://github.com/bondhugula/pluto.git /opt/pluto \
    && cd /opt/pluto \
    && git checkout "$PLUTO_COMMIT" \
    && git submodule update --init --recursive \
    && ./autogen.sh \
    && ./configure --with-clang-prefix=/usr/lib/llvm-17 \
         CC=clang-17 CXX=clang++-17 CXXFLAGS='-std=c++17 -include cstdint' \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig
# The /opt/pluto build tree is intentionally KEPT (not rm'd): polycc bakes absolute
# build-tree paths -- its pluto binary (tool/pluto), inscop, and getversion's .git --
# that `make install` does NOT relocate, so deleting the tree breaks the installed
# polycc. Cost is image size; a lean image would need to relocate all three paths.

# Fail the build if polycc did not build or run -- this image's whole purpose. A
# transform smoke-test (not `polycc --version`, which forwards to the pluto binary and
# exits non-zero on no-input): multidim scop so pet extracts it, output must be non-empty.
RUN printf '%s\n' \
      '#include <stdint.h>' \
      'void mm(const int64_t N, double (*restrict A)[N], double (*restrict B)[N], double (*restrict C)[N]) {' \
      '#pragma scop' \
      '  for (int64_t i=0;i<N;i++) for (int64_t j=0;j<N;j++) for (int64_t k=0;k<N;k++) C[i][j]+=A[i][k]*B[k][j];' \
      '#pragma endscop' \
      '}' > /tmp/smoke.c \
    && cd /tmp && polycc --pet smoke.c -o smoke_out.c && test -s smoke_out.c \
    && echo "polycc OK: $(command -v polycc)"
