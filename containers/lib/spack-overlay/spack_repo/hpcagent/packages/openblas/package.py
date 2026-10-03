# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""OpenBLAS with runtime dispatch left whole: no NO_AVX512 when +dynamic_dispatch.

The builtin recipe appends NO_AVX512=1 for every target below x86_64_v4, dynamic dispatch or not.
With DYNAMIC_ARCH=1 on an AVX-512 CPU (Zen 4, the MI300A host) that build segfaults in dgemm on a
tall row-major product (M >= 8192, K >= 512), single-threaded and under every OPENBLAS_CORETYPE;
upstream 0.3.34 built with the same flags minus NO_AVX512 does not. The images pin a portable
target (cpu_target.env) and reach AVX-512 through this runtime dispatch, so dropping the flag
keeps the common code portable and gives dispatch the kernels it selects. blas_gate.sh proves it
in every image build.
"""

from spack_repo.builtin.packages.openblas.package import MakefileBuilder as BuiltinMakefileBuilder
from spack_repo.builtin.packages.openblas.package import Openblas as BuiltinOpenblas

from spack.package import *  # noqa: F403


class Openblas(BuiltinOpenblas):
    """The builtin recipe; only its make definitions change (:class:`MakefileBuilder`)."""


class MakefileBuilder(BuiltinMakefileBuilder):
    @property
    def make_defs(self):
        defs = super().make_defs
        if self.spec.satisfies("+dynamic_dispatch"):
            defs = [d for d in defs if d != "NO_AVX512=1"]
        return defs
