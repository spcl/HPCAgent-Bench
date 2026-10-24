# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Polygeist (``cgeist`` + ``polygeist-opt``) built against the image's OWN LLVM, with Polymer's Pluto scheduler.

The ``polygeist_cpu`` / ``polygeist_gpu`` columns (hpcagent_bench/polyhedral/polygeist.py) need ``cgeist`` to lower
the emitted C scop to affine MLIR, ``polygeist-opt`` for Polymer and MLIR's parallelization passes, and MLIR's ROCm
runtime for the host half of a GPU binary.

Why this source and not llvm/Polygeist: upstream's newest LLVM pin is 26eb4285 (2023-09, 18.0.0git). The randreshg
fork migrates Polygeist and cgeist to LLVM 23-dev; ``llvm22.patch`` takes it the few API steps back to 22.1, ports
Polymer (which the fork left on the old API), lets ``raise-scf-to-affine`` raise loops nested in affine loops, and
drops the lit suites (an installed LLVM exports no FileCheck). Everything after MLIR -- opt, codegen, OpenMP, the
AMDGPU backend -- is the same LLVM every other clang column uses.

cgeist compiles clang's cc1 driver sources and includes clang's private CodeGen headers, and ConvertPolygeistToLLVM
includes MLIR's MemRefToLLVM.cpp, so the llvm-project source of the SAME release is a resource.

Polymer's Pluto backend needs the kumasento Pluto fork, OpenScop and CLooG at the pins of
``tools/polymer/build_polymer_deps.sh``. That script would also build LLVM 10 for pet, Pluto's C front end, which
Polymer never calls (it hands Pluto an OpenScop built from MLIR), so pet and polycc are dropped from the fork's build.
Each dependency installs into its own directory under the prefix, where Polymer's CMake looks, because the fork and
CLooG each carry their own isl.
"""

import re

from spack.package import *
from spack_repo.builtin.build_systems.cmake import CMakePackage, generator


class Polygeist(CMakePackage):
    """C/C++ to MLIR front end (cgeist) and optimizer (polygeist-opt) with Polymer's Pluto scheduler."""

    homepage = "https://github.com/llvm/Polygeist"
    git = "https://github.com/randreshg/Polygeist.git"

    license("Apache-2.0 WITH LLVM-exception")

    # randresh-local-fixes: Polygeist + cgeist migrated to LLVM 23-dev (llvm 23d8651de).
    version("2026.05.14", commit="e503e554c98d1293da4e9bf5bbfc8db96c590ec4")

    resource(
        name="pluto",
        git="https://github.com/kumasento/pluto.git",
        commit="5603283fb3e74fb33c380bb52874972b440d51a2",
        submodules=True,
        destination="polymer-deps",
        placement="pluto",
    )
    resource(
        name="openscop",
        git="https://github.com/periscop/openscop.git",
        commit="37805d8fef38c2d1b8aa8f5c26b40f79100322e7",
        destination="polymer-deps",
        placement="openscop",
    )
    resource(
        name="cloog",
        git="https://github.com/kumasento/cloog.git",
        commit="43cfb85ed1e1ba1c2f27b450498522b35467ace7",
        submodules=True,
        destination="polymer-deps",
        placement="cloog",
    )
    resource(
        name="llvm-project",
        url="https://github.com/llvm/llvm-project/releases/download/llvmorg-22.1.8/llvm-project-22.1.8.src.tar.xz",
        sha256="922f1817a0df7b1489272d18134ee0087a8b068828f87ac63b9861b1a9965888",
        placement="llvm-src",
    )

    patch("llvm22.patch")

    generator("ninja")

    depends_on("c", type="build")
    depends_on("cxx", type="build")
    depends_on("llvm@22.1.8 +mlir", type=("build", "link"))
    depends_on("hip", type=("build", "link"))
    depends_on("gmp")
    depends_on("gmake", type="build")
    depends_on("autoconf", type="build")
    depends_on("automake", type="build")
    depends_on("libtool", type="build")
    depends_on("pkgconfig", type="build")
    depends_on("flex", type="build")
    depends_on("bison", type="build")
    depends_on("texinfo", type="build")

    def polymer_dir(self, name: str) -> str:
        """Where Polymer's CMake (``tools/polymer/cmake/*.cmake``) looks for dependency ``name``'s install."""
        return join_path(self.prefix, "polymer-deps", name, "install")

    @run_before("cmake")
    def build_polymer_deps(self) -> None:
        src = join_path(self.stage.source_path, "polymer-deps")
        # K&R C throughout; gcc 16 defaults to C23, where `f()` takes no arguments. On CC, not CFLAGS: Pluto's
        # configure re-runs each submodule's configure with its own CFLAGS.
        kr_env = {"CC": f"{spack_cc} -std=gnu17"}

        with working_dir(join_path(src, "pluto")):
            configure_ac = open("configure.ac").read()
            configure_ac = re.sub(
                r"dnl Configure PET.*?configure in pet/ failed\]\)\]\)\n", "", configure_ac, flags=re.S
            )
            with open("configure.ac", "w") as f:
                f.write(configure_ac)
            filter_file(
                r"^SUBDIRS = .*$",
                "SUBDIRS = piplib polylib $(MAY_ISL) openscop cloog-isl clan candl lib",
                "Makefile.am",
            )
            Executable("./autogen.sh")()
            Executable("./configure")(f"--prefix={self.polymer_dir('pluto')}", extra_env={"FILECHECK": "yes", **kr_env})
            for sub in ("piplib", "polylib", "isl", "openscop", "cloog-isl", "clan", "candl", "lib"):
                make("-C", sub)
                make("-C", sub, "install", parallel=False)
            make("install-pkgincludeHEADERS", "install-internalincludeHEADERS", parallel=False)

        with working_dir(join_path(src, "openscop")):
            Executable("./autogen.sh")()
            Executable("./configure")(f"--prefix={self.polymer_dir('openscop')}", extra_env=kr_env)
            make()
            make("install", parallel=False)

        with working_dir(join_path(src, "cloog")):
            Executable("./autogen.sh")()
            Executable("./configure")(
                "LDFLAGS=-fPIC",
                "CFLAGS=-fPIC",
                "CPPFLAGS=-fPIC",
                f"--prefix={self.polymer_dir('cloog')}",
                f"--with-osl-prefix={self.polymer_dir('openscop')}",
                extra_env=kr_env,
            )
            make()
            make("install", parallel=False)

        # build_polymer_deps.sh (run by Polymer's CMake) skips every dependency whose .DONE exists.
        deps = join_path(self.prefix, "polymer-deps")
        for name in ("pluto", "openscop", "cloog"):
            touch(join_path(deps, name, ".DONE"))
        mkdirp(join_path(deps, "llvm", "install"))
        touch(join_path(deps, "llvm", "install", ".DONE"))

    def cmake_args(self) -> list[str]:
        llvm = self.spec["llvm"].prefix
        llvm_src = join_path(self.stage.source_path, "llvm-src")
        return [
            self.define("MLIR_DIR", join_path(llvm, "lib", "cmake", "mlir")),
            self.define("CLANG_DIR", join_path(llvm, "lib", "cmake", "clang")),
            self.define("LLVM_SOURCE_DIR", join_path(llvm_src, "llvm")),
            self.define("CMAKE_CXX_FLAGS", f"-idirafter {join_path(llvm_src, 'mlir', 'include')}"),
            self.define("LLVM_TARGETS_TO_BUILD", "X86;AMDGPU"),
            self.define("LLVM_EXTERNAL_LIT", "/bin/true"),
            self.define("POLYGEIST_ENABLE_POLYMER", True),
            self.define("POLYGEIST_POLYMER_ENABLE_PLUTO", True),
            self.define("POLYMER_DEP_DIR", join_path(self.prefix, "polymer-deps")),
        ]

    @run_after("install")
    def install_rocm_runtime(self) -> None:
        """``libmlir_rocm_runtime.so``, the ``mgpu*`` calls ``gpu-to-llvm`` emits: the image's LLVM is built without
        MLIR_ENABLE_ROCM_RUNNER, so it is built here from that LLVM's own source. Host code only, so the LLVM's
        clang++ compiles it against the HIP headers."""
        llvm = self.spec["llvm"].prefix
        hip = self.spec["hip"].prefix
        mkdirp(self.prefix.lib)
        Executable(join_path(llvm.bin, "clang++"))(
            "-O2",
            "-fPIC",
            "-shared",
            "-D__HIP_PLATFORM_AMD__",
            "-DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1",
            f"-I{hip.include}",
            f"-I{llvm.include}",
            join_path(self.stage.source_path, "llvm-src", "mlir", "lib", "ExecutionEngine", "RocmRuntimeWrappers.cpp"),
            "-o",
            join_path(self.prefix.lib, "libmlir_rocm_runtime.so"),
            join_path(llvm.lib, "libLLVMSupport.a"),
            join_path(llvm.lib, "libLLVMDemangle.a"),
            f"-L{hip.lib}",
            "-lamdhip64",
            f"-Wl,-rpath,{hip.lib}",
        )
