#!/usr/bin/env python3
"""Fail if anything the image needs to FUNCTION comes from outside the image.

Run it inside a container. Mounted host filesystems are fine as DATA -- benchmarks are read and
results are written there -- but nothing the software stack needs to run may resolve to one, and
an image that quietly picks up a host tree is not reproducible: its behaviour then depends on the
state of somebody's scratch directory rather than on its digest.

This has happened twice. flydsl used to arrive through PYTHONPATH=${SCRATCH}/pyprefix/..., an
upgrade invisible to the image digest, which is what the sglang consolidation exists to kill; and
${SCRATCH}/dace shadowed the image's own /opt/dace because sys.path starts with the CWD, handing
back an empty namespace package that looked like a packaging fault.

  python3 selfcontained_check.py [--modules numpy,torch,...]

Exit status is the number of things resolving outside, so a gate can use it directly.
"""

import argparse
import importlib
import pathlib
import os
import shutil
import subprocess
import sys

#: A path under any of these is host-mounted on this cluster, never part of an image.
OUTSIDE = ("/capstor", "/iopsstor", "/users", "/home")

#: Executables a graded kernel can reach for. Missing is reported, but only an OUTSIDE one fails:
#: an image without hipcc is a different complaint than an image borrowing the host's.
BINARIES = ("python3", "gcc", "g++", "gfortran", "mpicc", "mpiexec", "hipcc", "cmake", "ninja")

DEFAULT_MODULES = "numpy,scipy,pandas,sympy,networkx,torch,cupy,dace,mpi4py,islpy,z3,numba"


def drop_own_directory_from_path() -> None:
    """Take this script's own directory off sys.path before importing anything.

    THE CHECKER FOUND THIS ON ITSELF. Python puts the script's directory at sys.path[0], and this
    script lives beside directories named `sglang`, `vllm` and `vllm-0271` -- so `import sglang`
    resolved to the ce-images source tree as a namespace package and the gate reported an image
    that ships SGLang as depending on the host. Exactly the shadowing it exists to detect, which
    is precisely why it must not do it itself.
    """
    own = str(pathlib.Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if p not in ("", ".", own)]


def outside(path: str) -> bool:
    return path.startswith(OUTSIDE)


def check_modules(names: list[str]) -> list[str]:
    bad = []
    print("modules")
    for name in names:
        try:
            module = importlib.import_module(name)
            # A namespace package has __file__ None -- that is the shadowing failure, not an
            # absence, so it is reported as broken rather than skipped.
            path = module.__file__ or f"NAMESPACE PACKAGE (shadowed): {list(module.__path__)}"
        except ImportError as exc:
            print(f"  {name:12s} absent      {exc}"[:110])
            continue
        if module.__file__ is None or outside(path):
            bad.append(name)
            print(f"  {name:12s} OUTSIDE     {path}"[:110])
        else:
            print(f"  {name:12s} in-image    {path}"[:110])
    return bad


def check_binaries() -> list[str]:
    bad = []
    print("\nexecutables")
    for name in BINARIES:
        found = shutil.which(name)
        if found is None:
            print(f"  {name:12s} absent")
        elif outside(os.path.realpath(found)):
            bad.append(name)
            print(f"  {name:12s} OUTSIDE     {found} -> {os.path.realpath(found)}"[:110])
        else:
            print(f"  {name:12s} in-image    {found}")
    return bad


def check_loader() -> list[str]:
    """Shared libraries the MPI stack pulls in, as the loader actually resolves them."""
    bad = []
    print("\nlinked libraries (mpicc probe)")
    exe = shutil.which("mpicc")
    if exe is None:
        print("  mpicc absent; skipping")
        return bad
    src, out = "/tmp/.sc_probe.c", "/tmp/.sc_probe"
    with open(src, "w") as handle:
        handle.write("#include <mpi.h>\nint main(int c, char **v){MPI_Init(&c,&v);MPI_Finalize();return 0;}\n")
    if subprocess.run([exe, src, "-o", out], capture_output=True).returncode != 0:
        print("  probe did not compile; skipping")
        return bad
    ldd = subprocess.run(["ldd", out], capture_output=True, text=True).stdout
    for line in ldd.splitlines():
        if "=>" not in line:
            continue
        resolved = line.split("=>", 1)[1].strip().split(" ")[0]
        if resolved and outside(resolved):
            bad.append(resolved)
            print(f"  OUTSIDE  {line.strip()}"[:110])
    print(f"  {len(ldd.splitlines())} libraries, {len(bad)} outside")
    return bad


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modules", default=DEFAULT_MODULES)
    args = parser.parse_args()

    print(f"interpreter  {sys.executable}")
    print(f"PYTHONPATH   {os.environ.get('PYTHONPATH', '(unset)')}")
    print(f"PYTHONSAFEPATH {os.environ.get('PYTHONSAFEPATH', '(unset)')}")
    print(f"cwd          {os.getcwd()}\n")

    drop_own_directory_from_path()
    bad = check_modules([m for m in args.modules.split(",") if m])
    bad += check_binaries()
    bad += check_loader()

    print(f"\nresolving outside the image: {bad if bad else 'nothing'}")
    return len(bad)


if __name__ == "__main__":
    sys.exit(main())
