# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""An OpenMP shared library must still LOAD after it links.

LLVM 17 and later park ``libomp.so`` under a target-triple libdir
(``lib/x86_64-unknown-linux-gnu``) that no loader searches, and clang links it by absolute path
while writing no RUNPATH. The .so builds clean, reports success, and then dies at ``dlopen`` with
``libomp.so: cannot open shared object file``. Measured on spack clang 22.1.8, where it took the
REFERENCE build down and voided every graded call of four campaign arms -- a whole column of zeros
behind a build line that said OK.

The hermetic tests drive :func:`languages.driver_library_dir` with stub drivers, so the three
answers a driver can give are all covered on a host with no compiler at all. The build test is the
one that would have caught it: it links and then loads.
"""

import ctypes
import os
import pathlib
import stat

import pytest

from hpcagent_bench import languages

#: A translation unit that references the OpenMP runtime and nothing else, so a load failure can
#: only come from that runtime being unreachable.
_OMP_TU = """#include <omp.h>
double probe(double *a, int n) {
  double s = 0.0;
#pragma omp parallel for reduction(+ : s)
  for (int i = 0; i < n; ++i) {
    a[i] *= 2.0;
    s += a[i];
  }
  return s + omp_get_max_threads();
}
"""


def _stub_driver(tmp_path, answer):
    """A fake compiler that answers ``-print-file-name`` with ``answer``, as a real driver does."""
    script = tmp_path / "stubcc"
    script.write_text(f'#!/bin/sh\nprintf "%s\\n" "{answer}"\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def test_a_runtime_in_a_libdir_no_loader_searches_earns_an_rpath(tmp_path) -> None:
    libdir = tmp_path / "lib" / "x86_64-unknown-linux-gnu"
    libdir.mkdir(parents=True)
    (libdir / "libomp.so").write_bytes(b"")
    cc = _stub_driver(tmp_path, libdir / "libomp.so")
    languages.driver_library_dir.cache_clear()
    assert languages.driver_library_dir(cc, ("libomp.so",)) == str(libdir)


def test_a_runtime_the_loader_already_finds_earns_none(tmp_path) -> None:
    resident = pathlib.Path("/usr/lib/x86_64-linux-gnu/libgomp.so")
    if not resident.exists():
        pytest.skip(f"{resident} is not installed on this host")
    cc = _stub_driver(tmp_path, resident)
    languages.driver_library_dir.cache_clear()
    assert languages.driver_library_dir(cc, ("libgomp.so",)) == ""


def test_a_driver_that_cannot_place_the_name_earns_none(tmp_path) -> None:
    # What gcc answers for libomp.so: the name straight back, with no path in front of it.
    cc = _stub_driver(tmp_path, "libomp.so")
    languages.driver_library_dir.cache_clear()
    assert languages.driver_library_dir(cc, ("libomp.so",)) == ""


def test_a_library_only_library_path_can_reach_is_still_named(tmp_path, monkeypatch) -> None:
    # The allocator's case: the driver cannot place libmimalloc.so, and the only directory that
    # can is the one toolchain_env() is about to drop. Naming it is the whole fix.
    viewdir = tmp_path / "view" / "lib"
    viewdir.mkdir(parents=True)
    (viewdir / "libmimalloc.so").write_bytes(b"")
    cc = _stub_driver(tmp_path, "libmimalloc.so")  # what a driver answers when it cannot place it
    monkeypatch.setenv("LIBRARY_PATH", f"/nonexistent:{viewdir}")
    languages.driver_library_dir.cache_clear()
    assert languages.driver_library_dir(cc, ("libmimalloc.so",)) == str(viewdir)
    languages.driver_library_dir.cache_clear()


@pytest.mark.parametrize("block", ["clang", "gcc"])
def test_an_openmp_shared_library_loads_after_it_links(tmp_path, block) -> None:
    blocks = languages.compiler_names()
    if block not in blocks:
        pytest.skip(f"no {block!r} block in compilers.yaml")
    cc = languages.compiler_driver(block)
    if languages.resolve_compiler(cc) is None:
        pytest.skip(f"{cc} is not installed on this host")
    src = tmp_path / "omp_probe.c"
    src.write_text(_OMP_TU)
    lib = tmp_path / "libomp_probe.so"
    cmds = languages.build_shared_lib_commands("c", src, lib, compiler=block)
    failed, log = languages.run_build_commands(cmds, tmp_path)
    assert not failed, log
    assert lib.exists(), log
    # The environment is deliberately NOT the one the build ran under: a graded .so is dlopen'd by
    # a judge process whose LD_LIBRARY_PATH is nobody's business, so the rpath has to carry it.
    ctypes.CDLL(str(lib))


def test_the_link_line_carries_the_flag_and_its_runtime() -> None:
    if "clang" not in languages.compiler_names():
        pytest.skip("no 'clang' block in compilers.yaml")
    cc = languages.compiler_driver("clang")
    if languages.resolve_compiler(cc) is None:
        pytest.skip(f"{cc} is not installed on this host")
    cmds = languages.build_shared_lib_commands("c", pathlib.Path("k.c"), pathlib.Path("libk.so"), compiler="clang")
    link = cmds[-1]
    flag = next((t for t in link if t in languages.OPENMP_BASELINE_FLAGS), None)
    assert flag is not None, link
    runtime = languages.driver_library_dir(cc, languages.OPENMP_RUNTIME_SONAMES.get(flag, ("libomp.so", "libgomp.so")))
    if not runtime:
        pytest.skip(f"{cc} resolves its OpenMP runtime without help")
    assert f"-Wl,-rpath,{runtime}" in link, link
    assert os.path.exists(runtime)
