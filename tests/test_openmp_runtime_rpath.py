# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""An OpenMP shared library still LOADS after it links (LLVM >= 17 keeps ``libomp.so`` in a triple libdir with no
RUNPATH): :func:`languages.driver_library_dir` with stub drivers, plus a link-and-load test."""

import ctypes
import pathlib
import stat

import pytest

from hpcagent_bench import languages
from hpcagent_bench.categories import Language
from tests.own_process import isolated

#: A translation unit that references the OpenMP runtime and nothing else, so a load failure can
#: only come from that runtime being unreachable.
OMP_TU = """#include <omp.h>
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


def stub_driver(tmp_path: pathlib.Path, answer: pathlib.Path | str) -> str:
    """A fake compiler that answers ``-print-file-name`` with ``answer``, as a real driver does."""
    script = tmp_path / "stubcc"
    script.write_text(f'#!/bin/sh\nprintf "%s\\n" "{answer}"\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def test_a_runtime_in_a_libdir_no_loader_searches_earns_an_rpath(tmp_path: pathlib.Path) -> None:
    libdir = tmp_path / "lib" / "x86_64-unknown-linux-gnu"
    libdir.mkdir(parents=True)
    (libdir / "libomp.so").write_bytes(b"")
    cc = stub_driver(tmp_path, libdir / "libomp.so")
    languages.driver_library_dir.cache_clear()
    assert languages.driver_library_dir(cc, ("libomp.so",)) == str(libdir)


@pytest.mark.parametrize("resident", [True, False], ids=["in-a-loader-directory", "driver-cannot-place-it"])
def test_a_runtime_the_loader_already_finds_earns_none(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, resident: bool
) -> None:
    if resident:
        # A library in one of the loader's default directories needs no RUNPATH; the directory is the stub's own.
        loader_dir = tmp_path / "usr-lib"
        loader_dir.mkdir()
        library = loader_dir / "libgomp.so"
        library.write_bytes(b"")
        monkeypatch.setattr(languages, "DEFAULT_LOADER_DIRS", (str(loader_dir.resolve()),))
        cc = stub_driver(tmp_path, library)
        name = "libgomp.so"
    else:
        # What gcc answers for libomp.so: the name straight back, with no path in front of it.
        monkeypatch.delenv("LIBRARY_PATH", raising=False)
        cc = stub_driver(tmp_path, "libomp.so")
        name = "libomp.so"
    languages.driver_library_dir.cache_clear()
    assert languages.driver_library_dir(cc, (name,)) == ""


def test_a_library_only_library_path_can_reach_is_still_named(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The allocator's case: the driver cannot place libmimalloc.so, and the only directory that
    # can is the one toolchain_env() is about to drop. Naming it is the whole fix.
    viewdir = tmp_path / "view" / "lib"
    viewdir.mkdir(parents=True)
    (viewdir / "libmimalloc.so").write_bytes(b"")
    cc = stub_driver(tmp_path, "libmimalloc.so")  # what a driver answers when it cannot place it
    monkeypatch.setenv("LIBRARY_PATH", f"/nonexistent:{viewdir}")
    languages.driver_library_dir.cache_clear()
    assert languages.driver_library_dir(cc, ("libmimalloc.so",)) == str(viewdir)
    languages.driver_library_dir.cache_clear()


@pytest.mark.parametrize("block", ["clang", "gcc"])
@isolated
def test_an_openmp_shared_library_loads_after_it_links(tmp_path: pathlib.Path, block: str) -> None:
    blocks = languages.compiler_names()
    if block not in blocks:
        pytest.skip(f"no {block!r} block in compilers.yaml")
    cc = languages.compiler_driver(block)
    if languages.resolve_compiler(cc) is None:
        pytest.skip(f"{cc} is not installed on this host")
    src = tmp_path / "omp_probe.c"
    src.write_text(OMP_TU)
    lib = tmp_path / "libomp_probe.so"
    cmds = languages.build_shared_lib_commands(Language.C, src, lib, compiler=block)
    failed, log = languages.run_build_commands(cmds, tmp_path)
    assert not failed, log
    assert lib.exists(), log
    # The environment is deliberately NOT the one the build ran under: a graded .so is dlopen'd by
    # a judge process whose LD_LIBRARY_PATH is nobody's business, so the rpath has to carry it.
    ctypes.CDLL(str(lib))
