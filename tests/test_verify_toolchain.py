# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``scripts/verify_toolchain.py`` -- the CI gate that refuses a half-provisioned runner.

The gate must agree with the harness about what "present" means. When it was stricter,
CI went red on a toolchain every test then used successfully.
"""
import importlib.util
import pathlib
import stat

import pytest

from hpcagent_bench import languages
from hpcagent_bench.languages import resolve_compiler

REPO = pathlib.Path(__file__).resolve().parents[1]

#: How the runner actually spells its drivers: LLVM's Fortran driver arrives ONLY as
#: ``flang-new-<major>`` (apt's ``flang`` is a metapackage), while gcc ships unversioned.
FAKE_PATH_ENTRIES = ("make", "gcc", "g++", "gfortran", "clang", "clang++", "flang-new-18")


def load_script():
    """Import ``scripts/verify_toolchain.py`` as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("verify_toolchain", REPO / "scripts" / "verify_toolchain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_executable(path: pathlib.Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def fake_toolchain(tmp_path, monkeypatch):
    """A PATH holding nothing but the entries above, plus a pkg-config and a link probe that
    both say yes -- the link rows have their own test below."""
    for name in FAKE_PATH_ENTRIES:
        write_executable(tmp_path / name, "#!/bin/sh\nexit 0\n")
    write_executable(tmp_path / "pkg-config", "#!/bin/sh\necho -lopenblas\n")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(languages, "library_linkable", lambda soname: True)
    # resolve_compiler is lru_cached on the name, so a stale entry would survive the new PATH.
    resolve_compiler.cache_clear()
    yield tmp_path
    resolve_compiler.cache_clear()


def test_a_versioned_only_llvm_driver_counts_as_present(fake_toolchain):
    """The exact CI failure: `command -v flang` misses flang-new-18, the harness resolves it."""
    rows = dict(load_script().probe())
    assert rows["flang"] == str(fake_toolchain / "flang-new-18")
    assert all(found is not None for found in rows.values()), f"unexpected MISS: {rows}"


def test_a_complete_toolchain_exits_zero(fake_toolchain):
    assert load_script().main() == 0


def test_a_missing_driver_fails_loudly(fake_toolchain, capsys):
    """A gate that tolerated a missing driver would let the tests needing it skip silently."""
    (fake_toolchain / "gfortran").unlink()
    resolve_compiler.cache_clear()
    assert load_script().main() == 1
    assert "MISS  gfortran" in capsys.readouterr().out


def test_a_missing_library_fails_loudly(fake_toolchain, capsys):
    """openblas resolves through pkg-config, not PATH, so it needs its own MISS row."""
    write_executable(fake_toolchain / "pkg-config", "#!/bin/sh\nexit 1\n")
    assert load_script().main() == 1
    assert "MISS  pkg-config:openblas" in capsys.readouterr().out


def test_an_unlinkable_runtime_fails_loudly(fake_toolchain, monkeypatch, capsys):
    """libomp is a HARD requirement, not an extra: libgomp deadlocks across fork() and libomp
    recovers, so a runner with only libgomp cannot tell the fix from the forgiving runtime.
    apt's libomp-dev is a metapackage, so `installed` and `linkable` are different questions."""
    monkeypatch.setattr(languages, "library_linkable", lambda soname: soname != "omp")
    assert load_script().main() == 1
    assert "MISS  -lomp" in capsys.readouterr().out
