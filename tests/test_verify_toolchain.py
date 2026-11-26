# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``helpers/scripts/checks/verify_toolchain.py`` -- the CI gate that refuses a half-provisioned runner.

The gate must agree with the harness about what "present" means. When it was stricter,
CI went red on a toolchain every test then used successfully.
"""

import pathlib
import stat
import types

import pytest

from hpcagent_bench import languages, paths
from hpcagent_bench.languages import resolve_compiler
from tests.fresh_module import module_at

REPO = paths.ROOT

#: How the runner actually spells its drivers: LLVM's Fortran driver arrives ONLY as
#: ``flang-new-<major>`` (apt's ``flang`` is a metapackage), while gcc ships unversioned.
FAKE_PATH_ENTRIES = ("make", "gcc", "g++", "gfortran", "clang", "clang++", "flang-new-22")


def load_script() -> types.ModuleType:
    """Import ``helpers/scripts/checks/verify_toolchain.py`` as a module (helpers/scripts/ is not a package)."""
    return module_at(REPO / "helpers" / "scripts" / "checks" / "verify_toolchain.py")


def write_executable(path: pathlib.Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def fake_toolchain(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """A PATH holding nothing but the entries above, plus a pkg-config and a link probe that
    both say yes -- the link rows have their own test below."""
    for name in FAKE_PATH_ENTRIES:
        write_executable(tmp_path / name, "#!/bin/sh\nexit 0\n")
    for name in ("gcc", "gfortran"):  # -dumpversion: the graded major
        write_executable(tmp_path / name, "#!/bin/sh\necho 16\n")
    write_executable(tmp_path / "pkg-config", "#!/bin/sh\necho -lopenblas\n")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(languages, "library_linkable", lambda soname: True)
    # resolve_compiler is lru_cached on the name, so a stale entry would survive the new PATH.
    resolve_compiler.cache_clear()
    yield tmp_path
    resolve_compiler.cache_clear()


def test_a_complete_toolchain_exits_zero(fake_toolchain: pathlib.Path) -> None:
    """The gate agrees with the harness about what "present" means. The exact CI failure: `command -v flang` misses
    flang-new-22 (LLVM's Fortran driver arrives only versioned), the harness resolves it."""
    assert load_script().main() == 0
    rows = dict(load_script().probe())
    assert rows["flang"] == str(fake_toolchain / "flang-new-22")
    assert all(found is not None for found in rows.values()), f"unexpected MISS: {rows}"


@pytest.mark.parametrize(
    ("broken", "miss"),
    [
        # A gate that tolerated a missing driver would let the tests needing it skip silently.
        ("driver", "MISS  gfortran"),
        # openblas resolves through pkg-config, not PATH, so it needs its own MISS row.
        ("library", "MISS  pkg-config:openblas"),
        # libomp is a HARD requirement, not an extra: libgomp deadlocks across fork() and libomp recovers, so a runner
        # with only libgomp cannot tell a fork-safe harness from the forgiving runtime. apt's libomp-dev is a
        # metapackage, so `installed` and `linkable` are different questions.
        ("runtime", "MISS  -lomp"),
        # GCC 13 rejects -std=c23: every C build would fail as a test, so the gate refuses it first.
        ("old_gcc", "MISS  gcc>=15"),
    ],
)
def test_a_missing_driver_fails_loudly(
    fake_toolchain: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    broken: str,
    miss: str,
) -> None:
    if broken == "driver":
        (fake_toolchain / "gfortran").unlink()
        resolve_compiler.cache_clear()
    elif broken == "library":
        write_executable(fake_toolchain / "pkg-config", "#!/bin/sh\nexit 1\n")
    elif broken == "runtime":
        monkeypatch.setattr(languages, "library_linkable", lambda soname: soname != "omp")
    else:
        write_executable(fake_toolchain / "gcc", "#!/bin/sh\necho 13\n")
    assert load_script().main() == 1
    assert miss in capsys.readouterr().out
