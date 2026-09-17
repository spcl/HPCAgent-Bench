# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""scripts/cscs/container_runtime.sh: pyxis (`ce`) or the enroot fallback, decided by whether pyxis
could create its cache directory.

The first version asked whether the cache path's filesystem ROOT exists. On compute nodes /capstor
survives as an empty root-owned mount point, so the probe answered `ce` and job 640052 died in pyxis
with "mkdir: cannot create directory '/capstor/scratch/cscs'" before any role started.
"""

import os
import pathlib
import subprocess

import pytest

from hpcagent_bench import paths

PROBE = paths.ROOT / "scripts" / "cscs" / "container_runtime.sh"


def _probe(tmp_path: pathlib.Path, cache_line: str, **env: str) -> str:
    conf = tmp_path / "enroot.conf"
    conf.write_text(f"ENROOT_RUNTIME_PATH         /dev/shm/$(id -nu)/enrootrun\n{cache_line}\n")
    base = {"PATH": "/usr/bin:/bin", "SITE_ENROOT_CONF": str(conf)}
    result = subprocess.run(["bash", str(PROBE)], env={**base, **env}, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def test_a_cache_under_a_non_writable_mount_point_means_enroot(tmp_path: pathlib.Path) -> None:
    mount = tmp_path / "capstor" / "scratch"
    mount.mkdir(parents=True)
    mount.chmod(0o555)
    try:
        assert _probe(tmp_path, f"ENROOT_CACHE_PATH {mount}/cscs/$(id -nu)/.enroot") == "enroot"
    finally:
        mount.chmod(0o755)


def test_a_cache_on_a_filesystem_that_is_gone_means_enroot(tmp_path: pathlib.Path) -> None:
    assert _probe(tmp_path, "ENROOT_CACHE_PATH /no-such-filesystem-here/scratch/$(id -nu)/.enroot") == "enroot"


def test_a_cache_the_user_can_create_means_pyxis(tmp_path: pathlib.Path) -> None:
    """The username is expanded from the site's literal `$(id -nu)`, space and all."""
    assert _probe(tmp_path, f"ENROOT_CACHE_PATH   {tmp_path}/scratch/$(id -nu)/.enroot") == "ce"


def test_an_explicit_runtime_always_wins(tmp_path: pathlib.Path) -> None:
    assert _probe(tmp_path, "ENROOT_CACHE_PATH /no-such-filesystem-here/x", CONTAINER_RUNTIME="ce") == "ce"
