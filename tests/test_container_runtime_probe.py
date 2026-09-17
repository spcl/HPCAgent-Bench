# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""scripts/cscs/container_runtime.sh: `enroot` unless the caller names a runtime.

`ce` applies the EDF comm hooks to every step, and single-node inference then dies at RCCL init
("Failed to initialize any NET plugin", jobs 640160-640181). The chooser must therefore default to
`enroot` whatever the site's pyxis state is, and an explicit CONTAINER_RUNTIME must still win.
"""

import pathlib
import subprocess

import pytest

from hpcagent_bench import paths

CHOOSER = paths.ROOT / "scripts" / "cscs" / "container_runtime.sh"


def choose(tmp_path: pathlib.Path, **env: str) -> str:
    """Run the chooser in a minimal environment with a writable site enroot cache (pyxis usable)."""
    conf = tmp_path / "enroot.conf"
    conf.write_text(f"ENROOT_CACHE_PATH {tmp_path}/scratch/$(id -nu)/.enroot\n")
    base = {"PATH": "/usr/bin:/bin", "SITE_ENROOT_CONF": str(conf)}
    result = subprocess.run(["bash", str(CHOOSER)], env={**base, **env}, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def test_the_default_is_enroot_even_when_pyxis_could_start_containers(tmp_path: pathlib.Path) -> None:
    assert choose(tmp_path) == "enroot"


@pytest.mark.parametrize("runtime", ["ce", "enroot", "apptainer"])
def test_an_explicit_runtime_always_wins(tmp_path: pathlib.Path, runtime: str) -> None:
    assert choose(tmp_path, CONTAINER_RUNTIME=runtime) == runtime


def test_an_empty_runtime_counts_as_unset(tmp_path: pathlib.Path) -> None:
    assert choose(tmp_path, CONTAINER_RUNTIME="") == "enroot"
