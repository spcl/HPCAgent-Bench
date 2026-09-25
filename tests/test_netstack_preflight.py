# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""scripts/cscs/netstack_preflight.sh: gate on the artifact netstack, not "host".

"host" mode's only installed plugin (variant=rocm6) needs libamdhip64.so.6, which the ROCm-7.2
images do not ship, so RCCL falls back to sockets or hard-fails. These lock the gate to "artifact"
and prove it checks the pinned version/name tree, using a fixture instead of the site's store
(HPCAGENT_BENCH_NETSTACK_BASE, set by the site layer).
"""

import pathlib
import subprocess

import pytest

from hpcagent_bench import paths

PREFLIGHT = paths.ROOT / "scripts" / "cscs" / "netstack_preflight.sh"


def make_bundle(base: pathlib.Path, version: str, name: str, *, plugin: bool = True, libfabric: bool = True) -> None:
    bundle = base / "x86_64" / version / name
    bundle.mkdir(parents=True)
    if libfabric:
        (bundle / "libfabric.so.1").write_bytes(b"")
    if plugin:
        (bundle / "librccl-net.so").write_bytes(b"")


def run_preflight(tmp_path: pathlib.Path, **env: str) -> subprocess.CompletedProcess[str]:
    base = {"PATH": "/usr/bin:/bin", "HPCAGENT_BENCH_NETSTACK_BASE": str(tmp_path / "netstack")}
    return subprocess.run(["bash", str(PREFLIGHT)], env={**base, **env}, capture_output=True, text=True, check=False)


def test_artifact_with_the_pinned_bundle_present_passes(tmp_path: pathlib.Path) -> None:
    make_bundle(tmp_path / "netstack", "26.08.1", "gpu_rocm7-cxi_13.1.0-ofi_2.6.0-aws_1.20.0")
    result = run_preflight(tmp_path)
    assert result.returncode == 0
    assert "source=artifact version=26.08.1" in result.stdout


def test_host_source_is_rejected(tmp_path: pathlib.Path) -> None:
    make_bundle(tmp_path / "netstack", "26.08.1", "gpu_rocm7-cxi_13.1.0-ofi_2.6.0-aws_1.20.0")
    result = run_preflight(tmp_path, HPCAGENT_BENCH_NETSTACK_SOURCE="host")
    assert result.returncode == 1
    assert "libamdhip64.so.6" in result.stderr


def test_missing_version_dir_fails(tmp_path: pathlib.Path) -> None:
    (tmp_path / "netstack" / "x86_64").mkdir(parents=True)
    result = run_preflight(tmp_path)
    assert result.returncode == 1
    assert "MISSING version dir" in result.stderr


@pytest.mark.parametrize("plugin,libfabric", [(False, True), (True, False)])
def test_missing_a_bundle_file_fails(tmp_path: pathlib.Path, plugin: bool, libfabric: bool) -> None:
    make_bundle(
        tmp_path / "netstack",
        "26.08.1",
        "gpu_rocm7-cxi_13.1.0-ofi_2.6.0-aws_1.20.0",
        plugin=plugin,
        libfabric=libfabric,
    )
    result = run_preflight(tmp_path)
    assert result.returncode == 1
    assert "MISSING" in result.stderr


def test_a_repointed_name_is_a_loud_failure_not_a_silent_fallback(tmp_path: pathlib.Path) -> None:
    make_bundle(tmp_path / "netstack", "26.08.1", "some-other-build")
    result = run_preflight(tmp_path)
    assert result.returncode == 1
    assert "MISSING RCCL plugin" in result.stderr
    assert "some-other-build" in result.stderr


def test_a_site_without_a_netstack_base_skips_the_check(tmp_path: pathlib.Path) -> None:
    empty_site = tmp_path / "site.env"
    empty_site.write_text("")
    result = run_preflight(tmp_path, HPCAGENT_BENCH_NETSTACK_BASE="", HPCAGENT_BENCH_SITE_ENV=str(empty_site))
    assert result.returncode == 0, result.stderr
    assert "fabric check skipped" in result.stderr
