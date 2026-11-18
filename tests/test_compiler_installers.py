# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The hand installers of the cc_icx and cc_aocc compilers (``containers/images/lib/install-{icx,aocc}.sh``) run or
unpack a vendor download only when its digest is the CHECKSUM given (or pinned): a swapped or truncated download
never executes. The tests feed each a local stand-in, so nothing is fetched."""

import hashlib
import os
import pathlib
import subprocess
import tarfile

import pytest

from hpcagent_bench import paths

LIB = paths.ROOT / "containers" / "images" / "lib"


def run(script: str, args: list[str], tmp_path: pathlib.Path, **env: str) -> subprocess.CompletedProcess[str]:
    environment = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "TMPDIR": str(tmp_path)} | env
    return subprocess.run(
        ["bash", str(LIB / script), *args], env=environment, capture_output=True, text=True, check=False
    )


def fake_icx_installer(tmp_path: pathlib.Path) -> pathlib.Path:
    """A stand-in for Intel's offline installer: it lays out ``compiler/<v>/bin/icx`` under --install-dir."""
    installer = tmp_path / "icx_offline.sh"
    installer.write_text(
        'while [ "$#" -gt 0 ]; do [ "$1" = --install-dir ] && dir="$2"; shift; done\n'
        'mkdir -p "$dir/compiler/1.0/bin"\n'
        'printf "#!/bin/sh\\necho icx 1.0\\n" > "$dir/compiler/1.0/bin/icx"\n'
        'chmod +x "$dir/compiler/1.0/bin/icx"\n'
    )
    return installer


def fake_aocc_tarball(tmp_path: pathlib.Path, version: str) -> pathlib.Path:
    """A stand-in for AMD's tarball: ``aocc-compiler-<version>/`` with an install.sh and a bin/clang."""
    tree = tmp_path / "src" / f"aocc-compiler-{version}"
    (tree / "bin").mkdir(parents=True)
    (tree / "install.sh").write_text("#!/bin/sh\n")
    (tree / "bin" / "clang").write_text("#!/bin/sh\necho AOCC clang\n")
    for path in (tree / "install.sh", tree / "bin" / "clang"):
        path.chmod(0o755)
    tarball = tmp_path / f"aocc-compiler-{version}.tar"
    with tarfile.open(tarball, "w") as archive:
        archive.add(tree, arcname=tree.name)
    return tarball


def digest(path: pathlib.Path, algo: str = "sha256") -> str:
    return f"{algo}:{hashlib.new(algo, path.read_bytes()).hexdigest()}"


def test_the_icx_installer_runs_only_under_the_checksum_it_was_given(tmp_path: pathlib.Path) -> None:
    installer = fake_icx_installer(tmp_path)
    prefix = tmp_path / "icx"
    unnamed = run("install-icx.sh", [str(prefix), str(installer)], tmp_path)
    assert unnamed.returncode == 2 and "CHECKSUM" in unnamed.stderr
    wrong = run("install-icx.sh", [str(prefix), str(installer)], tmp_path, CHECKSUM="sha384:" + "0" * 96)
    assert wrong.returncode == 1, wrong.stderr
    assert not prefix.exists(), "a download that fails its checksum must never run"
    good = run("install-icx.sh", [str(prefix), str(installer)], tmp_path, CHECKSUM=digest(installer, "sha384"))
    assert good.returncode == 0, good.stderr
    assert "icx 1.0" in good.stdout


@pytest.mark.parametrize("given", ["", "md5:abc"])
def test_an_aocc_version_without_a_pinned_checksum_needs_a_valid_one(tmp_path: pathlib.Path, given: str) -> None:
    tarball = fake_aocc_tarball(tmp_path, "6.1.0")
    env = {"VERSION": "6.1.0"} | ({"CHECKSUM": given} if given else {})
    refused = run("install-aocc.sh", [str(tmp_path / "aocc"), str(tarball)], tmp_path, **env)
    assert refused.returncode == 2, refused.stderr
    assert not (tmp_path / "aocc").exists()


def test_the_aocc_tarball_unpacks_only_under_its_checksum(tmp_path: pathlib.Path) -> None:
    tarball = fake_aocc_tarball(tmp_path, "6.1.0")
    prefix = tmp_path / "aocc"
    wrong = run(
        "install-aocc.sh", [str(prefix), str(tarball)], tmp_path, VERSION="6.1.0", CHECKSUM="sha256:" + "0" * 64
    )
    assert wrong.returncode == 1, wrong.stderr
    assert not prefix.exists(), "a download that fails its checksum must never be unpacked"
    good = run("install-aocc.sh", [str(prefix), str(tarball)], tmp_path, VERSION="6.1.0", CHECKSUM=digest(tarball))
    assert good.returncode == 0, good.stderr
    assert "AOCC clang" in good.stdout


def test_the_pinned_aocc_checksum_refuses_another_tarball(tmp_path: pathlib.Path) -> None:
    """6.0.0's digest is pinned, so a file that is not AMD's tarball is refused without any CHECKSUM given."""
    tarball = fake_aocc_tarball(tmp_path, "6.0.0")
    refused = run("install-aocc.sh", [str(tmp_path / "aocc"), str(tarball)], tmp_path)
    assert refused.returncode == 1, refused.stderr
    assert "expected 28b97b6c" in refused.stderr
