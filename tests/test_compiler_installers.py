# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The hand installers of the cc_icx, cc_aocc and cc_nvhpc compilers (``containers/images/lib/install-*.sh``) run or
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
    unnamed = run("install-icx.sh", [str(prefix), str(installer)], tmp_path, VERSION="1.0")
    assert unnamed.returncode == 2 and "CHECKSUM" in unnamed.stderr
    pinned = run("install-icx.sh", [str(prefix), str(installer)], tmp_path)
    assert pinned.returncode == 1, "the default version's pinned digest must refuse a stand-in"
    assert "sha384" in pinned.stderr
    wrong = run("install-icx.sh", [str(prefix), str(installer)], tmp_path, CHECKSUM="sha384:" + "0" * 96)
    assert wrong.returncode == 1, wrong.stderr
    assert not prefix.exists(), "a download that fails its checksum must never run"
    good = run("install-icx.sh", [str(prefix), str(installer)], tmp_path, CHECKSUM=digest(installer, "sha384"))
    assert good.returncode == 0, good.stderr
    assert "icx 1.0" in good.stdout
    assert f"export HPCAGENT_BENCH_ICX_ROOT={prefix}/compiler/1.0" in good.stdout


def test_an_icx_version_without_a_pinned_installer_needs_its_url(tmp_path: pathlib.Path) -> None:
    refused = run("install-icx.sh", [str(tmp_path / "icx")], tmp_path, VERSION="1.0")
    assert refused.returncode == 2 and "none pinned" in refused.stderr
    assert not (tmp_path / "icx").exists()


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


def test_aocc_without_a_checksum_is_refused_before_any_download(tmp_path: pathlib.Path) -> None:
    """AMD publishes no digest and none is pinned, so the installer refuses to run without one."""
    tarball = fake_aocc_tarball(tmp_path, "6.0.0")
    refused = run("install-aocc.sh", [str(tmp_path / "aocc"), str(tarball)], tmp_path)
    assert refused.returncode == 2, refused.stderr
    assert "set CHECKSUM" in refused.stderr
    assert not (tmp_path / "aocc").exists()


def fake_nvhpc_tarball(tmp_path: pathlib.Path) -> pathlib.Path:
    """A stand-in for NVIDIA's tarball: an ``install`` that copies ``install_components`` into NVHPC_INSTALL_DIR, with
    a ``compilers/bin/nvc``, a ``makelocalrc`` that writes the localrc, the OpenMP runtime and the BLAS/LAPACK pair
    (empty shared libraries) in ``compilers/lib``, and a bundled ``cuda`` tree beside it."""
    tree = tmp_path / "src" / "nvhpc_2026_2611_Linux_x86_64_cuda_13.4"
    release = tree / "install_components" / "Linux_x86_64" / "26.11"
    (release / "compilers" / "bin").mkdir(parents=True)
    (release / "compilers" / "lib").mkdir(parents=True)
    (release / "cuda" / "bin").mkdir(parents=True)
    (release / "compilers" / "bin" / "nvc").write_text("#!/bin/sh\necho nvc 26.11-0\n")
    (release / "compilers" / "bin" / "makelocalrc").write_text('#!/bin/sh\ntouch "$(dirname "$0")/localrc"\n')
    (release / "cuda" / "bin" / "nvcc").write_text("#!/bin/sh\n")
    (tree / "install").write_text('#!/bin/sh\ncp -r "$(dirname "$0")/install_components/." "$NVHPC_INSTALL_DIR"\n')
    for path in (
        tree / "install",
        release / "compilers" / "bin" / "nvc",
        release / "compilers" / "bin" / "makelocalrc",
    ):
        path.chmod(0o755)
    empty = tmp_path / "empty.c"
    empty.write_text("int nvhpc_stand_in(void) { return 0; }\n")
    for name in ("libnvomp.so", "libblas.so", "liblapack.so"):
        lib = release / "compilers" / "lib" / name
        subprocess.run(["gcc", "-shared", "-fPIC", str(empty), "-o", str(lib), f"-Wl,-soname,{name}"], check=True)
    tarball = tmp_path / f"{tree.name}.tar.gz"
    with tarfile.open(tarball, "w:gz") as archive:
        archive.add(tree, arcname=tree.name)
    return tarball


def test_the_nvhpc_tarball_unpacks_only_under_its_checksum_and_leaves_cuda_out(tmp_path: pathlib.Path) -> None:
    """NVIDIA publishes md5 alone, so the NVHPC installer (and only it) takes one; a CPU column needs no CUDA."""
    if os.uname().machine != "x86_64":
        pytest.skip("the stand-in tarball is the x86_64 one")
    tarball = fake_nvhpc_tarball(tmp_path)
    prefix = tmp_path / "nvhpc"
    unnamed = run("install-nvhpc.sh", [str(prefix), str(tarball)], tmp_path, CUDA="multi")
    assert unnamed.returncode == 2 and "md5sum.txt" in unnamed.stderr
    pinned = run("install-nvhpc.sh", [str(prefix), str(tarball)], tmp_path)
    assert pinned.returncode == 1, "26.11's pinned md5 must refuse a stand-in"
    wrong = run("install-nvhpc.sh", [str(prefix), str(tarball)], tmp_path, CHECKSUM="md5:" + "0" * 32)
    assert wrong.returncode == 1, wrong.stderr
    assert not prefix.exists(), "a download that fails its checksum must never be unpacked"
    good = run("install-nvhpc.sh", [str(prefix), str(tarball)], tmp_path, CHECKSUM=digest(tarball, "md5"))
    assert good.returncode == 0, good.stderr
    root = prefix / "Linux_x86_64" / "26.11" / "compilers"
    assert f"export HPCAGENT_BENCH_NVHPC_ROOT={root}" in good.stdout
    assert not (prefix / "Linux_x86_64" / "26.11" / "cuda").exists()
    # The nvhpc OpenMP context in the install: one runtime by both names, numpy's BLAS soname on NVHPC's BLAS.
    context = root / "omp-context"
    assert (context / "lib" / "libgomp.so.1").resolve() == (root / "lib" / "libnvomp.so").resolve()
    shim = subprocess.run(
        ["readelf", "-d", str(context / "lib" / "libopenblas.so.0")], capture_output=True, text=True, check=True
    )
    assert "[libblas.so]" in shim.stdout and "[liblapack.so]" in shim.stdout
    assert f"prefix={context}/view" in (context / "view" / "lib" / "pkgconfig" / "openblas.pc").read_text()
    again = run("nvhpc_context.sh", [str(root)], tmp_path)
    assert again.returncode == 0, again.stderr
    assert (context / "lib" / "libopenblas.so.0").is_file() and not (root / "omp-context.new").exists()


def test_only_the_nvhpc_installer_accepts_an_md5(tmp_path: pathlib.Path) -> None:
    installer = fake_icx_installer(tmp_path)
    refused = run(
        "install-icx.sh", [str(tmp_path / "icx"), str(installer)], tmp_path, CHECKSUM=digest(installer, "md5")
    )
    assert refused.returncode == 2, refused.stderr
    assert not (tmp_path / "icx").exists()
