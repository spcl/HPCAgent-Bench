# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The build defaults in containers/images/build_common.sh: layer and build caches on, pull first.

A pull replaces a multi-hour build only when the registry image carries the build-inputs fingerprint
of this checkout, so the fingerprint must move with every input and with nothing else. The cache knob
must really build cold when it is off. IMAGE_REQUIREMENTS.md "Build defaults" documents both.
"""

import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CE = ROOT / "containers" / "images"
SHELL_PATH = "/usr/bin:/bin"
DOCKERFILE = """\
FROM ${BASE_IMAGE} AS agent
COPY containers/lib/agent.sh /agent.sh
RUN python3 - <<'PY'
from pathlib import Path
PY
FROM agent AS judge
COPY pkg /opt/pkg
"""


def bash(snippet: str, repo: pathlib.Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``snippet`` with the repo's build_common.sh sourced."""
    common = repo / "containers" / "images" / "build_common.sh"
    return subprocess.run(
        ["bash", "-c", f'source "$1"; {snippet}', "bash", str(common)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": SHELL_PATH, "USER": "someone", "HOME": str(repo), **(env or {})},
    )


def git(repo: pathlib.Path, *argv: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *argv],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """A committed checkout with the real build_common.sh and images.env and a two-stage Dockerfile."""
    images = tmp_path / "containers" / "images"
    images.mkdir(parents=True)
    for name in ("build_common.sh", "images.env"):
        shutil.copy2(CE / name, images / name)
    (images / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (tmp_path / "containers" / "lib").mkdir()
    (tmp_path / "containers" / "lib" / "agent.sh").write_text("echo agent\n", encoding="utf-8")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("X = 1\n", encoding="utf-8")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "fixture")
    return tmp_path


def fingerprint(repo: pathlib.Path, target: str, *args: str, base: str = "docker.io/x/base@sha256:1") -> str:
    done = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; ce_build_fingerprint "$2" "$3" "${@:4}"',
            "bash",
            str(repo / "containers" / "images" / "build_common.sh"),
            str(repo / "containers" / "images" / "Dockerfile"),
            target,
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": SHELL_PATH, "USER": "someone", "BASE_IMAGE": base},
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def test_the_fingerprint_is_stable_for_the_same_inputs(repo: pathlib.Path) -> None:
    assert fingerprint(repo, "agent", "--build-arg", "ROCM_ARCH=gfx90a") == fingerprint(
        repo, "agent", "--build-arg", "ROCM_ARCH=gfx90a"
    )


def test_a_build_arg_moves_the_fingerprint(repo: pathlib.Path) -> None:
    assert fingerprint(repo, "agent", "--build-arg", "ROCM_ARCH=gfx90a") != fingerprint(
        repo, "agent", "--build-arg", "ROCM_ARCH=gfx942"
    )


def test_mounts_and_a_local_base_copy_do_not_move_the_fingerprint(repo: pathlib.Path) -> None:
    """A cache mount or ce_cache_base_image's dir: rewrite is where the bytes come from, not what they are."""
    plain = fingerprint(repo, "agent", "--build-arg", "ROCM_ARCH=gfx90a")
    mounted = fingerprint(
        repo,
        "agent",
        "-v",
        "/somewhere:/pip-cache:rw",
        "--build-arg",
        "ROCM_ARCH=gfx90a",
        "--build-arg",
        "BASE_IMAGE=dir:/cache/base",
    )
    assert mounted == plain


def test_the_base_image_reference_moves_the_fingerprint(repo: pathlib.Path) -> None:
    assert fingerprint(repo, "agent", base="docker.io/x/base@sha256:1") != fingerprint(
        repo, "agent", base="docker.io/x/base@sha256:2"
    )


def test_a_committed_edit_to_a_copied_file_moves_only_the_stages_that_copy_it(repo: pathlib.Path) -> None:
    """The judge stage copies pkg/; the agent stage does not, so its pull must survive a pkg/ edit."""
    before = {t: fingerprint(repo, t) for t in ("agent", "judge")}
    (repo / "pkg" / "mod.py").write_text("X = 2\n", encoding="utf-8")
    git(repo, "commit", "-qam", "edit")
    after = {t: fingerprint(repo, t) for t in ("agent", "judge")}
    assert after["agent"] == before["agent"]
    assert after["judge"] != before["judge"]


def test_an_uncommitted_edit_to_a_copied_file_makes_the_inputs_unknown(repo: pathlib.Path) -> None:
    """The fingerprint reads git blobs; a dirty file would be built but not fingerprinted."""
    (repo / "containers" / "lib" / "agent.sh").write_text("echo edited\n", encoding="utf-8")
    done = bash('ce_build_fingerprint "$(dirname "$1")/Dockerfile" agent', repo)
    assert done.returncode == 1, done.stdout
    assert "no pull" in done.stderr


@pytest.mark.parametrize(
    ("mode", "registry_label", "role", "want_rc"),
    [
        ("1", "abc", "judge-agent-amd", 0),
        ("1", "other", "judge-agent-amd", 1),
        ("1", "", "judge-agent-amd", 1),
        ("0", "abc", "judge-agent-amd", 1),
        ("only", "other", "judge-agent-amd", 0),
        ("1", "abc", "sglang-mi200", 1),
        ("only", "abc", "sglang-mi200", 2),
    ],
)
def test_ce_pull_wanted_pulls_only_a_matching_or_explicitly_requested_image(
    repo: pathlib.Path, mode: str, registry_label: str, role: str, want_rc: int
) -> None:
    """sglang-mi200 has no registry tag in images.env: nothing to pull, and pull-only must fail."""
    stub = f"ce_registry_label() {{ printf '%s' {registry_label!r}; }}"
    done = bash(f"{stub}; ce_pull_wanted {role} abc", repo, {"CE_PULL": mode})
    assert done.returncode == want_rc, done.stdout + done.stderr


@pytest.mark.parametrize(
    ("knob", "want_flags", "store_kept"), [("1", "--layers=true", True), ("0", "--no-cache", False)]
)
def test_the_layer_store_survives_only_with_the_cache_on(
    repo: pathlib.Path, tmp_path: pathlib.Path, knob: str, want_flags: str, store_kept: bool
) -> None:
    tmpfs = tmp_path / "tmpfs"
    (tmpfs / "root" / "layers").mkdir(parents=True)
    (tmpfs / "runroot").mkdir()
    stubs = tmp_path / "bin"
    stubs.mkdir()
    podman = stubs / "podman"
    podman.write_text('#!/bin/sh\n[ "$1" = unshare ] && { shift; exec "$@"; }\nexit 0\n', encoding="ascii")
    podman.chmod(0o755)
    done = bash(
        'ce_podman_env >/dev/null; printf "%s" "${CE_BUILD_FLAGS[*]}"',
        repo,
        {"CE_BUILD_CACHE": knob, "CE_TMPFS": str(tmpfs), "PATH": f"{stubs}:{SHELL_PATH}"},
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout == want_flags
    assert (tmpfs / "root" / "layers").is_dir() == store_kept
    assert not (tmpfs / "runroot").exists(), "runtime state must never outlive its job"


def test_the_cache_knob_off_mounts_no_build_cache(repo: pathlib.Path) -> None:
    done = bash('ce_cache_args spack pip >/dev/null; printf "%s" "${#CACHE_ARGS[@]}"', repo, {"CE_BUILD_CACHE": "0"})
    assert (done.returncode, done.stdout) == (0, "0"), done.stderr


def test_the_cache_knob_on_mounts_the_spack_and_pip_caches(repo: pathlib.Path, tmp_path: pathlib.Path) -> None:
    done = bash(
        'ce_cache_args spack-buildcache pip-cache/gfx90a >/dev/null; printf "%s\\n" "${CACHE_ARGS[@]}"',
        repo,
        {"SCRATCH": str(tmp_path)},
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [
        "-v",
        f"{tmp_path}/spack-buildcache:/spack-buildcache:rw",
        "-v",
        f"{tmp_path}/pip-cache/gfx90a:/pip-cache:rw",
    ]
