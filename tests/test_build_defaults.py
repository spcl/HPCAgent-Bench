# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The build defaults in containers/images/build_common.sh: layer and build caches on, pull first.

A pull replaces a multi-hour build only when the registry image carries the build-inputs fingerprint
of this checkout, so the fingerprint must move with every input and with nothing else. The cache knob
must really build cold when it is off. IMAGE_REQUIREMENTS.md "Build defaults" documents both.
"""

import pathlib
import re
import shutil
import subprocess

import pytest

from hpcagent_bench import paths

ROOT = paths.ROOT
CE = ROOT / "containers" / "images"
SHELL_PATH = "/usr/bin:/bin"
DOCKERFILE = """\
FROM ${BASE_IMAGE} AS agent
COPY containers/images/lib/agent.sh /agent.sh
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
    (tmp_path / "containers" / "images" / "lib").mkdir()
    (tmp_path / "containers" / "images" / "lib" / "agent.sh").write_text("echo agent\n", encoding="utf-8")
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
        "/somewhere:/uv-cache:rw",
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
    (repo / "containers" / "images" / "lib" / "agent.sh").write_text("echo edited\n", encoding="utf-8")
    done = bash('ce_build_fingerprint "$(dirname "$1")/Dockerfile" agent', repo)
    assert done.returncode == 1, done.stdout
    assert "no pull" in done.stderr


def test_a_source_on_a_copy_continuation_line_is_a_build_input(repo: pathlib.Path) -> None:
    """Every line of a COPY names sources, so an edit to any of them moves the fingerprint or blocks the pull."""
    dockerfile = repo / "containers" / "images" / "Dockerfile"
    dockerfile.write_text(
        "FROM ${BASE_IMAGE} AS agent\nCOPY containers/images/lib/agent.sh \\\n     pkg/mod.py /opt/\n", encoding="utf-8"
    )
    git(repo, "commit", "-qam", "continuation")
    before = fingerprint(repo, "agent")
    (repo / "pkg" / "mod.py").write_text("X = 2\n", encoding="utf-8")
    done = bash('ce_build_fingerprint "$(dirname "$1")/Dockerfile" agent', repo)
    assert done.returncode == 1, done.stdout
    assert "pkg/mod.py" in done.stderr
    git(repo, "commit", "-qam", "edit")
    assert fingerprint(repo, "agent") != before


def move_locked(lock: pathlib.Path, package: str) -> None:
    """Change one hash of the package's lock entry, which is what any new release of it does."""
    text, moved = re.subn(
        rf'(name = "{package}"\n.*?hash = "sha256:)(.)',
        lambda found: found[1] + ("1" if found[2] == "0" else "0"),
        lock.read_text(encoding="utf-8"),
        count=1,
        flags=re.DOTALL,
    )
    assert moved, package
    lock.write_text(text, encoding="utf-8")


@pytest.mark.parametrize("image", ["judge-agent-amd", "judge-agent-cpu", "judge-agent-cuda"])
def test_only_a_package_the_image_installs_moves_its_fingerprint_with_the_lock(
    tmp_path: pathlib.Path, image: str
) -> None:
    """A wheel the launch venv installs must not rebuild the agent target, whose source builds take hours."""
    images = tmp_path / "containers" / "images"
    images.mkdir(parents=True)
    for name in ("build_common.sh", "images.env", f"{image}/Dockerfile"):
        (images / name).parent.mkdir(exist_ok=True)
        shutil.copy2(CE / name, images / name)
    for name in ("pyproject.toml", "uv.lock", "agent/pyproject.toml", "adapters/hpcagent_bench/pyproject.toml"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, tmp_path / name)
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "fixture")
    uv = pathlib.Path(str(shutil.which("uv"))).parent

    def agent_fingerprint() -> str:
        snippet = (
            "BUILD_ARGS=(); ce_lock_keys >/dev/null && ce_build_args IMAGE_PYTHON_LOCK && "
            f'ce_build_fingerprint {images / image / "Dockerfile"} agent "${{BUILD_ARGS[@]}}"'
        )
        done = bash(snippet, tmp_path, {"PATH": f"{uv}:{SHELL_PATH}", "BASE_IMAGE": "x@sha256:1"})
        assert done.returncode == 0, done.stderr
        return done.stdout.strip()

    before = agent_fingerprint()
    lock = tmp_path / "uv.lock"
    move_locked(lock, "jaxlib")
    project = tmp_path / "pyproject.toml"
    project.write_text(project.read_text(encoding="utf-8").replace('version = "', 'version = "9', 1), encoding="utf-8")
    git(tmp_path, "commit", "-qam", "a wheel and the release move")
    assert agent_fingerprint() == before
    # A harness bump: what the launch hook installs, from uv.lock and the npm lock, and the playwright crawl4ai drives.
    for package in ("litellm", "mini-swe-agent", "openai-agents", "anthropic", "playwright"):
        move_locked(lock, package)
    npm_lock = tmp_path / "agent" / "harness" / "node" / "package-lock.json"
    npm_lock.parent.mkdir(parents=True)
    npm_lock.write_text("{}", encoding="utf-8")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "the harnesses move")
    assert agent_fingerprint() == before
    for package in ("numpy", "mpi4py", "meson-python"):
        move_locked(lock, package)
        git(tmp_path, "commit", "-qam", package)
        moved = agent_fingerprint()
        assert moved != before, package
        before = moved


@pytest.mark.parametrize(
    ("mode", "registry_label", "role", "want_rc"),
    [
        ("1", "abc", "judge-agent-amd", 0),
        ("1", "other", "judge-agent-amd", 1),
        ("1", "", "judge-agent-amd", 1),
        ("0", "abc", "judge-agent-amd", 1),
        ("only", "other", "judge-agent-amd", 0),
        ("1", "abc", "judge-mlscale", 1),
        ("only", "abc", "judge-mlscale", 2),
    ],
)
def test_ce_pull_wanted_pulls_only_a_matching_or_explicitly_requested_image(
    repo: pathlib.Path, mode: str, registry_label: str, role: str, want_rc: int
) -> None:
    """judge-mlscale has no registry tag in images.env: nothing to pull, and pull-only must fail."""
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
        {"CE_BUILD_CACHE": knob, "CE_LAYER_CACHE": "0", "CE_TMPFS": str(tmpfs), "PATH": f"{stubs}:{SHELL_PATH}"},
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout == want_flags
    assert (tmpfs / "root" / "layers").is_dir() == store_kept
    assert not (tmpfs / "runroot").exists(), "runtime state must never outlive its job"


def test_a_layer_registry_that_cannot_start_costs_the_resume_not_the_build(
    repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """On by default, so a machine without its download must still build."""
    tmpfs = tmp_path / "tmpfs"
    tmpfs.mkdir()
    stubs = tmp_path / "bin"
    stubs.mkdir()
    for name, body in (("podman", '[ "$1" = unshare ] && { shift; exec "$@"; }\nexit 0'), ("curl", "exit 22")):
        (stubs / name).write_text(f"#!/bin/sh\n{body}\n", encoding="ascii")
        (stubs / name).chmod(0o755)
    done = bash(
        'ce_podman_env; printf "repo=[%s] flags=%s" "${CE_LAYER_REPO}" "${CE_BUILD_FLAGS[*]}"',
        repo,
        {"JIT_CACHE_ROOT": str(tmp_path / "jit"), "CE_TMPFS": str(tmpfs), "PATH": f"{stubs}:{SHELL_PATH}"},
    )
    assert done.returncode == 0, done.stderr
    assert "repo=[] flags=--layers=true" in done.stdout
    assert "only on this node" in done.stdout


def test_the_cache_knob_off_mounts_no_build_cache(repo: pathlib.Path) -> None:
    done = bash('ce_cache_args spack uv >/dev/null; printf "%s" "${#CACHE_ARGS[@]}"', repo, {"CE_BUILD_CACHE": "0"})
    assert (done.returncode, done.stdout) == (0, "0"), done.stderr


def test_the_cache_knob_on_mounts_the_spack_and_pip_caches(repo: pathlib.Path, tmp_path: pathlib.Path) -> None:
    done = bash(
        'ce_cache_args spack-buildcache uv-cache/gfx90a >/dev/null; printf "%s\\n" "${CACHE_ARGS[@]}"',
        repo,
        {"SCRATCH": str(tmp_path)},
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [
        "-v",
        f"{tmp_path}/spack-buildcache:/spack-buildcache:rw",
        "-v",
        f"{tmp_path}/uv-cache/gfx90a:/uv-cache:rw",
    ]
