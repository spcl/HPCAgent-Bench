# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""No tracked file invokes or names enroot: the one container-runtime seam is the CSCS Container Engine, whose
EDFs mount a squashfs that ``containers/images/build_common.sh`` writes with podman and mksquashfs. Markdown
and shell are scanned with the code, so an ``enroot import`` cannot come back into a script or a doc.
"""

import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]

ENROOT = re.compile(r"enroot", re.IGNORECASE)

#: Repo-relative path -> why that file may name it.
ALLOWED: dict[str, str] = {
    "tests/test_no_enroot.py": "this file spells the name it searches for",
    "tests/test_container_factory.py": "asserts that enroot is not a backend the factory resolves",
}


def tracked_files() -> list[pathlib.Path]:
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True)
    names = [name for name in listed.stdout.split("\0") if name]
    return [REPO / name for name in names if (REPO / name).is_file() and not (REPO / name).is_symlink()]


def mentions(path: pathlib.Path) -> list[int]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    return [number for number, line in enumerate(text.splitlines(), 1) if ENROOT.search(line)]


def test_no_tracked_file_names_enroot_outside_the_allowlist() -> None:
    offenders = []
    for path in tracked_files():
        rel = path.relative_to(REPO).as_posix()
        if rel not in ALLOWED:
            offenders.extend(f"{rel}:{number}" for number in mentions(path))
    assert not offenders, "enroot is dropped; the CE mounts a podman-built squashfs:\n" + "\n".join(offenders)


def test_every_allowlisted_file_still_names_enroot() -> None:
    stale = [rel for rel in ALLOWED if not (REPO / rel).is_file() or not mentions(REPO / rel)]
    assert not stale, f"ALLOWED entries that no longer name enroot: {stale}"
