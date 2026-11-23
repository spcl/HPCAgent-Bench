#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Patch-release every package we own whose main moved since its last release, dependencies first.

The dependency tree is read statically from each repository's pyproject.toml on main (dependencies, extras, groups and
[tool.uv.sources] that name one of our packages), so a package releases only after everything it pins. Each release
runs that repository's own script, which uploads to PyPI with twine and then creates the GitHub release:

* dace-extended: helpers/scripts/dace_extended_release.sh, when spcl/dace@extended moved past the last release;
* the libraries and plugins: helpers/release_patch.sh in a fresh clone of main. It first moves the pins of our
  packages to their newest releases; when one moved it opens a pins/* pull request and stops. This script waits for
  that PR's CI, merges it when green and runs the release again, so a dependency's release always reaches its
  dependents, tested where it lands;
* hpcagent-bench: in this checkout, pins.sh --bump (a pull request, CI-gated and merged the same way), then
  release_patch.sh --skip-libraries.

A pin-bump PR whose CI fails stops the script: the dependency's release broke this repository, so the fix goes into the
dependency, which releases again; then rerun.

Before every step the script waits until all CI runs on the repository's main head are green; a failed run stops it.
A rerun after a failure resumes: an unchanged package has nothing to release and is skipped.

    uv run helpers/scripts/release_all.py --dry-run   # the tree, the order and what each package has pending; change nothing
    uv run helpers/scripts/release_all.py             # release; needs gh (authenticated), uv, TWINE_USER and TWINE_TOKEN
    uv run helpers/scripts/release_all.py --github-only  # everything but the PyPI uploads (no TWINE_*); the next run
                                                         # without it first uploads exactly the dists GitHub carries

A dependent's pins move only to versions on PyPI, so a --github-only release reaches its dependents after that upload.
"""

import argparse
import graphlib
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
import tomllib

from hpcagent_bench import paths

ROOT = paths.ROOT
FORK = "dace-extended"
BENCH = "hpcagent-bench"
#: Our packages and their repositories. dace-extended is the root: the fork's release commit writes its metadata.
SLUGS = {
    FORK: "ThrudPrimrose/dace",
    "fp-arena": "spcl/FP-Arena",
    "dace-relayout": "ThrudPrimrose/dace-relayout",
    "dace-alloy": "ThrudPrimrose/dace-alloy",
    BENCH: "spcl/HPCAgent-Bench",
    "tactics": "spcl/tactics",
}
POLL_SECONDS = 120
REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9._-]+)")
RELEASE_TAG = re.compile(r"refs/tags/(v\d+\.\d+\.\d+)$")


def run(command: list[str], cwd: pathlib.Path = ROOT) -> str:
    """``command``'s stdout; a failure stops the script."""
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def step(command: list[str], cwd: pathlib.Path = ROOT) -> None:
    """Run ``command`` with its output on the terminal; a failure stops the script."""
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def pyproject(slug: str) -> dict[str, object]:
    """The pyproject.toml on ``slug``'s main."""
    text = run(["gh", "api", f"repos/{slug}/contents/pyproject.toml", "-H", "Accept: application/vnd.github.raw"])
    return tomllib.loads(text)


def requirement_names(table: dict[str, object]) -> set[str]:
    """Every distribution named by the project's dependencies, extras, dependency groups and uv sources."""
    project = table.get("project", {})
    lists: list[object] = [project.get("dependencies", []), *project.get("optional-dependencies", {}).values()]
    lists += table.get("dependency-groups", {}).values()
    names = {
        match.group(1).lower()
        for entries in lists
        for entry in entries
        if isinstance(entry, str) and (match := REQUIREMENT_NAME.match(entry))
    }
    return names | set(table.get("tool", {}).get("uv", {}).get("sources", {}))


def dependency_tree() -> dict[str, set[str]]:
    """Each package's dependencies among our packages (dace-extended has none)."""
    tree = {FORK: set()}
    for name, slug in SLUGS.items():
        if name != FORK:
            tree[name] = (requirement_names(pyproject(slug)) & set(SLUGS)) - {name}
    return tree


def main_head(slug: str) -> str:
    """The commit ``slug``'s main points at."""
    return run(["git", "ls-remote", f"https://github.com/{slug}.git", "refs/heads/main"]).split()[0]


def newest_tag(slug: str) -> str:
    """``slug``'s newest vX.Y.Z tag, or ""."""
    lines = run(["git", "ls-remote", "--tags", f"https://github.com/{slug}.git"]).splitlines()
    tags = [match.group(1) for line in lines if (match := RELEASE_TAG.search(line))]
    return max(tags, key=lambda tag: tuple(int(part) for part in tag[1:].split(".")), default="")


def pending(name: str) -> str:
    """What ``name`` would release now, in one line."""
    if name == FORK:
        upstream = run(["gh", "api", "repos/spcl/dace/commits/extended", "-q", ".sha"])
        tags = run(["git", "ls-remote", "--tags", "https://github.com/ThrudPrimrose/dace.git", "extended-v*"])
        released = run(["gh", "api", f"repos/{SLUGS[FORK]}/commits/extended", "-q", ".parents[0].sha"]) if tags else ""
        return (
            "nothing (spcl/dace@extended is released)" if released == upstream else f"spcl/dace@extended {upstream[:9]}"
        )
    slug = SLUGS[name]
    tag = newest_tag(slug)
    if not tag:
        return "no vX.Y.Z tag yet: release the first version by hand"
    count = run(["gh", "api", f"repos/{slug}/compare/{tag}...main", "-q", ".total_commits"])
    return f"{count} commits since {tag}" if count != "0" else f"nothing since {tag} (unless a dependency releases)"


def wait_green(slug: str, sha: str = "") -> str:
    """Wait until every CI run on ``sha`` (default: ``slug``'s main head, followed across newer pushes, which cancel
    the older runs) is green and return that commit; a failed run on it stops the script."""
    while True:
        head = sha or main_head(slug)
        runs = run(
            [
                "gh",
                "run",
                "list",
                "-R",
                slug,
                "-c",
                head,
                "-L",
                "100",
                "--json",
                "status,conclusion,workflowName",
                "-q",
                '.[] | "\\(.status) \\(.conclusion) \\(.workflowName)"',
            ]
        ).splitlines()
        states = [line.split(" ", 2) for line in runs]
        bad = [
            line
            for line, state in zip(runs, states, strict=True)
            if state[0] == "completed" and state[1] not in ("success", "skipped")
        ]
        if bad and (sha or main_head(slug)) == head:
            sys.exit(f"release_all.py: CI on {slug}@{head[:9]} is not green:\n  " + "\n  ".join(bad))
        if states and not bad and all(state[0] == "completed" for state in states):
            return head
        print(f"waiting for CI on {slug}@{head[:9]} ({len(states)} runs)", flush=True)
        time.sleep(POLL_SECONDS)


def pins_pr(slug: str) -> str:
    """The URL of ``slug``'s open pins/* pull request, or ""."""
    return run(
        [
            "gh",
            "pr",
            "list",
            "-R",
            slug,
            "--state",
            "open",
            "--json",
            "headRefName,url",
            "-q",
            '.[] | select(.headRefName | startswith("pins/")) | .url',
        ]
    ).split("\n")[0]


def merge_pins_pr(slug: str) -> bool:
    """Wait for green CI on ``slug``'s open pins/* PR and merge it; False when none is open."""
    url = pins_pr(slug)
    if not url:
        return False
    sha = run(["gh", "pr", "view", url, "--json", "headRefOid", "-q", ".headRefOid"])
    print(f"pin bump {url}: waiting for its CI", flush=True)
    wait_green(slug, sha)
    step(["gh", "pr", "merge", url, "--merge", "--delete-branch"])
    return True


def release_library(name: str, flags: list[str]) -> None:
    """Run ``name``'s helpers/release_patch.sh (with ``flags``) until it releases or has nothing to release."""
    slug = SLUGS[name]
    while True:
        wait_green(slug)
        if merge_pins_pr(slug):
            continue
        with tempfile.TemporaryDirectory(prefix="release-all.") as work:
            clone = pathlib.Path(work) / "repo"
            step(["git", "clone", "-q", "--branch", "main", f"https://github.com/{slug}.git", str(clone)])
            step(["helpers/release_patch.sh", *flags], cwd=clone)
        if not pins_pr(slug):
            return


def release_bench(flags: list[str]) -> None:
    """Bump this checkout's pins (a pull request, CI-gated and merged), then release hpcagent-bench and
    hpcagent-agent (``flags`` go to release_patch.sh)."""
    slug = SLUGS[BENCH]
    while True:
        wait_green(slug)
        if merge_pins_pr(slug):
            continue
        step(["git", "pull", "-q", "--ff-only", "origin", "main"])
        step(["helpers/scripts/pins.sh", "--bump"])
        if not run(["git", "status", "--porcelain"]):
            break
        branch = f"pins/{time.strftime('%Y%m%d-%H%M%S')}"
        step(["git", "switch", "-q", "-c", branch])
        step(["git", "add", "-A"])
        step(["git", "commit", "-q", "-m", "Pin the newest releases of our packages"])
        step(["git", "push", "-q", "origin", branch])
        step(["gh", "pr", "create", "-R", slug, "--base", "main", "--head", branch, "--fill"])
        step(["git", "switch", "-q", "main"])
    step(["helpers/scripts/release_patch.sh", "--skip-libraries", *flags])


def main() -> int:
    """Print the tree and the order; release each package in that order unless --dry-run."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="print the tree, the order and what is pending")
    parser.add_argument("--github-only", action="store_true", help="release on GitHub only; no PyPI upload")
    args = parser.parse_args()
    dry = args.dry_run
    flags = ["--github-only"] if args.github_only else []
    tree = dependency_tree()
    order = list(graphlib.TopologicalSorter(tree).static_order())
    print("dependency tree (package: what it pins of ours):")
    for name in order:
        print(f"  {name}: {', '.join(sorted(tree[name])) or '-'}")
    print("release order:", " -> ".join(order))
    if dry:
        for name in order:
            print(f"  {name}: {pending(name)}")
        return 0
    if not flags and not (os.environ.get("TWINE_USER") and os.environ.get("TWINE_TOKEN")):
        sys.exit("release_all.py: TWINE_USER and TWINE_TOKEN must be set")
    if run(["git", "branch", "--show-current"]) != "main" or run(["git", "status", "--porcelain"]):
        sys.exit("release_all.py: this checkout must be a clean main")
    for name in order:
        print(f"=== {name} ===", flush=True)
        if name == FORK:
            step(["helpers/scripts/dace_extended_release.sh", *flags])
        elif name == BENCH:
            release_bench(flags)
        else:
            release_library(name, flags)
    print("done: every package with changes is released")
    return 0


if __name__ == "__main__":
    sys.exit(main())
