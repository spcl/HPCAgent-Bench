# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""experiments/record_identity.sh's record_tag_version(): the frozen version stamp a launcher
appends alongside record_identity's own columns, so two runs of "the same tag name" can be told
apart once experiments/tags.yaml (or a kernels-<tag>.txt file) moves between them.

The run itself is already frozen the moment PROBLEMS_FILE is written (every submit-*.sh test in
this suite proves that separately); this is the OTHER half -- pooling two runs by (experiment,
tag_version) instead of (experiment) alone must not silently merge two different kernel sets.
"""

import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def run_record_tag_version(tag: str, registry_text: str, tmp_path: pathlib.Path) -> subprocess.CompletedProcess[str]:
    registry = tmp_path / "tags.yaml"
    registry.write_text(registry_text)
    env_file = tmp_path / "env"
    env_file.write_text("EXISTING=1\n")
    script = (
        f'. "{REPO}/experiments/record_identity.sh"; '
        f'record_tag_version "{env_file}" "{tag}"; rc=$?; '
        f'cat "{env_file}"; exit "${{rc}}"'
    )
    env = {
        **os.environ,
        "PY": sys.executable,
        "HPCAGENT_BENCH_TAGS_FILE": str(registry),
    }
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=60, check=False)


def test_a_registered_tag_gets_a_frozen_version_stamp(tmp_path: pathlib.Path) -> None:
    result = run_record_tag_version("mytag", "tags:\n  mytag:\n    list:\n      - explicit:kmp,dfa\n", tmp_path)
    assert result.returncode == 0, result.stderr
    assert "EXISTING=1" in result.stdout
    lines = [ln for ln in result.stdout.splitlines() if ln.startswith("HPCAGENT_BENCH_RECORD_TAG_VERSION=")]
    assert len(lines) == 1, result.stdout
    assert len(lines[0].split("=", 1)[1]) == 12


def test_the_stamp_matches_the_python_resolver_directly(tmp_path: pathlib.Path) -> None:
    """record_tag_version shells out to the same hpcagent_bench.tags.version every python consumer
    would call -- the stamp must be byte-identical to calling it directly."""
    registry_text = "tags:\n  mytag:\n    list:\n      - explicit:kmp,dfa\n"
    result = run_record_tag_version("mytag", registry_text, tmp_path)
    assert result.returncode == 0, result.stderr
    stamped = next(
        ln.split("=", 1)[1] for ln in result.stdout.splitlines() if ln.startswith("HPCAGENT_BENCH_RECORD_TAG_VERSION=")
    )
    registry = tmp_path / "tags.yaml"
    direct = subprocess.run(
        [sys.executable, "-m", "hpcagent_bench.tags", "version", "mytag"],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HPCAGENT_BENCH_TAGS_FILE": str(registry),
            "PYTHONPATH": f"{REPO}",
        },
        timeout=60,
        check=True,
    )
    assert stamped == direct.stdout.strip()


def test_a_tag_hpcagent_bench_tags_cannot_resolve_is_refused_not_silently_skipped(
    tmp_path: pathlib.Path,
) -> None:
    """record_tag_version itself always fails loudly for an unresolvable tag; callers that want
    best-effort (most launchers, whose TAG default is a plain manifest-scan tag with no
    tags.yaml entry) chain ``|| true`` at the call site -- this function must never swallow the
    failure itself, or a caller that DOES want it fatal could never get that."""
    result = run_record_tag_version("no-such-tag-anywhere", "tags: {}\n", tmp_path)
    assert result.returncode == 2
    assert "could not resolve a version" in result.stderr
    assert "EXISTING=1" in result.stdout
    assert "HPCAGENT_BENCH_RECORD_TAG_VERSION" not in result.stdout
