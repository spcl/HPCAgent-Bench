# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A registry tag and an EDF name are different namespaces that look alike.

A tag names bytes in a Docker repository; an EDF name names a rendered file in ~/.edf. Both are
kebab-case strings in images.env, one line apart, and a rename that walked the tree rewriting EDF
names also rewrote two tags. The pull then asked the registry for a tag nobody ever pushed and
got a 404 -- after a compute node had already spent twenty minutes fetching the other three.

Published tags are role- and partition-scoped (`agent-mi300-latest`, `judge-mi300-latest`, `sglang-mi200-latest`, `vllm-0.23-mi300`);
EDF names carry the project prefix. So a tag that begins with the project prefix is the signature
of exactly this mistake.
"""

import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
IMAGES_ENV = REPO / "containers" / "images" / "images.env"
PROJECT_PREFIX = "hpcagent-bench-"


def _vars_matching(suffix: str) -> dict[str, str]:
    script = f'set -a; SCRATCH=/nonexistent; . "{IMAGES_ENV}"; set +a; env'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    found = {}
    for line in out.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.endswith(suffix):
            found[k] = v.strip()
    return found


def test_registry_tags_do_not_carry_the_edf_prefix() -> None:
    tags = _vars_matching("_TAG")
    assert tags, "images.env defined no *_TAG variables"
    wrong = {k: v for k, v in tags.items() if v.startswith(PROJECT_PREFIX)}
    assert not wrong, (
        "these are registry tags, not EDF names, and no such tag is published:\n  "
        + "\n  ".join(f"{k}={v}" for k, v in sorted(wrong.items()))
        + "\nPublished tags are role-scoped, e.g. agent-mi300-latest."
    )


def test_edf_names_do_carry_the_prefix() -> None:
    """The converse, so the two cannot be swapped in the other direction either."""
    edfs = _vars_matching("_EDF_LATEST")
    assert edfs, "images.env defined no *_EDF_LATEST variables"
    wrong = {k: v for k, v in edfs.items() if not v.startswith(PROJECT_PREFIX)}
    assert not wrong, "EDF names must carry the project prefix:\n  " + "\n  ".join(
        f"{k}={v}" for k, v in sorted(wrong.items())
    )
