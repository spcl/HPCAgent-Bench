# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The agent images' launch gate builds hpcagent-agent from a stub tree: the stub carries every file the build reads."""

import tomllib

import pytest

from hpcagent_bench import paths

REPO = paths.repo_root()
STUB = "/opt/hpcagent-bench-agent"


# Reproducer: the judge-agent-cuda gate failed "failed to open file /opt/hpcagent-bench/agent/README.md" (Daint 5455921)
@pytest.mark.parametrize("image", ["judge-agent-cpu", "judge-agent-cuda", "judge-agent-amd"])
def test_the_launch_gate_stub_holds_the_agents_readme(image: str) -> None:
    readme = tomllib.loads((REPO / "agent" / "pyproject.toml").read_text())["project"]["readme"]
    gate = next(
        line
        for line in (REPO / "containers" / "images" / image / "Dockerfile").read_text().splitlines()
        if "touch " + STUB in line
    )
    assert f"{STUB}/{readme}" in gate.split(), gate
