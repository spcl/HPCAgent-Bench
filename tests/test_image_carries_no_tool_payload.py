# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The judge-agent images carry tool dependencies and the hpcagent_agent hook, never the tool code.

The agent runtime (hpcagent_agent) is bound from the submitting checkout at launch; the image holds only its
editable-install hook (containers/lib/package_hook.sh). A tool script or registry copied
into an image goes stale the next commit, which is how a stale registry reached the suite. Static checks
over the recipes and the verifier, plus a run of the launch check against a fake agent tree.
"""

import pathlib
import re
import subprocess
import sys

import pytest

ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
CE_IMAGES: pathlib.Path = ROOT / "containers" / "images"
JUDGE_AGENT_DOCKERFILES: tuple[str, ...] = (
    "judge-agent-amd/Dockerfile",
    "judge-agent-cuda/Dockerfile",
    "judge-agent-cpu/Dockerfile",
)
LAUNCH_CHECK: pathlib.Path = CE_IMAGES / "tools_launch_check.py"
#: Image paths the tool code is bound at; a recipe names them only to install the package hook.
TOOL_MOUNTS: tuple[str, ...] = ("/opt/hpcagent-bench-agent", "/opt/hpcagent-bench-judge")
#: The kinds of recipe line that may name a tool mount: the hook's pyproject, the workspace link to it and the
#: hook call.
HOOK_LINE: re.Pattern[str] = re.compile(
    r"COPY agent/pyproject\.toml /opt/hpcagent-bench-agent/pyproject\.toml"
    r"|(?:RUN set -eux; \\\s*)?ln -s /opt/hpcagent-bench-agent /opt/hpcagent-bench/agent;.*"
    r"|.*package_hook\.sh /opt/hpcagent-bench /opt/hpcagent-bench-agent/hpcagent_agent .*"
)
HARNESS_BUILD_INPUT: re.Pattern[str] = re.compile(
    r"agent/harness/(?:pins\.env|install_tools\.sh|node/package(?:-lock)?\.json)|agent/pyproject\.toml"
)


def recipe(dockerfile: str) -> str:
    return (CE_IMAGES / dockerfile).read_text(encoding="utf-8")


def copy_sources(text: str) -> list[str]:
    """Every source path of every COPY instruction, continuation lines joined, flags dropped."""
    sources: list[str] = []
    pending = ""
    for line in text.splitlines():
        pending += line.rstrip().removesuffix("\\") + " "
        if line.rstrip().endswith("\\"):
            continue
        tokens = pending.split()
        pending = ""
        if tokens[:1] == ["COPY"]:
            sources.extend(token for token in tokens[1:-1] if not token.startswith("--"))
    return sources


def agent_copies_that_are_not_build_inputs(text: str) -> list[str]:
    agent = [source for source in copy_sources(text) if source.startswith("agent")]
    return [source for source in agent if HARNESS_BUILD_INPUT.fullmatch(source) is None]


@pytest.mark.parametrize("dockerfile", JUDGE_AGENT_DOCKERFILES)
def test_no_judge_agent_image_creates_or_reads_a_tool_mount(dockerfile: str) -> None:
    code = [line.strip() for line in recipe(dockerfile).splitlines() if not line.lstrip().startswith("#")]
    lines = [line for line in code if HOOK_LINE.fullmatch(line) is None]
    assert [mount for mount in TOOL_MOUNTS if any(mount in line for line in lines)] == []


@pytest.mark.parametrize("dockerfile", JUDGE_AGENT_DOCKERFILES)
def test_a_judge_agent_image_copies_only_harness_build_inputs_from_containers_agent(dockerfile: str) -> None:
    text = recipe(dockerfile)
    assert any(source.startswith("agent/harness/") for source in copy_sources(text)), dockerfile
    assert agent_copies_that_are_not_build_inputs(text) == []


@pytest.mark.parametrize(
    ("text", "flagged"),
    [
        ("COPY agent /opt/hpcagent-bench-agent\n", ["agent"]),
        (
            "COPY --chown=1:1 agent/hpcagent_agent/tools/mcp_server.py /x/\n",
            ["agent/hpcagent_agent/tools/mcp_server.py"],
        ),
        (
            "COPY agent/harness/pins.env \\\n     agent/hpcagent_agent/harness/run_miniswe.py /h/\n",
            ["agent/hpcagent_agent/harness/run_miniswe.py"],
        ),
        ("COPY agent/harness/install_tools.sh agent/harness/node/package.json /h/\n", []),
    ],
)
def test_the_copy_scan_flags_each_agent_tree_copy_that_is_not_a_build_input(text: str, flagged: list[str]) -> None:
    assert agent_copies_that_are_not_build_inputs(text) == flagged


def test_verify_image_binds_the_checkout_agent_tree_and_runs_its_checks_from_repo() -> None:
    text = (CE_IMAGES / "verify_image.sbatch").read_text(encoding="utf-8")
    assert 'REPO="${REPO:-${SLURM_SUBMIT_DIR:?' in text
    assert '"${REPO}/agent:/opt/hpcagent-bench-agent"' in text
    assert 'python3 "${REPO}/containers/images/tools_launch_check.py"' in text
    assert "exit $(( rc + sc + mpi_rc + tools_rc ))" in text
    assert "/hpcagent-bench}" not in text, "a hard-coded checkout path"


def test_build_and_verify_hands_its_checkout_to_verify_image() -> None:
    text = (CE_IMAGES / "build_and_verify.sbatch").read_text(encoding="utf-8")
    assert re.search(r'REPO="\$\{REPO\}" bash "\$\{REPO\}/containers/images/verify_image\.sbatch"', text)


def launch_check(agent_dir: pathlib.Path, web_search: pathlib.Path) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-W", "error", str(LAUNCH_CHECK), "--agent-dir", str(agent_dir)]
    return subprocess.run(
        [*command, "--judge-web-search", str(web_search)], capture_output=True, text=True, check=False
    )


def fake_checkout(root: pathlib.Path, registry: str) -> tuple[pathlib.Path, pathlib.Path]:
    agent_dir, web_search = root / "agent", root / "judge_web_search.py"
    (agent_dir / "hpcagent_agent" / "tools").mkdir(parents=True)
    (agent_dir / "hpcagent_agent" / "tools" / "mcp_server.py").write_text(registry, encoding="utf-8")
    web_search.write_text("QUERY_LIMIT = 1\n", encoding="utf-8")
    return agent_dir, web_search


def test_the_launch_check_passes_when_the_bound_registry_lists_a_tool(tmp_path: pathlib.Path) -> None:
    agent_dir, web_search = fake_checkout(tmp_path, 'import json\nprint(json.dumps({"allowed_tools": ["score"]}))\n')
    result = launch_check(agent_dir, web_search)
    assert result.returncode == 0, result.stderr
    assert "agent tools: score" in result.stdout


def test_the_launch_check_fails_naming_the_registry_that_offers_no_tools(tmp_path: pathlib.Path) -> None:
    agent_dir, web_search = fake_checkout(tmp_path, 'import json\nprint(json.dumps({"allowed_tools": []}))\n')
    result = launch_check(agent_dir, web_search)
    assert result.returncode != 0
    assert (
        str(agent_dir / "hpcagent_agent" / "tools" / "mcp_server.py") in result.stderr
        and "offers no tools" in result.stderr
    )
