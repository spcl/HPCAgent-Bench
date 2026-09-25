# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""run_cluster.sh's derived_edf switches the comm hooks off for a role that never crosses a node.

An EDF's cxi/aws_ofi_nccl hooks and its forced NCCL_NET/NCCL_NET_PLUGIN serve cross-node
collectives only; a single-node tensor-parallel server fails at init with them ("Failed to
initialize any NET plugin"). The agent and single-node inference therefore get an EDF with both
switched off; the judge (MPI gang ranks reuse its EDF) and multi-node inference keep them.

``run_cluster.sh`` cannot be sourced (its top level needs a real allocation), so ``role_srun`` and
the functions it calls are cut out of the shipped text and run standalone with a fake ``srun``
that records its argv -- the same technique tests/test_derived_edf.py uses.
"""

import pathlib
import re
import shlex
import subprocess
import tomllib
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "experiments" / "run_cluster.sh"

_FUNCS = {
    "agent_ro_binds": re.compile(r"^agent_ro_binds\(\) \{$.*?^\}$", re.MULTILINE | re.DOTALL),
    "role_mounts": re.compile(r"^role_mounts\(\) \{$.*?^\}$", re.MULTILINE | re.DOTALL),
    "derived_edf": re.compile(r"^derived_edf\(\) \{$.*?^\}$", re.MULTILINE | re.DOTALL),
    "role_srun": re.compile(r"^role_srun\(\) \{$.*?^\}$", re.MULTILINE | re.DOTALL),
}


def _function_text() -> str:
    text = SCRIPT.read_text()
    out = []
    for name, pattern in _FUNCS.items():
        match = pattern.search(text)
        assert match, f"{name}() not found in {SCRIPT} -- this test runs its shipped text"
        out.append(match.group(0))
    return "\n".join(out)


def _stub(path: pathlib.Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/usr/bin/env bash\n{body}\n")
    path.chmod(0o755)


REGISTERED_EDF = (
    'image = "x"\nmounts = [\n    "/tmp:/tmp",\n]\nworkdir = "/old"\n\n'
    '[env]\nNCCL_NET = "AWS Libfabric"\nNCCL_NET_PLUGIN = "ofi"\nFI_CXI_RX_MATCH_MODE = "software"\n\n'
    '[annotations]\ncom.hooks.netstack.source = "artifact"\ncom.hooks.cxi.enabled = "true"\n'
    'com.hooks.aws_ofi_nccl.enabled = "true"\n'
)


def run_role_srun(tmp_path: pathlib.Path, role_flag: str, *, inference_nodes: int = 1) -> dict[str, Any]:
    """Runs ``role_srun 1 nid001 bench-ce-env "" <role_flag>`` with CONTAINER_RUNTIME=ce and returns
    the parsed TOML of the EDF srun was handed through --environment."""
    edf_dir = tmp_path / "edf"
    edf_dir.mkdir()
    (edf_dir / "bench-ce-env.toml").write_text(REGISTERED_EDF)
    repo = tmp_path / "repo"
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    bin_dir = tmp_path / "bin"
    _stub(bin_dir / "srun", f'printf \'%s\\n\' "$@" >"{marker_dir}/srun-argv"\n')

    run_dir = tmp_path / "run"
    script = "\n".join(
        [
            "set -eo pipefail",
            f'PATH="{bin_dir}:$PATH"',
            f"RUN_DIR={shlex.quote(str(run_dir))}",
            f"SHARED_HOST_DIR={shlex.quote(str(run_dir / 'shared'))}",
            "SHARED_MOUNT=/shared",
            f"EDF_PATH={shlex.quote(str(edf_dir))}",
            f"HPCAGENT_BENCH_REPO={shlex.quote(str(repo))}",
            f"SCRIPT_DIR={shlex.quote(str(repo / 'experiments'))}",
            f"RUN_ROOT={shlex.quote(str(run_dir))}",
            # role_mounts' inference branch falls back to ${SCRATCH}/.hpcagentbench-cache.
            f"SCRATCH={shlex.quote(str(tmp_path))}",
            "AGENT_PAYLOAD_MOUNT=/opt/hpcagent-bench-agent",
            f"AGENT_LAUNCH_DIR={shlex.quote(str(run_dir / '.agent-launch'))}",
            'CONTAINER_MOUNTS=""',
            f"GENERATED_CACHE_HOST={shlex.quote(str(run_dir / 'generated'))}",
            "GENERATED_CACHE_MOUNT=/opt/generated",
            "CONTAINER_RUNTIME=ce",
            f"INFERENCE_NODES={inference_nodes}",
            "CONTAINER_GPU_FLAGS=",
            "COLOCATE=0",
            "DRY_RUN=0",
            "GRADE_CPUS=4",
            "JUDGES_PER_NODE=1",
            _function_text(),
            f"role_srun 1 nid001 bench-ce-env '' {shlex.quote(role_flag)}",
            'wait "${ROLE_PID}"',
        ]
    )
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    argv = (marker_dir / "srun-argv").read_text().splitlines()
    edf = [a.removeprefix("--environment=") for a in argv if a.startswith("--environment=")]
    assert len(edf) == 1, argv
    return tomllib.loads(pathlib.Path(edf[0]).read_text())


def comm_hooks(edf: dict[str, Any]) -> tuple[str, str]:
    hooks = edf["annotations"]["com"]["hooks"]
    return hooks["cxi"]["enabled"], hooks["aws_ofi_nccl"]["enabled"]


def assert_hooks_off(edf: dict[str, Any]) -> None:
    assert comm_hooks(edf) == ("false", "false")
    env = edf["env"]
    assert "NCCL_NET" not in env and "NCCL_NET_PLUGIN" not in env
    assert env["FI_CXI_RX_MATCH_MODE"] == "software"
    assert edf["annotations"]["com"]["hooks"]["netstack"]["source"] == "artifact"


def assert_hooks_kept(edf: dict[str, Any]) -> None:
    assert comm_hooks(edf) == ("true", "true")
    assert edf["env"]["NCCL_NET"] == "AWS Libfabric"


def test_the_agent_gets_no_comm_hooks(tmp_path: pathlib.Path) -> None:
    """The agent runs no collective, so its EDF has the hooks and the forced NCCL_NET removed."""
    assert_hooks_off(run_role_srun(tmp_path, "--agent-node"))


def test_single_node_inference_gets_no_comm_hooks(tmp_path: pathlib.Path) -> None:
    """INFERENCE_NODES=1: no cross-node collective, and a forced NCCL_NET would fail TP init."""
    assert_hooks_off(run_role_srun(tmp_path, "--vllm-node", inference_nodes=1))


def test_multi_node_inference_keeps_the_comm_hooks(tmp_path: pathlib.Path) -> None:
    """INFERENCE_NODES=4: the collective crosses nodes and needs the fabric plugin."""
    assert_hooks_kept(run_role_srun(tmp_path, "--vllm-node", inference_nodes=4))


def test_the_judge_keeps_the_comm_hooks(tmp_path: pathlib.Path) -> None:
    """MPI gang ranks reuse the judge's EDF, so it keeps the hooks whatever INFERENCE_NODES is."""
    assert_hooks_kept(run_role_srun(tmp_path, "--judge-node", inference_nodes=1))
