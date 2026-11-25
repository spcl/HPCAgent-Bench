# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""run_cluster.sh's derived_edf switches the comm hooks off for a role that never crosses a node.

An EDF's cxi/aws_ofi_nccl hooks and its forced NCCL_NET/NCCL_NET_PLUGIN serve cross-node
collectives only; a single-node tensor-parallel server fails at init with them ("Failed to
initialize any NET plugin"). The agent and single-node inference therefore get an EDF with both
switched off; the judge (MPI gang ranks reuse its EDF) and multi-node inference keep them.

``run_cluster.sh`` cannot be sourced (its top level needs a real allocation), so ``role_srun`` is cut out of the
shipped text and run with the sourced container-runtime seam (``container_runtime.sh``) and a fake ``srun`` that
records its argv.
"""

import pathlib
import re
import shlex
import subprocess
import tomllib
from typing import Any

import pytest

from hpcagent_bench import paths
from tests.bash_stub import bash_stub

REPO = paths.ROOT
SCRIPT = REPO / "hpcagent_bench" / "cluster" / "run_cluster.sh"

SEAM = REPO / "hpcagent_bench" / "cluster" / "container_runtime.sh"
ROLE_SRUN = re.compile(r"^role_srun\(\) \{$.*?^\}$", re.MULTILINE | re.DOTALL)


def _function_text() -> str:
    match = ROLE_SRUN.search(SCRIPT.read_text())
    assert match, f"role_srun() not found in {SCRIPT} -- this test runs its shipped text"
    return match.group(0)


REGISTERED_EDF = (
    'image = "x"\nmounts = [\n    "/tmp:/tmp",\n]\nworkdir = "/old"\n\n'
    '[env]\nNCCL_NET = "AWS Libfabric"\nNCCL_NET_PLUGIN = "ofi"\nFI_CXI_RX_MATCH_MODE = "software"\n\n'
    '[annotations]\ncom.hooks.netstack.source = "artifact"\ncom.hooks.cxi.enabled = "true"\n'
    'com.hooks.aws_ofi_nccl.enabled = "true"\n'
)


def run_role_srun(
    tmp_path: pathlib.Path, role_flag: str, *, inference_nodes: int = 1, inference_mode: str = "pp"
) -> dict[str, Any]:
    """Runs ``role_srun 1 nid001 bench-ce-env "" <role_flag>`` with CONTAINER_RUNTIME=ce and returns
    the parsed TOML of the EDF srun was handed through --environment."""
    edf_dir = tmp_path / "edf"
    edf_dir.mkdir()
    (edf_dir / "bench-ce-env.toml").write_text(REGISTERED_EDF)
    repo = tmp_path / "repo"
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    bin_dir = tmp_path / "bin"
    bash_stub(bin_dir, "srun", f'printf \'%s\\n\' "$@" >"{marker_dir}/srun-argv"\n')

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
            f"SCRIPT_DIR={shlex.quote(str(repo / 'hpcagent_bench' / 'cluster'))}",
            f"RUN_ROOT={shlex.quote(str(run_dir))}",
            # role_mounts' inference branch reads cache_env.sh's JIT_CACHE_ROOT.
            f"SCRATCH={shlex.quote(str(tmp_path))}",
            f"JIT_CACHE_ROOT={shlex.quote(str(tmp_path / '.hpcagentbench-cache'))}",
            "AGENT_PAYLOAD_MOUNT=/opt/hpcagent-bench-agent",
            f"AGENT_LAUNCH_DIR={shlex.quote(str(run_dir / '.agent-launch'))}",
            'CONTAINER_MOUNTS=""',
            f"GENERATED_CACHE_HOST={shlex.quote(str(run_dir / 'generated'))}",
            "GENERATED_CACHE_MOUNT=/opt/generated",
            "CONTAINER_RUNTIME=ce",
            f"INFERENCE_NODES={inference_nodes}",
            f"INFERENCE_MODE={inference_mode}",
            "CONTAINER_GPU_FLAGS=",
            "COLOCATE=0",
            "DRY_RUN=0",
            "GRADE_CPUS=4",
            f". {shlex.quote(str(SEAM))}",
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
    assert "NCCL_NET" not in env
    assert "NCCL_NET_PLUGIN" not in env
    assert env["FI_CXI_RX_MATCH_MODE"] == "software"
    assert edf["annotations"]["com"]["hooks"]["netstack"]["source"] == "artifact"


def assert_hooks_kept(edf: dict[str, Any]) -> None:
    assert comm_hooks(edf) == ("true", "true")
    assert edf["env"]["NCCL_NET"] == "AWS Libfabric"
    assert edf["env"]["NCCL_NET_PLUGIN"] == "ofi"


@pytest.mark.parametrize(
    ("role_flag", "inference_nodes", "inference_mode"),
    [
        pytest.param("--agent-node", 1, "pp", id="agent"),  # the agent runs no collective
        pytest.param("--vllm-node", 1, "pp", id="single_node_inference"),  # a forced NCCL_NET would fail TP init
        pytest.param("--vllm-node", 2, "replicas", id="replica_inference_on_several_nodes"),  # no cross-node collective
    ],
)
def test_a_role_with_no_cross_node_collective_gets_no_comm_hooks(
    tmp_path: pathlib.Path, role_flag: str, inference_nodes: int, inference_mode: str
) -> None:
    """The EDF has the hooks and the forced NCCL_NET removed. Replicas are a standalone server per node, so no
    collective crosses nodes however many there are, and a forced NCCL_NET would fail every replica's TP init."""
    assert_hooks_off(run_role_srun(tmp_path, role_flag, inference_nodes=inference_nodes, inference_mode=inference_mode))


@pytest.mark.parametrize(
    ("role_flag", "inference_nodes"),
    [
        pytest.param("--vllm-node", 4, id="multi_node_inference"),  # the collective crosses nodes
        pytest.param("--judge-node", 1, id="judge"),  # MPI gang ranks reuse the judge's EDF
    ],
)
def test_a_role_with_a_cross_node_collective_keeps_the_comm_hooks(
    tmp_path: pathlib.Path, role_flag: str, inference_nodes: int
) -> None:
    """Multi-node inference needs the fabric plugin; the judge keeps the hooks whatever INFERENCE_NODES is."""
    assert_hooks_kept(run_role_srun(tmp_path, role_flag, inference_nodes=inference_nodes))
