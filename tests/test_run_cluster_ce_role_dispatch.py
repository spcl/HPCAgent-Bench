# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""run_cluster.sh's role_srun must not force pyxis comm hooks onto a role that never crosses a node.

Under CONTAINER_RUNTIME=ce, pyxis applies a registered EDF's [annotations] comm hooks (netstack,
cxi, aws_ofi_nccl) and its forced NCCL_NET/NCCL_NET_PLUGIN unconditionally -- derived_edf only
rewrites the mounts/workdir block, never [annotations] or [env] -- so a role that never runs a
cross-node GPU collective still gets them. A single-node inference step then fails tensor-parallel
init with "NCCL error ... Failed to initialize any NET plugin": the 2026-09-17 17:00 wave,
640160-640181, 22 arms, every one INFERENCE_NODES=1 (container_runtime.sh's own comment records
the same failure). submit-mlscale.sh pins CONTAINER_RUNTIME=ce globally because JUDGE_GANG_NODES
(the multi-node judge/grade gang) needs pyxis; agent-node and a single-node vllm-node must instead
take the SAME hook-gated enroot path every non-mlscale wave already gets, judge-node and a
multi-node vllm-node must still go through pyxis.

``run_cluster.sh`` cannot be sourced (its top level needs a real allocation), so ``role_srun`` and
the functions it calls are cut out of the shipped text and run standalone, with fake ``srun`` and
``enroot_srun.sh`` binaries recording which one launched and with what -- the same technique
tests/test_derived_edf.py and tests/test_run_cluster_gang_judge.py use.
"""

import pathlib
import re
import shlex
import subprocess

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


REGISTERED_EDF = 'image = "x"\nmounts = [\n    "/tmp:/tmp",\n]\nworkdir = "/old"\n'


def run_role_srun(tmp_path: pathlib.Path, role_flag: str, *, inference_nodes: int = 1) -> dict[str, str]:
    """Runs ``role_srun 1 nid001 bench-ce-env "" <role_flag>`` with CONTAINER_RUNTIME=ce, and
    returns {"launcher": "srun"|"enroot"|"none", "argv": <recorded argv>, "hooks": <env var seen>}."""
    edf_dir = tmp_path / "edf"
    edf_dir.mkdir()
    (edf_dir / "bench-ce-env.toml").write_text(REGISTERED_EDF)
    repo = tmp_path / "repo"
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    _stub(
        repo / "scripts" / "cscs" / "enroot_srun.sh",
        f'printf \'%s\\n\' "$@" >"{marker_dir}/enroot-argv"\n'
        f'printf \'%s\' "${{HPCAGENT_BENCH_COMM_HOOKS-<unset>}}" >"{marker_dir}/enroot-hooks"\n',
    )
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
            # role_mounts' vllm*/inference* branch falls back to ${SCRATCH}/.hpcagentbench-cache
            # when JIT_CACHE_ROOT is unset (run_cluster.sh:1195); CI's shell has no ambient
            # SCRATCH, so the extracted function's own `${SCRATCH:?set SCRATCH}` guard aborted the
            # subprocess with "environment: line N: SCRATCH: set SCRATCH" -- the script under test
            # is unchanged, this only supplies the variable a real submission's env.sh exports.
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

    srun_argv = marker_dir / "srun-argv"
    enroot_argv = marker_dir / "enroot-argv"
    if srun_argv.exists():
        assert not enroot_argv.exists(), "both launchers ran -- role_srun invoked twice"
        return {"launcher": "srun", "argv": srun_argv.read_text(), "hooks": ""}
    if enroot_argv.exists():
        return {
            "launcher": "enroot",
            "argv": enroot_argv.read_text(),
            "hooks": (marker_dir / "enroot-hooks").read_text(),
        }
    raise AssertionError(f"neither launcher ran: {proc.stdout}\n{proc.stderr}")


def test_agent_node_under_ce_never_takes_the_pyxis_fabric_hooks(tmp_path: pathlib.Path) -> None:
    """The agent never runs an MPI/RCCL collective; forcing pyxis's comm hooks onto it under ce
    buys nothing and (via the SAME registered EDF the judge uses) risks the identical NET-plugin
    failure the inference role hit in the 2026-09-17 wave."""
    result = run_role_srun(tmp_path, "--agent-node")
    assert result["launcher"] == "enroot"
    assert result["hooks"] == "off"


def test_single_node_inference_under_ce_takes_the_hook_gated_path(tmp_path: pathlib.Path) -> None:
    """INFERENCE_NODES=1 (qwen38, oss120b): no cross-node collective, so this must route through
    enroot_srun.sh with comm hooks off, exactly like every non-mlscale single-node wave -- not
    through pyxis --environment=, which applies the registered EDF's hooks unconditionally."""
    result = run_role_srun(tmp_path, "--vllm-node", inference_nodes=1)
    assert result["launcher"] == "enroot"
    assert result["hooks"] == "off"


def test_multi_node_inference_under_ce_keeps_the_pyxis_fabric(tmp_path: pathlib.Path) -> None:
    """INFERENCE_NODES=4 (kimi27sglang, glm53): a real cross-node NCCL collective, so this arm
    still needs pyxis's comm hooks -- this must NOT be rerouted onto the single-node path."""
    result = run_role_srun(tmp_path, "--vllm-node", inference_nodes=4)
    assert result["launcher"] == "srun"
    assert "--environment=" in result["argv"]


def test_judge_node_under_ce_always_keeps_the_pyxis_fabric(tmp_path: pathlib.Path) -> None:
    """The judge gang's rank launches need CE (run_cluster.sh's own JUDGE_GANG_NODES gate); the
    judge-node role step itself must stay on pyxis --environment= whatever INFERENCE_NODES is."""
    result = run_role_srun(tmp_path, "--judge-node", inference_nodes=1)
    assert result["launcher"] == "srun"
    assert "--environment=" in result["argv"]
