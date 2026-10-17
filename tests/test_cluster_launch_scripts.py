# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Launch-environment lines of the cluster job scripts, lifted from the files and run through bash.

The scripts cannot run end to end in a unit test (they need Slurm, containers and images), so each test
takes the EXACT text it checks out of the script (never retyped) and runs only that.
"""

import pathlib
import shutil
import subprocess

import pytest

from hpcagent_bench import paths

CLUSTER = paths.ROOT / "hpcagent_bench" / "cluster"


def lines_between(path: pathlib.Path, first: str, last: str) -> str:
    """The lines of ``path`` from the one starting with ``first`` through the next one starting with ``last``."""
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(first))
    stop = next(i for i in range(start, len(lines)) if lines[i].startswith(last))
    return "\n".join(lines[start : stop + 1])


def bash(script: str, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script, "script", *args],
        env={"PATH": "/usr/bin:/bin", **(env or {})},
        capture_output=True,
        text=True,
        check=False,
    )


def quote(text: str) -> str:
    """``text`` as one single-quoted bash word."""
    return "'" + text.replace("'", "'\\''") + "'"


@pytest.mark.skipif(shutil.which("taskset") is None, reason="needs taskset to own one CPU")
def test_a_role_step_derives_its_thread_limit_from_the_cores_it_owns() -> None:
    """srun --export=ALL hands a role step the batch shell's whole-node OMP_THREAD_LIMIT; a judge owning 24
    cores must clamp at 24, not at the node's 192, or a 4 * omp_get_num_procs() team is not clamped."""
    block = lines_between(CLUSTER / "run_cluster.sh", "ulimit -s ", "export OMP_THREAD_LIMIT=")
    script = f'{block}\necho "${{OMP_THREAD_LIMIT}}"'
    role = bash(f"exec taskset -c 0 bash -c {quote(script)} script --judge-node", env={"OMP_THREAD_LIMIT": "192"})
    assert role.returncode == 0, role.stderr
    assert role.stdout.strip() == "1"
    batch = bash(f"exec taskset -c 0 bash -c {quote(script)} script", env={"OMP_THREAD_LIMIT": "192"})
    assert batch.stdout.strip() == "192", "the batch shell keeps a limit its launcher set"


def test_serve_only_reads_the_agents_readiness_key_first() -> None:
    """The agent driver waits AGENT_READY_TIMEOUT_SECONDS first; a model layer sets it (Kimi: 7200) above the
    VLLM_READY_TIMEOUT_SECONDS a base layer may carry, so serve-only must not give up at the lower one."""
    line = next(x for x in (CLUSTER / "serve-only.sbatch").read_text().splitlines() if x.startswith("READY_TIMEOUT="))
    done = bash(
        f'{line}\necho "${{READY_TIMEOUT}}"',
        env={"VLLM_READY_TIMEOUT_SECONDS": "3600", "AGENT_READY_TIMEOUT_SECONDS": "7200"},
    )
    assert done.stdout.strip() == "7200"
    assert (
        bash(f'{line}\necho "${{READY_TIMEOUT}}"', env={"VLLM_READY_TIMEOUT_SECONDS": "3600"}).stdout.strip() == "3600"
    )


def test_grade_under_cleanup_survives_a_relay_that_already_died(tmp_path: pathlib.Path) -> None:
    """A gang relay that crashed mid-job fails its kill at EXIT; the trap must still remove the job's files
    and leave the job's own exit status."""
    cleanup = lines_between(CLUSTER / "grade-under.sbatch", "cleanup() {", "}")
    leftover = tmp_path / "judge-1.edf.toml"
    leftover.write_text("x")
    script = "\n".join(
        [
            "set -euo pipefail",
            'true & dead=$!; wait "${dead}"',
            'relay_pids=("${dead}")',
            f"job_files=({quote(str(leftover))})",
            cleanup,
            "trap cleanup EXIT",
            "exit 0",
        ]
    )
    done = bash(script)
    assert done.returncode == 0, done.stderr
    assert not leftover.exists()


#: A registered inference EDF as install_edfs.sh renders it: the forced fabric plugin and both comm hooks on.
FABRIC_EDF = """image = "/images/sglang.sqsh"
workdir = "/workspace"
mounts = [
    "/scratch:/scratch",
]

[env]
NCCL_NET_PLUGIN = "ofi"
NCCL_NET = "AWS Libfabric"
FI_PROVIDER = "cxi"

[annotations]
com.hooks.cxi.enabled = "true"
com.hooks.aws_ofi_nccl.enabled = "true"
"""


@pytest.mark.parametrize(("nodes", "mode", "hooks"), [("1", "pp", False), ("2", "replicas", False), ("2", "pp", True)])
def test_serve_only_serves_from_the_inference_roles_derived_edf(
    tmp_path: pathlib.Path, nodes: str, mode: str, hooks: bool
) -> None:
    """The registered EDF forces NCCL_NET and the cxi / aws-ofi-nccl hooks, with which a single-node or replica
    server fails building its tensor-parallel group; serve-only launches what an experiment launches, the
    role's derived EDF, which drops them there and keeps them for a multi-node pp server."""
    text = (CLUSTER / "serve-only.sbatch").read_text()
    assert '--environment="${INFERENCE_CE_ENV' not in text
    assert '"${CONTAINER_SRUN_ARGS[@]}"' in lines_between(CLUSTER / "serve-only.sbatch", "srun --nodes=", '    "')
    block = lines_between(CLUSTER / "serve-only.sbatch", "export RUN_ROOT RUN_DIR", "container_wrap vllm-node")
    edfs = tmp_path / "edf"
    edfs.mkdir()
    (edfs / "sglang.toml").write_text(FABRIC_EDF)
    run_dir = tmp_path / "root" / "1"
    script = f'set -euo pipefail\n{block}\nprintf "%s\\n" "${{CONTAINER_SRUN_ARGS[@]}}"'
    done = bash(
        script,
        env={
            "RUN_ROOT": str(tmp_path / "root"),
            "RUN_DIR": str(run_dir),
            "SCRIPT_DIR": str(CLUSTER),
            "HPCAGENT_BENCH_REPO": str(CLUSTER.parents[1]),
            "HPCAGENT_BENCH_GENERATED_CACHE_HOST": str(tmp_path / "generated"),
            "SCRATCH": str(tmp_path),
            "FAST_SCRATCH": str(tmp_path / "fast"),
            "EDF_PATH": str(edfs),
            "INFERENCE_CE_ENV": "sglang",
            "INFERENCE_NODES": nodes,
            "INFERENCE_MODE": mode,
        },
    )
    assert done.returncode == 0, done.stderr
    edf = pathlib.Path(done.stdout.strip().removeprefix("--environment="))
    assert edf.parent == run_dir / "edf"
    rendered = edf.read_text()
    assert ("NCCL_NET =" in rendered) is hooks
    assert ('com.hooks.cxi.enabled = "false"' not in rendered) is hooks
    assert f"{run_dir}/shared:/shared" in rendered
