# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``hpcagent_bench/cluster/run_cluster.sh``'s OWN cache-root derivation, inside ``run_vllm_node``.

``run_vllm_node`` cannot be invoked end to end in a unit test: it snapshots a huggingface_hub
repo over the network, backgrounds a monitor process and ends by exec-ing an inference engine, none
of which belongs in a login-node test run. So these tests lift the EXACT shell expressions the
function uses for its cache root and its node-local JIT directory straight out of the file (never
retyped) and run only those, through bash -- the same "read the real text, run only the real text"
approach ``tests/test_jit_cache_layer.py`` already takes for this function's body.
"""

import pathlib
import re
import subprocess

import pytest

from hpcagent_bench import paths

REPO = paths.ROOT
SCRIPT = REPO / "hpcagent_bench" / "cluster" / "run_cluster.sh"
TEXT = SCRIPT.read_text()
BODY = TEXT[TEXT.index("run_vllm_node() {") : TEXT.index('case "${1:-}" in')]


def _assignment(var: str) -> str:
    """The right-hand side of ``local <var>="..."`` inside run_vllm_node, exactly as written."""
    match = re.search(rf'local {re.escape(var)}="([^\n]*)"\n', BODY)
    assert match, f"run_vllm_node no longer assigns {var} the way this test expects; update it"
    return match.group(1)


def run(script: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    full_env = {"PATH": "/usr/bin:/bin", **env}
    return subprocess.run(["bash", "-c", script], env=full_env, capture_output=True, text=True, check=False)


@pytest.mark.parametrize("jit_root_set", [True, False], ids=["jit_cache_root_set", "jit_cache_root_missing"])
def test_cache_root_is_jit_cache_root(tmp_path: pathlib.Path, jit_root_set: bool) -> None:
    """An inference node without JIT_CACHE_ROOT refuses to pick a cache root, even with SCRATCH set, rather
    than deriving one that could disagree with cache_env.sh's."""
    expr = _assignment("cache_root")
    root = tmp_path / "jit"
    env = {"JIT_CACHE_ROOT": str(root)} if jit_root_set else {"SCRATCH": str(tmp_path)}
    proc = run(f'cache_root="{expr}"; echo "$cache_root"', env)
    if jit_root_set:
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == str(root)
    else:
        assert proc.returncode != 0, proc.stdout
        assert "cache_env.sh exports it" in proc.stderr, proc.stderr


@pytest.mark.parametrize(
    ("ranks", "job_ids"),
    [
        pytest.param(("0", "1"), ("640200", "640200"), id="two_node_ranks_of_one_job"),
        pytest.param(("0", "0"), ("640200", "640201"), id="two_concurrent_jobs_on_one_node_rank"),
    ],
)
def test_distinct_ranks_or_jobs_get_distinct_local_roots(
    tmp_path: pathlib.Path, ranks: tuple[str, str], job_ids: tuple[str, str]
) -> None:
    expr = _assignment("local_root")
    roots = set()
    for rank, job_id in zip(ranks, job_ids, strict=True):
        proc = run(
            f'node_rank="{rank}"; local_root="{expr}"; echo "$local_root"',
            {"SLURM_JOB_ID": job_id, "TMPDIR": str(tmp_path)},
        )
        assert proc.returncode == 0, proc.stderr
        roots.add(proc.stdout.strip())
    assert len(roots) == 2, roots
