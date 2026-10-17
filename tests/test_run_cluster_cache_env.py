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


def test_the_cache_root_derivation_reads_only_jit_cache_root() -> None:
    """Pins the exact expression the two behavioural tests below execute: the root is cache_env.sh's
    JIT_CACHE_ROOT (exported by the batch shell's env.sh), with no second default spelled here."""
    assert _assignment("cache_root") == "${JIT_CACHE_ROOT:?helpers/scripts/cache_env.sh exports it}"


def test_cache_root_fails_loudly_without_jit_cache_root(tmp_path: pathlib.Path) -> None:
    """An inference node without JIT_CACHE_ROOT refuses to pick a cache root, even with SCRATCH set, rather
    than deriving one that could disagree with cache_env.sh's."""
    expr = _assignment("cache_root")
    proc = run(f'cache_root="{expr}"; echo "$cache_root"', {"SCRATCH": str(tmp_path)})
    assert proc.returncode != 0, proc.stdout
    assert "cache_env.sh exports it" in proc.stderr, proc.stderr


def test_cache_root_is_jit_cache_root(tmp_path: pathlib.Path) -> None:
    expr = _assignment("cache_root")
    root = tmp_path / "jit"
    proc = run(f'cache_root="{expr}"; echo "$cache_root"', {"JIT_CACHE_ROOT": str(root)})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(root)


def test_the_node_local_jit_root_is_keyed_by_both_job_and_node_rank() -> None:
    """The node-local write layer (jit_cache_layer.sh) exists because several engines compiling into
    the SAME directory at once corrupt each other's reads. That guarantee
    depends on no two node ranks -- of the same job or of two jobs running at once -- ever computing
    the same local_root; both the job id and the node rank must appear in it."""
    expr = _assignment("local_root")
    assert "${SLURM_JOB_ID:-$$}" in expr, expr
    assert "${node_rank}" in expr, expr


def test_two_node_ranks_of_the_same_job_get_distinct_local_roots(tmp_path: pathlib.Path) -> None:
    expr = _assignment("local_root")
    roots = set()
    for rank in ("0", "1"):
        proc = run(
            f'node_rank="{rank}"; local_root="{expr}"; echo "$local_root"',
            {"SLURM_JOB_ID": "640200", "TMPDIR": str(tmp_path)},
        )
        assert proc.returncode == 0, proc.stderr
        roots.add(proc.stdout.strip())
    assert len(roots) == 2, roots


def test_two_concurrent_jobs_on_the_same_node_rank_get_distinct_local_roots(tmp_path: pathlib.Path) -> None:
    expr = _assignment("local_root")
    roots = set()
    for job_id in ("640200", "640201"):
        proc = run(
            f'node_rank="0"; local_root="{expr}"; echo "$local_root"',
            {"SLURM_JOB_ID": job_id, "TMPDIR": str(tmp_path)},
        )
        assert proc.returncode == 0, proc.stderr
        roots.add(proc.stdout.strip())
    assert len(roots) == 2, roots
