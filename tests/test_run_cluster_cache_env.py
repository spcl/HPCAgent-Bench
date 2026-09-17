# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``experiments/run_cluster.sh``'s OWN cache-root derivation, inside ``run_vllm_node``.

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

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "experiments" / "run_cluster.sh"
TEXT = SCRIPT.read_text()
BODY = TEXT[TEXT.index("run_vllm_node() {") : TEXT.index('case "${1:-}" in')]


def _assignment(var: str) -> str:
    """The right-hand side of ``local <var>="..."`` inside run_vllm_node, exactly as written."""
    match = re.search(rf'local {re.escape(var)}="([^\n]*)"\n', BODY)
    assert match, f"run_vllm_node no longer assigns {var} the way this test expects; update it"
    return match.group(1)


def run(script: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    full_env = {"PATH": "/usr/bin:/bin", **env}
    return subprocess.run(["bash", "-c", script], env=full_env, capture_output=True, text=True)


def test_the_cache_root_derivation_still_reads_jit_cache_root_then_scratch() -> None:
    """Pins the exact expression the two behavioural tests below execute, so a rewrite that changes
    the fallback chain fails here first rather than silently invalidating those tests."""
    assert _assignment("cache_root") == '${JIT_CACHE_ROOT:-${SCRATCH:?set SCRATCH}/.hpcagentbench-cache}'


def test_cache_root_fails_loudly_when_neither_jit_cache_root_nor_scratch_is_set() -> None:
    """Same contract as scripts/cache_env.sh's own JIT_CACHE_ROOT: an inference node with no SCRATCH
    must refuse to pick a cache root rather than silently compiling into $HOME or an ephemeral /tmp
    that vanishes with the container."""
    expr = _assignment("cache_root")
    proc = run(f'cache_root="{expr}"; echo "$cache_root"', {})
    assert proc.returncode != 0, proc.stdout
    assert "set SCRATCH" in proc.stderr, proc.stderr


def test_cache_root_resolves_under_scratch_when_jit_cache_root_is_unset(tmp_path: pathlib.Path) -> None:
    expr = _assignment("cache_root")
    scratch = tmp_path / "scratch"
    proc = run(f'cache_root="{expr}"; echo "$cache_root"', {"SCRATCH": str(scratch)})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"{scratch}/.hpcagentbench-cache"


def test_the_node_local_jit_root_is_keyed_by_both_job_and_node_rank() -> None:
    """The node-local write layer (jit_cache_layer.sh) exists because several engines compiling into
    the SAME directory at once corrupted each other's reads (640074/640075/640090). That guarantee
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
