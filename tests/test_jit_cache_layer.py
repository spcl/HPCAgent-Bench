# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""hpcagent_bench/cluster/jit_cache_layer.sh: engines compile node-locally and publish add-only to the shared cache.

The shared cache is NFS; engines writing it concurrently turn each other's rewrites into ESTALE.
These tests pin the contract that makes the layer safe to share.
"""

import os
import pathlib
import subprocess

import pytest

from hpcagent_bench import paths
from tests.bash_stub import bash_stub

REPO = paths.ROOT
LAYER = REPO / "hpcagent_bench" / "cluster" / "jit_cache_layer.sh"
STAGE_PREFIX = ".jit-layer-staging"


def run(*args: str) -> None:
    subprocess.run(["bash", str(LAYER), *args], check=True, capture_output=True, text=True)


def write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def tree(root: pathlib.Path) -> dict[str, str]:
    return {str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.parametrize(
    ("local_files", "shared_files", "expected"),
    [
        pytest.param(
            {"abc123/kernel.so": "binary", "abc123/__grp__kernel.json": "group"},
            {},
            {"abc123/kernel.so": "binary", "abc123/__grp__kernel.json": "group"},
            id="new-directory-published-whole",
        ),
        # Entries are content-addressed: an existing one is another engine's valid result, and
        # rewriting it under a concurrent reader is exactly what produced ESTALE.
        pytest.param(
            {"abc123/kernel.so": "mine"},
            {"abc123/kernel.so": "theirs"},
            {"abc123/kernel.so": "theirs"},
            id="existing-entry-never-overwritten",
        ),
        pytest.param(
            {"fxgraph/ab/new-graph": "new"},
            {"fxgraph/ab/old-graph": "old"},
            {"fxgraph/ab/new-graph": "new", "fxgraph/ab/old-graph": "old"},
            id="missing-files-inside-a-shared-directory-are-added",
        ),
        pytest.param({"d1/f": "x", "top-file": "y"}, {"d2/g": "z"}, None, id="no-staging-entry-left-behind"),
    ],
)
def test_a_kernel_directory_the_shared_cache_lacks_is_published_whole(
    tmp_path: pathlib.Path,
    local_files: dict[str, str],
    shared_files: dict[str, str],
    expected: dict[str, str] | None,
) -> None:
    local, shared = tmp_path / "local", tmp_path / "shared"
    for name, text in local_files.items():
        write(local / name, text)
    shared.mkdir()
    for name, text in shared_files.items():
        write(shared / name, text)
    run("publish", str(local), str(shared))
    if expected is not None:
        assert tree(shared) == expected, tree(shared)
    leftovers = [p for p in shared.rglob("*") if p.name.startswith(STAGE_PREFIX)]
    assert leftovers == [], leftovers


@pytest.mark.parametrize(
    ("shared_files", "local_files", "expected"),
    [
        pytest.param(
            {"abc123/kernel.so": "binary", f"{STAGE_PREFIX}.nid1.42.def456/kernel.so": "half"},
            {},
            {"abc123/kernel.so": "binary"},
            id="staging-entry-not-copied",
        ),
        pytest.param(
            {"abc123/kernel.so": "shared"},
            {"abc123/kernel.so": "local"},
            {"abc123/kernel.so": "local"},
            id="local-entry-never-replaced",
        ),
        pytest.param(None, {}, {}, id="missing-shared-cache-creates-an-empty-layer"),
    ],
)
def test_seed_copies_the_shared_cache_but_not_a_killed_publishers_staging_entry(
    tmp_path: pathlib.Path,
    shared_files: dict[str, str] | None,
    local_files: dict[str, str],
    expected: dict[str, str],
) -> None:
    shared, local = tmp_path / "shared", tmp_path / "local"
    for name, text in (shared_files or {}).items():
        write(shared / name, text)
    for name, text in local_files.items():
        write(local / name, text)
    run("seed", str(shared if shared_files is not None else tmp_path / "never-created"), str(local))
    assert local.is_dir()
    assert tree(local) == expected, tree(local)


def test_a_seed_copy_that_fails_part_way_leaves_no_entry(tmp_path: pathlib.Path) -> None:
    shared, local = tmp_path / "shared", tmp_path / "local"
    write(shared / "abc123" / "kernel.so", "binary")
    bash_stub(tmp_path / "bin", "cp", '/bin/cp "$@"; exit 1')
    subprocess.run(
        ["bash", str(LAYER), "seed", str(shared), str(local)],
        check=True,
        env={**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}"},
        capture_output=True,
    )
    assert tree(local) == {}


def test_seed_sweeps_a_dead_publishers_staging_entry_and_keeps_a_fresh_one(tmp_path: pathlib.Path) -> None:
    shared, local = tmp_path / "shared", tmp_path / "local"
    dead = shared / f"{STAGE_PREFIX}.nid1.42.def456"
    fresh = shared / f"{STAGE_PREFIX}.nid2.43.ghi789"
    write(dead / "kernel.so", "half")
    write(fresh / "kernel.so", "in-flight")
    subprocess.run(["touch", "-d", "2 days ago", str(dead)], check=True)
    run("seed", str(shared), str(local))
    assert not dead.exists()
    assert fresh.exists()


@pytest.mark.parametrize("variable", ["TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT"])
def test_the_engine_is_pointed_at_the_local_layer_before_it_starts(variable: str) -> None:
    """Static: the redirect has to happen in run_vllm_node before its exec, or the engine writes
    NFS directly again."""
    text = (REPO / "hpcagent_bench" / "cluster" / "run_cluster.sh").read_text()
    body = text[text.index("run_vllm_node() {") :]
    body = body[: body.index('exec "${command[@]}"')]
    redirect = body.index('export TRITON_CACHE_DIR="${local_dirs[0]}"')
    assert f"{variable}=" in body[redirect : redirect + 200], body[redirect : redirect + 200]


def compile_cache(artifacts: list[pathlib.Path]) -> str:
    """The shape vLLM writes: a dict whose values name each artifact by ABSOLUTE path."""
    entries = ", ".join(f"({i}, 0): {{'artifact': '{p}'}}" for i, p in enumerate(artifacts))
    return "{" + entries + "}\n"


def test_seeding_drops_an_entry_whose_recorded_artifact_path_is_gone(tmp_path: pathlib.Path) -> None:
    """A vLLM entry compiled under another job's node-local root is a TRAP, not a miss: the engine
    opens the recorded path and dies with FileNotFoundError.
    """
    local, shared = tmp_path / "local", tmp_path / "shared"
    entry = shared / "95ea305896" / "rank_0_0" / "backbone"
    write(entry / "artifact_compile_range_1_8192_subgraph_0", "code")
    write(
        entry / "vllm_compile_cache.py",
        compile_cache([tmp_path / "gone" / "vllm" / "artifact_compile_range_1_8192_subgraph_0"]),
    )
    run("seed", str(shared), str(local))
    assert not (local / "95ea305896" / "rank_0_0" / "backbone").exists(), tree(local)


def test_seeding_keeps_an_entry_whose_recorded_artifacts_all_resolve(tmp_path: pathlib.Path) -> None:
    """The stable node-local root makes the recorded path the path seed restores the entry to."""
    local, shared = tmp_path / "local", tmp_path / "shared"
    backbone = local / "95ea305896" / "rank_0_0" / "backbone"
    write(shared / "95ea305896" / "rank_0_0" / "backbone" / "artifact_compile_range_1_8192_subgraph_0", "code")
    write(
        shared / "95ea305896" / "rank_0_0" / "backbone" / "vllm_compile_cache.py",
        compile_cache([backbone / "artifact_compile_range_1_8192_subgraph_0"]),
    )
    run("seed", str(shared), str(local))
    assert (backbone / "artifact_compile_range_1_8192_subgraph_0").read_text() == "code"
    assert (backbone / "vllm_compile_cache.py").exists()


def test_scrubbing_a_dead_entry_leaves_the_other_ranks_alone(tmp_path: pathlib.Path) -> None:
    local, shared = tmp_path / "local", tmp_path / "shared"
    good = local / "95ea305896" / "rank_1_0" / "backbone"
    for rank, recorded in (("rank_0_0", tmp_path / "gone" / "art"), ("rank_1_0", good / "art")):
        entry = shared / "95ea305896" / rank / "backbone"
        write(entry / "art", "code")
        write(entry / "vllm_compile_cache.py", compile_cache([recorded]))
    run("seed", str(shared), str(local))
    assert not (local / "95ea305896" / "rank_0_0" / "backbone").exists(), tree(local)
    assert (good / "art").read_text() == "code"
