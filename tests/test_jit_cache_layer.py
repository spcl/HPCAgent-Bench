# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""experiments/jit_cache_layer.sh: engines compile node-locally and publish add-only to the shared cache.

The shared cache is NFS; engines writing it concurrently turned each other's rewrites into ESTALE
(640074/640075/640090). These tests pin the contract that makes the layer safe to share.
"""

import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LAYER = REPO / "experiments" / "jit_cache_layer.sh"
STAGE_PREFIX = ".jit-layer-staging"


def run(*args: str) -> None:
    subprocess.run(["bash", str(LAYER), *args], check=True, capture_output=True, text=True)


def write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def tree(root: pathlib.Path) -> dict[str, str]:
    return {str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*")) if p.is_file()}


def test_a_kernel_directory_the_shared_cache_lacks_is_published_whole(tmp_path: pathlib.Path) -> None:
    local, shared = tmp_path / "local", tmp_path / "shared"
    write(local / "abc123" / "kernel.so", "binary")
    write(local / "abc123" / "__grp__kernel.json", "group")
    shared.mkdir()
    run("publish", str(local), str(shared))
    assert tree(shared) == {"abc123/kernel.so": "binary", "abc123/__grp__kernel.json": "group"}, tree(shared)


def test_publish_never_overwrites_an_entry_the_shared_cache_already_has(tmp_path: pathlib.Path) -> None:
    """Entries are content-addressed: an existing one is another engine's valid result, and
    rewriting it under a concurrent reader is exactly what produced ESTALE."""
    local, shared = tmp_path / "local", tmp_path / "shared"
    write(local / "abc123" / "kernel.so", "mine")
    write(shared / "abc123" / "kernel.so", "theirs")
    run("publish", str(local), str(shared))
    assert (shared / "abc123" / "kernel.so").read_text() == "theirs"


def test_files_missing_inside_a_directory_both_sides_have_are_added(tmp_path: pathlib.Path) -> None:
    local, shared = tmp_path / "local", tmp_path / "shared"
    write(local / "fxgraph" / "ab" / "new-graph", "new")
    write(shared / "fxgraph" / "ab" / "old-graph", "old")
    run("publish", str(local), str(shared))
    assert tree(shared) == {"fxgraph/ab/new-graph": "new", "fxgraph/ab/old-graph": "old"}, tree(shared)


def test_publish_leaves_no_staging_entry_behind(tmp_path: pathlib.Path) -> None:
    local, shared = tmp_path / "local", tmp_path / "shared"
    write(local / "d1" / "f", "x")
    write(local / "top-file", "y")
    write(shared / "d2" / "g", "z")
    run("publish", str(local), str(shared))
    leftovers = [p for p in shared.rglob("*") if p.name.startswith(STAGE_PREFIX)]
    assert leftovers == [], leftovers


def test_seed_copies_the_shared_cache_but_not_a_killed_publishers_staging_entry(tmp_path: pathlib.Path) -> None:
    shared, local = tmp_path / "shared", tmp_path / "local"
    write(shared / "abc123" / "kernel.so", "binary")
    write(shared / f"{STAGE_PREFIX}.nid1.42.def456" / "kernel.so", "half")
    run("seed", str(shared), str(local))
    assert tree(local) == {"abc123/kernel.so": "binary"}, tree(local)


def test_seed_never_replaces_an_entry_already_in_the_local_layer(tmp_path: pathlib.Path) -> None:
    shared, local = tmp_path / "shared", tmp_path / "local"
    write(shared / "abc123" / "kernel.so", "shared")
    write(local / "abc123" / "kernel.so", "local")
    run("seed", str(shared), str(local))
    assert (local / "abc123" / "kernel.so").read_text() == "local"


def test_seed_of_a_cache_that_does_not_exist_yet_creates_an_empty_layer(tmp_path: pathlib.Path) -> None:
    local = tmp_path / "local"
    run("seed", str(tmp_path / "never-created"), str(local))
    assert local.is_dir() and not any(local.iterdir())


@pytest.mark.parametrize("variable", ["TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT"])
def test_the_engine_is_pointed_at_the_local_layer_before_it_starts(variable: str) -> None:
    """Static: the redirect has to happen in run_vllm_node before its exec, or the engine writes
    NFS directly again."""
    text = (REPO / "experiments" / "run_cluster.sh").read_text()
    body = text[text.index("run_vllm_node() {") :]
    body = body[: body.index('exec "${command[@]}"')]
    redirect = body.index('export TRITON_CACHE_DIR="${local_dirs[0]}"')
    assert f"{variable}=" in body[redirect : redirect + 200], body[redirect : redirect + 200]
