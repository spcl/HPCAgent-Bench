# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""2026-09-23 USER: an INDEPENDENT tile oracle for :func:`mpi_descriptor.local_slice` -- not a
round-trip against the harness's own math (:mod:`test_ml_layout_flexible`'s job), a check against
someone else's implementation of the same idea.

For a BLOCK layout (any axis, or a 2-D 2x2 grid), the independent oracle is PyTorch's own
``torch.distributed.tensor``: ``distribute_tensor(global, mesh, [Shard(axis), ...])`` shards with
``torch.chunk`` semantics, and at an extent divisible by the rank count (the 64-rule's own
condition) that is BIT-IDENTICAL to ``_block_bounds``'s load-balanced block. For cyclic /
block_cyclic there is no DTensor placement that matches this harness's scheme, so the oracle there
is a formula written directly in THIS file: ``owner(i) = i % P`` (cyclic) or ``(i // b) % P``
(block_cyclic) -- never a call into :mod:`mpi_descriptor`.

Runs under real ``torch.distributed`` (backend ``gloo``, CPU, ``torch.multiprocessing.spawn``) --
DTensor's ``init_device_mesh`` needs an active process group even on CPU."""

import socket
from collections.abc import Callable

import pytest

torch = pytest.importorskip("torch")
dtensor = pytest.importorskip("torch.distributed.tensor")
import torch.distributed as dist
import torch.multiprocessing as mp

from hpcagent_bench.harness.mpi_descriptor import ArrayDist, AxisDist, Grid, local_slice


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker_block(rank: int, world: int, port: int, axis: int, shape: tuple, q: "mp.Queue") -> None:
    dist.init_process_group(backend="gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=world, rank=rank)
    try:
        from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

        g = torch.arange(int(torch.tensor(shape).prod()), dtype=torch.int64).reshape(shape)
        mesh = init_device_mesh("cpu", (world,))
        oracle_local = distribute_tensor(g, mesh, [Shard(axis)]).to_local()

        dist_ = ArrayDist(axes=tuple(AxisDist(0, "block") if d == axis else AxisDist() for d in range(len(shape))))
        ours = local_slice(g.numpy(), dist_, Grid((world,)), rank)
        q.put((rank, bool(torch.equal(oracle_local, torch.from_numpy(ours)))))
    finally:
        dist.destroy_process_group()


def _worker_block_2d(rank: int, world: int, port: int, shape: tuple, q: "mp.Queue") -> None:
    dist.init_process_group(backend="gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=world, rank=rank)
    try:
        from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

        g = torch.arange(int(torch.tensor(shape).prod()), dtype=torch.int64).reshape(shape)
        mesh = init_device_mesh("cpu", (2, 2))
        oracle_local = distribute_tensor(g, mesh, [Shard(0), Shard(1)]).to_local()

        dist_ = ArrayDist(axes=(AxisDist(0, "block"), AxisDist(1, "block")))
        ours = local_slice(g.numpy(), dist_, Grid((2, 2)), rank)
        q.put((rank, bool(torch.equal(oracle_local, torch.from_numpy(ours)))))
    finally:
        dist.destroy_process_group()


def run_spawn(worker: "Callable[..., None]", world: int, *extra_args: object) -> dict[int, bool]:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = free_port()
    mp.spawn(worker, args=(world, port, *extra_args, q), nprocs=world, join=True)
    return dict(q.get(timeout=60) for _ in range(world))


@pytest.mark.parametrize("axis", [0, 1])
def test_block_layout_matches_pytorch_dtensor(axis: int) -> None:
    """Any axis, 1-D grid: our local_slice == DTensor's to_local(), bitwise, at every rank."""
    results = run_spawn(_worker_block, 4, axis, (32, 16))
    assert results == {0: True, 1: True, 2: True, 3: True}, results


def test_2d_block_grid_matches_pytorch_dtensor() -> None:
    """A 2x2 grid splitting both of an array's first two axes: our local_slice == DTensor's
    to_local() under [Shard(0), Shard(1)] on a (2,2) mesh, bitwise, at every rank."""
    results = run_spawn(_worker_block_2d, 4, (16, 16))
    assert results == {0: True, 1: True, 2: True, 3: True}, results


@pytest.mark.parametrize("scheme,block_size", [("cyclic", 1), ("block_cyclic", 4)])
@pytest.mark.parametrize("axis", [0, 1])
def test_cyclic_and_block_cyclic_match_an_independent_index_formula(scheme: str, block_size: int, axis: int) -> None:
    """cyclic: owner(i) = i % P. block_cyclic: owner(i) = (i // block_size) % P. Written here, not
    borrowed from mpi_descriptor.owned_indices -- an independent check of the same claim."""
    import numpy as np

    shape = (32, 16)
    world = 4
    g = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    n = shape[axis]
    dist_ = ArrayDist(axes=tuple(AxisDist(0, scheme, block_size) if d == axis else AxisDist() for d in range(2)))
    grid = Grid((world,))
    for rank in range(world):
        if scheme == "cyclic":
            owned = [i for i in range(n) if i % world == rank]
        else:
            owned = [i for i in range(n) if (i // block_size) % world == rank]
        expected = np.take(g, owned, axis=axis)
        ours = local_slice(g, dist_, grid, rank)
        assert np.array_equal(ours, expected), (scheme, axis, rank)
