# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The MPI driver wire format (``optarena/agent_bench/mpi_wire.py``).

Both harness-owned drivers read the SAME infile and write the SAME outfile, so a mismatch
between them (or between a driver and the gather) would silently corrupt a distributed result.
These tests pin the layout end to end WITHOUT a cluster: the descriptor partitions, the wire
serialises, an in-process ``numpy`` "kernel" stands in for the driver, and the gather must
reconstruct the global array bit-for-bit -- the same round-trip the C / mpi4py drivers perform.
"""
import numpy as np
import pytest

from optarena.agent_bench.mpi_descriptor import ArrayDist, AxisDist, Descriptor, Grid
from optarena.agent_bench.mpi_wire import pack_infile, pack_outfile, unpack_infile, unpack_outfile
from optarena.bindings.contract import Arg, Binding
from optarena.bindings.stubs import LANGS


def _binding(*args) -> Binding:
    return Binding(kernel="k", config="dense", args=tuple(args), symbols={lang: "k_fp64" for lang in LANGS})


def _yax_binding() -> Binding:
    # y = a*x : pointers x (in), y (out) ; symbol N ; value scalar a. Canonical order already.
    return _binding(
        Arg(name="x", kind="ptr", dtype="float64", is_const=True),
        Arg(name="y", kind="ptr", dtype="float64", is_const=False, role="output"),
        Arg(name="N", kind="scalar", dtype="int64", is_const=True, role="symbol"),
        Arg(name="a", kind="scalar", dtype="float64", is_const=True),
    )


def _block0(nranks: int, arrays) -> Descriptor:
    ad = ArrayDist(axes=(AxisDist(grid_dim=0, scheme="block"), ))
    return Descriptor(grid=Grid((nranks, )), arrays={n: ad for n in arrays}, symbol_axes={"N": [("x", 0)]})


# --------------------------------------------------------------------------------------- #
# Infile: tiles, localised scalars, per-rank workspace
# --------------------------------------------------------------------------------------- #
def test_infile_roundtrip_localises_symbol_and_workspace():
    N, R = 10, 4  # ragged: 10 over 4 -> 3,3,2,2
    b, desc = _yax_binding(), _block0(4, ("x", "y"))
    x = np.arange(N, dtype=np.float64) + 1.0
    raw = pack_infile(b, desc, {"x": x, "y": np.zeros(N)}, {"N": N, "a": 2.5}, k_repeats=3, workspace_expr="8*N")
    p = unpack_infile(raw)

    assert p.nranks == R and p.k_repeats == 3
    localNs = [3, 3, 2, 2]
    for r in range(R):
        local_n, a = p.scalar_values[r]
        assert local_n == localNs[r]  # size symbol N is the LOCAL extent
        assert a == 2.5  # a value scalar is replicated unchanged
        assert p.workspace_bytes[r] == 8 * localNs[r]  # 8*N over the LOCAL N
        assert p.ptrs[0].tiles[r].shape == (localNs[r], )  # x tile local-shaped
    # the x tiles reassemble to the original global array
    assert np.array_equal(np.concatenate([p.ptrs[0].tiles[r] for r in range(R)]), x)


def test_infile_no_workspace_is_zero_per_rank():
    b, desc = _yax_binding(), _block0(2, ("x", "y"))
    raw = pack_infile(b, desc, {"x": np.arange(6.0), "y": np.zeros(6)}, {"N": 6, "a": 1.0}, k_repeats=1)
    p = unpack_infile(raw)
    assert p.workspace_bytes == [0, 0]  # no request -> 0 bytes everywhere (ABI §11)


def test_infile_replicated_array_full_copy_each_rank():
    # A replicated pointer: every rank gets the whole array (scalars/length-1 rule + explicit).
    b = _binding(
        Arg(name="w", kind="ptr", dtype="float64", is_const=True),
        Arg(name="y", kind="ptr", dtype="float64", is_const=False, role="output"),
        Arg(name="N", kind="scalar", dtype="int64", is_const=True, role="symbol"),
    )
    desc = Descriptor(grid=Grid((3, )),
                      arrays={
                          "w": ArrayDist(replicated=True),
                          "y": ArrayDist(axes=(AxisDist(grid_dim=0, scheme="block"), ))
                      },
                      symbol_axes={"N": [("y", 0)]})
    w = np.arange(5.0)
    p = unpack_infile(pack_infile(b, desc, {"w": w, "y": np.zeros(5)}, {"N": 5}, k_repeats=1))
    for r in range(3):
        assert np.array_equal(p.ptrs[0].tiles[r], w)  # whole array on every rank
    # N is tied to y (distributed), not w -> localised by y's split
    assert [p.scalar_values[r][0] for r in range(3)] == [2, 2, 1]


def test_infile_2d_block_rows():
    b = _binding(
        Arg(name="A", kind="ptr", dtype="float64", is_const=False, role="output"),
        Arg(name="N", kind="scalar", dtype="int64", is_const=True, role="symbol"),
    )
    desc = Descriptor(grid=Grid((2, )),
                      arrays={"A": ArrayDist(axes=(AxisDist(grid_dim=0, scheme="block"), AxisDist(grid_dim=None)))},
                      symbol_axes={"N": [("A", 0)]})
    A = np.arange(20.0).reshape(5, 4)
    p = unpack_infile(pack_infile(b, desc, {"A": A}, {"N": 5}, k_repeats=1))
    assert p.ptrs[0].tiles[0].shape == (3, 4) and p.ptrs[0].tiles[1].shape == (2, 4)
    assert np.array_equal(np.concatenate(p.ptrs[0].tiles, axis=0), A)


@pytest.mark.parametrize("dtype", ["float64", "float32", "int64", "int32"])
def test_infile_dtypes_roundtrip(dtype):
    b = _binding(
        Arg(name="A", kind="ptr", dtype=dtype, is_const=False, role="output"),
        Arg(name="N", kind="scalar", dtype="int64", is_const=True, role="symbol"),
    )
    desc = Descriptor(grid=Grid((4, )),
                      arrays={"A": ArrayDist(axes=(AxisDist(grid_dim=0, scheme="block"), ))},
                      symbol_axes={"N": [("A", 0)]})
    A = np.arange(9, dtype=dtype)
    p = unpack_infile(pack_infile(b, desc, {"A": A}, {"N": 9}, k_repeats=1))
    rebuilt = np.concatenate([p.ptrs[0].tiles[r] for r in range(4)])
    assert rebuilt.dtype == np.dtype(dtype) and np.array_equal(rebuilt, A)


def test_infile_rejects_unserialisable_dtype():
    b = _binding(Arg(name="A", kind="ptr", dtype="complex128", is_const=False, role="output"))
    desc = Descriptor(grid=Grid((1, )), arrays={"A": ArrayDist(replicated=True)})
    with pytest.raises(ValueError, match="not wire-serialisable"):
        pack_infile(b, desc, {"A": np.zeros(3, dtype="complex128")}, {}, k_repeats=1)


def test_unpack_rejects_bad_magic():
    with pytest.raises(ValueError, match="magic"):
        unpack_infile(b"\x00" * 64)


# --------------------------------------------------------------------------------------- #
# Outfile + the full scatter -> compute -> gather round-trip (what the drivers actually do)
# --------------------------------------------------------------------------------------- #
def test_outfile_roundtrip():
    tiles = [np.arange(3.0), np.arange(3.0, 5.0)]  # ragged per-rank output tiles
    raw = pack_outfile(2, 4, [0.1, 0.2, 0.3, 0.05], [("y", "float64", tiles)])
    samples, outputs = unpack_outfile(raw)
    assert samples == [0.1, 0.2, 0.3, 0.05]
    dtype, got = outputs[0]
    assert dtype == "float64"
    assert np.array_equal(got[0], tiles[0]) and np.array_equal(got[1], tiles[1])


def _full_roundtrip(b, desc, data, scalars, N, kernel, expected, dtype=np.float64):
    """scatter (pack_infile) -> per-rank numpy kernel -> gather (pack/unpack_outfile) -> global."""
    p = unpack_infile(pack_infile(b, desc, data, scalars, k_repeats=2))
    out_names = [a.name for a in b.pointers if a.role == "output"]
    out_idx = [i for i, a in enumerate(b.pointers) if a.role == "output"]
    per_rank_out = {name: [] for name in out_names}
    for r in range(desc.grid.nranks):
        local = {a.name: p.ptrs[i].tiles[r] for i, a in enumerate(b.pointers)}
        local_scalars = {a.name: p.scalar_values[r][j] for j, a in enumerate(b.scalars)}
        produced = kernel(local, local_scalars)  # {out_name: local_out_tile}
        for name in out_names:
            per_rank_out[name].append(np.asarray(produced[name]))
    outputs = [(b.pointers[i].name, b.pointers[i].dtype, per_rank_out[b.pointers[i].name]) for i in out_idx]
    _samples, decoded = unpack_outfile(pack_outfile(desc.grid.nranks, 2, [1.0, 2.0], outputs))
    result = {}
    for (odtype, tiles), i in zip(decoded, out_idx):
        name = b.pointers[i].name
        shaped = [t.reshape(desc.local_shape(name, np.shape(data[name]), r)) for r, t in enumerate(tiles)]
        result[name] = desc.gather(name, shaped, np.shape(data[name]), np.dtype(odtype))
    for name in out_names:
        assert np.allclose(result[name], expected[name]), (name, result[name], expected[name])


def test_full_roundtrip_yax_block():
    N = 13
    b, desc = _yax_binding(), _block0(4, ("x", "y"))
    x = np.arange(N, dtype=np.float64) + 1.0
    _full_roundtrip(b,
                    desc, {
                        "x": x,
                        "y": np.zeros(N)
                    }, {
                        "N": N,
                        "a": 3.0
                    },
                    N,
                    kernel=lambda loc, sc: {"y": sc["a"] * loc["x"]},
                    expected={"y": 3.0 * x})


def test_full_roundtrip_replicated_reduction_reads_rank0():
    # A length-1 (replicated) output: every rank computes it; gather reads rank 0's copy.
    b = _binding(
        Arg(name="x", kind="ptr", dtype="float64", is_const=True),
        Arg(name="s", kind="ptr", dtype="float64", is_const=False, role="output"),
        Arg(name="N", kind="scalar", dtype="int64", is_const=True, role="symbol"),
    )
    desc = Descriptor(grid=Grid((4, )),
                      arrays={
                          "x": ArrayDist(axes=(AxisDist(grid_dim=0, scheme="block"), )),
                          "s": ArrayDist(replicated=True)
                      },
                      symbol_axes={"N": [("x", 0)]})
    x = np.arange(8.0) + 1.0
    # Each rank writes the GLOBAL sum into its replicated s (coherent across ranks); gather
    # takes rank 0's -> the global reduction. (dist_for coerces length-1 s to replicated too.)
    _full_roundtrip(b,
                    desc, {
                        "x": x,
                        "s": np.zeros(1)
                    }, {"N": 8},
                    8,
                    kernel=lambda loc, sc: {"s": np.array([x.sum()])},
                    expected={"s": np.array([x.sum()])})
