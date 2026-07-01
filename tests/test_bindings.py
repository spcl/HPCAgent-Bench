# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Canonical C-ABI binding generation (``optarena/bindings/``).

Pins the load-bearing guarantees of ``optarena/docs/abi_contract.md`` for one
dense kernel (``gemm``) and one sparse kernel (``spmv``):

* §4 canonical order -- pointers alpha-sorted, then scalars+symbols
  alpha-sorted, then ``time_ns`` (described separately, position trailing);
* §5 const-ness -- scalars/symbols const, input pointers const, output
  pointers non-const;
* §3 packed group -- a sparse logical array unpacks into ordered member
  pointers;
* §2 phantom filter -- a captured ``np`` numpy module param never reaches the
  signature;
* §7 stub -- contains the signature symbol + a TODO and NOT a reference body;
* §8 JSON round-trip.
"""
import pytest

from optarena.bindings import (
    PackedGroup,
    binding_from_spec,
    gen_call_stub,
    gen_host_glue,
)
from optarena.bindings.stubs import LANGS
from optarena.spec import BenchSpec


def _load(name):
    try:
        return BenchSpec.load(name)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"{name} spec unavailable: {exc}")


# --------------------------------------------------------------------------- #
# Dense kernel: gemm
# --------------------------------------------------------------------------- #


def test_gemm_canonical_order_and_constness():
    spec = _load("gemm")
    b = binding_from_spec(spec)

    ptr_names = [a.name for a in b.args if a.kind == "ptr"]
    scal_names = [a.name for a in b.args if a.kind == "scalar"]

    # gemm: pointers A, B, C ; scalars alpha, beta ; symbols NI, NJ, NK.
    assert ptr_names == ["A", "B", "C"]
    assert ptr_names == sorted(ptr_names)
    assert scal_names == sorted(scal_names)
    # All pointers precede all scalars (§4).
    kinds = [a.kind for a in b.args]
    assert kinds == sorted(kinds, key=lambda k: 0 if k == "ptr" else 1)

    by = {a.name: a for a in b.args}
    # §5 const-ness: C is the output -> non-const; A, B inputs -> const.
    assert by["C"].is_const is False
    assert by["C"].role == "output"
    assert by["A"].is_const is True
    assert by["B"].is_const is True
    # Every scalar/symbol is const.
    for s in (a for a in b.args if a.kind == "scalar"):
        assert s.is_const is True
    # Symbols are tagged as such.
    assert by["NI"].role == "symbol"
    assert by["NI"].dtype == "int64"
    assert by["alpha"].role is None
    assert by["alpha"].dtype == "float64"


def test_gemm_time_ns_trailing_and_separate():
    spec = _load("gemm")
    b = binding_from_spec(spec)
    # time_ns is NOT in args; it is the separate trailing descriptor (§6).
    assert all(a.name != "time_ns" for a in b.args)
    j = b.to_json()
    assert j["time_ns"]["position"] == "trailing"
    assert j["time_ns"]["name"] == "time_ns"
    assert j["time_ns"]["dtype"] == "int64"
    assert j["packed"] == {}


def test_gemm_stub_has_signature_and_todo_not_reference():
    spec = _load("gemm")
    b = binding_from_spec(spec)
    for lang in LANGS:
        stub = gen_call_stub(b, lang)
        assert b.symbols[lang] in stub, lang
        assert "TODO" in stub, lang
        assert "time_ns" in stub, lang
        assert "workspace" in stub and "workspace_size" in stub, lang  # §11 always present
        # Never the reference solution.
        assert "alpha * A @ B" not in stub
        assert "A[i]" not in stub and "C[i * NJ" not in stub

    c_stub = gen_call_stub(b, "c")
    # The canonical C signature shape (§7 / §9).
    assert "const double *restrict A" in c_stub
    assert "double *restrict C" in c_stub  # output, non-const
    assert "const long" not in c_stub  # symbols are int64_t
    assert "const int64_t NI" in c_stub
    assert "int64_t *restrict time_ns" in c_stub
    # §11 reserved scratch pair, after time_ns.
    assert "uint8_t *restrict workspace" in c_stub
    assert "const int64_t workspace_size" in c_stub
    assert c_stub.index("time_ns") < c_stub.index("workspace")


def test_gemm_host_glue_brackets_pure_with_timer():
    spec = _load("gemm")
    b = binding_from_spec(spec)
    glue = gen_host_glue(b)
    assert "gemm_pure" in glue
    assert "timer_start" in glue and "timer_end" in glue
    assert "time_ns[0] =" in glue
    assert b.symbols["c"] in glue


def test_gemm_json_round_trip():
    spec = _load("gemm")
    b = binding_from_spec(spec)
    j = b.to_json()
    assert j["kernel"] == "gemm"
    assert j["abi"] == "c-abi-v2"
    assert j["symbol"] == "gemm_fp64"
    # §11 reserved scratch pair described after time_ns, NULLable + never in args.
    assert j["workspace"]["name"] == "workspace" and j["workspace"]["dtype"] == "uint8"
    assert j["workspace"]["size_name"] == "workspace_size" and j["workspace"]["nullable"] is True
    assert set(j["symbols"]) == set(LANGS)
    names = [a["name"] for a in j["args"]]
    assert names == ["A", "B", "C", "NI", "NJ", "NK", "alpha", "beta"]
    # const flags carried through.
    cmap = {a["name"]: a["const"] for a in j["args"]}
    assert cmap["C"] is False and cmap["A"] is True and cmap["alpha"] is True


# --------------------------------------------------------------------------- #
# Sparse kernel: spmv (packed group)
# --------------------------------------------------------------------------- #


def test_spmv_packed_group_and_order():
    spec = _load("spmv")
    if not spec.configurations:
        pytest.skip("spmv has no sparse configurations")
    b = binding_from_spec(spec, config="csr")

    # §3: A is a packed group with ordered member buffers.
    assert len(b.packed) == 1
    g = b.packed[0]
    assert isinstance(g, PackedGroup)
    assert g.logical == "A"
    assert g.fmt == "csr"
    # Members sorted by member name.
    assert list(g.members) == sorted(g.members)
    assert set(g.members) == {"A_data", "A_indices", "A_indptr"}

    # The members appear in the flat pointer block as ordinary const pointers,
    # alpha-sorted with the other pointers (x), all before scalars (§4).
    ptr_names = [a.name for a in b.args if a.kind == "ptr"]
    assert ptr_names == sorted(ptr_names)
    assert {"A_data", "A_indices", "A_indptr", "x"} <= set(ptr_names)
    for m in g.members:
        arg = next(a for a in b.args if a.name == m)
        assert arg.is_const is True
        assert arg.kind == "ptr"
    # Index buffers carry an int dtype, data buffer the float dtype.
    by = {a.name: a for a in b.args}
    assert by["A_indptr"].dtype == "int64"
    assert by["A_data"].dtype == "float64"

    # JSON records the packed group (§8).
    j = b.to_json()
    assert j["packed"]["A"]["format"] == "csr"
    assert j["packed"]["A"]["members"] == sorted(g.members)


def test_spmv_host_glue_unpacks_handle():
    spec = _load("spmv")
    if not spec.configurations:
        pytest.skip("spmv has no sparse configurations")
    b = binding_from_spec(spec, config="csr")
    glue = gen_host_glue(b)
    # The wrapper documents the unpack of the logical handle into members (§3).
    assert "packed handle A [csr]" in glue
    for m in b.packed[0].members:
        assert m in glue


# --------------------------------------------------------------------------- #
# Phantom-arg filter (§2)
# --------------------------------------------------------------------------- #


def test_phantom_np_arg_filtered():
    # A synthetic spec carrying a captured ``np`` numpy module param: it must
    # never reach the binding (§2).
    raw = {
        "short_name": "phantom",
        "name": "phantom",
        "relative_path": "phantom",
        "module_name": "phantom",
        "func_name": "kernel",
        "parameters": {
            "S": {
                "N": 16
            }
        },
        "input_args": ["x", "y", "N", "np"],
        "array_args": ["x", "y"],
        "output_args": ["y"],
    }
    spec = BenchSpec.from_dict(raw, source="<test>")
    b = binding_from_spec(spec)
    names = [a.name for a in b.args]
    assert "np" not in names
    assert names == ["x", "y", "N"]  # x,y pointers then N symbol
    by = {a.name: a for a in b.args}
    assert by["y"].is_const is False and by["y"].role == "output"
    assert by["x"].is_const is True
    assert by["N"].role == "symbol" and by["N"].is_const is True
