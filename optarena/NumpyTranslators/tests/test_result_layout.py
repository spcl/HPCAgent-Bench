"""Sparse-op result-layout rule (directive #3) + the no-JSON size convention
(directive #2). A pure-logic unit test: reads no bench_info JSON, needs no
toolchain. Imports resolve via PYTHONPATH (the suite's convention)."""
import numpy as np

from numpyto_common.sparse_emit import DENSE, FRAMEWORK_SPARSE_CAPS, result_layout
from numpyto_common.testing import sizes


def test_sparse_times_dense_is_dense():
    # sparse @ dense and dense @ sparse -> dense, every target.
    assert result_layout("csr", None, "c") == DENSE
    assert result_layout(None, "csr", "c") == DENSE
    assert result_layout("csr", None, "jax") == DENSE
    assert result_layout(None, "dia", "fortran") == DENSE


def test_sparse_times_sparse_follows_caps():
    # Backends with no sparse-result SpGEMM in the lowering path densify.
    assert result_layout("csr", "csr", "c") == DENSE
    assert result_layout("csr", "csr", "fortran") == DENSE
    assert result_layout("csr", "csr", "jax") == DENSE
    # Backends that follow the scipy source (CSR @ CSR -> CSR) keep the layout.
    assert result_layout("csr", "csr", "numba") == "csr"
    assert result_layout("csr", "csr", "cupy") == "csr"


def test_jax_spmm_skip_is_a_consequence_of_the_rule():
    # spmm is CSR @ CSR on JAX; BCOO@BCOO densifies, so the rule yields DENSE
    # at benchmark size -> the documented skip is the rule, not an ad-hoc call.
    assert FRAMEWORK_SPARSE_CAPS["jax"] is False
    assert result_layout("csr", "csr", "jax") == DENSE


def test_sizes_come_from_frozen_small_fixture_not_json():
    # Directive #2: a unit test sizes its inputs from the frozen small fixture,
    # never from bench_info/*.json (which can change under it). The shapes are
    # tiny (so the test runs fast) and distinct per axis (to catch index bugs).
    g = sizes("gemm")
    assert g == {"NI": 32, "NJ": 48, "NK": 64}
    assert len(set(g.values())) == len(g)  # distinct per axis
    a = np.zeros((g["NI"], g["NK"]))
    assert a.shape == (32, 64)
