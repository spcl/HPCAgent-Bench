# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Direct tests for :func:`validate_sparse_config`, the 11-rule sparse-layout validator.

The module had no direct test file (a name-grep survey confirmed no ``tests/test_*.py`` imports
it). Every rule raises on the FIRST violation with a specific message, and rule 10/11 pin exact
identity / naming conventions the rest of the sparse ABI depends on -- both worth pinning tightly
rather than trusting only the end-to-end emit path that happens to exercise a subset of them.
"""
from typing import Dict

import pytest

from hpcagent_bench.spec import SparseBuffer, SparseConfiguration, SparseDistribution, SparseLayout, SparseLayoutVariant
from hpcagent_bench.validate_sparse import SparseConfigError, validate_sparse_config


def _buf(role: str, name: str, dtype: str = "float64") -> SparseBuffer:
    return SparseBuffer(role=role, name=name, shape=("n", ), dtype=dtype)


def _csr_layout(logical: str = "A") -> SparseLayout:
    """One valid CSR layout for a logical array, buffers named per the rule-11 convention."""
    variant = SparseLayoutVariant(
        format="csr",
        buffers=(
            _buf("indptr", f"{logical}_indptr", "int32"),
            _buf("indices", f"{logical}_indices", "int32"),
            _buf("data", f"{logical}_data", "float64"),
        ),
    )
    return SparseLayout(logical_shape=("NI", "NJ"), default_dtype="float64", variants={"csr": variant})


def _dense_layout(logical: str = "B") -> SparseLayout:
    variant = SparseLayoutVariant(format="dense", buffers=(_buf("data", f"{logical}_data", "float64"), ))
    return SparseLayout(logical_shape=("NI", "NJ"), default_dtype="float64", variants={"dense": variant})


def _coo_layout(logical: str = "B") -> SparseLayout:
    variant = SparseLayoutVariant(
        format="coo",
        buffers=(
            _buf("row", f"{logical}_row", "int32"),
            _buf("col", f"{logical}_col", "int32"),
            _buf("data", f"{logical}_data", "float64"),
        ),
    )
    return SparseLayout(logical_shape=("NI", "NJ"), default_dtype="float64", variants={"coo": variant})


def _valid_single_array() -> Dict:
    """One logical array 'A' with a valid CSR layout, configuration and distribution."""
    return {
        "sparse_layouts": {
            "A": _csr_layout("A")
        },
        "configurations": {
            "csr_only": SparseConfiguration(arrays={"A": "csr"})
        },
        "distributions": {
            "uniform": SparseDistribution(configuration="csr_only", distribution="uniform")
        },
        "array_args": ["A"],
    }


# --- baseline / edge cases -------------------------------------------------------------------------
def test_all_empty_is_trivially_valid():
    # No arrays declared at all -- every rule's loop is over an empty collection.
    validate_sparse_config({}, {}, {}, [])


def test_a_fully_valid_single_array_config_passes():
    kw = _valid_single_array()
    assert validate_sparse_config(kw["sparse_layouts"], kw["configurations"], kw["distributions"],
                                  kw["array_args"]) is None


def test_non_sparse_array_args_are_untouched():
    # array_args entries with no sparse_layouts entry at all (dense-only arrays) never trip rule 9.
    kw = _valid_single_array()
    kw["array_args"] = ["A", "alpha", "beta"]
    validate_sparse_config(kw["sparse_layouts"], kw["configurations"], kw["distributions"], kw["array_args"])


# --- rule 1: format must be supported ----------------------------------------------------------
def test_rule1_unsupported_format_key_is_rejected():
    layout = SparseLayout(logical_shape=("NI", ),
                          default_dtype="float64",
                          variants={"not_a_real_format": _csr_layout("A").variants["csr"]})
    with pytest.raises(SparseConfigError, match="unsupported format 'not_a_real_format'"):
        validate_sparse_config({"A": layout}, {}, {}, [])


# --- rule 2: required buffer roles per format ----------------------------------------------------
def test_rule2_missing_required_role_is_rejected():
    # CSR needs indptr + indices + data; drop indices.
    variant = SparseLayoutVariant(format="csr", buffers=(_buf("indptr", "A_indptr", "int32"), _buf("data", "A_data")))
    layout = SparseLayout(logical_shape=("NI", ), default_dtype="float64", variants={"csr": variant})
    with pytest.raises(SparseConfigError, match=r"missing required buffer roles \['indices'\]"):
        validate_sparse_config({"A": layout}, {}, {}, [])


# --- rule 3: numeric dtype -----------------------------------------------------------------------
def test_rule3_unsupported_dtype_is_rejected():
    layout = _csr_layout("A")
    bad = SparseLayoutVariant(format="csr",
                              buffers=(
                                  _buf("indptr", "A_indptr", "int32"),
                                  _buf("indices", "A_indices", "int32"),
                                  _buf("data", "A_data", "not_a_dtype"),
                              ))
    layout.variants["csr"] = bad
    with pytest.raises(SparseConfigError, match="unsupported dtype 'not_a_dtype'"):
        validate_sparse_config({"A": layout}, {}, {}, [])


# --- rule 4: index buffers must be int32/int64 ---------------------------------------------------
def test_rule4_index_role_with_float_dtype_is_rejected():
    layout = _csr_layout("A")
    bad = SparseLayoutVariant(
        format="csr",
        buffers=(
            _buf("indptr", "A_indptr", "float64"),  # index role, wrong dtype
            _buf("indices", "A_indices", "int32"),
            _buf("data", "A_data", "float64"),
        ))
    layout.variants["csr"] = bad
    with pytest.raises(SparseConfigError, match="index buffer must be int32 or int64"):
        validate_sparse_config({"A": layout}, {}, {}, [])


def test_rule4_int64_index_buffer_is_accepted():
    layout = _csr_layout("A")
    ok = SparseLayoutVariant(format="csr",
                             buffers=(
                                 _buf("indptr", "A_indptr", "int64"),
                                 _buf("indices", "A_indices", "int64"),
                                 _buf("data", "A_data", "float64"),
                             ))
    layout.variants["csr"] = ok
    validate_sparse_config({"A": layout}, {}, {}, [])


# --- rule 5: configuration must name every layout-bearing array ----------------------------------
def test_rule5_configuration_missing_an_array_entry_is_rejected():
    layout = _csr_layout("A")
    cfg = SparseConfiguration(arrays={})  # 'A' has a layout but no chosen format
    with pytest.raises(SparseConfigError, match="missing entry for array 'A'"):
        validate_sparse_config({"A": layout}, {"only": cfg}, {}, [])


# --- rule 6: configuration's format must be declared on the layout -------------------------------
def test_rule6_configuration_format_not_in_layout_variants_is_rejected():
    layout = _csr_layout("A")  # only declares 'csr'
    cfg = SparseConfiguration(arrays={"A": "csc"})
    with pytest.raises(SparseConfigError, match=r"not in sparse_layouts\.A\.variants"):
        validate_sparse_config({"A": layout}, {"bad": cfg}, {}, [])


# --- rule 7: at most one non-dense sparse format per configuration --------------------------------
def test_rule7_mixing_two_sparse_formats_in_one_configuration_is_rejected():
    layouts = {"A": _csr_layout("A"), "B": _coo_layout("B")}
    cfg = SparseConfiguration(arrays={"A": "csr", "B": "coo"})
    with pytest.raises(SparseConfigError, match="cannot mix sparse formats"):
        validate_sparse_config(layouts, {"mixed": cfg}, {}, [])


def test_rule7_dense_plus_one_sparse_format_is_allowed():
    layouts = {"A": _csr_layout("A"), "C": _dense_layout("C")}
    cfg = SparseConfiguration(arrays={"A": "csr", "C": "dense"})
    validate_sparse_config(layouts, {"mixed": cfg}, {}, [])


# --- rule 8: distribution must point at a real configuration --------------------------------------
def test_rule8_distribution_pointing_at_unknown_configuration_is_rejected():
    kw = _valid_single_array()
    dist = SparseDistribution(configuration="nonexistent", distribution="uniform")
    with pytest.raises(SparseConfigError, match="configuration 'nonexistent' not in configurations"):
        validate_sparse_config(kw["sparse_layouts"], kw["configurations"], {"d": dist}, kw["array_args"])


# --- rule 9: array_args must be logical names, not physical buffer names --------------------------
def test_rule9_physical_buffer_name_in_array_args_is_rejected():
    kw = _valid_single_array()
    with pytest.raises(SparseConfigError, match="'A_indptr' is a physical buffer name"):
        validate_sparse_config(kw["sparse_layouts"], kw["configurations"], kw["distributions"], ["A_indptr"])


# --- rule 10: distinct configurations must be distinct mappings (order-independent) ---------------
def test_rule10_duplicate_configuration_mappings_are_rejected():
    layout = _csr_layout("A")
    cfg_a = SparseConfiguration(arrays={"A": "csr"})
    cfg_b = SparseConfiguration(arrays={"A": "csr"})
    with pytest.raises(SparseConfigError, match="'second' and 'first' are"):
        validate_sparse_config({"A": layout}, {"first": cfg_a, "second": cfg_b}, {}, [])


def test_rule10_duplicate_detection_ignores_dict_key_insertion_order():
    # Same {array: format} content, keys inserted in a different order -- still a duplicate,
    # because the fingerprint is a frozenset of items, not the insertion-ordered dict. Uses a
    # second array pinned to 'dense' (not another sparse format) so rule 7 does not fire first.
    layouts = {"A": _csr_layout("A"), "C": _dense_layout("C")}
    cfg_a = SparseConfiguration(arrays={"A": "csr", "C": "dense"})
    cfg_b = SparseConfiguration(arrays={"C": "dense", "A": "csr"})
    with pytest.raises(SparseConfigError, match="are identical"):
        validate_sparse_config(layouts, {"first": cfg_a, "second": cfg_b}, {}, [])


def test_rule10_different_configurations_are_not_flagged_as_duplicates():
    layout = _csr_layout("A")
    dense = SparseLayoutVariant(format="dense", buffers=(_buf("data", "A_data"), ))
    layout.variants["dense"] = dense
    cfg_a = SparseConfiguration(arrays={"A": "csr"})
    cfg_b = SparseConfiguration(arrays={"A": "dense"})
    validate_sparse_config({"A": layout}, {"sparse": cfg_a, "dflt": cfg_b}, {}, [])


# --- rule 11: physical buffer names follow <logical>_<role> ---------------------------------------
def test_rule11_buffer_name_not_matching_logical_role_convention_is_rejected():
    variant = SparseLayoutVariant(
        format="csr",
        buffers=(
            _buf("indptr", "A_ptr", "int32"),  # should be 'A_indptr'
            _buf("indices", "A_indices", "int32"),
            _buf("data", "A_data", "float64"),
        ))
    layout = SparseLayout(logical_shape=("NI", ), default_dtype="float64", variants={"csr": variant})
    with pytest.raises(SparseConfigError, match=r"expected 'A_indptr'"):
        validate_sparse_config({"A": layout}, {}, {}, [])


# --- ordering guarantee: rules fire in ascending order, first violation wins ----------------------
def test_first_violation_wins_when_multiple_rules_are_broken():
    # Both rule 1 (bad format key) and rule 11 (bad buffer name) are broken here; rule 1 must fire.
    variant = SparseLayoutVariant(format="csr",
                                  buffers=(_buf("indptr", "wrong_name",
                                                "int32"), _buf("indices", "A_indices",
                                                               "int32"), _buf("data", "A_data", "float64")))
    layout = SparseLayout(logical_shape=("NI", ), default_dtype="float64", variants={"nonsense": variant})
    with pytest.raises(SparseConfigError, match="unsupported format"):
        validate_sparse_config({"A": layout}, {}, {}, [])


def test_error_message_includes_the_source_label():
    layout = SparseLayout(logical_shape=("NI", ),
                          default_dtype="float64",
                          variants={"nonsense": _csr_layout("A").variants["csr"]})
    with pytest.raises(SparseConfigError, match=r"^my_bench\.yaml: "):
        validate_sparse_config({"A": layout}, {}, {}, [], source="my_bench.yaml")


def test_layout_of_wrong_type_is_rejected():
    with pytest.raises(SparseConfigError, match="expected SparseLayout, got str"):
        validate_sparse_config({"A": "not_a_layout"}, {}, {}, [])
