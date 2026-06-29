"""Regression tests for ``_dtypes_from_initialize`` return-target mapping.

The cloudsc flux-accumulation miscompile (native c/cpp/fortran emitted a
spurious ``(int64_t)`` cast on float flux arrays, truncating their tiny values
to 0) was caused by an UNGATED positional ``zip`` between the kernel's array
args and the ``initialize`` return tuple. When those two lists differ in length
or order, the zip mis-assigns one array's dtype to an unrelated array. cloudsc's
``initialize`` returns 58 values while the kernel takes 53 array args in a
different order, so ``ktype``/``ldcum`` (int32) leaked onto ``pfsqrf`` /
``pfsqltur`` / ``pvfi`` (float64).

The fix gates the positional fallback on EQUAL lengths (the only case where the
correspondence is provably 1:1); the by-name ``init.dtypes`` block stays the
authoritative source. These tests pin both directions of that gate. A full
emit+compile+run numerical check of the fix lives in
``test_translator_feature_fixes::test_feature_kernels_e2e[cloudsc]``.
"""
import pathlib
import textwrap

from numpyto_common.frontend import _dtypes_from_initialize


def _write_harness(tmp_path: pathlib.Path, body: str, short: str = "k") -> pathlib.Path:
    """Write ``<short>.py`` (the harness) and return the sibling
    ``<short>_numpy.py`` path the parser derives the harness location from."""
    (tmp_path / f"{short}.py").write_text(textwrap.dedent(body))
    numpy_py = tmp_path / f"{short}_numpy.py"
    numpy_py.write_text("def k():\n    pass\n")
    return numpy_py


def test_mismatched_length_skips_positional_mapping(tmp_path):
    # 3 returns vs 2 array args, different order: the unsound positional zip
    # would put ``flags``'s int32 onto ``flux`` (float). The length gate must
    # skip it -- only the by-name int32 of ``flags`` survives.
    numpy_py = _write_harness(
        tmp_path, """
        import numpy as np
        def initialize():
            flux = np.zeros((4,))
            temp = np.zeros((4,))
            flags = np.zeros((4,)).astype(np.int32)
            return temp, flux, flags
        """)
    info = {"init": {"func_name": "initialize"}, "input_args": ["flux", "flags"], "array_args": ["flux", "flags"]}
    dtypes = _dtypes_from_initialize(numpy_py, info)
    assert dtypes.get("flags") == "int32"  # by-name: correct
    assert "flux" not in dtypes  # not corrupted by the misaligned zip


def test_equal_length_positional_mapping_renamed(tmp_path):
    # Equal length AND order: a kernel that RENAMES the harness locals (idx_in
    # <- idx) inherits the int32 via the gated positional fallback.
    numpy_py = _write_harness(
        tmp_path, """
        import numpy as np
        def initialize():
            data = np.zeros((4,))
            idx = np.zeros((4,)).astype(np.int32)
            return data, idx
        """)
    info = {
        "init": {
            "func_name": "initialize"
        },
        "input_args": ["data_in", "idx_in"],
        "array_args": ["data_in", "idx_in"]
    }
    dtypes = _dtypes_from_initialize(numpy_py, info)
    assert dtypes.get("idx_in") == "int32"  # positional rename mapping applied


def test_by_name_dtype_is_recorded(tmp_path):
    # The harness local name == kernel arg name: the dtype is recorded under that
    # name regardless of any positional consideration.
    numpy_py = _write_harness(
        tmp_path, """
        import numpy as np
        def initialize():
            mask = np.zeros((4,)).astype(np.int32)
            val = np.zeros((4,))
            return val, mask
        """)
    info = {"init": {"func_name": "initialize"}, "input_args": ["val", "mask"], "array_args": ["val", "mask"]}
    dtypes = _dtypes_from_initialize(numpy_py, info)
    assert dtypes.get("mask") == "int32"
    assert "val" not in dtypes  # float default, never recorded
