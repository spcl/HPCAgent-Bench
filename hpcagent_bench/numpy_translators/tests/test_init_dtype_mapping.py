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

import pytest

from _bench_yaml import bench_info_for

from numpyto_common.frontend import _dtypes_from_initialize, declared_dtypes, parse_kernel


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


# --------------------------------------------------------------------------- #
# The DECLARED half: a manifest states an array's element type on its           #
# ``init.arrays`` entry, and the reader must pick it up from THAT spelling.     #
# Reading only ``init.dtypes`` (the retired one, which now carries symbols)     #
# silently defaulted every declared array to float64: complex buffers lost      #
# their imaginary part and int index arrays emitted as doubles.                 #
# --------------------------------------------------------------------------- #


def test_declared_dtypes_reads_the_arrays_entry():
    init = {
        "arrays": {
            "ip": {
                "shape": "(N,)",
                "dtype": "int32"
            },
            "z": {
                "shape": "(N,)",
                "dtype": "complex128"
            },
            "a": "(N,)",  # shorthand: shape only, no dtype to report
        }
    }
    assert declared_dtypes(init) == {"ip": "int32", "z": "complex128"}


def test_declared_dtypes_still_accepts_the_legacy_block():
    # A bench_info JSON on disk may predate the move, and ``init.dtypes`` is also where a
    # non-array name (a symbol / plain scalar) is typed -- both must come through.
    init = {"arrays": {"a": {"shape": "(N,)", "dtype": "int32"}}, "dtypes": {"b": "float32", "n_iter": "int64"}}
    assert declared_dtypes(init) == {"a": "int32", "b": "float32", "n_iter": "int64"}


def test_declared_dtypes_prefers_the_arrays_entry_over_the_legacy_block():
    init = {"arrays": {"a": {"shape": "(N,)", "dtype": "complex128"}}, "dtypes": {"a": "float64"}}
    assert declared_dtypes(init) == {"a": "complex128"}, "the current spelling wins"


@pytest.mark.parametrize("short,array,dtype", [
    ("tsvc_2_s4114", "ip", "int32"),
    ("fft_1d", "x", "complex128"),
])
def test_declared_array_dtype_reaches_the_ir(short, array, dtype):
    """The whole seam, on real manifests: manifest -> emit_bridge export -> parse_kernel.

    This is the assertion that was missing when the export moved to ``init.arrays``: each of
    these arrays came back out of the frontend as the float default, which emits a complex
    buffer as real (dropping the imaginary part) and an index array as a double."""
    with bench_info_for(short) as (_, numpy_py, bi):
        kir = parse_kernel(numpy_py, bi)
    assert {a.name: a.dtype for a in kir.arrays}[array] == dtype
