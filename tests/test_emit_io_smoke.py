# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The emit_io write path and its hand-override clobber guard.

:func:`numpyto_common.emit_io.write_generated` owns the single "if a file with the
canonical name already exists, use it; otherwise emit" rule. The clobber guard is the
override branch: a file present WITHOUT a generation marker on its first line is a
hand-written override and is never overwritten (``write_generated`` returns
``"override"`` and leaves it byte-for-byte untouched); a file carrying the marker is
auto-generated and is refreshed (``"ok"``).

These tests pin that contract -- new write, refresh-of-generated, refuse-to-clobber-
override, the subtle "a marker MENTIONED in a line-1 docstring is NOT a marker",
legacy-marker recognition, the C/Fortran comment leads, missing-path classification,
and parent-dir creation. No toolchain is involved, so they always run; all writes are
confined to ``tmp_path``.
"""
from numpyto_common.emit_io import AUTO_MARKER, is_generated, is_override, write_generated


def test_writes_to_new_path_and_stamps_marker(tmp_path):
    out = tmp_path / "gemm_cupy.py"
    status = write_generated(out, "print(1)\n", source="gemm_numpy.py")
    assert status == "ok"
    assert out.exists()
    first_line = out.read_text().splitlines()[0]
    # the marker sits right after the comment lead, and names its source.
    assert first_line.startswith("# " + AUTO_MARKER)
    assert "gemm_numpy.py" in first_line
    assert "print(1)" in out.read_text()
    assert is_generated(out) and not is_override(out)


def test_regenerate_refreshes_a_generated_file(tmp_path):
    out = tmp_path / "gemm_cupy.py"
    assert write_generated(out, "print(1)\n", source="gemm_numpy.py") == "ok"
    # a marked (generated) file is NOT protected -- a re-run refreshes it in place.
    assert write_generated(out, "print(2)\n", source="gemm_numpy.py") == "ok"
    body = out.read_text()
    assert "print(2)" in body and "print(1)" not in body


def test_hand_override_is_never_clobbered(tmp_path):
    """The clobber guard: a file present with an ordinary (non-marker) first line is a
    hand override -- ``write_generated`` refuses it and leaves the bytes untouched."""
    out = tmp_path / "gemm_numba.py"
    original = "# my own file\nprint('mine')\n"
    out.write_text(original)
    assert is_override(out) and not is_generated(out)

    status = write_generated(out, "print('GENERATED')\n", source="gemm_numpy.py")
    assert status == "override"
    assert out.read_text() == original  # not a single byte rewritten


def test_marker_mention_in_line1_docstring_is_not_generated(tmp_path):
    """A hand file whose line-1 docstring merely NAMES the marker is not a generated
    file (the marker must follow a comment lead), so it is protected as an override --
    exactly the misclassification the guard is written to avoid."""
    out = tmp_path / "gemm_pythran.py"
    original = '"""uses the ' + AUTO_MARKER + ' token"""\ncode = 1\n'
    out.write_text(original)
    assert not is_generated(out)
    assert is_override(out)
    assert write_generated(out, "clobbered = 1\n") == "override"
    assert out.read_text() == original


def test_legacy_marker_is_recognized_as_generated(tmp_path):
    """A file stamped by an earlier generator (legacy marker on line 1) is recognized
    as generated, so migration to the canonical name refreshes it rather than mistaking
    it for a hand override."""
    out = tmp_path / "legacy.py"
    out.write_text("# auto-generated from the numpy reference; do not edit\ncode = 1\n")
    assert is_generated(out) and not is_override(out)
    assert write_generated(out, "code = 2\n") == "ok"
    assert "code = 2" in out.read_text()


def test_c_and_fortran_comment_leads(tmp_path):
    c_out = tmp_path / "kernel.c"
    assert write_generated(c_out, "int main(){ return 0; }\n", line_comment="// ") == "ok"
    assert c_out.read_text().splitlines()[0].startswith("// " + AUTO_MARKER)
    assert is_generated(c_out)
    # a marked C file refreshes, it does not clobber-protect.
    assert write_generated(c_out, "int main(){ return 1; }\n", line_comment="// ") == "ok"

    f_out = tmp_path / "kernel.f90"
    assert write_generated(f_out, "end program\n", line_comment="! ") == "ok"
    assert f_out.read_text().splitlines()[0].startswith("! " + AUTO_MARKER)
    assert is_generated(f_out)


def test_missing_path_is_neither_generated_nor_override(tmp_path):
    missing = tmp_path / "does_not_exist.py"
    assert not is_generated(missing)
    assert not is_override(missing)


def test_parent_directories_are_created(tmp_path):
    nested = tmp_path / "sub" / "deeper" / "gemm_numba.py"
    assert not nested.parent.exists()
    assert write_generated(nested, "x = 1\n") == "ok"
    assert nested.exists() and is_generated(nested)
