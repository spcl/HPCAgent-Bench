# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The large-file guard separates a binary blob from source, and holds each to its own limit.

Written when a thirteen-line fix to ``lowering.py`` failed the commit: the module was already at
499.9 KiB and one shared 500 KiB ceiling made a routine source edit indistinguishable from staging
a dataset. The two limits only mean something while the text/binary split keeps working, and that
split is a content sniff -- nothing about it is visible in a diff, so it is pinned here.
"""
import importlib.util
import pathlib
import sys

import pytest

from hpcagent_bench import paths

SPEC = importlib.util.spec_from_file_location("check_large_files", paths.ROOT / "scripts" / "check_large_files.py")
check_large_files = importlib.util.module_from_spec(SPEC)
sys.modules["check_large_files"] = check_large_files
SPEC.loader.exec_module(check_large_files)

KB = check_large_files.BYTES_PER_KB


def write(tmp_path: pathlib.Path, name: str, payload: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(payload)
    return str(path)


def test_source_over_the_binary_limit_is_allowed_through(tmp_path: pathlib.Path) -> None:
    """The case that motivated the split: 600 KiB of real source is not a stray artifact."""
    big = write(tmp_path, "lowering.py", b"# a line of source\n" * (600 * KB // 19))
    assert check_large_files.main([big]) == 0


def test_source_past_the_text_limit_is_still_caught(tmp_path: pathlib.Path) -> None:
    """The higher ceiling is a ceiling, not an exemption -- a generated .py table still fails."""
    huge = write(tmp_path, "generated_table.py", b"TABLE = [\n" + b"    0,\n" * (1100 * KB // 7))
    assert check_large_files.main([huge]) == 1


def test_a_binary_blob_is_held_to_the_lower_limit(tmp_path: pathlib.Path) -> None:
    """600 KiB that a source file may spend is 100 KiB more than a dump gets."""
    blob = write(tmp_path, "weights.bin", bytes(600 * KB))
    assert check_large_files.main([blob]) == 1
    assert check_large_files.main([blob, "--max-kb", "700"]) == 0


def test_the_sniff_reads_content_not_the_extension(tmp_path: pathlib.Path) -> None:
    """A ``.py`` holding a NUL is a blob wearing a source name, and an extensionless fixture that
    decodes is text. Keying on the suffix would get both backwards."""
    assert not check_large_files.is_text(pathlib.Path(write(tmp_path, "fake.py", b"x = 1\n\0\n")))
    assert check_large_files.is_text(pathlib.Path(write(tmp_path, "LICENSE", "GPL © 2021\n".encode())))


def test_a_multibyte_character_split_by_the_read_boundary_is_not_called_binary(tmp_path: pathlib.Path) -> None:
    """The sniff reads a prefix, so a UTF-8 sequence can straddle the cut. Treating that as binary
    would drop a large translated text file onto the 500 KiB limit for no reason."""
    padded = ("a" * 8191).encode() + "é".encode() + b"more text\n"
    assert check_large_files.is_text(pathlib.Path(write(tmp_path, "notes.md", padded)))


def test_files_within_their_limits_report_nothing(tmp_path: pathlib.Path) -> None:
    small_text = write(tmp_path, "ok.py", b"x = 1\n")
    small_blob = write(tmp_path, "ok.bin", bytes(16))
    assert check_large_files.main([small_text, small_blob]) == 0


def test_a_path_that_is_not_a_regular_file_is_skipped(tmp_path: pathlib.Path) -> None:
    """pre-commit passes deleted paths too; opening one to sniff it would crash the hook."""
    assert check_large_files.main([str(tmp_path / "deleted.py"), str(tmp_path)]) == 0


@pytest.mark.parametrize("name", ["lowering.py"])
def test_the_module_that_motivated_this_fits_under_the_text_limit(name: str) -> None:
    """A guard nobody can satisfy is a guard that gets bypassed. If this fails, the limit is not
    the answer any more -- split the module."""
    path = paths.ROOT / "hpcagent_bench" / "numpy_translators" / "src" / "numpyto_common" / name
    assert check_large_files.main([str(path)]) == 0, f"{name} outgrew the text limit; split it"
