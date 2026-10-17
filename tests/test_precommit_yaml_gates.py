# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pin the pass/fail behaviour of the YAML house-style pre-commit gate (``tests/check_yaml_style.py``, hook id
``hpcagent_bench-yaml-style``) on crafted fixtures. ``tests/test_yaml_style.py`` pins that the current tree conforms;
a manifest's schema is ``BenchSpec.load`` and ``validate_kernel`` over every kernel (``tests/test_tree_structure.py``).
"""

from pathlib import Path

from tests.check_yaml_style import violations as yaml_style_violations


# hpcagent_bench-yaml-style (tests/check_yaml_style.py)


def test_yaml_style_passes_a_well_formed_file(tmp_path: Path) -> None:
    f = tmp_path / "ok.yaml"
    f.write_text("# a well-formed file\nfoo: 1\nbar:\n  baz: 2\n")
    assert yaml_style_violations(f) == []


def test_yaml_style_catches_a_missing_header(tmp_path: Path) -> None:
    f = tmp_path / "no_header.yaml"
    f.write_text("foo: 1\n")
    probs = yaml_style_violations(f)
    assert any("header" in p for p in probs)


def test_yaml_style_catches_a_tab_and_trailing_whitespace(tmp_path: Path) -> None:
    f = tmp_path / "bad.yaml"
    # The tab sits inside a quoted scalar so the file still PARSES (a tab used for
    # structural indent is a YAML syntax error, which would be caught as "does not
    # parse" instead -- a distinct violation this test is not pinning).
    f.write_text('# header\nfoo: 1  \nbar: "a\tb"\n')
    probs = yaml_style_violations(f)
    assert any("tab" in p for p in probs)
    assert any("trailing whitespace" in p for p in probs)
