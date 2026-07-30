# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pin the pass/fail BEHAVIOR of the two pre-commit YAML gates on crafted fixtures --
``tests/check_yaml_style.py`` (house style, hook id ``hpcagent_bench-yaml-style``) and
``scripts/check_manifest_structure.py`` (manifest schema, hook id
``hpcagent_bench-manifest-structure``, reusing ``hpcagent_bench.spec.BenchSpec``).

``tests/test_yaml_style.py`` already pins that the CURRENT tree conforms; this file pins
the checkers THEMSELVES the way ``tests/test_header_hook.py`` pins ``check_headers.py``:
a deliberately good fixture passes, a deliberately bad one is caught with a clear message.
"""
import importlib.util
from pathlib import Path
from typing import Any, List, Optional

import pytest

from tests.check_yaml_style import violations as yaml_style_violations

REPO = Path(__file__).resolve().parent.parent


def load_check_manifest_structure() -> Any:
    """Import ``scripts/check_manifest_structure.py`` as a module (it is not an installed
    package, same technique ``test_header_hook.py`` uses for ``check_headers.py``)."""
    spec = importlib.util.spec_from_file_location("check_manifest_structure",
                                                  REPO / "scripts" / "check_manifest_structure.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- hpcagent_bench-yaml-style (tests/check_yaml_style.py) ------------------------------


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


# --- hpcagent_bench-manifest-structure (scripts/check_manifest_structure.py) ------------

GOOD_NUMPY = "def kern(a, out):\n    out[0] = a[0]\n    return out\n"

GOOD_MANIFEST = """# test manifest
parameters:
  S:
    N: 4
init:
  arrays:
    a: (N,)
    out: (1,)
output_args:
- out
"""


def make_kernel(module: Any,
                tmp_path: Path,
                monkeypatch: pytest.MonkeyPatch,
                manifest_text: str,
                kernel_name: str,
                numpy_text: str = GOOD_NUMPY) -> Path:
    """A minimal on-disk kernel (manifest + numpy reference) under a fake ``benchmarks/``
    root, with ``hpcagent_bench.paths.BENCHMARKS`` patched to it so ``BenchSpec.from_yaml``'s
    own path-derived lookups (relative_path, func_name, ...) resolve against the fixture
    instead of the real repo tree (same idiom as ``tests/test_prompt_hints.py``)."""
    bench_root = tmp_path / "benchmarks"
    kdir = bench_root / kernel_name
    kdir.mkdir(parents=True)
    (kdir / f"{kernel_name}_numpy.py").write_text(numpy_text)
    manifest_path = kdir / f"{kernel_name}.yaml"
    manifest_path.write_text(manifest_text)
    monkeypatch.setattr(module.paths, "BENCHMARKS", bench_root)
    return manifest_path


def violations_of(module: Any, path: Path) -> Optional[List[str]]:
    return module.violations(str(path))


def test_manifest_structure_passes_a_well_formed_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_check_manifest_structure()
    p = make_kernel(module, tmp_path, monkeypatch, GOOD_MANIFEST, "kern_good")
    assert violations_of(module, p) is None


def test_manifest_structure_catches_an_unknown_top_level_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_check_manifest_structure()
    bad = GOOD_MANIFEST + "not_a_real_key: 1\n"
    p = make_kernel(module, tmp_path, monkeypatch, bad, "kern_badkey")
    probs = violations_of(module, p)
    assert probs is not None and any("not_a_real_key" in msg for msg in probs)


def test_manifest_structure_catches_a_missing_required_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_check_manifest_structure()
    bad = GOOD_MANIFEST.replace("output_args:\n- out\n", "")
    p = make_kernel(module, tmp_path, monkeypatch, bad, "kern_missing")
    probs = violations_of(module, p)
    assert probs is not None and any("output_args" in msg for msg in probs)


def test_manifest_structure_catches_malformed_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_check_manifest_structure()
    p = make_kernel(module, tmp_path, monkeypatch, "parameters: [1, 2\n", "kern_yaml")
    probs = violations_of(module, p)
    assert probs is not None and "parse" in probs[0]
