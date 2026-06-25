"""Tests for optarena.sanitize (Workstream J).

Covers comment-stripping + name-mangling for a small C snippet and a small
Python snippet:

* all comments removed;
* mapped names rewritten everywhere and consistently;
* mangled C still compiles (``gcc -fsyntax-only``; skipped if gcc absent);
* a non-mapped keyword / identifier is left untouched.

tree-sitter-only assertions are guarded behind ``find_spec`` so the suite
passes on the stdlib fallback (tree-sitter is not installed in CI here).
"""
import importlib.util
import shutil
import subprocess
import tempfile

import pytest

from optarena.sanitize import build_name_map, mangle, strip_comments
from optarena.sanitize.comments import tree_sitter_available

TREE_SITTER = tree_sitter_available()


# --------------------------------------------------------------------------- #
# Sample snippets
# --------------------------------------------------------------------------- #

C_SRC = """\
// leading comment mentioning relu and kernel
#include <stdint.h>

/* block comment: do not mangle the word relu in here */
double helper(double x) {
    return x * 2.0;  // inline comment with relu word
}

void relu(double *a, int64_t n) {
    const char *label = "relu stays in this string // not a comment";
    for (int i = 0; i < n; i++) {
        a[i] = helper(a[i]);  /* call helper */
    }
}
"""

PY_SRC = '''\
# top comment about relu
def helper(x):
    """docstring mentioning relu (a string, not a comment)"""
    return x * 2  # double it

def relu(a):
    note = "relu in a string # not a comment"
    return [helper(v) for v in a]
'''


# --------------------------------------------------------------------------- #
# Comment stripping
# --------------------------------------------------------------------------- #

def test_strip_comments_c_removes_all_comments():
    out = strip_comments(C_SRC, "c")
    # (a bare ``//`` survives inside the string literal asserted below; we check
    # that comment *content* is gone rather than the comment delimiters.)
    assert "/*" not in out
    assert "*/" not in out
    assert "leading comment" not in out
    assert "block comment" not in out
    assert "inline comment" not in out
    assert "call helper" not in out
    # The string literal containing comment-like text must survive verbatim.
    assert '"relu stays in this string // not a comment"' in out
    # Code is intact.
    assert "void relu(double *a, int64_t n)" in out
    assert "double helper(double x)" in out


def test_strip_comments_python_removes_comments_and_keeps_strings():
    out = strip_comments(PY_SRC, "python")
    assert "# top comment" not in out
    assert "# double it" not in out
    # The "# not a comment" text lives inside a string literal -> must survive.
    assert "relu in a string # not a comment" in out
    # Code survives.
    assert "def relu(a):" in out
    assert "def helper(x):" in out


# --------------------------------------------------------------------------- #
# Name map construction
# --------------------------------------------------------------------------- #

def test_build_name_map_ordering_and_precedence():
    nm = build_name_map(["relu", "conv"], ["helper", "pad", "relu"])
    assert nm["relu"] == "kernel1"
    assert nm["conv"] == "kernel2"
    assert nm["helper"] == "f1"
    assert nm["pad"] == "f2"
    # "relu" already an entry kernel -> not re-numbered as an f-name.
    assert nm["relu"] == "kernel1"


def test_build_name_map_dedups():
    nm = build_name_map(["relu", "relu"], ["helper", "helper"])
    assert nm["relu"] == "kernel1"
    assert nm["helper"] == "f1"
    assert len(nm) == 2


# --------------------------------------------------------------------------- #
# Mangling
# --------------------------------------------------------------------------- #

def test_mangle_c_consistent_and_boundary_safe():
    name_map = build_name_map(["relu"], ["helper"])
    stripped = strip_comments(C_SRC, "c")
    out = mangle(stripped, "c", name_map)

    # Mapped names rewritten everywhere they appear as identifiers.
    assert "void kernel1(double *a, int64_t n)" in out
    assert "double f1(double x)" in out
    assert "a[i] = f1(a[i]);" in out
    # Original identifiers gone from code.
    assert "relu(" not in out
    assert "helper(" not in out
    # Non-mapped keyword / identifier untouched.
    assert "double" in out
    assert "for (int i = 0; i < n; i++)" in out
    assert "int64_t" in out


def test_mangle_does_not_touch_strings_or_substrings():
    name_map = build_name_map(["relu"], ["helper"])
    stripped = strip_comments(C_SRC, "c")
    out = mangle(stripped, "c", name_map)
    # The word "relu" inside the surviving string literal must NOT be mangled.
    assert "relu stays in this string" in out


def test_mangle_substring_not_corrupted():
    # "relu" must not be rewritten inside "relufoo" or "prerelu".
    src = "int relu; int relufoo; int prerelu;"
    name_map = build_name_map(["relu"], [])
    out = mangle(src, "c", name_map)
    assert "int kernel1;" in out
    assert "relufoo" in out
    assert "prerelu" in out
    assert "kernel1foo" not in out
    assert "prekernel1" not in out


def test_mangle_python_consistent():
    name_map = build_name_map(["relu"], ["helper"])
    stripped = strip_comments(PY_SRC, "python")
    out = mangle(stripped, "python", name_map)
    assert "def kernel1(a):" in out
    assert "def f1(x):" in out
    assert "f1(v) for v in a" in out
    # The "relu" inside the surviving string literal stays.
    assert "relu in a string" in out
    # def / for / return keywords untouched.
    assert "return" in out
    assert "for v in a" in out


# --------------------------------------------------------------------------- #
# Compilation gate
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not available")
def test_mangled_c_still_compiles():
    name_map = build_name_map(["relu"], ["helper"])
    stripped = strip_comments(C_SRC, "c")
    out = mangle(stripped, "c", name_map)
    with tempfile.NamedTemporaryFile("w", suffix=".c", delete=True) as fh:
        fh.write(out)
        fh.flush()
        proc = subprocess.run(
            ["gcc", "-fsyntax-only", fh.name],
            capture_output=True, text=True)
    assert proc.returncode == 0, f"gcc rejected mangled C:\n{proc.stderr}"


# --------------------------------------------------------------------------- #
# tree-sitter parity (only when installed)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    importlib.util.find_spec("tree_sitter_language_pack") is None,
    reason="tree-sitter (tree-sitter-language-pack) not installed")
def test_tree_sitter_path_used_when_available():
    assert TREE_SITTER is True
    # Comment strip + mangle still satisfy the core contract on the ts path.
    out = mangle(strip_comments(C_SRC, "c"), "c", build_name_map(["relu"], ["helper"]))
    # Every comment is gone (line, block, inline).
    assert "leading comment" not in out
    assert "block comment" not in out
    assert "inline comment" not in out
    assert "call helper" not in out
    # The entry kernel is renamed.
    assert "void kernel1(" in out
    # The string literal is untouched -- its `//` and the word `relu` survive
    # (mangle never rewrites inside strings; strip never treats `//` in a string
    # as a comment).
    assert "relu stays in this string // not a comment" in out


def test_unsupported_lang_rejected():
    with pytest.raises(ValueError):
        strip_comments("x", "haskell")
    with pytest.raises(ValueError):
        mangle("x", "haskell", {"a": "b"})
