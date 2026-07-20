"""Directive #4 sanitize pass: strip comments/docstrings (+ optional mangle)
for the Python-emitting backends before container handoff. Pure-logic unit
test; imports resolve via PYTHONPATH."""
import ast

from numpyto_common.sanitize import sanitize


def test_strips_hash_comments():
    out = sanitize("x = 1  # inline note\n# standalone note\ny = x + 2\n")
    assert "#" not in out
    assert "x = 1" in out and "y = x + 2" in out


def test_strips_docstrings_by_default():
    src = '"""module doc"""\ndef f(a):\n    """fn doc"""\n    return a + 1\n'
    out = sanitize(src)
    assert "doc" not in out
    assert "def f(a):" in out and "return a + 1" in out


def test_keeps_docstrings_when_asked():
    out = sanitize('"""keep me"""\nx = 1\n', strip_docstrings=False)
    assert "keep me" in out


def test_mangle_renames_bound_names_not_attribute_members():
    src = "import cupy as cp\ndef k(a):\n    t = cp.zeros(3)\n    return t + a\n"
    out = sanitize(src, name_registry={"t": "_v0", "a": "_a0", "k": "_k"})
    assert "_v0" in out and "_a0" in out and "def _k(" in out
    assert "cp.zeros" in out  # attribute member NOT mangled
    assert "\n    t " not in out  # original local gone


def test_output_is_valid_python():
    ast.parse(sanitize("def f(x):\n    return x * 2\n"))


def test_docstring_only_function_stays_valid():
    # stripping the sole docstring leaves a `pass`, not an empty body.
    out = sanitize('def f():\n    """only a doc"""\n')
    ast.parse(out)
    assert "pass" in out
