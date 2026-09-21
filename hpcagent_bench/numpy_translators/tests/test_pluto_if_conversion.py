"""A value-dependent ``if`` inside a loop reaches the pluto scop as predicated assignments.

Before, the emitter left any loop holding ``if (a[i] > x)`` outside every ``#pragma scop``, so Pluto
never saw an argmax, a conditional sum or TSVC s2710 and the column recorded "the translator emitted
no #pragma scop". ``numpyto_c.pluto_predicate`` rewrites such an ``if`` into ``t = c ? e : t``: in
place when no test reads what the branches write, and through a flag set once where the ``if`` stood
otherwise. The semantic tests execute the rewritten Python against the original, so the ordering
argument is checked by running it, not by reading the emitted text.
"""

import ast
import json
import pathlib
import re
import tempfile
import textwrap
from typing import Callable

import numpy as np
import pytest

from _op_oracle import _bench_info
from numpyto_c.emit import emit_pluto
from numpyto_c.pluto_predicate import FLAG_PREFIX, if_convert
from numpyto_common.frontend import parse_kernel
from numpyto_common.lowering import lower


def pluto_c(src: str, fn: str, inputs: list[str], outputs: list[str], shapes: dict[str, str]) -> str:
    """The pluto translation unit of the kernel ``src``, sized by one symbol ``N``."""
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "k_numpy.py").write_text(src)
    (d / "bi.json").write_text(json.dumps(_bench_info(fn, inputs, outputs, shapes, {"N": 64})))
    return emit_pluto(lower(parse_kernel(d / "k_numpy.py", d / "bi.json")), fn_name=fn)


def regions(text: str) -> list[str]:
    return re.findall(r"#pragma scop(.*?)#pragma endscop", text, re.S)


ARGMAX = (
    "import numpy as np\n"
    "def argmax(a, out_value, out_index, N):\n"
    "    x = a[0]\n"
    "    idx = 0\n"
    "    for i in range(1, N):\n"
    "        if a[i] > x:\n"
    "            x = a[i]\n"
    "            idx = i\n"
    "    out_value[0] = x\n"
    "    out_index[0] = idx\n"
)


def test_an_argmax_loop_is_inside_a_scop_with_its_test_taken_once_into_a_flag() -> None:
    """The test reads ``x`` and the branch writes it, so the test is evaluated once into a flag that
    both assignments read; the flag is declared outside the region, where locals belong."""
    text = pluto_c(
        ARGMAX, "argmax", ["a"], ["out_value", "out_index"], {"a": "(N,)", "out_value": "(1,)", "out_index": "(1,)"}
    )
    body = "".join(regions(text))
    assert "for (" in body and "if (" not in body
    flag = f"{FLAG_PREFIX}0"
    assert re.search(rf"{flag} = \(\(a\[i\] > x\) \? 1 : 0\);", body)
    assert re.search(rf"x = \({flag} \? a\[i\] : x\);", body)
    assert re.search(rf"idx = \({flag} \? i : idx\);", body)
    assert f"int64_t {flag};" in text.split("#pragma scop")[0]


def test_a_conditional_sum_is_guarded_in_place_without_a_flag() -> None:
    """Nothing the test reads is written, so the guard stays in the statement and no scalar is added
    that would serialize the loop."""
    text = pluto_c(
        "import numpy as np\n"
        "def csum(a, b, N):\n"
        "    s = 0.0\n"
        "    for i in range(N):\n"
        "        if a[i] > 0.0:\n"
        "            s = s + a[i]\n"
        "    b[0] = s\n",
        "csum",
        ["a"],
        ["b"],
        {"a": "(N,)", "b": "(1,)"},
    )
    body = "".join(regions(text))
    assert re.search(r"s = \(\(a\[i\] > 0\.0\) \? \(s \+ a\[i\]\) : s\);", body)
    assert FLAG_PREFIX not in text


def test_a_loop_with_an_early_exit_stays_outside_every_scop() -> None:
    """A ``break`` makes the trip count depend on data; no predication expresses that, so the loop is
    left as written, outside the regions."""
    text = pluto_c(
        "import numpy as np\n"
        "def first(a, out, N):\n"
        "    out[0] = -1\n"
        "    for i in range(N):\n"
        "        if a[i] > 0.5:\n"
        "            out[0] = i\n"
        "            break\n",
        "first",
        ["a"],
        ["out"],
        {"a": "(N,)", "out": "(1,)"},
    )
    assert all("for (" not in body for body in regions(text))
    assert "break;" in text


def test_an_affine_if_stays_an_if() -> None:
    """A test on the loop index is part of the iteration domain, which Pluto models directly."""
    text = pluto_c(
        "import numpy as np\n"
        "def half(a, b, N):\n"
        "    for i in range(N):\n"
        "        if i < 8:\n"
        "            b[i] = a[i] * 2.0\n",
        "half",
        ["a"],
        ["b"],
        {"a": "(N,)", "b": "(N,)"},
    )
    body = "".join(regions(text))
    assert "if (" in body and "?" not in body


def subscripted(test: ast.expr) -> bool:
    """The test the emitter applies, reduced to what these fixtures need: it reads an element."""
    return any(isinstance(node, ast.Subscript) for node in ast.walk(test))


def run_both(src: str, name: str, args: Callable[[np.random.Generator], tuple]) -> None:
    """Execute ``src`` and its if-converted form on the same random inputs; the outputs must agree."""
    original: dict = {}
    exec(compile(src, "original", "exec"), original)
    tree = ast.parse(src)
    if_convert(tree, subscripted)
    converted: dict = {}
    exec(compile(tree, "converted", "exec"), converted)
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)]
    assert loops and not any(isinstance(node, ast.If) for loop in loops for node in ast.walk(loop))
    rng = np.random.default_rng(0)
    for _ in range(50):
        a_args = args(rng)
        b_args = tuple(x.copy() if isinstance(x, np.ndarray) else x for x in a_args)
        original[name](*a_args)
        converted[name](*b_args)
        for x, y in zip(a_args, b_args):
            if isinstance(x, np.ndarray):
                np.testing.assert_array_equal(x, y)


def test_the_flagged_form_computes_what_the_argmax_computes() -> None:
    src = textwrap.dedent("""
        def argmax(a, out, n):
            x = a[0]
            idx = 0
            for i in range(1, n):
                if a[i] > x:
                    x = a[i]
                    idx = i
            out[0] = x
            out[1] = idx
    """)
    run_both(src, "argmax", lambda rng: (rng.integers(0, 5, 32).astype(float), np.zeros(2), 32))


def test_the_flagged_form_computes_what_s2710_computes() -> None:
    """Both branches write an input of the outer test and nest a further test: the case where
    re-evaluating any test after a branch assignment would change the result."""
    src = textwrap.dedent("""
        def s2710(a, b, c, d, e, x, n):
            for i in range(n):
                if a[i] > b[i]:
                    a[i] = a[i] + b[i] * d[i]
                    if n > 10:
                        c[i] = c[i] + d[i] * d[i]
                    else:
                        c[i] = d[i] * e[i] + 1.0
                else:
                    b[i] = a[i] + e[i] * e[i]
                    if x[0] > 0.0:
                        c[i] = a[i] + d[i] * d[i]
                    else:
                        c[i] = c[i] + e[i] * e[i]
    """)
    run_both(src, "s2710", lambda rng: (*(rng.standard_normal(16) for _ in range(6)), 16))


def test_the_inline_form_computes_what_a_conditional_sum_computes() -> None:
    src = textwrap.dedent("""
        def csum(a, out, n):
            s = 0.0
            for i in range(n):
                if a[i] > 0.0:
                    s += a[i]
                else:
                    s -= 0.5 * a[i]
            out[0] = s
    """)
    run_both(src, "csum", lambda rng: (rng.standard_normal(32), np.zeros(1), 32))


@pytest.mark.parametrize("statement", ["break", "print(i)"])
def test_an_if_holding_anything_but_assignments_is_left_alone(statement: str) -> None:
    tree = ast.parse(f"def f(a, n):\n    for i in range(n):\n        if a[i] > 0:\n            {statement}\n")
    assert if_convert(tree, subscripted) == []
    assert any(isinstance(node, ast.If) for node in ast.walk(tree))
