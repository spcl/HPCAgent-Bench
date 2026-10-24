# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A kernel's ``from __future__`` import stays the first statement of its Python siblings.

The numba emitter keep the kernel body verbatim and put their own docstring and
imports in front of it. A future import is only legal at the top of a module (after one docstring),
so a kernel opening with ``from __future__ import annotations`` produced a numba sibling that did
not import, and one that also carried a docstring broke the sibling.
"""

import ast
from collections.abc import Callable

import pytest

from hpcagent_bench.translators.numpyto_common.emit_io import assemble_python_module
from hpcagent_bench.translators.numpyto_numba.emit import emit_numba

BODY = "import numpy as np\ndef f(a, out):\n    for i in range(a.shape[0]):\n        out[i] = a[i] * 2.0\n"

SOURCES = {
    "future_first": "from __future__ import annotations\n" + BODY,
    "docstring_then_future": '"""A kernel."""\nfrom __future__ import annotations\n' + BODY,
}


EMITTERS: dict[str, Callable[[str], str]] = {"numba": emit_numba}


@pytest.mark.parametrize("source", sorted(SOURCES))
@pytest.mark.parametrize("backend", sorted(EMITTERS))
def test_the_sibling_compiles_with_the_future_import_first(backend: str, source: str) -> None:
    out = EMITTERS[backend](SOURCES[source])
    compile(out, f"f_{backend}.py", "exec")  # SyntaxError when the future import is not first
    stmts = ast.parse(out).body
    assert isinstance(stmts[1], ast.ImportFrom) and stmts[1].module == "__future__", out


def test_assemble_keeps_the_body_verbatim_and_no_future_twice() -> None:
    src = "from __future__ import annotations\n# keep me\nx = 1\n"
    out = assemble_python_module('"""h."""\n', "import y\n", src)
    assert out == '"""h."""\nfrom __future__ import annotations\nimport y\n# keep me\nx = 1\n'
