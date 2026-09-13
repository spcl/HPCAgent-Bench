"""An emitter that prepends to the numpy reference must keep the reference's future import first.

A reference module opens with a docstring and ``from __future__ import annotations``. The jax, numba
and cupy emitters put their own header and imports above the source they transform, and a future
import anywhere but the top is a SyntaxError on import. ``ast.parse`` does not enforce that rule, so
only ``compile`` sees it.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from numpyto_cupy.emit import emit_cupy
from numpyto_jax.core import emit_jax
from numpyto_numba.emit import emit_numba

REFERENCE = '''"""A kernel as the corpus writes one."""

from __future__ import annotations

import numpy as np


def kernel(a, out):
    out[:] = np.sqrt(a) + 1.0
'''

EMITTERS: dict[str, Callable[[str], str]] = {
    "jax": lambda source: emit_jax(source, "kernel"),
    "numba": emit_numba,
    "cupy": emit_cupy,
}


@pytest.mark.parametrize("backend", sorted(EMITTERS))
def test_an_emitted_module_with_a_future_import_compiles(backend: str) -> None:
    emitted = EMITTERS[backend](REFERENCE)
    assert "from __future__ import annotations" in emitted, emitted
    compile(emitted, f"<{backend}>", "exec")
