# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Skip guard for optional native-backed dependencies (tvm, triton, cupy, jax, numba, dace, ...).

``pytest.importorskip`` only converts ``ImportError`` into a skip. A pip wheel whose compiled
runtime fails to dlopen raises ``OSError`` instead, which ``pytest.importorskip`` does not catch --
so a broken/ABI-mismatched wheel FAILS the test instead of skipping it. That is exactly what broke
CI: the apache-tvm wheel on PyPI has an ABI break against its own runtime .so, so ``import tvm``
raised

    OSError: .../libtvm_runtime.so: undefined symbol: _ZN3tvm3ffi4json9StringifyE...

from ctypes (tvm ``ctypes.CDLL``-loads its runtime itself, outside the normal import machinery),
not ImportError.
"""
from __future__ import annotations

import importlib

import pytest


def import_or_skip(modname: str):
    """Import ``modname``, skipping the test (not failing it) when the wheel itself won't load.

    Catches exactly ``ImportError`` (module/submodule not found -- pytest.importorskip's own case)
    and ``OSError`` (a compiled shared library failed to dlopen -- an ABI-broken wheel). Nothing
    broader: the try wraps only the import of ``modname``, so any other exception is a real bug
    surfacing at that module's import time and must fail the test, not be read as "not installed".
    """
    try:
        return importlib.import_module(modname)
    except (ImportError, OSError) as exc:
        pytest.skip(f"{modname} import failed: {type(exc).__name__}: {exc}")
