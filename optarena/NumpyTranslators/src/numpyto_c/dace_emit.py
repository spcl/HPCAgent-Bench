"""Emit a DaCe ``@dc.program`` from the canonical numpy reference.

The Foundation-track numpy kernels are already dace-shaped (they were
ported FROM ``@dace.program`` bodies), and they now carry every size
symbol as an explicit scalar argument plus a per-array shape. That is
exactly the metadata a numpy->dace converter needs:

* size SYMBOLS (``LEN_1D``, ``K``, ...) become module-level
  ``dc.symbol(...)`` declarations and are DROPPED from the program
  signature (dace passes them implicitly via the array shapes);
* ARRAY arguments are typed ``<dctype>[shape]`` -- ``dc_float`` for the
  precision-driven floats, ``dc.int32`` / ``dc.int64`` for index arrays
  (the dtype the original ``dace.int*`` annotation carried, recovered
  from bench_info ``init.dtypes``);
* kernel SCALARS (``alpha``, s174's ``M``, ...) stay typed scalar args.

The body is the numpy reference verbatim. Reuses :func:`parse_kernel`
so the array/symbol/scalar classification is the single source of
truth shared with the C/Fortran emitters.
"""

import ast
from typing import List

from numpyto_c.ir import KernelIR


#: numpy dtype tag -> dace type expression. Floats route through the
#: precision-driven ``dc_float`` / ``dc_complex_float`` globals the dace
#: framework rebinds per run; integers carry a fixed width.
_DTYPE_TO_DACE = {
    "float64": "dc_float",
    "float32": "dc_float",
    "complex128": "dc_complex_float",
    "complex64": "dc_complex_float",
    "int64": "dc.int64",
    "int32": "dc.int32",
    "int16": "dc.int16",
    "int8": "dc.int8",
    "uint64": "dc.uint64",
    "uint32": "dc.uint32",
    "uint16": "dc.uint16",
    "uint8": "dc.uint8",
    "int": "dc.int64",
    "bool": "dc.bool",
}


def _dace_dtype(tag: str) -> str:
    return _DTYPE_TO_DACE.get(tag, "dc_float")


def _array_annotation(arr) -> str:
    """``a`` of shape ``(LEN_1D,)`` float64 -> ``dc_float[LEN_1D]``."""
    shape = ", ".join(str(s) for s in arr.shape) if arr.shape else "1"
    return f"{_dace_dtype(arr.dtype)}[{shape}]"


def emit_dace(kir: KernelIR, fn_name: str | None = None) -> str:
    """Return the source of a ``<short>_dace.py`` module for ``kir``."""
    name = fn_name or kir.kernel_name
    arrays = {a.name: a for a in kir.arrays}
    scalars = {s.name: s for s in kir.scalars}
    symbol_names = [s.name for s in kir.symbols]

    # Program signature: arrays + scalars in their original input_args
    # order; symbols are module-level, not parameters.
    params: List[str] = []
    for arg in kir.input_args:
        if arg in arrays:
            params.append(f"{arg}: {_array_annotation(arrays[arg])}")
        elif arg in scalars:
            params.append(f"{arg}: {_dace_dtype(scalars[arg].dtype)}")
        # symbols: skip (declared at module scope below)

    needs_complex = any(
        _dace_dtype(a.dtype) == "dc_complex_float" for a in kir.arrays
    ) or any(_dace_dtype(s.dtype) == "dc_complex_float" for s in kir.scalars)

    out: List[str] = []
    out.append('"""DaCe program auto-generated from the numpy reference '
               'by numpyto_c.dace_emit."""')
    out.append("import numpy as np")
    out.append("import dace as dc")
    imp = "dc_float, dc_complex_float" if needs_complex else "dc_float"
    out.append(f"from optarena.infrastructure.dace_framework import {imp}")
    out.append("from math import sin, cos, log, exp, pow")
    out.append("")
    if symbol_names:
        names = ", ".join(symbol_names)
        srcs = ", ".join(f"'{s}'" for s in symbol_names)
        if len(symbol_names) == 1:
            out.append(f"{names} = dc.symbol({srcs}, dtype=dc.int64)")
        else:
            out.append(f"{names} = (dc.symbol(s, dtype=dc.int64) "
                       f"for s in ({srcs}))")
        out.append("")
    out.append("")
    out.append("@dc.program")
    out.append(f"def {name}({', '.join(params)}):")

    # Body: every statement of the source function except its docstring
    # (a string-literal Expr first statement), re-indented under the def.
    body = list(kir.tree.body)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    if not body:
        out.append("    pass")
    else:
        for stmt in body:
            for line in ast.unparse(stmt).splitlines():
                out.append("    " + line)
    return "\n".join(out) + "\n"
