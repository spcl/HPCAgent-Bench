"""Python -> Fortran 2008 emitter, reusing NumpyToC's IR + frontend + lowering."""

from __future__ import annotations
from numpyto_fortran.emit import emit_fortran

__all__ = ["emit_fortran"]
