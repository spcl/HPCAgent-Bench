"""Python -> Fortran 2008 emitter. Reuses NumpyToC's IR + frontend + lowering.

Top-level surface mirrors :mod:`numpyto_c`:

* :func:`emit_fortran` -- generate one Fortran subroutine.
"""
from numpyto_fortran.emit import emit_fortran

__all__ = ["emit_fortran"]
