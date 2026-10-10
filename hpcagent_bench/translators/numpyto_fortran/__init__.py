"""Python -> Fortran 2008 emitter, reusing NumpyToC's IR + frontend + lowering."""

from hpcagent_bench.translators.numpyto_fortran.emit import emit_fortran

__all__ = ["emit_fortran"]
