"""Compatibility shim: ``frontend`` moved to :mod:`numpyto_common.frontend` (Phase 1 of the
NumpyToX unified-core migration). Re-exports the module namespace (public and private)
so existing ``numpyto_c.frontend`` importers keep working.
"""
from numpyto_common import frontend as _src
globals().update({k: v for k, v in vars(_src).items() if not k.startswith("__")})
del _src
