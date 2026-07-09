"""Compatibility shim: ``lowering`` moved to :mod:`numpyto_common.lowering` (Phase 1 of the
NumpyToX unified-core migration). Re-exports the module namespace (public and private)
so existing ``numpyto_c.lowering`` importers keep working.
"""
from numpyto_common import lowering as _src
globals().update({k: v for k, v in vars(_src).items() if not k.startswith("__")})
del _src
