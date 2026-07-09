"""Compatibility shim: ``sparse_emit`` moved to :mod:`numpyto_common.sparse_emit` (Phase 1 of the
NumpyToX unified-core migration). Re-exports the module namespace (public and private)
so existing ``numpyto_c.sparse_emit`` importers keep working.
"""
from numpyto_common import sparse_emit as _src
globals().update({k: v for k, v in vars(_src).items() if not k.startswith("__")})
del _src
