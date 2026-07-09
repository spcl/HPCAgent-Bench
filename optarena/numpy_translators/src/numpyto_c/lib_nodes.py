"""Compatibility shim: ``lib_nodes`` moved to :mod:`numpyto_common.lib_nodes` (Phase 1 of the
NumpyToX unified-core migration). Re-exports the module namespace (public and private)
so existing ``numpyto_c.lib_nodes`` importers keep working.
"""
from numpyto_common import lib_nodes as _src
globals().update({k: v for k, v in vars(_src).items() if not k.startswith("__")})
del _src
