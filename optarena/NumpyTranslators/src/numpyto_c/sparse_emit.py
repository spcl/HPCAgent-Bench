"""Compatibility shim: ``sparse_emit`` moved to :mod:`numpyto_common.sparse_emit` (Phase 1 of the
NumpyToX unified-core migration). Transparently re-exports the full module
namespace (public *and* private names) so existing ``numpyto_c.sparse_emit`` importers
-- including the test suite -- keep working unchanged.
"""
from numpyto_common import sparse_emit as _src
globals().update({k: v for k, v in vars(_src).items() if not k.startswith("__")})
del _src
