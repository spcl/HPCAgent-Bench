"""Compatibility shim: ``ir`` moved to :mod:`numpyto_common.ir` (Phase 1 of the
NumpyToX unified-core migration). Transparently re-exports the full module
namespace (public *and* private names) so existing ``numpyto_c.ir`` importers
-- including the test suite -- keep working unchanged.
"""
from numpyto_common import ir as _src
globals().update({k: v for k, v in vars(_src).items() if not k.startswith("__")})
del _src
