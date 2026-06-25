"""Compatibility shim: ``frontend`` moved to :mod:`numpyto_common.frontend` (Phase 1 of the
NumpyToX unified-core migration). Transparently re-exports the full module
namespace (public *and* private names) so existing ``numpyto_c.frontend`` importers
-- including the test suite -- keep working unchanged.
"""
from numpyto_common import frontend as _src
globals().update({k: v for k, v in vars(_src).items() if not k.startswith("__")})
del _src
