"""Write one kernel's numba sibling (target ``numba``): one ``@njit(parallel=True)`` build."""

from hpcagent_bench.translators.numpyto_common.emit_helpers.request import EmitRequest
from hpcagent_bench.translators.numpyto_common.emit_io import write_python_sibling
from hpcagent_bench.translators.numpyto_common.frontend import parse_kernel
from hpcagent_bench.translators.numpyto_common.sanitize import sanitize
from hpcagent_bench.translators.numpyto_numba.emit import emit_numba

__all__ = ["write_numba"]


def write_numba(request: EmitRequest) -> str:
    """``<short>[_<config>]_numba.py``; ``"override"`` when a hand-written file holds the name, else ``"ok"``."""
    # The IR carries the array ranks the desugarer needs to tell a batched (>=3-D) matmul from a 2-D one; without
    # bench_info the emit is verbatim.
    kir = (
        None if request.bench_info is None else parse_kernel(request.kernel, request.bench_info, config=request.config)
    )
    source = emit_numba(request.kernel.read_text(), fastmath=request.fastmath, kir=kir)
    if request.sanitize:
        source = sanitize(source)
    return write_python_sibling(request.kernel, request.out, request.config, "numba", source)
