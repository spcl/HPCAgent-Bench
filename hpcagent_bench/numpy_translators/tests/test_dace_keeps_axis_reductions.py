"""The DaCe backend keeps a float axis reduction as the numpy call, which canon lowers to a library reduction."""

import pytest

from numpyto_common.numpy_desugar import desugar_for_python_backend


class Kir:
    """The fields ``desugar_for_python_backend`` reads off a KernelIR."""

    class Arr:
        def __init__(self, name: str, shape: tuple[str, ...], dtype: str) -> None:
            self.name, self.shape, self.dtype = name, shape, dtype

    sparse = None
    kernel_name = "rows"

    def __init__(self, dtype: str) -> None:
        self.arrays = [Kir.Arr("a", ("N", "M"), dtype), Kir.Arr("out", ("N",), dtype)]


def desugared(body: str, dtype: str = "float64", backend: str = "dace") -> str:
    src = f"import numpy as np\ndef rows(a, out):\n    out[:] = {body}\n"
    return desugar_for_python_backend(src, Kir(dtype), backend=backend)


@pytest.mark.parametrize("body", ["a.sum(axis=1)", "np.sum(a, axis=1)", "a.max(axis=1)", "np.mean(a, axis=1)"])
def test_a_float_axis_reduction_stays_a_call_for_dace(body: str) -> None:
    out = desugared(body)
    assert body in out and "__rdo" not in out, out


@pytest.mark.parametrize(
    "body, dtype, backend",
    [
        ("a.sum(axis=1)", "int32", "dace"),
        ("np.sum(a, axis=1, keepdims=True)[:, 0]", "float64", "dace"),
        ("a.sum(axis=1)", "float64", "numba"),
    ],
)
def test_a_reduction_the_dace_frontend_cannot_take_is_still_a_loop(body: str, dtype: str, backend: str) -> None:
    assert "__rdo" in desugared(body, dtype, backend)
