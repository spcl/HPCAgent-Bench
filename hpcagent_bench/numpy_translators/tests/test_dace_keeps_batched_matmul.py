"""The DaCe backend keeps a stacked ``@`` as one product, which canon lowers to a batched GEMM."""

import pytest

from numpyto_common.numpy_desugar import desugar_for_python_backend


class Kir:
    """The fields ``desugar_for_python_backend`` reads off a KernelIR."""

    class Arr:
        def __init__(self, name: str, shape: tuple[str, ...]) -> None:
            self.name, self.shape, self.dtype = name, shape, "float64"

    sparse = None
    kernel_name = "bmm"

    def __init__(self) -> None:
        self.arrays = [Kir.Arr("a", ("B", "M", "K")), Kir.Arr("b", ("B", "K", "N")), Kir.Arr("out", ("B", "M", "N"))]


def desugared(body: str, backend: str) -> str:
    return desugar_for_python_backend(f"import numpy as np\ndef bmm(a, b, out):\n    out[:] = {body}\n", Kir(), backend)


@pytest.mark.parametrize("body", ["a @ b", "np.matmul(a, b)", "out + a @ b"])
def test_a_batched_matmul_stays_one_product_for_dace(body: str) -> None:
    out = desugared(body, "dace")
    assert body in out and "__bm" not in out, out


def test_numba_still_gets_the_per_batch_loop() -> None:
    assert "__bm" in desugared("a @ b", "numba")
