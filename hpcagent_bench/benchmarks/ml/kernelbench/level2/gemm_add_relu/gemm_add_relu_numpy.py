import numpy as np


def gemm_add_relu(x, gemm_weight, gemm_bias, bias, out):
    x = ((x) @ gemm_weight.T + gemm_bias)
    x = (x + bias)
    x = np.maximum(x, 0)
    out[:] = x
