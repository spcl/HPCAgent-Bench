import numpy as np


def sum_reduction_over_a_dimension(x, dim, out):
    out[:] = np.sum(x, axis=dim, keepdims=True)
