import numpy as np


def max_reduction_over_a_dimension(x, dim, out):
    out[:] = np.max(x, axis=dim, keepdims=False)
