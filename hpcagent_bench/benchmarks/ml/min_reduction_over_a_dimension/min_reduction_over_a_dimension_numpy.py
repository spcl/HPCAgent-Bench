import numpy as np


def min_reduction_over_a_dimension(x, dim, out):
    out[:] = np.min(x, axis=dim, keepdims=False)
