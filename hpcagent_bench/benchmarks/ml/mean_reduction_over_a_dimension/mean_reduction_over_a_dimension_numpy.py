import numpy as np


def mean_reduction_over_a_dimension(x, dim, out):
    out[:] = np.mean(x, axis=dim, keepdims=False)
