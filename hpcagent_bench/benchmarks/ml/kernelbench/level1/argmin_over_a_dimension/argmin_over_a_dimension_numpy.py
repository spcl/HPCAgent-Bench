import numpy as np


def argmin_over_a_dimension(x, dim, out):
    out[:] = np.argmin(x, axis=dim, keepdims=False)
