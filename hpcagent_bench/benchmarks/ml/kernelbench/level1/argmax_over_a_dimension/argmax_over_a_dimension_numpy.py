import numpy as np


def argmax_over_a_dimension(x, dim, out):
    out[:] = np.argmax(x, axis=dim, keepdims=False)
