import numpy as np


def cumsum(x, dim, out):
    out[:] = np.cumsum(x, axis=dim)
