import numpy as np


def cumprod(x, dim, out):
    out[:] = np.cumprod(x, axis=dim)
