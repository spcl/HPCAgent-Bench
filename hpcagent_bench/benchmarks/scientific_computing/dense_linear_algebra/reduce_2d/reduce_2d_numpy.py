import numpy as np


def row_reduce(matrix, out, N, M):
    out[:] = np.sum(matrix, axis=1)
