import numpy as np


def initialize(N, M):
    matrix = np.random.rand(N, M).astype(np.float64)
    out = np.zeros(N, dtype=np.float64)
    return matrix, out
