import numpy as np


def dist_matmul_large_k(A, B, out):
    out[:] = np.matmul(A, B)
