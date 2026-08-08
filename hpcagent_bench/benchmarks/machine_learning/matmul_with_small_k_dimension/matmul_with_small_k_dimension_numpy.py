import numpy as np

def matmul_with_small_k_dimension(A, B, out):
    out[:] = np.matmul(A, B)
