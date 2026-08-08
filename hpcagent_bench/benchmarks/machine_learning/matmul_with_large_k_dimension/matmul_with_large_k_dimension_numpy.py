import numpy as np

def matmul_with_large_k_dimension(A, B, out):
    out[:] = np.matmul(A, B)
