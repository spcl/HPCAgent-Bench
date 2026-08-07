import numpy as np

def matmul_with_transposed_b(A, B, out):
    out[:] = np.matmul(A, B.T)
