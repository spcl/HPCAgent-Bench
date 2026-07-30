import numpy as np

def matmul_with_transposed_a(A, B, out):
    out[:] = np.matmul(A.T, B)
