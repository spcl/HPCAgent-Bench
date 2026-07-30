import numpy as np

def matmul_for_symmetric_matrices(A, B, out):
    out[:] = np.matmul(A, B)
