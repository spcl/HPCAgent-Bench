import numpy as np

def matmul_for_upper_triangular_matrices(A, B, out):
    out[:] = np.triu(np.matmul(A, B))
