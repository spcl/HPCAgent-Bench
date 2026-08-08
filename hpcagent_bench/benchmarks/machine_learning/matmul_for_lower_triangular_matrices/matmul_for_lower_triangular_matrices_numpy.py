import numpy as np

def matmul_for_lower_triangular_matrices(A, B, out):
    out[:] = np.tril(np.matmul(A, B))
