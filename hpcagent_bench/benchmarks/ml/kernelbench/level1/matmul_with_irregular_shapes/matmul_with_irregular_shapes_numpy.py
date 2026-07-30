import numpy as np

def matmul_with_irregular_shapes(A, B, out):
    out[:] = np.matmul(A, B)
