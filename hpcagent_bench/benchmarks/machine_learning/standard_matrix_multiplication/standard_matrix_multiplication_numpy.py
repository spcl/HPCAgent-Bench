import numpy as np

def standard_matrix_multiplication(A, B, out):
    out[:] = np.matmul(A, B)
