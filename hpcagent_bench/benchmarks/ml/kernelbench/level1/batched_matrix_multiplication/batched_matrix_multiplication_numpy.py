import numpy as np

def batched_matrix_multiplication(A, B, out):
    out[:] = np.matmul(A, B)
