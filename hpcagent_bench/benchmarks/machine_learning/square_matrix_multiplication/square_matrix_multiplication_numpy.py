import numpy as np

def square_matrix_multiplication(A, B, out):
    out[:] = np.matmul(A, B)
