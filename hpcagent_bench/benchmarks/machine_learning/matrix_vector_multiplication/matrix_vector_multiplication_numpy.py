import numpy as np

def matrix_vector_multiplication(A, B, out):
    out[:] = np.matmul(A, B)
