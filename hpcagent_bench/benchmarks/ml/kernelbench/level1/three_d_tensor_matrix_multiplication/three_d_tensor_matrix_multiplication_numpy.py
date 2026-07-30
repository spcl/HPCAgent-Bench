import numpy as np

def three_d_tensor_matrix_multiplication(A, B, out):
    out[:] = np.matmul(A, B)
