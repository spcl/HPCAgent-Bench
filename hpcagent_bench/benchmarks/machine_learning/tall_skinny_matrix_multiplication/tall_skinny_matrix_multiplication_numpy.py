from __future__ import annotations
import numpy as np


def tall_skinny_matrix_multiplication(A, B, out):
    out[:] = np.matmul(A, B)
