from __future__ import annotations
import numpy as np


def matrix_scalar_multiplication(A, s, out):
    out[:] = A * s
