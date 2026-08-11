import numpy as np

def frobenius_norm(x, out):
    norm = np.linalg.norm(x, axis=None, keepdims=False)
    out[:] = x / norm
