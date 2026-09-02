import numpy as np


def rms_norm(x, eps, out):
    rms = np.sqrt(np.mean(x**2, axis=1, keepdims=True) + eps)
    out[:] = x / rms
