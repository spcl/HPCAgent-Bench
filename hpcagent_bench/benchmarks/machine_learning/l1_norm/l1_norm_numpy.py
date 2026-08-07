import numpy as np

def l1_norm(x, out):
    out[:] = x / np.mean(np.abs(x), axis=1, keepdims=True)
