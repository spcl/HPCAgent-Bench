import numpy as np

def rms_norm(x, num_features, eps, out):
    rms = np.sqrt(np.mean(x ** 2, axis=1, keepdims=True) + eps)
    out[:] = x / rms
