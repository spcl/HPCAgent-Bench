import numpy as np

def relu(x, out):
    out[:] = np.maximum(x, 0)
