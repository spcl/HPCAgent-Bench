import numpy as np

def softsign(x, out):
    out[:] = x / (1 + np.abs(x))
