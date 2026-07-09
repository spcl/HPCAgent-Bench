import numpy as np

def forward(x, out):
    out[:] = x / np.mean(np.abs(x), axis=1, keepdims=True)
