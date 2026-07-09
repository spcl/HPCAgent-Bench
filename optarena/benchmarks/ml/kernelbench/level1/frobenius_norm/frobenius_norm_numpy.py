import numpy as np

def forward(x, out):
    norm = np.linalg.norm(x, axis=None, keepdims=False)
    out[:] = x / norm
