import numpy as np

def forward(x, out):
    out[:] = x / np.linalg.norm(x, axis=1, keepdims=True)
