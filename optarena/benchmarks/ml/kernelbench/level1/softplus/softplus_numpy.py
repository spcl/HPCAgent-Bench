import numpy as np

def forward(x, out):
    out[:] = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
