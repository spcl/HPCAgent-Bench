import numpy as np

def forward(x, out):
    out[:] = np.maximum(x, 0)
