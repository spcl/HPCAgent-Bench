import numpy as np

def forward(x, out):
    out[:] = x * (1.0 / (1.0 + np.exp(-x)))
