import numpy as np

def forward(x, out):
    out[:] = x / (1 + np.abs(x))
