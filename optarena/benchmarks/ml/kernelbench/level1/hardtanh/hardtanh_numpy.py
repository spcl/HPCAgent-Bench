import numpy as np

def forward(x, out):
    out[:] = np.clip(x, -1.0, 1.0)
