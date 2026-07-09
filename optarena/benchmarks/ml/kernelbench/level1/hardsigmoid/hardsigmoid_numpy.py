import numpy as np

def forward(x, out):
    out[:] = np.clip((x + 3.0) / 6.0, 0.0, 1.0)
