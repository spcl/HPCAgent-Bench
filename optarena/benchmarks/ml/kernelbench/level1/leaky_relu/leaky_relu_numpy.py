import numpy as np

def forward(x, negative_slope, out):
    out[:] = np.where(x > 0, x, negative_slope * x)
