import numpy as np


def forward(x, dim, out):
    out[:] = np.cumprod(x, axis=dim)
