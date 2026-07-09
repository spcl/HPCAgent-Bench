import numpy as np


def forward(x, dim, out):
    out[:] = np.cumsum(x, axis=dim)
