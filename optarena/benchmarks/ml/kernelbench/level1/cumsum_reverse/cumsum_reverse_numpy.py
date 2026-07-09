import numpy as np


def forward(x, dim, out):
    out[:] = np.flip(np.cumsum(np.flip(x, axis=dim), axis=dim), axis=dim)
