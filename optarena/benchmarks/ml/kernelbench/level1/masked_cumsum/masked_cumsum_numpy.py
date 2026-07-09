import numpy as np


def forward(x, mask, dim, out):
    out[:] = np.cumsum((x * mask), axis=dim)
