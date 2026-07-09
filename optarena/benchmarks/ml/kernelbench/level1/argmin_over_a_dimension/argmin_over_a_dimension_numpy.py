import numpy as np


def forward(x, dim, out):
    out[:] = np.argmin(x, axis=dim, keepdims=False)
