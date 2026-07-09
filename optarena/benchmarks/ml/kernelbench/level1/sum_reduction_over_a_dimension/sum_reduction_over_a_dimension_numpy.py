import numpy as np


def forward(x, dim, out):
    out[:] = np.sum(x, axis=dim, keepdims=True)
