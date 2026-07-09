import numpy as np


def forward(x, dim, out):
    out[:] = np.min(x, axis=dim, keepdims=False)
