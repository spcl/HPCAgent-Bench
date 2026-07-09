import numpy as np


def forward(x, dim, out):
    out[:] = np.max(x, axis=dim, keepdims=False)
