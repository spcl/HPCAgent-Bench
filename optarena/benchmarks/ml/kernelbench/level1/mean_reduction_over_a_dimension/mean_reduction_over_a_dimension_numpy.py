import numpy as np


def forward(x, dim, out):
    out[:] = np.mean(x, axis=dim, keepdims=False)
