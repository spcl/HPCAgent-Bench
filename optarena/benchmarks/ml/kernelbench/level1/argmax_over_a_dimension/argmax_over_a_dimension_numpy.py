import numpy as np


def forward(x, dim, out):
    out[:] = np.argmax(x, axis=dim, keepdims=False)
