from __future__ import annotations
import numpy as np


def tanh(x, out):
    out[:] = np.tanh(x)
