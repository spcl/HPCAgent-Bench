from __future__ import annotations
import numpy as np


def relu(x, out):
    out[:] = np.maximum(x, 0)
