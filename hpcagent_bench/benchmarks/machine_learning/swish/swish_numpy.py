from __future__ import annotations
import numpy as np


def swish(x, out):
    out[:] = x * (1.0 / (1.0 + np.exp(-x)))
