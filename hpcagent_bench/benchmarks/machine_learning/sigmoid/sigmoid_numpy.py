from __future__ import annotations
import numpy as np


def sigmoid(x, out):
    out[:] = 1.0 / (1.0 + np.exp(-x))
