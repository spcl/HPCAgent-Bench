from scipy.linalg import eigh as _sci_eigh
import numpy as np


def eigh_test(a, b, wout, vout, lower=False):
    w, v = _sci_eigh(a, b, lower=lower)
    wout[:] = w
    vout[:] = v
