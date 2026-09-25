import numpy as np


def dist_all_to_all_transpose(x, out):
    out[:] = np.transpose(x)
