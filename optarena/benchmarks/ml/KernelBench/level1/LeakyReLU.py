import numpy as np

batch_size = 4096
dim = 393216

def init(negative_slope=0.01):
    pass

def forward(x, negative_slope=0.01):
    return np.where((x) > 0, (x), (negative_slope) * (x))

