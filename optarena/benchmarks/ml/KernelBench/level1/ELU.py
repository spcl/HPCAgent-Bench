import numpy as np

batch_size = 4096
dim = 393216

def init(alpha=1.0):
    pass

def forward(x, alpha=1.0):
    return np.where((x) > 0, (x), (alpha) * (np.exp(x) - 1.0))

