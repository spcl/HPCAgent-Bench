import numpy as np

batch_size = 4096
dim = 393216

def init():
    pass

def forward(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

