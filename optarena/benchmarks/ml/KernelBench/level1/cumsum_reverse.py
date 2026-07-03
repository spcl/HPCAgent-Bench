import numpy as np

batch_size = 32768
input_shape = (32768,)
dim = 1

def init(dim):
    pass

def forward(x, dim=dim):
    return np.flip(np.cumsum(np.flip(x, axis=dim), axis=dim), axis=dim)

