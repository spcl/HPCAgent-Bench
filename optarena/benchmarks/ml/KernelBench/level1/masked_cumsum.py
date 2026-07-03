import numpy as np

batch_size = 32768
input_shape = (32768,)
dim = 1

def init(dim):
    pass

def forward(x, mask, dim=dim):
    return np.cumsum((x * mask), axis=dim)

