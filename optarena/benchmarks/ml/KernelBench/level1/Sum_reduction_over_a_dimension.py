import numpy as np

batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

def init(dim):
    pass

def forward(x, dim=reduce_dim):
    return np.sum(x, axis=dim, keepdims=True)

