import numpy as np

batch_size = 128
dim1 = 4096
dim2 = 4095
dim = 1

def init(dim):
    pass

def forward(x, dim=dim):
    return np.argmin(x, axis=dim, keepdims=False)

