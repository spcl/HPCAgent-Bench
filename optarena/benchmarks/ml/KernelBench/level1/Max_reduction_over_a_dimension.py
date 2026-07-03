import numpy as np

batch_size = 128
dim1 = 4096
dim2 = 4095

def init(dim):
    pass

def forward(x, dim=1):
    return np.max(x, axis=dim, keepdims=False)

