import numpy as np

batch_size = 112
features = 64
dim1 = 512
dim2 = 512

def init():
    pass

def forward(x):
    norm = np.linalg.norm(x, axis=None, keepdims=False)
    return (x / norm)

