import numpy as np

batch_size = 32768
dim = 65535

def init():
    pass

def forward(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True))

