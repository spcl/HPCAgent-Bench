import numpy as np

M = 4096
N = 4096

def init():
    pass

def forward(A, B):
    return (np.expand_dims(A, axis=1) * B)

