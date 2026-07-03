import numpy as np

M = 4096

def init():
    pass

def forward(A, B):
    return np.tril(np.matmul(A, B))

