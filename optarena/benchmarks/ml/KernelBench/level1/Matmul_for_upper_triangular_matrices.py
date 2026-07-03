import numpy as np

N = 4096

def init():
    pass

def forward(A, B):
    return np.triu(np.matmul(A, B))

