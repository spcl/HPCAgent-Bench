import numpy as np

M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def init():
    pass

def forward(A, B):
    return np.matmul(A.T, B)

