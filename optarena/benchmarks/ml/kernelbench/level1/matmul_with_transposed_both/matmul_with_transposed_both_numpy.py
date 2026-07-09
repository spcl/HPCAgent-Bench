import numpy as np

def forward(A, B, out):
    out[:] = np.matmul(A.T, B.T)
