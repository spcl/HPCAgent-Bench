import numpy as np

def forward(A, B, out):
    out[:] = np.matmul(A, B.T)
