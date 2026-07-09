import numpy as np

def forward(A, B, out):
    out[:] = np.triu(np.matmul(A, B))
