import numpy as np

def forward(A, B, out):
    out[:] = np.tril(np.matmul(A, B))
