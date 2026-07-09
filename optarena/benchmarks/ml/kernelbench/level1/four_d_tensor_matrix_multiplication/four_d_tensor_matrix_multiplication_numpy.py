import numpy as np

def forward(A, B, out):
    out[:] = np.einsum('bijl,lk->bijk', A, B)
