import numpy as np

b = 8
i = 256
j = 512
l = 256
k = 768

def init():
    pass

def forward(A, B):
    return np.einsum('bijl,lk->bijk', A, B)

