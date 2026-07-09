import numpy as np

def forward(A, B, out):
    out[:] = np.expand_dims(A, axis=1) * B
