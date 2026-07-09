import numpy as np

def forward(x, out):
    out[:] = np.tanh(x)
