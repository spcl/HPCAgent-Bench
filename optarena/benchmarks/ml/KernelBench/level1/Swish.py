import numpy as np

batch_size = 4096
dim = 393216

def init():
    pass

def forward(x):
    return (x * (1.0 / (1.0 + np.exp(-(x)))))

