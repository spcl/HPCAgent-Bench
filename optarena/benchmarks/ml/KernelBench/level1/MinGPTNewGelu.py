import numpy as np

batch_size = 8192
dim = 8192

def init():
    pass

def forward(x):
    return ((0.5 * x) * (1.0 + np.tanh((np.sqrt((2.0 / np.pi)) * (x + (0.044715 * np.power(x, 3.0)))))))

