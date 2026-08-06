import numpy as np

def min_gpt_new_gelu(x, out):
    out[:] = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3.0))))
