import numpy as np


def forward(A, s, out):
    out[:] = (A * s)
