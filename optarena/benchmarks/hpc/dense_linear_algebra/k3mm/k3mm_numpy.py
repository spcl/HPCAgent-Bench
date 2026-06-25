

def kernel(A, B, C, D, out):

    out[:] = A @ B @ C @ D
