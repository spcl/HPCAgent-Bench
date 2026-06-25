

def kernel(A, x, out):

    out[:] = (A @ x) @ A
