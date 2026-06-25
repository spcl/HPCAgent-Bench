

def kernel(alpha, beta, C, A, B):

    C[:] = alpha * A @ B + beta * C
