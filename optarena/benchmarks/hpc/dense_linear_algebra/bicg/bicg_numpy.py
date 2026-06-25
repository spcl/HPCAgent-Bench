

def kernel(A, p, r, out0, out1):

    out0[:] = r @ A
    out1[:] = A @ p
