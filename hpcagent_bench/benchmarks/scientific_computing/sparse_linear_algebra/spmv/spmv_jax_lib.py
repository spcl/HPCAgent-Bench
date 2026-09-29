# Sparse Matrix-Vector Multiplication (SpMV)


# The JAX framework hands ``A`` over as a BCOO matrix (converted outside the timed call).
def spmv(A, x):
    return A @ x
