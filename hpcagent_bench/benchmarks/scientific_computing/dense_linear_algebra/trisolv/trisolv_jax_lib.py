import jax


@jax.jit
def kernel(L, x, b):

    x = jax.scipy.linalg.solve_triangular(L, b, lower=True)
    return x
