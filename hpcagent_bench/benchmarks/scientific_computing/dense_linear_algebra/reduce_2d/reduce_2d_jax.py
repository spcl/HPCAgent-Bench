# JAX version written for HPCAgent-Bench from the NumPy reference (NPBench ships none).

import jax
import jax.numpy as jnp


@jax.jit
def row_reduce(matrix):
    return jnp.sum(matrix, axis=1)
