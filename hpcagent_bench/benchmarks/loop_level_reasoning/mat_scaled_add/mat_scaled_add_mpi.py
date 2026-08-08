"""Reference distributed kernel_mpi for mat_scaled_add: elementwise ``B += alpha*A``, no cross-rank dependence."""


def kernel_mpi(A, B, M, N, alpha, *, comm, workspace):
    B[...] = B + alpha * A
