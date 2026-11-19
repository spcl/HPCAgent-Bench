# NumpyToFortran

Python (numpy-numeric subset) -> Fortran emitter. Shares the IR, frontend and lowering passes
of `numpyto_common` with the C family; only the body walker differs.

Why Fortran is the smallest delta from NumpyToC's IR:

| numpy idiom            | Fortran equivalent           |
|------------------------|------------------------------|
| `A @ B`                | `MATMUL(A, B)`               |
| `np.dot(a, b)`         | `DOT_PRODUCT(a, b)`          |
| `np.sum(A)`            | `SUM(A)`                     |
| `np.maximum(A, c)`     | `MAX(A, c)` (ELEMENTAL)      |
| `np.argmax(A)`         | `MAXLOC(A, DIM=1)`           |
| `np.where(M, A, B)`    | `MERGE(A, B, M)`             |
| `np.exp(A)`            | `EXP(A)` (ELEMENTAL)         |
| `A[1:N-1] = expr`      | `A(2:N-1) = expr`            |

Every kernel emits a thin `bind(C)` wrapper. The kernel carries no
in-kernel timer; the harness times the call externally:

```fortran
subroutine tsvc_2_s111_fp64(a, b, LEN_1D) bind(C, name="tsvc_2_s111_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    integer(c_int64_t) :: i_l0

    do i_l0 = 1, (LEN_1D) - 1, 2
        a((i_l0) + 1) = a(((i_l0 - 1)) + 1) + b((i_l0) + 1)
    end do

end subroutine tsvc_2_s111_fp64
```

Notes:

* The subroutine is `bind(C, name="tsvc_2_s111_fp64")` so the harness
  ctypes call links straight against it -- same convention as the
  C / C++ outputs.
* 1-based Fortran indexing offsets Python's 0-based loops; the
  walker tracks this so the emitted code reads `a((i_l0) + 1)` for the
  Python loop `for i in range(1, LEN_1D, 2)`.
* `intent(inout)` for the LHS array, `intent(in)` for read-only.
