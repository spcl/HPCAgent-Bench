# NumpyToFortran

Python (numpy-numeric subset) → Fortran 2008 emitter. Reuses
NumpyToC's IR + frontend + lowering passes; only the body walker
and the timing prelude differ.

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

Every kernel emits a thin timing wrapper:

```fortran
subroutine s111_d_auto(iterations, len_1d, a, b, time_ns)
    use, intrinsic :: iso_c_binding
    use, intrinsic :: iso_fortran_env, only: int64
    integer, intent(in) :: iterations, len_1d
    real(c_double), intent(inout) :: a(len_1d)
    real(c_double), intent(in)    :: b(len_1d)
    integer(int64), intent(out) :: time_ns

    integer(int64) :: t1, t2, rate
    integer :: nl, i

    call system_clock(t1, rate)
    do nl = 0, 2*iterations - 1
        do i = 1, len_1d - 1, 2
            a(i+1) = a(i) + b(i+1)
        end do
    end do
    call system_clock(t2)
    time_ns = (t2 - t1) * 1000000000_int64 / rate
end subroutine
```

Notes:

* The subroutine is `bind(C, name="s111_d_auto")` so the harness
  ctypes call links straight against it -- same convention as the
  C / C++ outputs.
* 1-based Fortran indexing offsets Python's 0-based loops; the
  walker tracks this so the emitted code reads `a(i+1)` for a
  loop variable that ranges `[0, N-1)` in Python.
* `intent(inout)` for the LHS array, `intent(in)` for read-only.
