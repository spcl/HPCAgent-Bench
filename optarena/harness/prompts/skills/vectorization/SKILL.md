---
name: vectorization
description: Getting the inner loop into SIMD -- contiguity, aliasing, alignment, reductions.
---

The compiler vectorizes an inner loop only when it can prove the loop is safe. Most of
this skill is removing the reasons it cannot.

- **Contiguity.** The vectorized axis must be unit-stride. If it is not, interchange or
  transpose first -- a gathered load throws away most of the win.
- **Aliasing.** Two pointers the compiler cannot separate serialize the loop. Say they are
  distinct: `restrict` in C, `__restrict__` in C++, distinct dummy arguments in Fortran.
- **Alignment.** Align hot arrays to the vector width and tell the compiler
  (`assume_aligned`, `!$omp simd aligned(...)`). Misalignment costs a peel loop.
- **Trip count.** A loop bound the compiler cannot see forces a scalar remainder. Where
  the bound is a known multiple, say so.
- **Reductions.** A floating-point reduction is not reassociable by default, so the
  compiler will refuse. Declare it (`#pragma omp simd reduction(+:acc)`) -- but only when
  the reassociated result stays inside the tolerance.
- **Branches.** Turn data-dependent branches in the inner loop into arithmetic (select /
  masked blend) so the whole vector stays on one path.

Verify it worked -- do not assume. Read the vectorization report, or `objdump -d` the
symbol and look for the wide registers.
