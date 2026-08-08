---
name: lang-fortran
description: "Writing correct Fortran 2018 for this harness: explicit kinds and intents, and the gates that check them."
---

# lang-fortran

Two jobs: (A) QUALITY-CHECK an existing Fortran 2018 file through the gate ladder;
(B) enforce modern Fortran 2018 idioms when WRITING Fortran. `<file>.f90` is the
placeholder for the target throughout -- swap in the real path. Every command is
copy-pasteable.

## Golden rule

**All gates run. Warnings are errors. A clean pass = zero diagnostics from every
tool + a clean `-fcheck=all` RUN + a clean ASan RUN + a clean UBSan RUN.** Do not
report "looks good" until every gate is green. Fix findings at the source -- never
silence a warning to pass.

House conventions: **single-TU free-form `.f90` sources, line length 120.**
If a `.fprettify.rc` sits at or above the file, it wins (typical: `indent=2`,
`line-length=120`). Tools: `fprettify`, `gfortran` (primary gate -- needs a recent
version, 13+/15+, for full F2018 + `-fanalyzer` + sanitizers), and `flang`
(optional second front-end, only if installed). Probe availability first; run the
flang gate only where a flang driver exists.

## A. The gates (run in this order)

### 1. fprettify -- format first, in place
```bash
# project style if a config is present at/above the file, else the house default:
cfg="$(dirname <file>.f90)/.fprettify.rc"; [ -f "$cfg" ] || cfg="$(git -C "$(dirname <file>.f90)" rev-parse --show-toplevel 2>/dev/null)/.fprettify.rc"
if [ -f "$cfg" ]; then
  fprettify --config-file "$cfg" <file>.f90
else
  fprettify --indent 2 --line-length 120 <file>.f90
fi
```
fprettify edits in place by default: consistent indentation, whitespace around
operators/delimiters, aligned continuations. To exempt a hand-aligned block (e.g.
a literal matrix), guard it with `!&<` ... `!&>` (or a trailing `!&` on one line).

### 2. Compile with ALL warnings -- warnings are errors (both compilers when available)
gfortran (primary gate -- this is the strong one for Fortran):
```bash
gfortran -std=f2018 -Wall -Wextra -Wimplicit-interface -Wimplicit-procedure \
  -Wconversion -Wconversion-extra -fimplicit-none -Werror -c <file>.f90 -o /tmp/fq.o
```
Add `-pedantic` to also flag non-standard extensions. `-Wimplicit-interface`
`-Wimplicit-procedure` catch any call going through an implicit (uncheckable)
interface -- in clean modern code there are none. `-Wconversion` `-Wconversion-extra`
flag every implicit type/kind conversion (mixed-mode `real`/`integer` arithmetic,
`kind` promotions) -- with `-Werror` they are build failures; fix them with an
explicit intrinsic conversion, never by widening the warning set down.

LLVM flang, only if installed -- weaker warnings today, but a useful second
front-end opinion and the path to LLVM sanitizers for `bind(c)` code. LLVM 20
renamed the driver `flang-new` -> `flang`, so probe the new name first and fall
back, the way `hpcagent_bench/languages.py::resolve_compiler` does -- probing only
`flang-new` means the gate never runs on a current toolchain:
```bash
FLANG="$(command -v flang || command -v flang-new)"
[ -n "$FLANG" ] && "$FLANG" -std=f2018 -Wall -c <file>.f90 -o /tmp/fq_flang.o
```

### 3. gfortran static analyzer (`-fanalyzer`, syntax-only, no link)
```bash
gfortran -std=f2018 -fsyntax-only -fanalyzer -Wall -Wextra -Wconversion -Wconversion-extra <file>.f90
```
`-fanalyzer` enables the `-Wanalyzer-*` path-sensitive family (double-free,
use-after-free, null/leak). **Advisory, not a gate, and it does not count toward
"clean".** GCC documents it as "only suitable for use on C code in this release":
on pure Fortran it fires on nothing, so a zero-finding run is the tool declining to
look, not evidence the file is sound. Run it only where the file has a `bind(c)`
surface whose C side you compile too, and there any `-Wanalyzer-*` line is a real
defect. Do not add `-Werror` here. Gate 4 is what actually decides.

### 4. Runtime-checked build + RUN (gfortran) -- the real Fortran gate
Static checks are not enough: the file must actually run with checks armed.
```bash
gfortran -std=f2018 -fcheck=all -fbacktrace -finit-real=snan \
  -finit-integer=-2147483648 -g -O0 <file>.f90 -o /tmp/fq_check
/tmp/fq_check    # exercise the real entry point / driver / test
```
`-fcheck=all` traps array-bounds, invalid do-loop index modification, pointer/
allocatable misuse, `mem`, `recursion`, and array-temp creation. `-finit-real=snan`
+ `-finit-integer=-2147483648` poison uninitialized storage so use-before-set shows
up as an obvious NaN / sentinel. To actually **trap** on touching a poisoned real,
add floating-point traps:
```bash
gfortran -std=f2018 -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow \
  -finit-real=snan -finit-integer=-2147483648 -g -O0 <file>.f90 -o /tmp/fq_fpe && /tmp/fq_fpe
```
A clean run = exits 0 with no bounds/pointer/temporary/FPE message on stderr.

### 5. AddressSanitizer -- build and RUN once
```bash
gfortran -std=f2018 -fsanitize=address -fno-omit-frame-pointer -g -O1 \
  <file>.f90 -o /tmp/fq_asan
/tmp/fq_asan    # exercise the real entry point
```
ASan catches heap/stack out-of-bounds and use-after-free -- most valuable for
`allocatable`/`pointer` and the C-interop (`bind(c)`, `iso_c_binding`) surface.

### 6. UndefinedBehaviorSanitizer -- build and RUN once
```bash
gfortran -std=f2018 -fsanitize=undefined -fno-omit-frame-pointer -g -O1 \
  <file>.f90 -o /tmp/fq_ubsan
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 /tmp/fq_ubsan
```

**Be honest about Fortran sanitizer limits.** gfortran's UBSan is thin for Fortran
(it mostly instruments C-like UB); `-fcheck=all` from gate 4 is the primary
runtime correctness gate for Fortran semantics, and ASan is the primary memory
gate. LLVM's ASan/UBSan are stronger for the `bind(c)`/C-interop parts -- but
`flang-new` does not yet ship working sanitizers, so gfortran is the sanitizer
toolchain here. Run gates 4+5+6 together for coverage; do not treat any one as
redundant. ASan+UBSan can share a build (`-fsanitize=address,undefined`) when
convenient; keeping them separate isolates which fired.

**Report** each gate's status. Only "clean" when every gate passes with zero output.

## B. Writing modern Fortran 2018 (no legacy bloat)

Free-form `.f90`, single translation unit, line length 120. Prefer plain
module procedures over elaborate derived-type hierarchies (KISS/YAGNI). Apply:

- **`implicit none` everywhere.** At module scope use the F2018 form
  `implicit none (type, external)` -- it also forbids implicit *external* interfaces,
  so every called procedure must be explicitly known.
- **`intent(in|out|inout)` on every dummy argument**, no exceptions. Mark
  read-only pointers/targets and use `value` for small C-interop scalars.
- **`pure` / `elemental` wherever the procedure has no side effects** -- enables
  optimization, `do concurrent`, and reasoning. `elemental` implies `pure`.
- **Modules + explicit interfaces only.** No external procedures with implicit
  interfaces, no `include`d bodies. Default to `private`, then `public ::` the
  exported names. Use explicit, named `use, only:` imports.
- **`contains`ed module/internal procedures** so interfaces are always explicit.
- **Parameterized `kind` from `iso_fortran_env`** (`real64`, `real32`, `int32`,
  `int64`), never legacy `real*8` / `double precision` / `integer*4`. Suffix every
  literal with its kind: `1.0_real64`, `0_int32`. Declare
  `use, intrinsic :: iso_fortran_env, only: real64, int32`.
- **No implicit type/kind conversions -- convert EXPLICITLY.** Never rely on silent
  mixed-mode arithmetic or `kind` promotion (`integer`<->`real`, `real32`<->`real64`,
  `real`<->`complex`). Write the intrinsic: `real(i, kind=real64)`, `int(x, kind=int32)`,
  `cmplx(re, im, kind=real64)`, `nint(x)` for rounding. Keep every operand of an
  expression the SAME kind, and suffix literals with their kind so no promotion sneaks
  in (`0.5_real64 * x`, not `0.5 * x`). The `-Wconversion -Wconversion-extra -Werror`
  gate fails the build on any implicit conversion -- fix it with an intrinsic, and never
  narrow the kind of a stored result by accident.
- **`allocatable` over `pointer`** whenever ownership is not shared -- automatic
  cleanup, no leaks, no dangling. **Always check `stat=`** on `allocate`/
  `deallocate` and act on `errmsg=`:
  `allocate(a(n), stat=ierr, errmsg=msg); if (ierr /= 0) error stop msg`.
- **`associate`** to name subexpressions / slices for clarity.
- **`do concurrent`** for genuinely data-parallel loops (no cross-iteration
  dependence) instead of a plain `do`.
- **`error stop "msg"`** for fatal errors (not bare `stop`; never `pause`).
- **Never** `common`, `equivalence`, `goto`/arithmetic-`if`/computed-`goto`,
  `entry`, `data`, fixed-form, or vendor extensions.
- Lowercase all keywords; name `end` blocks (`end subroutine foo`,
  `end module bar`); one-or-two-syllable names, underscores when longer.

After writing or modernizing, run all gates in section A on the result.

## References

Consulted 2026-08-04 (web access available):
- Fortran best practices -- https://fortran-lang.org/learn/best_practices/
- Fortran style guide -- https://fortran-lang.org/learn/best_practices/style_guide/
- fortran90.org best practices (implicit none, intent, allocatable, kinds) -- https://www.fortran90.org/src/best-practices.html
- stdlib style guide -- https://github.com/fortran-lang/stdlib/blob/master/STYLE_GUIDE.md
- fprettify README (CLI, config, `!&` guards) -- https://github.com/fortran-lang/fprettify/blob/master/README.md
- gfortran code-gen / debug options (`-fcheck`, `-finit-real=snan`, `-finit-integer`, `-fbacktrace`) -- https://gcc.gnu.org/onlinedocs/gfortran/Code-Gen-Options.html
- GCC instrumentation options (`-fsanitize=address`, `-fsanitize=undefined`) -- https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html
- GCC Fortran debug flags + `-fcheck` vs sanitizer tradeoffs -- https://gjbex.github.io/Defensive_programming_and_debugging/BugsAtRuntime/Verification/Compilers/gfortran_flags/
- Clang UBSan (limits, C-oriented) -- https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html
