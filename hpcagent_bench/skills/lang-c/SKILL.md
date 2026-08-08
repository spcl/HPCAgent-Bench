---
name: lang-c
description: "Writing correct C17 for this harness: explicit casts, const/restrict, and the six gates that check it."
---

# lang-c

Two jobs: (A) QUALITY-CHECK an existing C file through six gates; (B) enforce
C17 idioms when WRITING C. `<file>.c` is the placeholder for the target
throughout -- swap in the real path. Every command is copy-pasteable. This is C,
not C++: compile with `gcc`/`clang` (not `g++`), `-std=c17`, `--language=c`.
`-std=c17` is what the harness builds with; `languages.std_flag("c")` is the source of truth.

## Golden rule

**All six gates run. Warnings are errors. A clean pass = zero diagnostics from
every tool + a clean ASan run + a clean UBSan run.** Do not report "looks good"
until all six are green. Fix findings at the source (no suppress-to-pass); the
cppcheck suppressions below are only for third-party/system noise.

Hand-written vs generated code -- this changes the clang-tidy/cppcheck check set:
- **Hand-written** code (the default here): the COMPREHENSIVE set below.
- **Machine-GENERATED** code (e.g. codegen output): narrow clang-tidy to
  `clang-analyzer-*` only, `bugprone-*`/style/naming OFF -- emitted code trips
  every style rule and most `bugprone-*` are false positives. The path-sensitive
  analyzer is the only useful compile-time gate; the ASan run is the real heap gate.

## A. The six gates (run in this order)

### 1. clang-format (format first, in place)
Use the project's `.clang-format` if one exists at or above the file; else a modern default.
(clang-format's `Standard:` knob is C++-only; for C files there is no `-std` to set.)
```bash
# project style if present, else a modern default (fallback only when none is found):
if find "$(dirname <file>.c)" -maxdepth 4 -name .clang-format | grep -q .; then
  clang-format -i --style=file <file>.c
else
  clang-format -i --style='{BasedOnStyle: LLVM, ColumnLimit: 120}' <file>.c
fi
```

### 2. clang-tidy (COMPREHENSIVE for hand-written C)
For C, drop the C++-only families (`modernize-*`, `cppcoreguidelines-*`) and add `cert-*`.
```bash
clang-tidy \
  --checks='-*,bugprone-*,cert-*,clang-analyzer-*,performance-*,portability-*,readability-*' \
  --header-filter='.*' \
  --warnings-as-errors='*' \
  <file>.c -- -std=c17 -Wall -Wextra -Wconversion -Wsign-conversion -Wfloat-conversion -Wdouble-promotion -Wbad-function-cast
```
`--header-filter=.*` so the file's own headers are checked too. Prefer
`clang-tidy-21` if installed. If a CMake compile DB
exists, add `-p <build-dir>` so includes/macros resolve (configure it with
`cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON <build-dir>`).

GENERATED-code variant (narrow set, analyzer only):
```bash
clang-tidy --checks='-*,clang-analyzer-*' --header-filter='$^' <generated>.c -- -std=c17
```

### 3. cppcheck
```bash
cppcheck --enable=warning,performance,portability,style \
  --std=c17 --language=c \
  --inline-suppr --error-exitcode=1 --quiet \
  --suppress=preprocessorErrorDirective \
  --suppress=missingIncludeSystem \
  --suppress='*:*/external/*' \
  <file>.c
```
Suppressions cover third-party/system noise only (vendored-header platform `#error`s,
findings inside `external/`, system-include gaps) -- never our own bugs. Add
`--check-level=exhaustive` for a deeper (slower) pass. If a compile DB exists,
prefer `--project=<build-dir>/compile_commands.json` over the bare file.

### 4. gcc static analyzer (syntax-only, no build)
```bash
gcc -std=c17 -fsyntax-only -fanalyzer -Wall -Wextra -Wconversion -Wsign-conversion -Wfloat-conversion -Wdouble-promotion -Wbad-function-cast <file>.c
```
`-fanalyzer` turns on the whole `-Wanalyzer-*` family (double-free, use-after-free,
null-deref, malloc/file leaks, mismatched dealloc, tainted-array-index, write-to-const).
Treat every `-Wanalyzer-*` line as a defect to fix. Add `-Werror` to make it hard-fail.
The analyzer is stronger at higher `-O`, but `-fsyntax-only` keeps it a no-build gate;
use `-O2 -c -o /dev/null` instead if you want the optimizer's extra reach.

### 5. AddressSanitizer -- build and RUN once
Static analysis is not enough; the file must actually run under ASan.
```bash
gcc -std=c17 -fsanitize=address -fno-omit-frame-pointer -g -O1 <file>.c -o /tmp/cq_asan
ASAN_OPTIONS=detect_leaks=1 /tmp/cq_asan   # exercise the real entry point / test
```
Catches heap/stack/global overflows, use-after-free, use-after-return, leaks.
`detect_leaks=1` is the Linux default. Use `detect_leaks=0` ONLY when the process
is dominated by an external runtime whose leaks you don't own -- state the rationale
when you do. For a `dlopen`'d object, build it with the same flags and
`LD_PRELOAD=$(gcc -print-file-name=libasan.so)` into the host process.

### 6. UndefinedBehaviorSanitizer -- build and RUN once
```bash
gcc -std=c17 -fsanitize=undefined -fno-omit-frame-pointer -g -O1 <file>.c -o /tmp/cq_ubsan
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 /tmp/cq_ubsan
```
`halt_on_error=1` so the first UB aborts with a trace -- any hit is a bug. Catches
signed-overflow, out-of-range shifts, null-deref, misalignment, bad float<->int
casts, integer div-by-zero, and invalid `bool`/enum loads.
`-fno-sanitize-recover=all` also aborts on first hit if you prefer it baked into the
binary. ASan and UBSan can share one build (`-fsanitize=address,undefined`); keeping
them separate isolates which sanitizer fired.

**Report** each gate's status. Only "clean" when all six pass with zero output.

## B. Writing C17 (lean, and only what C17 actually has)

Prefer plain functions + small concrete structs + tight scope. C has NO templates, NO
concepts, NO overloading -- for generic code use `_Generic` or macros.

**This section is C17, not C23, because that is what the harness compiles with.** C23
adds `constexpr` objects, `nullptr`, `typeof`, `_BitInt(N)`, `auto`, `unreachable()`,
`#embed`, `enum E : uint8_t` and the `[[...]]` attribute syntax -- **none of them are
available here**, and reaching for one gets you a compile error, not a nicer kernel.
Check `hpcagent_bench/languages.py::std_flag("c")` before assuming otherwise; it is the
single source of truth and this page follows it rather than restating a standard.

The C17 spellings of the same intents:

- **Compile-time constants**: `enum { CAP = 256 };` for integers (typed, scoped, usable
  in array bounds and `case` labels) and `static const double PI = 3.14159...;` for
  non-integers. A `static const` is not a constant expression in C, so it cannot size an
  array at file scope -- that is the one place `enum` or a macro is still required.
- **`static_assert`**: `_Static_assert(sizeof(T) == 8, "ABI");` is a C11 keyword and
  needs no header; `#include <assert.h>` also gives the `static_assert` spelling.
- **`bool` / `true` / `false`**: `#include <stdbool.h>`. They are macros here, not
  keywords -- so do not `#undef` them and do not assume `sizeof(bool) == 1` in an ABI.
- **Null pointer**: `NULL` from `<stddef.h>`. There is no `nullptr` / `nullptr_t`.
- **Attributes**: `__attribute__((warn_unused_result))`, `((unused))`, `((noreturn))`,
  `((fallthrough))` -- gcc and clang both take them, and `_Noreturn` is standard C11.
  Put `warn_unused_result` on any must-check return (allocators, parse/IO results).
- **Type-generic locals/macros**: GNU `__typeof__(*p) tmp = *p;`. Not portable C17, but
  both compilers this repo uses accept it; say so where you rely on it.
- **Exact widths**: `int32_t` / `uint64_t` / `size_t` / `ptrdiff_t` from `<stdint.h>` and
  `<stddef.h>`. There is no `_BitInt(N)`, so a 24-bit field is a bitfield or manual
  masking.
- **Enums**: an enum's underlying type is implementation-defined and promotes to `int`.
  If a fixed size matters (an ABI struct, a packed array), use an explicit `uint8_t` and
  named `enum` constants, not the enum type itself.
- **Impossible branches**: `__builtin_unreachable()`, or better, an `assert(0)` in debug
  builds -- there is no standard `unreachable()`.
- **Binary literals** `0b1010` are a GNU extension, not C17. Use hex.
- **No silent implicit conversions -- cast EXPLICITLY.** C has no `static_cast`, so
  write every lossy / narrowing / sign-changing / int<->float conversion as a deliberate
  `(type)` cast so the intent (and the truncation) is visible at the call site. Watch
  the usual C traps: integer promotions, `unsigned`/`signed` mixing, `size_t` vs `int`,
  `double`->`float`, implicit `int` from a bool context. The
  `-Wconversion -Wsign-conversion -Wfloat-conversion -Wdouble-promotion -Wbad-function-cast`
  flags above make implicit conversions fail the build -- fix them with an explicit cast
  at the source, never by silencing the warning. Keep casts rare and intentional; a
  cast you cannot justify is usually a type or design bug.

The rest, which C23 would not have changed anyway:
- **`const` and `restrict` correctness** -- `const` on non-written pointees; `restrict`
  on non-aliasing pointer params in hot paths (only when aliasing is truly impossible).
- **Designated initializers** with `= {0}` zeroing the rest: never leave fields indeterminate.
- **`static inline` functions over function-like macros** -- no double-evaluation, real
  types. Reserve macros for token pasting, `X`-macros, conditional compilation.
- **Check every return code** (`malloc`, `realloc`, `fopen`, `snprintf`, `pthread_*`);
  mark the APIs `__attribute__((warn_unused_result))` -- `[[nodiscard]]` is C23 syntax
  and does not compile at the `-std=c17` this harness builds with.
- **`sizeof(*ptr)` in allocations**, not the type name: `p = malloc(n * sizeof(*p));`
  (or `calloc(n, sizeof(*p))` for overflow-safe zeroing).
- **No VLAs in headers / public interfaces**, and avoid VLAs generally.
- **Minimal scope for declarations** -- declare at first use, initialize on declaration,
  loop counters inside the `for`; `static` (internal linkage) for anything not exported.

After writing or modernizing, run all six gates in section A on the result.

## References

Consulted 2026-08-04:
- Clang-Tidy checks & usage -- https://clang.llvm.org/extra/clang-tidy/
- Cppcheck manual -- https://cppcheck.sourceforge.io/manual.html
- GCC `-fanalyzer` / `-Wanalyzer-*` options -- https://gcc.gnu.org/onlinedocs/gcc/Static-Analyzer-Options.html
- GCC sanitizer (ASan/UBSan) instrumentation flags -- https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html
- Clang UndefinedBehaviorSanitizer -- https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html
- "A gentle introduction to static analyzers for C" (nrk) -- https://nrk.neocities.org/articles/c-static-analyzers
- Chris Wellons / nullprogram, modern C practices -- https://nullprogram.com/blog/2023/10/08/
- C17/C11 library and language reference -- https://en.cppreference.com/w/c
