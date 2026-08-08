---
name: lang-cpp
description: "Writing correct C++23 for this harness: static_cast over silent conversion, and the six gates that check it."
---

# lang-cpp

Two jobs: (A) QUALITY-CHECK an existing C++ file through six gates; (B) enforce
modern C++23 idioms when WRITING C++. `<file>.cpp` is the placeholder for the
target throughout -- swap in the real path. Every command is copy-pasteable.

## Golden rule

**All six gates run. Warnings are errors. A clean pass = zero diagnostics from
every tool + a clean ASan run + a clean UBSan run.** Do not report "looks good"
until all six are green. Fix findings at the source (no suppress-to-pass); the
cppcheck suppressions below are only for third-party/system noise.

**First decide which kind of C++ this is -- it changes the clang-tidy/cppcheck set
AND whether Section B applies:**

- **Agent SELF-WRITTEN code** (the default -- C++ an agent/human authored by hand,
  including an optimization agent's own code): the **COMPREHENSIVE** set below, AND
  the modern-C++ writing rules in Section B are in force. The author writes ordinary,
  idiomatic C++ -- it does NOT need to know anything about DaCe or any code generator;
  it is judged as plain hand-written C++.

- **Machine-GENERATED outside code** (emitted by a tool the agent does not author and
  is not expected to understand internally -- e.g. DaCe codegen in
  `.dacecache/<name>/src/cpu/*.cpp`): treat as OPAQUE. Narrow clang-tidy to
  `clang-analyzer-*` only -- `bugprone-*` OFF, style/naming/modernize OFF -- because
  emitted code trips every style rule and even `bugprone-*` is ~all false positives
  (200+ lines of noise). **Section B does NOT apply** (do not "modernize" generated
  output; fix its generator instead). The path-sensitive analyzer is the only useful
  compile-time gate; the **ASan run is the real gate**. The whole check set is the
  GENERATED-code variant under gate 2 below (`--checks='-*,clang-analyzer-*'`,
  `--header-filter='$^'`) plus the sanitizer runs in gates 5 and 6.

Rule of thumb: if the agent wrote it (or would edit it by hand), it's self-written --
full gates + Section B. If a generator emitted it, it's machine-generated -- analyzer
+ sanitizers only, and never restyle it.

## A. The six gates (run in this order)

### 1. clang-format (format first, in place)
Use the project's `.clang-format` if one exists at or above the file; else a modern default.
```bash
# project style if present, else a modern default (fallback only when none is found):
if git -C "$(dirname <file>.cpp)" ls-files --error-unmatch .clang-format >/dev/null 2>&1 \
   || find "$(dirname <file>.cpp)" -name .clang-format | grep -q .; then
  clang-format -i --style=file <file>.cpp
else
  clang-format -i --style='{BasedOnStyle: LLVM, Standard: c++23, ColumnLimit: 120}' <file>.cpp
fi
```

### 2. clang-tidy (COMPREHENSIVE for hand-written code)
```bash
clang-tidy \
  --checks='-*,bugprone-*,cppcoreguidelines-*,modernize-*,performance-*,portability-*,readability-*,clang-analyzer-*' \
  --header-filter='.*' \
  --warnings-as-errors='*' \
  <file>.cpp -- -std=c++23 -Wall -Wextra -Wconversion -Wsign-conversion -Wfloat-conversion -Wdouble-promotion -Wold-style-cast
```
`-header-filter=.*` so the file's own headers are checked too. Prefer `clang-tidy-21`
if installed (`clang-tidy-21 ...`). If a CMake compile DB exists, add `-p <build-dir>`
so includes/macros resolve (the build dir needs
`cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON <build-dir>`).

GENERATED-code variant (narrow set, analyzer only):
```bash
clang-tidy --checks='-*,clang-analyzer-*' --header-filter='$^' -p <build-dir> <generated>.cpp
```

### 3. cppcheck
```bash
cppcheck --enable=warning,performance,portability,style \
  --std=c++23 --language=c++ \
  --inline-suppr --error-exitcode=1 --quiet \
  --suppress=preprocessorErrorDirective \
  --suppress=missingIncludeSystem \
  --suppress='*:*/external/*' \
  <file>.cpp
```
Suppressions cover third-party/system noise only (vendored-header platform `#error`s,
findings inside `external/`, system-include gaps) -- never our own bugs. If a compile
DB exists, prefer `--project=<build-dir>/compile_commands.json` over the bare file.

### 4. gcc static analyzer (syntax-only, no build)
```bash
g++ -std=c++23 -fsyntax-only -fanalyzer -Wall -Wextra -Wconversion -Wsign-conversion -Wfloat-conversion -Wdouble-promotion -Wold-style-cast <file>.cpp
```
`-fanalyzer` turns on the `-Wanalyzer-*` family (double-free, use-after-free,
null-deref, leaks, taint). Treat every `-Wanalyzer-*` line as a defect to fix.
Add `-Werror` to make it hard-fail.

### 5. AddressSanitizer -- build and RUN once
Static analysis is not enough; the file must actually run under ASan.
```bash
g++ -std=c++23 -fsanitize=address -fno-omit-frame-pointer -g -O1 <file>.cpp -o /tmp/cppq_asan
ASAN_OPTIONS=detect_leaks=1 /tmp/cppq_asan   # exercise the real entry point / test
```
`detect_leaks=1` by default. Use `detect_leaks=0` ONLY when the process is
dominated by an external runtime whose leaks you don't own (e.g. a kernel dlopen'd
into a leaky Python host) -- state the rationale when you do. For a dlopen'd kernel,
build it with the same flags and `LD_PRELOAD=$(gcc -print-file-name=libasan.so)`
into the host process (use the matching clang RT if the code is built with clang).

### 6. UndefinedBehaviorSanitizer -- build and RUN once
```bash
g++ -std=c++23 -fsanitize=undefined -fno-omit-frame-pointer -g -O1 <file>.cpp -o /tmp/cppq_ubsan
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 /tmp/cppq_ubsan
```
`halt_on_error=1` so the first UB aborts with a trace -- any hit is a bug.
ASan and UBSan can be combined in one build (`-fsanitize=address,undefined`) when
convenient; keeping them separate isolates which sanitizer fired.

**Report** each gate's status. Only "clean" when all six pass with zero output.

## B. Writing modern C++23 (no OO bloat)

Prefer plain functions + small concrete data types + RAII. Do NOT invent class
hierarchies, factories, or indirection layers that aren't needed (YAGNI). Apply:

- **Concepts** to constrain templates; drop SFINAE/`enable_if` trickery.
- **`if constexpr`** over tag-dispatch / overload-set tricks for compile-time branching.
- **`constexpr` / `consteval`** on anything evaluable at compile time; add
  **`static_assert`** to lock in invariants (sizes, ranges, type traits).
- **NO macros.** Replace `#define` constants with `constexpr` values; replace
  function-like macros with `constexpr`/`consteval` (or `inline`) functions.
- **Ranges & views** (`std::ranges`, `|` pipelines) over hand-rolled index loops.
- **`std::format`** for formatting; **`std::expected`** for recoverable errors;
  **`std::span`** for non-owning array views; **`std::string_view`** for borrowed text.
- **No implicit conversions -- make every cast EXPLICIT.** Never rely on a silent
  narrowing / sign-changing / int<->float / promotion conversion. Write it out with a
  named cast (`static_cast<T>`, `gsl::narrow_cast`/`narrow` when intended), never a
  C-style or functional cast, never `const_cast`/`reinterpret_cast` unless truly
  unavoidable (justify). Brace-initialize (`T x{expr};`, `{}` in ctor args) so a
  narrowing conversion is a compile error, not a silent truncation. The
  `-Wconversion -Wsign-conversion -Wfloat-conversion -Wdouble-promotion -Wold-style-cast`
  flags above make implicit conversions fail the build; fix them at the source with an
  explicit cast, do not silence the warning.
- **Value semantics + RAII.** Prefer values and RAII for resource lifetime. **Raw
  pointers are fine** -- for non-owning/observing references and performance-sensitive
  interfaces; do not force `unique_ptr`/`shared_ptr` where a raw pointer or reference
  is clearer. Use smart pointers when they genuinely simplify ownership. Avoid leaking
  manual `new`/`delete`; no C-style casts (see the explicit-cast rule above).
- `auto`, range-`for`, `enum class`, `[[nodiscard]]`, `noexcept` where it holds.

After writing or modernizing, run all six gates in section A on the result.
