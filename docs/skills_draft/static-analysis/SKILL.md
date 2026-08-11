---
name: static-analysis
description: Catch undefined behaviour at compile time -- gcc and clang warning gates, -fanalyzer, the clang analyzer, cppcheck, what each one misses, and when a sanitizer is the right tool.
---

An ICON halo body shipped for months on a buffer sized from an uninitialised local. The only symptom
was a glibc abort inside an unrelated `free()`, kernels away from the cause. The compiler had named
it at every build; nothing read the output.

Split the diagnostics in two first -- a reader drowning in style findings stops reading. **UB class,
gate on these, zero tolerance:** `uninitialized`, `maybe-uninitialized`, `sometimes-uninitialized`,
`array-bounds`, `stringop-overflow`, `free-nonheap-object`, `nonnull`, `return-type`,
`sizeof-pointer-memaccess` -- each means the program has no defined meaning and the optimizer is
entitled to anything. **Style class, never gate:** unused variable, shadowed name, naming.

## Gate 1: the compiler you already run
```sh
g++ -c -o /dev/null -O2 -Wall -Wextra -std=c++23 \
    -Werror=uninitialized -Werror=maybe-uninitialized -Werror=array-bounds \
    -Werror=stringop-overflow -Werror=free-nonheap-object -Werror=nonnull \
    -Werror=return-type -Werror=sizeof-pointer-memaccess kernel.cpp
```

**The `-std=` is the harness's, not a habit.** `c++23` and `c17`, from `hpcagent_bench/envs/compilers.yaml`
-- analysing at a standard the build never selects analyses a translation unit that never ships.

Clang spells a subset: drop `maybe-uninitialized`, `stringop-overflow`, `free-nonheap-object`, add
`-Werror=sometimes-uninitialized`. An unknown `-W` name is only a warning to clang, so an unpruned
list gates on less than you think.

**`-O2` is load-bearing.** Measured on gcc 15.2: same TU, same flags, `-O0` reported NOTHING and
`-fsyntax-only` nothing; `-O2` reported `maybe-uninitialized` plus four `array-bounds`. That dataflow
runs only under optimization, so analysing a debug build at its own `-O0` is the failure mode.
**Match tags by prefix**, too: gcc 15 prints `[-Warray-bounds=]`, trailing `=`, and clang printed
`[-Wsometimes-uninitialized]` where the grep wanted `[-Wuninitialized]`. An exact-string filter
drops the diagnostic and the run reads clean.

## Gate 2: deep analysis FOLLOWS the compiler

gcc build gets `-fanalyzer`, clang build gets the LLVM analyzer, so the analysis matches the
toolchain that made the binary -- the other one's model of your flags is a guess.
```sh
gcc -c -o /dev/null -fanalyzer -std=c17 -Werror=analyzer-use-of-uninitialized-value \
    -Werror=analyzer-possible-null-dereference -Werror=analyzer-out-of-bounds \
    -Werror=analyzer-malloc-leak kernel.c   # also: -use-after-free, -double-free
```

The GCC manual, still at 15.2: "The analyzer is only suitable for use on C code in this release."
Measured, it does run on C++ and reported the uninitialised extent -- but on the identical C file it
also found two possible-null dereferences it missed in C++, so on C++ it is a bonus, not the gate.
It does not want `-O` either: some warnings are documented as unlikely to fire under optimization,
the opposite of gate 1 -- run it separately.

On clang, reach the analyzer through clang-tidy rather than `clang --analyze`: same engine, measured
identical findings, but `--analyze` exits 0 with findings and clang-tidy can be made to exit 1.

```sh
clang-tidy --quiet --header-filter= --system-headers=false --warnings-as-errors='*' \
  --checks='-*,clang-analyzer-core.*,clang-analyzer-unix.*,clang-analyzer-deadcode.*,bugprone-integer-division,bugprone-misplaced-widening-cast,bugprone-sizeof-expression,bugprone-undefined-memory-manipulation' \
  kernel.cpp -- -std=c++23 -O2
```

**The compile database is where people get stuck.** Everything after `--` is the compile line for
that one file. For a project drop the `--` and point at the build --
`cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`, then `clang-tidy -p build src/kernel.cpp`; that
variable works only with the Makefile and Ninja generators. Wrong flags, wrong program, wrong findings.

`--checks` is an allowlist over `-*`, every exclusion with a written reason. Absent on purpose:
`readability-*`, `modernize-*`, `cppcoreguidelines-*` -- style verdicts, and on generated code
`bugprone-reserved-identifier` fires on every `__i` loop counter. Never `--fix`. `--header-filter=`
(empty) drops diagnostics from headers you do not own, so a vendored one cannot bury your file.
**A misspelled check is silent:** measured, `--checks='-*,clang-analyzer-core.*,bugprone-integer-divison'`
runs, reports the core findings, exits 0 -- the typo'd check just never existed, and only when EVERY
name is bad do you get `Error: no checks enabled` and exit 1. Print
`clang-tidy --list-checks --checks='<your list>'` once and read what is actually on.

**cppcheck is a second opinion from a different engine** -- not a compiler, not tied to one, so it
disagrees with both. On the test file it found the out-of-bounds store, the leak and the
null-on-allocation-failure in one pass. Give it the `-I`/`-D` that matter; the two suppressions are
its noise about the ones it cannot resolve, plus its own coverage nag.
```sh
cppcheck --enable=warning --check-level=exhaustive --inline-suppr --error-exitcode=2 \
  --suppress=missingIncludeSystem --suppress=checkersReport --quiet kernel.cpp
```

## What they CANNOT find, or a clean run reads as proof
Measured on one 30-line file: gcc 15.2, clang 21.1.8, cppcheck 2.19.

| bug | gcc `-Wall -Wextra -O2` | clang `-Wall -Wextra -O2` | clang analyzer | cppcheck |
|---|---|---|---|---|
| `new double[uninit_extent]` | yes | yes | yes | no |
| loop stores past `double a[4]` | yes, 4 iterations | **no** | no | yes |
| leaked `malloc` | no | no | yes | yes |
| unchecked null from `malloc` | no | no | yes | yes |
| `2147483600 + argc*100` | **no** | **no** | **no** | **no** |

Clang's `-Warray-bounds` is a frontend check on constant subscripts and a loop index is not one, so it
is silent where gcc's optimizer pass names four out-of-range iterations. The signed overflow, textbook
UB, was missed by all four -- clang-tidy included, on the full `clang-analyzer-*,bugprone-*` set.
**And they invent bugs.** Measured on a provably-correct kernel:
```c++
double *b = new double[n];
for (int i = 0; i < n; ++i) b[i] = 0.0;
out[0] = b[0];  // warning: Assigned value is uninitialized [clang-analyzer-core.uninitialized.Assign]
```
The analyzer walks the path where the loop runs zero times, and every symbolic-extent loop has one, so
on numeric code this fires everywhere. `if (n <= 0) return;` silenced it; gcc and cppcheck never
reported it. Chase a finding to a concrete input or discard it -- never "fix" what it could not prove.

## Sanitizers: the complement, not the competitor

A static analyzer proves absence badly, as the table shows; a sanitizer proves presence exactly, on the
one path your input took. Reach for one when the static tools are clean and the program still
misbehaves, when a finding on a symbolic extent needs a witness, or when the bug class is arithmetic.
```sh
clang++ -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
    -fno-sanitize-recover=all -o run.bin kernel.cpp main.cpp && ./run.bin
```
- **ASAN**: overruns, use-after-free, leaks at exit. Roughly 2x slower, 3x memory.
- **UBSAN**: signed overflow, bad shifts, misaligned access, bad casts. Without
  `-fno-sanitize-recover=all` it prints, continues, and still exits 0.
- **MSAN**: reads of uninitialised memory. Needs every TU instrumented, C++ runtime included -- an
  uninstrumented libstdc++ gives false reports. Not combinable with ASAN.

Measured trap: that same binary with those same bugs, run with `n=3`, printed nothing and exited 0.
The faulty branch needs `n<=0`. **A sanitizer given the wrong input is not evidence.**

## Exit codes, and a missing tool
| invocation | findings present | exit |
|---|---|---|
| `g++ -Wall -Wextra` / `clang++ --analyze` / `clang-tidy` / `cppcheck` | yes | **0** |
| `g++ -Werror=<tag>` / `clang-tidy --warnings-as-errors='*'` | yes | 1 |
| `cppcheck --error-exitcode=2` | yes | 2 |
| ASAN/UBSAN binary, fault reached | yes | 1 |

Every tool defaults to zero, so a CI step that runs one and tests `$?` is green forever. And check the
tool exists first -- `command -v clang-tidy >/dev/null || { echo "no clang-tidy" >&2; exit 1; }`.
Degrading to "no findings" is how the ICON bug survived: the report said clean when it meant absent,
and downstream those are indistinguishable. A tool that is optional on a host must say WHY its section
is empty -- "clang-tidy: not installed on this host, no findings collected" -- never render an empty pass.

## Documentation
- GCC warning options, and the exact spelling of every `-W` above -- https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html
- GCC static analyzer options: the full `-Wanalyzer-*` list and the C-only caveat -- https://gcc.gnu.org/onlinedocs/gcc/Static-Analyzer-Options.html
- Clang diagnostics reference, for which `-W` names clang actually has -- https://clang.llvm.org/docs/DiagnosticsReference.html
- clang-tidy check list with per-check docs -- https://clang.llvm.org/extra/clang-tidy/checks/list.html -- and the compile database format, if you build one by hand -- https://clang.llvm.org/docs/JSONCompilationDatabase.html
- Clang analyzer checkers: what each `core.*`/`unix.*`/`security.*` checker models -- https://clang.llvm.org/docs/analyzer/checkers.html
- Cppcheck manual: severities, suppressions, `--check-level` -- https://cppcheck.sourceforge.io/manual.pdf
- Sanitizer flags and runtime options -- https://clang.llvm.org/docs/AddressSanitizer.html and https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html
