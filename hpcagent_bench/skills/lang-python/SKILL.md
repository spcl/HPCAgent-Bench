---
name: lang-python
description: "Writing correct modern Python for this harness: type hints, explicit conversion, and the gate ladder."
---

# lang-python

Two jobs: (A) QUALITY-CHECK an existing Python file through the gate ladder;
(B) enforce modern Python (>= 3.10) idioms + this repo's house rules when WRITING
Python. `<file>.py` is the placeholder for the target throughout -- swap in the
real path. Every command is copy-pasteable.

## Golden rule

**All gates run. Warnings are errors. Type errors are errors.** A clean pass =
zero diagnostics from yapf (`--diff` shows nothing), ruff, pyright (or mypy), and
the warnings-as-errors smoke, **plus** a clean `pre-commit run` and green pytest
consumers. Do not report "looks good" until every gate is green. Fix findings at
the source -- never silence a warning, `# type: ignore`, or `# noqa` to pass
(a targeted `# noqa: CODE` with a reason is allowed only for a genuine
third-party/false-positive, same discipline as the C++/Fortran skills).

Tools used: `yapf`, `ruff` (with `pyflakes`/`flake8` as fallbacks), `pyright` or
`mypy` for the type gate, `pre-commit`, `pytest`. Probe what's actually available
before running (`ruff --version`, `pyright --version`, etc.) and adapt. If a tool is
absent, run its gate where the project provides it (repo config / CI) and report that
gate as DEFERRED -- never skip silently, and **do NOT `pip install` anything** to make
a gate pass. Use `python` (>= 3.10); if the project pins an interpreter (a pyenv venv,
a `.python-version`), use that one.

## A. The gates (run in this order)

### 1. yapf -- format first, in place (column 120)
yapf auto-discovers a project `.style.yapf` / `setup.cfg [yapf]` /
`pyproject.toml [tool.yapf]` at or above the file; the explicit `--style` below is
the fallback used only when none exists (house default: pep8 base, 120 columns).
```bash
cfg="$(dirname <file>.py)/.style.yapf"
[ -f "$cfg" ] || cfg="$(git -C "$(dirname <file>.py)" rev-parse --show-toplevel 2>/dev/null)/.style.yapf"
if [ -f "$cfg" ]; then
  yapf -i --style="$cfg" <file>.py
else
  yapf -i --style='{based_on_style: pep8, column_limit: 120}' <file>.py
fi
```
yapf is the established formatter here -- do NOT switch to black or ruff-format;
either would reflow the whole tree to a different style. To CHECK without editing
(the form the golden rule scores), use `--diff` -- it exits non-zero if anything
would change:
```bash
yapf --diff --style='{based_on_style: pep8, column_limit: 120}' <file>.py
```

### 2. ruff -- lint (fast: unused imports, undefined names, bugbear, pyupgrade)
```bash
ruff check --line-length 120 <file>.py
```
Stronger, explicit rule set (recommended when the repo has no `ruff` config of its
own): pyflakes + pycodestyle + bugbear + comprehensions + pyupgrade + simplify:
```bash
ruff check --select E,F,W,B,C4,UP,SIM --target-version py310 --line-length 120 <file>.py
```
**Always pass `--line-length 120`** unless the repo's own `ruff` config sets it.
ruff defaults to **88**, while gate 1 formats at **120** -- so the two gates disagree
and every line yapf just produced between 89 and 120 columns comes back as a wall of
`E501`. That is a bug in the invocation, not in the file: read the codes before
reflowing anything, and if they are all `E501`, re-run at 120 first.
`--fix` applies the autofixable subset (re-run yapf after). If `ruff` is absent,
fall back to `flake8 --max-line-length 120 <file>.py`, or at minimum
`pyflakes <file>.py` -- these catch unused imports and undefined names but far less
than ruff. flake8's default is **79**, tighter still than ruff's 88, so the same
width caveat applies with more force; `pyflakes` has no width check at all.

### 3. Type check -- pyright (strict) and/or mypy (strict)
The strong correctness gate. Treat every type error as a failure.
```bash
pyright <file>.py                 # honors pyrightconfig.json / [tool.pyright]; add --strict for full strict mode
mypy --strict <file>.py           # alternative / second opinion
```
If neither `pyright` nor `mypy` is on `PATH`, run this gate the way the project
provides it -- many repos configure pyright via `pyrightconfig.json` /
`[tool.pyright]` (driven by the editor's bundled pyright or a repo dev-dep) or run
mypy in CI. So run it from inside the repo that provides it; if the target repo
configures neither, this gate is DEFERRED -- say so loudly in the report rather than
skipping silently, and do NOT `pip install`/`npm install` a checker to force it.

### 4. Warnings-as-errors import / compile smoke
Surface `Deprecation`/`Syntax`/`Resource` warnings as hard errors, and catch any
import-time or byte-compile failure.
```bash
python -W error -m py_compile <file>.py                 # SyntaxWarning + byte-compile, no execution
python -W error -c "import package.module"              # import path -- runs module top-level with warnings fatal
```
Use the interpreter the module's dependencies require (a project pyenv venv /
`.python-version` if it pins one); prefer plain `python` in scripts and switch only
when a version-specific dependency forces it. `python -We <file>.py`
executes the file directly with warnings fatal -- use it when the file IS a runnable
script rather than an importable module.

### 5. pre-commit -- the user runs this on EVERY touched file
```bash
root=$(git -C "$(dirname <file>.py)" rev-parse --show-toplevel 2>/dev/null)
if [ -f "$root/.pre-commit-config.yaml" ]; then
  pre-commit run --files <file>.py
else                                             # silence here would read as a green gate
  echo "pre-commit: DEFERRED (no config)" >&2
fi
```
Standing mandate: yapf + pre-commit on every file you touch, no exceptions. If a
new import was added, ensure the dep is declared (e.g. `setup.py`/`pyproject.toml`)
so the hooks and CI resolve it. A failing hook is a failing gate -- fix the code,
do not `--no-verify`.

### 6. Tests -- run the file's pytest consumers
Tests are consumers, not dead code: exercise whatever imports/covers this file.
```bash
pytest -q --maxfail=10 path/to/test_<thing>.py           # the matching test module(s)
pytest -q --maxfail=10 -k "<thing>" path/to/tests/        # or select by keyword
```
Run from the repo root so the package prefix (`from pkg.sub import ...`) resolves --
never `sys.path` hacks. `--maxfail=10` per house policy. Green == every consumer
passes; a new warning during the run is a failure too (zero-warning policy).

**Report** each gate's status. Only "clean" when 1-6 all pass with zero output
(and note explicitly if gate 3 was deferred for lack of an in-repo checker).

## B. Writing modern Python (>= 3.10, no OO bloat)

Decision ladder first (KISS/YAGNI): does it need to exist? -> in the codebase
already? -> stdlib? -> native? -> installed dep? -> one line? -> else the minimum that
works. Prefer plain **functions + small dataclasses** over class hierarchies,
factories, or indirection. New code is a liability. Then apply:

- **Type hints ALWAYS.** Every function signature -- every parameter and the return
  -- and every non-trivial local. Modern 3.10+ syntax: `X | None` (PEP 604), not
  `Optional[X]`; `list[int]` / `dict[str, int]` / `tuple[int, ...]`, not
  `typing.List`/`Dict`/`Tuple`. Reach into `typing` only for what has no builtin
  form (`Callable`, `Protocol`, `TypeVar`, `Iterable`, `Self`, `Literal`).

- **No implicit conversions -- convert EXPLICITLY.** Don't lean on Python's silent
  coercions: wrap with `int()` / `float()` / `str()` / `bool()` at the point a type
  changes, and use `//` (not `int(a / b)`) when you want integer division. Never use
  `bool`/`int` interchangeably (`True + 1`), and prefer explicit comparisons
  (`if n != 0:`, `if s is not None:`) over bare truthiness when the intent is a
  specific check, not "is it falsy". Keep numeric kinds consistent in hot loops (no
  int<->float churn). The strict type checker (gate 3) is what enforces this -- it flags
  implicit `Any`, incompatible assignments, and int/float/None mismatches; fix them
  with an explicit conversion or a corrected annotation, never a `# type: ignore`.

- **Imports top-level and absolute.** All imports at module top. Absolute,
  package-qualified (`from pkg.sub.mod import fn`) -- **never** relative
  (`from .x import y` / `from ..pkg import z`). A function-local/deferred import is
  allowed ONLY to break a genuine import cycle or to defer a heavy optional
  dependency -- and then it carries a one-line comment saying which. Do NOT run
  `python -c "<script>"` for real work; write a `.py` file with top-level imports.

- **Static attribute schemas + `__slots__`.** Classes declare every attribute up
  front; no dynamically added/removed attributes. Use `__slots__` (for a dataclass:
  `@dataclass(slots=True)`, Py3.10+, NOT a hand-written `__slots__` -- field
  defaults break otherwise) so attributes are fixed and typos become `AttributeError`
  at write time. The win is **memory + cache locality**, not raw attribute speed on
  3.11+ (that edge is ~gone by 3.13). Skip `__slots__` (with a one-line reason)
  where it is unsafe: a non-slotted base reintroduces `__dict__`, the class is a
  mixin, monkey-patched, or weakref'd/pickled in a way slots break.

- **No optional attributes -- sentinel-default slots.** Every attribute is a
  declared slot ALWAYS assigned in `__init__`. A logically-absent one gets a
  module-level `SENTINEL = object()` (or `None`) default; code DETECTS the default
  (`if obj.attr is SENTINEL`) and never asks whether the attribute exists. Direct
  attribute access everywhere.

- **No `getattr` / `hasattr`** for control flow -- attributes are known statically.
  Substitution ladder (house rule, `getattr`/`hasattr` are banned outright):
  - static slotted schema -> direct access (`obj.attr`), sentinel to mark "unset";
  - type/shape/capability probe -> `isinstance(x, np.ndarray)` then `.ndim`
    directly, never `hasattr(x, "shape")`;
  - genuinely optional attr on an object with a real `__dict__` (e.g.
    dynamically-set AST-node attrs) -> `vars(obj).get("name", default)`;
  - dynamic member of a module by name -> `vars(mod)["name"]`.
  `vars()` sees only the instance `__dict__` -- unsafe for class attrs, properties,
  slots, or base-class attrs (and raises on `__slots__` objects). The ONE kept
  exception: C-extension objects with no `__dict__` (tree-sitter etc.) -- `getattr`
  stays there. Note the tension with EAFP: `try/except AttributeError` for a
  genuinely optional external interface is a last resort only, because
  exceptions-as-control-flow is itself discouraged (below) -- prefer `isinstance` /
  `vars().get()` / a sentinel slot.

- **`functools.lru_cache(maxsize=..., typed=True)` -- always `typed=True`.** Never
  bare `@lru_cache`/`@lru_cache()`, never `@functools.cache` (it is
  `lru_cache(maxsize=None)` with no `typed`; for unbounded write
  `@functools.lru_cache(maxsize=None, typed=True)`). Untyped collapses `1`, `1.0`,
  `True` -- and `numpy.float32(x)` vs a Python float -- onto one key: in dtype/symbol/
  sympy-keyed code that silently returns a value computed for the wrong type, a
  miscompile not a nit. Also: **never `lru_cache` a sympy object or any mutable
  object** (equality/hash ignore dtype metadata / the object mutates -> stale entry).

- **No leading-underscore names.** Never prefix a function, class, or module with
  `_` -- public names everywhere; there is no "genuinely-private helper" carve-out
  (this house rule overrides PEP 8's `_private` convention). Module-level DATA
  constants prefer public names too; the hard rule is functions/classes/modules.

- **Cheap checks before expensive.** Order `and`/`or` chains and guard clauses
  cheapest-first so short-circuit skips the costly term. Cheap = int/float compare,
  identity test, `len()`, small set/dict membership. Expensive = a numpy call, regex,
  isinstance chain, attribute walk, `str` build, anything touching filesystem/
  imports/re-parse. `if name in KNOWN and expensive_probe(name)`, never the reverse.
  Only where semantically equivalent -- a `None` check that makes a later attribute
  read safe must stay first. When adding a guard to an existing chain, find the
  cheapest position that is still correct, don't just append.

- **No hardcoded paths.** No absolute literals (`/home/...`), no `sys.path.insert`,
  no `importlib` file-loading. Current interpreter -> `sys.executable`; repo root ->
  `Path(__file__).resolve().parents[N]` (or `git rev-parse --show-toplevel`, or an
  env/config var); scratch -> `tempfile`; fixtures -> a package-relative import.

- **Formatting: yapf, column 120, no nested f-strings.** Never reuse the outer
  quote char inside an f-string expression -- `f"{d['k']}"`, never `f"{d["k"]}"`:
  yapf's bundled parser can hard-crash on PEP 701 same-quote nesting, and keeping
  f-strings un-nested is clearer regardless of tooling.

- **Perf patterns** (apply ONLY in a proven hot path -- per-node/edge/item/match --
  and only when behavior-preserving; cold code stays readable):
  - local-alias a bound method reused across thousands of iters (`append = lst.append`);
    cache a deep attr leaf before the loop (`fn = obj.a.b.c`), never `obj.a.b.c()` per iter;
  - comprehensions over manual `.append` loops (LIST_APPEND in C);
    `list(filter(None, data))` over `[x for x in data if x]` when dropping falsy;
  - hoist loop-invariant work out of the loop; don't re-derive what the caller
    already has; avoid `list.index()` / linear identity scans (they go quadratic) --
    keep an `{id(obj): index}` map beside the list;
  - membership: frozen-literal `x in {1, 2, 3}` / `x in (1, 2, 3)` is fine, but
    **never** swap an ordered list/dict -> set to speed membership where iteration
    order is observed (that is a determinism bug);
  - never probe a `defaultdict` with `d[k]` (it INSERTS) -- use `.get(k)` / `k in d`;
    `if seq:` over `if len(seq) != 0:`; `x.keys().isdisjoint(y)` over
    `len(x.keys() & y) > 0`;
  - don't deepcopy a result the callers discard; give a hot class a custom
    `__deepcopy__`; reset a regenerable cache to its `__init__` state, don't deepcopy it;
  - sympy: cheap structural `a == b` before `(a - b).simplify()`; use `.is_*`
    assumptions, `xreplace` over `subs`, the cached `symbolic.simplify`; never
    `simplify()` in a loop.
  - Do NOT cargo-cult dead folklore on 3.11+: `__slots__`-for-speed, "cache builtins
    into locals", and "`s += x` in a loop is always quadratic" are obsolete (`+=` is
    linear on CPython -- reach for `"".join(...)` only when slicing the accumulator
    defeats the in-place fast path). Don't let exceptions fire in normal control flow
    (creation cost); don't override `__getattribute__` unless proxying.

- **KISS / YAGNI / no OO bloat.** Prefer functions + small dataclasses
  (`@dataclass(slots=True)`, add `frozen=True` where immutable) over deep hierarchies.
  Reuse existing utilities and stdlib before writing new. No speculative generality,
  no config knobs "just in case", no single-call-site abstractions. Delete dead code.

- **Comments/docstrings: zero fluff.** Explain only the non-obvious **why**, never
  the **what**; never restate signatures or types; single line where possible.

After writing or modernizing, run all gates in section A on the result.
