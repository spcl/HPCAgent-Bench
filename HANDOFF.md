# Handoff: release-v0.1 cleanup

State at handoff, and how to continue. `HANDOFF.md`, `PROGRESS.md` and `handoff/` are working files:
delete all three before the release merges.

## Landed on release-v0.1 (tested)

See `PROGRESS.md` for the checkpoint table. In short:

- CI fixes: translator test stand-ins, gemm c-autopar fallback, mi200 hosted/served arms, GCC >= 15 gate.
- Dead code removed (vulture).
- Docs: one owner per topic, 6 files merged/deleted, stale references fixed (10,474 -> 8,911 markdown lines).
- Two translator bug fixes, each with a test:
  - `numpyto_c`: the OpenMP variants (`emit_c_omp`, `emit_cpp_omp`) dropped the pinned-constant
    declarations, so a body reading a pinned knob named an undeclared identifier.
  - `numpyto_fortran`: a contained helper mapped every non-int64 integer local to int32 (int8/int16 took
    `_c_int32_t` literals and IAND got mismatched kinds); it now uses `implicit_int_kinds` like the kernel.

Verification in this container: tests for touched code pass except environment-only failures (GCC 13 has no
`-std=c23`, no gfortran, the container runs as root so read-only snapshot tests fail). CI uses GCC 16.

## Not landed: saved as patches in `handoff/`

The four workstreams were stopped by the usage limit. Three of them ran in worktrees created from `main`
(`bde5761`), not from `release-v0.1`, so their diffs do not apply cleanly here. None is finished.

| Patch | Base | State | Applies to release-v0.1 |
|---|---|---|---|
| `registries.patch` (52 files) | `main` | Nearly complete: `hpcagent_bench/registry.py` (generic `Registry`, decorator or call, duplicate error, close-match hint, pkgutil autoload); per-base `frameworks/<base>_flavors.py` replacing the `FRAMEWORK_META` literal; derived lists in preflight, wave_board, collect_canon, plot scripts; `containers/agent/tools/syntax_checks/<lang>.py` + `tool_registry.py` (`@syntax_check`, `@tool`); harness registry; registry.yaml trimmed; `tests/test_framework_registry.py`. On `main` it adds 1 real failure (`test_baseline_model::test_numba_baseline_times_the_parallel_njit_build`: no numba baseline timed) plus a stale annotation baseline. | Conflicts in harnesses.py, inference_service.py, autogen.py, registry.yaml, cupy/dace frameworks; verify_image.py and submit-canon-llr40.sh moved/removed on release-v0.1 |
| `anticheat.patch` (16 files) | `main` | Partial, written before the scope correction below: `hpcagent_bench/anticheat/` with `speedup_bound`, `roofline_bound`, `device_code`, `static_source`, `portability`, `synchronization`; `anticheat_cli.py`; hooks in scoring/recording/service/run_cluster.sh; `docs/extending/anticheat.md`; `tests/test_anticheat.py`. CLI hook unfinished. | Conflicts in cli.py, recording.py, scoring.py, service.py; `scripts/cscs/daint_worklist.py` does not exist on release-v0.1 |
| `kernels.patch` (2 files) | `main` | Started: `hpcagent_bench/kernels.py` and an initializer-inference helper in spec.py. No manifest pruning yet. | Small; re-do rather than port |
| `translators-wip.patch` (23 files) | `release-v0.1` | Uncommitted step: shared `numpyto_common/ast_names.py` (rename/substitute transformers) wired into ~20 files. Not verified against the golden corpus. | Applies (same base); verify before committing |

Porting recipe: `git apply --3way handoff/<name>.patch` on a branch from `release-v0.1`, resolve conflicts,
then run the tests listed for that workstream below.

## Design decisions (from the maintainer)

- No package rename.
- One shared `hpcagent_bench/registry.py`; decorators for frameworks, kernels, syntax checks, MCP tools,
  harnesses, anti-cheat measures. Adding one = files in one folder + a registration.
- Kernels: yaml and `@kernel(...)` interchangeable. The decorator is read statically (ast + literal_eval,
  no import) into the same dict `BenchSpec.from_dict` gets from yaml; both for one kernel is an error.
- Yaml keeps only what code reads (experiment tags, input/output names, ...); unread keys such as
  taxonomy go. Audit every `KNOWN_MANIFEST_KEYS` entry against readers before deleting.
- Anti-cheat, matching the paper (§4.2, App. D.1): `@anticheat(name, stage, action)`, one file per
  measure under `hpcagent_bench/anticheat/`, autoloaded; job launch records the active set with the run;
  `hpcagent-bench anticheat list`.
  - A measure is a predicate over one answer's evidence (source, built artifact, runtime observations,
    timings) returning a finding or None.
  - Measures: speedup bound (2000x host / 16,000x device), roofline bound (time below bytes / 2x HBM peak),
    no GPU code in CPU tasks (device runtime mapped, embedded code object, device symbols), static source
    patterns (sleep, timers, file I/O, runtime loading), aarch64 portability probe.
  - Protocol guarantees, NOT measures (document separately): sandbox namespaces (a launch/judge-API
    matter), the two secret seeds and which route uses which, speedup-only score tool, input cycling
    (D=4), the image-has-no-seed check.
  - Grading must stay identical (same rows flagged/refused/credited).
- De-duplication at all risk levels, behaviour identical. Translators are verified by a golden corpus:
  emit every kernel on every backend before and after, and diff.

## Remaining work, in order

1. Port `registries.patch` onto release-v0.1; fix the numba-baseline failure; regenerate the annotation
   baseline (`python tests/test_annotation_ratchet.py --write`).
2. Redo kernels: manifest key audit and pruning across the 702 yaml files (keep `# provenance:` comments;
   `tests/test_attribution.py`), `@kernel`, `docs/extending/benchmark.md` recipe. Prove no change with a
   canonical JSON dump of every BenchSpec before/after.
3. Port `anticheat.patch` with the scope above; switch its local registry to `hpcagent_bench/registry.py`;
   finish `anticheat list` and the launch record; list any paper claim the code does not implement.
4. Translators: verify and land `translators-wip.patch` against a golden corpus, then the rest of the
   audit (BaseEmitter scaffolding, Fortran kernel/helper path merge, `__all__` trimming, dace_emit split,
   shared C/Fortran type oracle).
5. Queued in `PROGRESS.md`: shared `util/` helpers (env files, read-only SQLite with the `?`/`#` URI bug,
   coercion, paths, git HEAD, HTTP JSON), one test loader fixture, module splits (agent_driver, scoring,
   remaining_kernels into owed, lazy CLI, importable experiments), config knobs through `config`,
   `run_cluster.sh` math into Python, comment trimming.

## Environment notes

- Python >= 3.12 is required; this container's default python3 is 3.11: use `/usr/bin/python3.13`.
- Install with `pip install -e .` (base deps); the `[cpu]` extra pulls torch from a blocked index.
- Only `release-v0.1` can be pushed from these sessions; `spcl/HPCAgent-Bench` is not reachable, so PRs to
  spcl are opened from a compare link.
