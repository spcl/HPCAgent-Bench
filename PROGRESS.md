# release-v0.1 cleanup: progress and design changes

Running log of the unbloat/unslop/registry work on `release-v0.1`. Deleted before the release PR merges.

## Decisions

- No package rename; `hpcagent_bench` stays.
- Registries move to decorators on one shared `hpcagent_bench/registry.py`: frameworks, kernels (yaml and
  `@kernel` interchangeable), syntax checks + MCP tools, harnesses. Adding one = files in one folder + a
  registration.
- Yaml keeps only what code reads (experiment tags, input/output names, ...); unread entries such as
  taxonomy go.
- De-duplication and design pass: everything, including high-risk items, with behaviour kept identical
  (translators proven against a golden corpus of emitted code).
- Code matches the paper (HPCAgent_Bench, ICLR 2027 submission): anything can be registered with a
  decorator, including anti-cheat measures. Job launch collects every registered measure and records the
  active set with the run.
- CI toolchain: GCC >= 15 is enforced by `scripts/checks/verify_toolchain.py` (CI installs GCC 16).

## Checkpoints

| # | Commit | What |
|---|---|---|
| 1 | `1ae90de` | Merged CI fixes: translator test stand-ins carry `dtype=None`; gemm score accepts the c-autopar fallback. |
| 2 | `eedbe2b` | mi200 arms: a hosted model needs only `partition-mi200.env`; a served model needs its own `partition-mi200-<model>.env`. GCC >= 15 toolchain gate. |
| 3 | `9121285` | Dead code found by vulture removed (~90 lines). Most vulture hits were false positives (kernel entry points loaded by name, FastAPI routes, http.server overrides, CLI enum choices, TYPE_CHECKING imports). |
| 4 | `1fddcc1` | Docs: one owner per topic. Merged and deleted `perf_protocol`, `job_submission`, `owed_and_checkpointing`, `AMD-SUBMISSION`, `docs/README`, `local_coding_agents`. README 286->169, CONTRIBUTING 325->146 lines; mi200 serving docs removed; stale `.env.base-*`, sbatch and script references fixed; 0 broken links (was 5). Markdown 10,474 -> 8,911 lines. |
| 5 | `2cdd441` | Translator bug fixes with tests: C OpenMP variants declare the pinned config constants their body reads; Fortran contained helpers take integer kinds from `implicit_int_kinds` like the kernel body. |

## Stopped (usage limit), saved in `handoff/`

Registries, kernels, anti-cheat and the translator de-duplication were stopped mid-task and are not merged.
Their work is saved as patches in `handoff/`; `HANDOFF.md` has each patch's state, how it applies to
release-v0.1, and the order to finish them.

## Queued

1. Shared utilities: `util/` for env files (~12 parsers), read-only SQLite (~20 opens; fixes `?`/`#` path bug),
   coercion helpers, repo paths, `git HEAD`, HTTP JSON + polling (16 sites); one test loader fixture.
2. Deletions: one-shot porting scripts; tests that only assert deleted symbols are gone.
3. Module splits: `agent_driver.py` (3.3k lines) into a package; `scoring.py` distributed/ML split and
   `graded_score` stages; `remaining_kernels.py` into `owed.py`; lazy CLI subcommands; importable
   `experiments` (drops `sys.path` hacks).
4. Config knobs: every `HPCAGENT_BENCH_*` read through `config`, prune single-use knobs, one kernel timeout;
   `run_cluster.sh` topology/port math into Python.
5. Comment and docstring trimming in the heaviest files; stale script references left in source (`finalize_grade_owed.py` account_env.sh, `remaining_kernels.py` submit-llrblind.sh, `install_dace.sh` rebuild_venv.sh, `judge_web_search.py --env-file` help).
