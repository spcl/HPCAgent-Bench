# Adding an optimizer or framework (no LLM)

Two seams. LLM agents have their own page, [writing_an_agent.md](../writing_an_agent.md).

| You have | Seam | Runs through | Recorded as |
|---|---|---|---|
| a tool that emits C/C++/Fortran source or a C-ABI `.so` per kernel | optimizer (`Agent.solve`) | `agent <name>` | JSONL `agent`, DB `optimizer` |
| a Python-callable backend `<module>_<postfix>.py` | framework column (`Framework`) | `run`, `run-benchmark`, `run-framework`, `run-sparse` | DB `framework` + `flavor` |

A new PyPI dependency goes in a `pyproject.toml` extra, then `python scripts/sync_requirements.py`
(`--check` diffs). Images install the `pyproject.toml` hardware extra, so rebuild them after.

## A. Optimizer

Change one file: `hpcagent_bench/harness/optimizers.py` (a subclass plus an `optimizer_registry()`
entry). `cli._agent_registry()` merges that dict, so the CLI needs no edit.

`LibraryOptimizer` fits a tool that produces source: `_deliver` returns the source in `restricted`
mode and builds and submits a `.so` in `any` mode. Subclass `Agent` directly only for another
delivery (`NoOpMPIOptimizer`). The smallest real optimizer:

```python
class NoOpOptimizer(LibraryOptimizer):
    name = "noop"

    def solve(self, task: Task, prompt: str = "", budget: Optional[int] = None) -> Submission:
        source = reference_source(task)
        return self._deliver(task, source)
```

Register it with `NoOpOptimizer.name: NoOpOptimizer` in `optimizer_registry()`.

- `task` carries `kernel`, `language`, `source_mode` and `residency`. Symbol and argument order come
  from `binding_from_spec(BenchSpec.load(task.kernel))` and `gen_call_stub`.
- Unsupported kernel or language: raise `NotImplementedError` with the reason
  (`BlasReductionOptimizer.solve`); the row becomes `status="agent_error"`.
- Extra `-I`/`-D`/`-L`/`-l` go in `Submission.build`; the judge owns `-O3` and `-march`.
- `name` is the recorded identity (JSONL `agent`, DB `optimizer`, `HPCAGENT_BENCH_OPTIMIZER` on the
  distributed path). Choose it once.

The harness grades the submission like an agent's: correctness against `--oracle` on public and
held-out inputs, timing against `--baseline`. Rules: [abi_contract.md](../../hpcagent_bench/docs/abi_contract.md),
[numerical_validation.md](../../hpcagent_bench/docs/numerical_validation.md).

```sh
export HPCAGENT_BENCH_RECORD_DB_PATH=$SCRATCH/smoke.db   # on disk, not tmpfs
PYTHONHASHSEED=0 python -m hpcagent_bench agent noop --kernels scaled_add --preset S --repeat 20 --record --run-id smoke
python -m pytest --maxfail=10 tests/test_optimizer_plugin.py
```

Expect `agentbench noop: 1/1 correct, ...`. Keep `--repeat` at 20 or more: the default
`mannwhitney_delta` timing backend refuses fewer samples (`score_error`). Exit status is 0 unless
`--fail-if-none-correct`. Preset `S` checks wiring only; measure at `XL`.

## B. Framework column

| File | Change | When |
|---|---|---|
| `hpcagent_bench/frameworks/framework.py` | one `FRAMEWORK_META` entry | always |
| `hpcagent_bench/frameworks/<base>_framework.py` | adapter class `<Base>Framework` | new `base` only |
| `hpcagent_bench/envs/registry.yaml` | display name appended under `frameworks:` | always |

The `FRAMEWORK_META` entry for Pythran:

```python
"pythran": {
    "base": "pythran",
    "sweep_deterministic": False,
    "full_name": "Pythran",
    "postfix": "pythran",
    "arch": "cpu",
    "precisions": IEEE_PRECISIONS,
},
```

- `base` picks the adapter class, `postfix` the file `<module>_<postfix>.py` beside the NumPy
  reference, `arch` is `cpu` or `gpu`; other precisions record `skip`. `sweep_deterministic` admits
  the column to unjudged batch sweeps (`preflight.DETERMINISTIC_FRAMEWORKS` derives from it).
- An entry with `column` and `flavor` is keyed `<column>_<flavor>` (`check_flavor_registry()`).
- A `base: native` column also declares `language`, plus `emit_language` when it starts from another
  translator's output. Add `FRAMEWORK_COMPILER` in `benchmarks/cpp_runtime.py` for a non-default compiler.
- New base: subclass `Framework` in `<base>_framework.py` (smallest: `pythran_framework.py`);
  `framework_class()` finds it by name. To generate the implementation file, add a target to
  `autogen_targets()` and to `TARGETS`/`_emit_target` in `hpcagent_bench/autogen.py`; otherwise commit
  a hand-written `<module>_<postfix>.py` per kernel.
- `frameworks:` key order in `registry.yaml` assigns colors, so append only (`tests/test_palette.py`).

```sh
PYTHONHASHSEED=0 python -m hpcagent_bench run --benchmark scaled_add --framework <key> --precision fp64 --preset S --repeat 1 --output $SCRATCH/smoke.jsonl
python -m pytest --maxfail=10 tests/test_frameworks.py
```

Expect `"status": "ok"` and `"validated": true` under every implementation in `impls`.
