# DESIGN: one word, one meaning -- kill `variants:`

## Problem

Four unrelated things are called *variant*. Measured, not guessed:

| name | what it is | where |
|---|---|---|
| `variants:` manifest block | legacy sparse: one entry = one matrix format + one distribution | 7 manifests |
| `hidden.VARIANTS` | held-out input rotation h1..h5, the correctness gate | `support/distributions/hidden.py` |
| `variant_spec` | the dict `variants:` feeds into `initialize` and into hand-written init funcs | `initialize.py:133`, 11 kernel `.py` |
| ~~`SCORED_VARIANTS`~~ | **DONE** — now each flavor's `pipelines`, and the word "variant" is gone from the DaCe side | `framework.py` FRAMEWORK_META, `dace_framework.DEFAULT_PIPELINES` |

Only the second one is modern and documented. The rest are the same word wearing
other jobs.

## The live bug this causes

6 of the 7 `variants:` manifests ALSO declare `configurations:`. `expand_layouts`
(`spec.py:1464`) returns on the `configurations` branch and never reads `variants:` --
but `Benchmark.get_data` (`frameworks/benchmark.py:75`) still reads `info["variants"]`
to pick `variant_spec` and its `distribution` override.

So the LAYOUT comes from one block and the DISTRIBUTION comes from the other. Nothing
checks they agree. `sp_cg` can resolve as `[coo]` while its data is generated from
`csr_uniform`. Two mechanisms, one kernel, no cross-check.

`banded_mmt` is the only `variants:`-only kernel left.

## The modern surface already covers it

- `sparse_layouts:` -- what buffers a format has.
- `configurations:` -- array to format map. The emit-distinct unit.
- `distributions:` -- named runtime distribution, targeting a configuration.
- `config:` -- a dimension drawn independently of size (`selects: branch`).
- `init.arrays[].dist` / `.domain` -- per-array distribution and value domain.

Nothing in `variants:` is unexpressible there. It is duplication, not capability.

## Decision

Reserve *variant* for the hidden rotation. Delete or rename the other three.

1. `banded_mmt`: write its `sparse_layouts` + `configurations` + `distributions` from
   its `variants:` block. One manifest, mechanical.
2. Other 6: delete `variants:`. Their layout set is already what `configurations:`
   says -- prove it by diffing `expand_layouts()` ids before and after. Move each
   variant's `distribution` onto `distributions:`, keyed to its configuration.
3. Delete `spec.variants`, the legacy branch of `expand_layouts`,
   `_legacy_sparse_variants`, `_legacy_sparse_matrix` (both also violate the
   no-underscore rule), and `Benchmark.get_data`'s variant selection.
4. `variant_spec` dies with them. Drop the kwarg from all 11 kernel initializers --
   the 4 non-sparse ones accept it and never read it.
5. `--variant` CLI flag becomes `--configuration`. That is what it selects.
6. ~~`SCORED_VARIANTS` becomes `SCORED_PIPELINES`.~~ **Done**, and better than proposed: the
   set is not a module constant at all any more but a property of the framework FLAVOR
   (`FRAMEWORK_META[...]["pipelines"]`), so `dace_cpu_canonicalize` is one column running one
   pipeline instead of a search whose winner the DB never recorded.

## What must not change

The hidden rotation. `hidden.VARIANTS`, `TIMED_VARIANT`, `--hidden-variant` keep their
names -- after this they are the ONLY variants, so the qualifier can be dropped later
but not in the same change.

## Gate

`expand_layouts()` id set is byte-identical for all 7 kernels before and after, and
preset-S data for each resolved id is bitwise identical. If a kernel's data changes,
the two blocks disagreed and that kernel was already wrong -- record which one was
right before fixing it.
