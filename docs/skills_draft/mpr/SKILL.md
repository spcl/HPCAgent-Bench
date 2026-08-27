---
name: mpr
description: "Render a kernel's DaCe SDFG as ONE self-contained, already-parallel C/C++ file: how to run it, what MPR refuses, and why the work left is specialization."
---

# mpr

A REPO tool, not an optimization page. MPR (maximal parallel rendering) turns a kernel's SDFG into
one translation unit that builds with a bare host compiler -- no `-I`, no `libdace`, no BLAS -- and
computes what the SDFG computes. What comes out is the CANONICAL form: every loop DaCe could prove
independent is already an OpenMP region -- a parallel baseline to read, diff, and specialize.

## Invoke

The tool is a subcommand of the benchmark CLI. The installed console script is `hpcagent-bench`
(its own usage line prints `agentbench`, which is only the parser's program name -- typing
`agentbench` finds nothing):

```bash
hpcagent-bench mpr --kernel arc_distance --out /tmp/mpr             # C++20, fp64
hpcagent-bench mpr --kernel arc_distance --out /tmp/mpr --language c --precision fp32
hpcagent-bench mpr --track scientific_computing --out /tmp/mpr --jsonl /tmp/mpr.jsonl
python -m hpcagent_bench.cli mpr --kernel arc_distance --out /tmp/mpr   # same, no install needed
```

The C render targets **C23** and the C++ render **C++20**: the emitted loops declare their
induction variable with `auto`, so `-std=c11`/`-std=c17` is a hard error
(`type defaults to 'int' in declaration of '_loop_it_7'`). A bare `gcc` happens to work only
because current gcc already defaults to `gnu23` -- build with `-std=c23` explicitly.

Exactly one of `--kernel` / `--track` is required, as is `--out`. Exit status is 1 only when a
render FAILED or timed out -- a refusal is a result, and exits 0. In process, with the SDFG:

```python
from hpcagent_bench import mpr_bridge
from hpcagent_bench.spec import BenchSpec
record = mpr_bridge.render_kernel(BenchSpec.load("arc_distance"), "/tmp/mpr", language="c")
```

Four steps, each somebody else's code: `autogen.emit_targets` writes the `<module>_dace.py`
sibling, its `@dace.program` parses to an SDFG, `canonicalize` + `finalize_for_target` produce the
parallel CPU form, `dace.codegen.mpr.render` emits the text. It runs in a CHILD PROCESS -- the DaCe
frontend wedges on some kernels, and a sweep must lose the kernel, not the sweep. Outputs land
beside each other: `<short>_<fptype>_mpr.{c,cpp}` and `<short>_<fptype>_mpr_binding.json`.

## The parallelism is already found -- your job is SPECIALIZATION

- **Do not re-parallelize.** Every `#pragma omp parallel for` in the file is a dependence DaCe
  PROVED absent; adding your own on top competes with a decision already made correctly.
- **A sequential loop is a claim, not an oversight.** It means a carried dependence -- a scan, a
  reduction into a scalar, an in-place update -- or a canonicalization gap. Establish which before
  touching it; if it is a genuine gap, the fix belongs in the DaCe pass, not in the emitted C. A
  render with NO pragma AT ALL is different again: correct and entirely sequential, meaning the
  schedules were lost between the SDFG and the text. Distrust it, and conclude nothing about
  canonicalization from it.
- **What is left is the machine.** Tile and block for cache, vectorize the innermost loop, pick a
  `schedule(...)` for load imbalance, `collapse(...)` a too-short outer loop, fuse adjacent regions
  for locality, fix layout so the parallel axis is not the strided one. Maximal parallelism gives
  none of that, and all of it is what makes the parallel form fast.
- **Measure against the render, not against serial.** It is the parallel baseline. A speedup over
  a sequential reference proves nothing about whether your specialization helped.

`<array>_idx` helpers spell the index arithmetic; a `reduction(...)` clause rather than an atomic
means the WCR folded; the preamble lists exactly which runtime helpers this kernel reached.

## The binding is not the native one

The entry is `<short>_<fptype>_mpr`, deliberately NOT the native `entry_symbol`. MPR's argument
list is the SDFG's -- arrays first then scalars, each group sorted, plus the free SYMBOLS the C
emitter never passes -- so a caller assuming the native ABI passes the right pointers in the wrong
order and gets numbers. Read the order out of the `*_mpr_binding.json` beside the source; its `abi`
field says `mpr/1` for exactly this reason.

A written scalar arrives as a length-1 ARRAY: MPR promotes it, because a C entry cannot return.

## What it refuses, and the rewrite

MPR refuses loudly and names the construct. Every refusal is a real gap in the single-host-unit
promise, so the answer is to change the SDFG, never to force the render:

| refusal names | why | rewrite |
|---|---|---|
| a GPU/FPGA schedule or storage | needs another compiler | render the CPU form: `canonicalize(target='cpu')`, no offload pass |
| `dace::Stream` | a lock-free queue with no standalone spelling | put the producer and consumer around an array |
| a consume scope (`dace::Consume`) | queue plus quiescence detection | express the work as a map |
| a vector WCR (`dace::vec<T,N>`) | a SIMD element type is a runtime template | reduce on the scalar element type |
| a Scalar returned BY VALUE | the entry signature cannot pass it back | make the output a length-1 array |
| `dace::CopyND` | a runtime copy helper | insert explicit copies before rendering |
| two translation units | the single-file contract | turn off the split-translation-unit codegen parameters |
| `could not expand <node> after N rounds` | the library node has no pure implementation | give it one, or replace the node with the map it stands for |

If the message instead says the result is "not self-contained" and quotes a line, that is MPR's own
gate on the FINISHED text: something leaked that no table lowered -- a bug in the lowering, not in
your SDFG. Report it with the quoted line.

## Three rules that decide whether the numbers match

1. **Declared precision applies to ACCUMULATION too.** `--precision fp32` renders every buffer AND
   every temporary at fp32; an untyped allocation that defaults to fp64 is the usual leak. Compare
   against a numpy reference computed in the same precision, not against an fp64 oracle.
2. **Buffers are ROW-MAJOR.** The descriptors are C-order and the emitted index helpers assume it.
   A Fortran-ordered input array produces a clean build and wrong numbers.
3. **Name the manifest SYMBOL, never `arr.shape[k]`.** The symbol is what reaches the entry
   signature; a shape read folds to a constant at parse time and the rendering stops being
   parameterised in the extent you meant to sweep.

## Documentation
- DaCe manual: SDFG semantics, transformations, and what `canonicalize` is doing before MPR runs -- https://spcldace.readthedocs.io/en/latest/
- DaCe source, which is the authority on the renderer itself (`dace/codegen/mpr/`) and on any construct it refuses -- https://github.com/spcl/dace
- OpenMP API specification, for the pragmas the rendered file carries -- https://www.openmp.org/specifications/
