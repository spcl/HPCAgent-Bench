---
name: pytorch-to-numpy-translator
description: Translate PyTorch KernelBench kernels into NumpyToC-compatible numpy, build or improve the translator under src, generate result/level1 and result/level2 outputs, and write parity tests comparing PyTorch against numpy.
---

# PyTorch to NumPy Translator

## Operating Contract

`CONTRIBUTOR_GUIDE.md` is the compatibility contract for generated numpy: static-shape, buffer-oriented where possible, no `torch` imports, limited to the numpy/control-flow surface it documents.

Each result file is read in isolation by NumpyToC/NumpyToFortran, so it must be a clean, minimal, standalone numpy implementation of that kernel's math. Inline only what the kernel needs; emit no shared runtime imports, helper libraries, or compatibility layers. A numpy-returning form is an acceptable temporary fallback where buffer-form is too hard, tracked in test/status output.

Do not weaken the guide to force a pass. If a PyTorch feature does not fit the guide, stop and explain the missing rule before editing the guide.

## Required Layout

- Translator code under `src/`; parity tests under `test/`.
- Converted kernels under `result/level1/` and `result/level2/`, preserving source filenames.
- Treat the `KernelBench/` submodule sources as read-only upstream.
- Do not keep separate project notes that override this skill or `CONTRIBUTOR_GUIDE.md`.

## Translation Workflow

1. Read a representative sample before changing translator logic.
2. Implement behavior in reusable translator code, not one-off edits to results.
3. Generate minimal standalone numpy results, preferring buffer-form signatures.
4. Run parity tests against the original PyTorch files.
5. Classify failures: unsupported construct, shape/init issue, tolerance, or harness.
6. Improve by level, level 1 then level 2.

## Conversion Rules

- Replace `torch` tensor ops with the `numpy` equivalents in `CONTRIBUTOR_GUIDE.md`.
- Strip autograd-only behavior (`requires_grad`, `.detach()`, `.cpu()`, `.cuda()`, `.to()`, training-only state) unless it changes inference numerics.
- `.view` / `.reshape` -> `np.reshape`; `.permute` -> `np.transpose` where the guide supports it, else explicit loops.
- `.size(dim)` / `.shape[dim]` -> numpy shape reads; `dim=` reductions -> `axis=`; in-place ops -> augmented assignment.
- For `nn.Module` models, preserve inference semantics: tests seed weights from the torch model, the numpy forward consumes equivalent parameter arrays.
- Emit only the concrete numpy a kernel needs (conv, batchnorm, pooling, linear, activations, eval-mode dropout, sequential); no local runtime helper imports.
- Result files import `numpy` only -- never `torch`, scipy, or project-local files -- and contain nothing outside the kernel's math.

## Test Expectations

- Import each PyTorch file dynamically; call `get_init_inputs()` / `get_inputs()` where present.
- Instantiate the torch `Model`, `eval()` where available, compare its forward to the numpy output.
- Convert torch tensors/parameters to numpy without changing values. Tests may import `torch`; result files may not.
- Reduce oversized dims to run on CPU, keeping representative structure (not trivially small).
- ~150s per-test timeout; on timeout, shrink that case and rerun before calling it a translator failure.
- Tolerance by dtype/depth: start `rtol=1e-4, atol=1e-5` for float32-heavy kernels, tighten when stable.
- Report per-file failures with exception type, missing op, shape mismatch, or max error.

## Project References

- `CONTRIBUTOR_GUIDE.md` -- the allowed generated-numpy surface.
