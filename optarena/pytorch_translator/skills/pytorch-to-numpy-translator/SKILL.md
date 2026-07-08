---
name: pytorch-to-numpy-translator
description: Translate PyTorch model/kernel Python files into NumpyToC-compatible numpy, build or improve the translator under src, generate result/level1 result/level2 result/level3 outputs, and create parity tests comparing original PyTorch behavior against numpy implementations.
---

# PyTorch To NumPy Translator

## Operating Contract

Use this skill when working on this repository's PyTorch-to-numpy translator. Treat `CONTRIBUTOR_GUIDE.md` as the compatibility contract for generated numpy: outputs should be static-shape, buffer-oriented wherever possible, free of `torch` imports, and limited to the numpy/control-flow surface documented there.

Generated kernels are meant to be consumed later by NumpyToC/NumpyToFortran, which will read each result file in isolation. Each generated result file must therefore be a clean, minimal, standalone numpy implementation of the original kernel math. Do not emit shared runtime imports, helper libraries, broad generic compatibility layers, or unused helper code into result files. Inline only the math needed by that specific source kernel. In rare cases where buffer-form output is too hard for a given construct, a numpy-array-returning implementation is acceptable as a temporary fallback and should be tracked by tests/status output.

Do not weaken the contributor guide to make a conversion pass. If an original PyTorch feature cannot be represented within the guide, stop and explain the missing rule before editing the guide.

## Required Layout

- Build translator code under `src/`.
- Write converted kernels under `result/level1/`, `result/level2/`, and `result/level3/`, preserving source filenames.
- Put parity test infrastructure under `test/`.
- Keep source inputs in `level1/`, `level2/`, and `level3/` unchanged unless the user explicitly asks otherwise.
- Do not keep separate project notes that override this skill and `CONTRIBUTOR_GUIDE.md`.

## Translation Workflow

1. Read a representative sample from each level before changing translator logic.
2. Implement translator behavior in reusable code, not by one-off manual edits to results.
3. Generate minimal standalone numpy results from the translator, preferring buffer-form signatures.
4. Run parity tests against the original PyTorch files.
5. Classify failures by unsupported PyTorch construct, shape/state initialization issue, numerical tolerance issue, or test harness issue.
6. Improve the translator progressively by level: level 1 first, then level 2, then level 3. Level 3 contains difficult model families, so incomplete level 3 coverage is acceptable only when failures are clearly reported and tied to unsupported constructs.

## Conversion Rules

- Replace `torch` tensor operations with `numpy` equivalents from `CONTRIBUTOR_GUIDE.md`.
- Strip autograd-only behavior such as `requires_grad`, `.detach()`, `.cpu()`, `.cuda()`, `.to()` device movement, and training-only state unless it affects inference numerics.
- Convert `.view(...)` and `.reshape(...)` to `np.reshape(...)`; convert `.permute(...)` to `np.transpose(...)` only when the downstream guide supports it, otherwise emit explicit loops.
- Convert `.size(dim)` and `.shape[dim]` to numpy shape reads.
- Convert reductions using `dim=` to numpy reductions using `axis=`.
- Convert PyTorch in-place operations to numpy augmented assignment.
- For `nn.Module` models, preserve inference semantics: initialize weights from the PyTorch model in tests, and emit numpy forward logic that consumes equivalent parameter arrays.
- For layers such as convolution, batch normalization, pooling, linear, activation, dropout in eval mode, and sequential containers, emit only the concrete numpy code needed by that kernel. Do not import a local runtime helper from generated result files.
- Generated result files may import `numpy` only. They must not import `torch`, scipy, project-local runtime files, or any other dependency.
- Generated kernels should not include code that is not part of the original kernel's math.

## Test Expectations

- Tests should import each original PyTorch file dynamically and call `get_init_inputs()` and `get_inputs()` where present.
- Tests should instantiate the PyTorch `Model`, set `eval()` when available, and compare its forward output to the generated numpy implementation.
- Convert torch tensors to numpy arrays for numpy execution; convert learned PyTorch parameters to numpy arrays without changing values.
- Tests may import `torch`; generated result files may not.
- Tests may reduce very large benchmark dimensions so they run on CPU, while preserving representative operation structure. Avoid making reduced cases trivially small.
- Add per-test timeout handling around 150 seconds. If a case times out, reduce that case's test size and rerun before treating it as a translator failure.
- Use tolerance appropriate to dtype and operation depth. Start with `rtol=1e-4, atol=1e-5` for float32-heavy neural kernels, and tighten when stable.
- Report per-file failures with enough context to improve the translator quickly: exception type, missing operation, shape mismatch, or max numerical error.

## Project References

- Read `CONTRIBUTOR_GUIDE.md` for the allowed generated-numpy surface.
- This skill is the repository-specific operating note for the translator.
