# numpyto_gpu: NumPy -> HIP / CUDA

Emits a numpy kernel as ONE GPU translation unit (`<base>.hip`; `<base>.cu` is the same text) plus the C-ABI
binding JSON. The unit's `extern "C"` entry has the kernel's ABI (`hpcagent_bench/docs/abi_contract.md` Sec. 7)
and receives **device-resident** pointers and host scalars (Sec. 10): it launches, never copies.

```bash
python -m hpcagent_bench.translators.numpyto_gpu jacobi_2d tsvc_2_s3111 -o /tmp/gpu
```

```python
from hpcagent_bench.categories import Language
from hpcagent_bench.harness.agent import emit_reference_source

hip = emit_reference_source("jacobi_2d", Language.HIP)  # also Language.CUDA; with the workspace pair
```

## Model

The kernel is parsed and lowered by `numpyto_common` like the C target, with helpers inlined and without library
calls (no BLAS / FFTW on the device), and its statements are printed by the C++ emitter. `kernelize.py` rewrites
the lowered body for one cooperative launch, in which every thread runs the whole body:

| Body construct | GPU form |
|---|---|
| loop the numba translator proves parallel (`numpyto_numba.prange.outermost_safe_loops`), independent | grid-stride loop, one iteration per thread; a perfect rectangular nest (`parallelism.collapsible_depth`) linearized, innermost index fastest |
| the same, a scalar reduction (`+ * max min`) | per-thread partial, block tree in shared memory, one compare-and-swap per block into a frame slot |
| array store, call, or compound statement without a parallel loop | lead thread only; the scalars it assigns are broadcast through the frame |
| scalar assignment, loop / `if` holding a parallel loop, `break` / `continue` | every thread (uniform control flow) |
| `np.zeros`-style local array | kernel parameter; the host entry `Malloc`s / `Free`s it, fills are parallel nests |

A grid barrier (`cooperative_groups::this_grid().sync()`) opens and closes every write epoch. Blocks have 256
threads; the grid is the resident maximum the runtime reports for this kernel (occupancy x compute units). A
kernel with no parallel loop launches one thread.

`emit.py` assembles the unit: the portability layer (`NPB_GPU(Malloc)` is `hipMalloc` under hipcc, `cudaMalloc`
under nvcc; hipify-perl output still takes the HIP branch), the C++ prelude, the device runtime, the kernel and
the host entry.

## Refusals (`GpuUnsupported`)

Complex values, 8-bit float storage, a helper that cannot be inlined, a local array sized by a loop index or a
computed value or whose extent changes, a proven loop with no GPU form, and any host-only call left in the body.
