# NumpyToNumba

Python (numpy) -> Python (numba) emitter. A dense kernel keeps its body; every top-level `def`
gains `@nb.njit(parallel=True, cache=True)` (`fastmath=True` with `--fastmath`).

```bash
numpyto --target numba --kernel k_numpy.py --bench-info k.json --out DIR   # writes DIR/k_numba_np.py
```

| module | does |
|---|---|
| `emit.py` | `emit_numba`: decorator, imports, header |
| `parfor.py` | drops `parallel=True` for a body numba's parfor pass answers differently from numpy, and turns at most one provably independent unit-step `range` loop into `nb.prange` |
| `objmode_fft.py` | runs a 1-D `np.fft.fft`/`ifft` in `objmode` (nopython mode cannot type `np.fft`) |
| `sparse.py` | lowers a sparse `A @ x` onto the unpacked CSR/CSC buffer ABI the manifest declares |
