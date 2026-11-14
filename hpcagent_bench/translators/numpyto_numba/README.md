# NumpyToNumba

Python (numpy) -> Python (numba) emitter. A dense kernel keeps its body; every top-level `def`
gains `@nb.njit(parallel=True, cache=True)` (`fastmath=True` with `--fastmath`), except the serial
helpers the parfor fixes add and the functions that run inside a prange, which gain `@nb.njit(cache=True)`.

```bash
numpyto --target numba --kernel k_numpy.py --bench-info k.json --out DIR   # writes DIR/k_numba.py
```

| module | does |
|---|---|
| `emit.py` | `emit_numba`: decorator, imports, header; drops `functools.lru_cache` / `cache` decorators, which nopython code cannot call through |
| `parfor.py` | keeps `parallel=True` where numba's parfor pass answers differently from numpy by fixing the body: a `max` / `min` / `eye` call runs through a serial module function (`serial_max`, ...), a slice store over cells its right-hand side reads stores a copy (`materialized(...)`), an augmented store of a reshaped operand is spelled out (`out += b.reshape(1, -1, 1)`, which the parfor pass misbroadcasts). `parallel=True` is dropped only for a form no fix takes (an overlapping `@=`, a starred unsafe call) |
| `prange.py` | in a parallel build, turns the outermost proven loop of every loop nest into `nb.prange`: independent by `loop_is_parallel_safe` once a per-iteration scalar's updates are discounted, or a `+` / `*` / `max` / `min` reduction of a literal-seeded scalar; its written arrays alias nothing, no helper it calls writes an array it hands over, no `break` / `continue` / `return`, its index dead after it. A function running inside a prange gets no prange and no `parallel=True` (no nested parallel region) |
| `objmode_fft.py` | runs a 1-D `np.fft.fft`/`ifft` in `objmode` (nopython mode cannot type `np.fft`) |
| `lstsq.py` | spells numpy's default `rcond=None` cutoff (`eps * max(M, N)`) as a float, the only `rcond` numba types |
| `sparse.py` | lowers a sparse `A @ x` onto the unpacked CSR/CSC buffer ABI the manifest declares. The dense operand must be a name proven rank 1 or a subscript the rank table proves rank 1 (`Q[:, k]`: an integer index drops its axis); the view is bound to a temp first |
