# NumpyToPythran

Python (numpy) -> Python (Pythran AOT) emitter. Pythran reads a `#pythran export` comment for the
entry function's argument types; the emitter synthesises it from the kernel IR and prepends it.

| module | does |
|---|---|
| `emit.py` | `emit_pythran`: the export line over the def's own parameters |
| `export.py` | numpy dtype -> pythran type spelling (refuses an unknown dtype) |
| `rewrites.py` | source rewrites pythran needs to compile the module and agree with numpy (unreachable helpers dropped, lazy templates materialised, NaN-propagating min/max/sign, zero-width GEMV, `...` expanded, keyword calls made positional) |
