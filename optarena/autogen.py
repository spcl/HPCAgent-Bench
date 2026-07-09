"""Auto-generate framework sibling files from the numpy reference.

ONE canonical file per (kernel, framework): ``<module>_<fw>.py``
``fw`` in :data:`TARGETS` (``dace`` / ``cupy`` / ``numba_n`` / ``numba_np`` /
``pythran`` / ``jax``). A file already present that does NOT carry the
``optarena-autogen`` marker is a hand-written OVERRIDE and is never overwritten
(so the committed microbench ``*_jax.py`` overrides win over autogen).

Entry point:

* :func:`ensure` -- emit any MISSING target for one kernel. The framework
  loaders call this so a sibling is generated **on demand** the first time it is
  needed (``run_benchmark.py -f cupy`` with no ``<k>_cupy.py`` yet just works).

The emitter reads a bench_info JSON synthesized from the co-located YAML
(:mod:`optarena.emit_bridge`); the flat ``bench_info/`` corpus is gone. native
C/C++/Fortran siblings (``cpp_backend/``, precision-specialised) are a separate
path and are NOT generated here.
"""
from __future__ import annotations

import ast
import pathlib
import subprocess
import sys
from typing import Dict, Iterable, List, Optional

from optarena import paths

#: Auto-generatable Python targets and the canonical filename each produces
#: (``{m}`` = the kernel's module_name). dace and jax are generated in-process;
#: the rest shell out to their per-package CLI (which writes the canonical name).
TARGETS = ("dace", "cupy", "numba_n", "numba_np", "pythran", "jax")


def _file_for(module_name: str, target: str) -> str:
    return f"{module_name}_{target}.py"


def _emit_dace(numpy_py: pathlib.Path, bench_info: pathlib.Path, out: pathlib.Path) -> str:
    from numpyto_c.frontend import parse_kernel
    from numpyto_c.dace_emit import emit_dace
    from numpyto_common.emit_io import write_generated
    src = emit_dace(parse_kernel(numpy_py, bench_info))
    ast.parse(src)  # syntactic self-check before writing
    return write_generated(out, src, source=numpy_py.name)


def _emit_jax(numpy_py: pathlib.Path, bench_info: pathlib.Path, out: pathlib.Path) -> str:
    # In-process like _emit_dace: numpyto_jax.emit_jax is a pure-AST np->jnp
    # translation (it imports no jax), emitted in EAGER mode -- the faithful 1:1
    # form that covers the widest kernel set. write_generated's marker guard
    # leaves a hand-written *_jax.py override (the committed microbench ones)
    # untouched.
    import json
    from numpyto_jax import emit_jax
    from numpyto_common.emit_io import write_generated
    func = json.loads(bench_info.read_text())["benchmark"]["func_name"]
    src = emit_jax(numpy_py.read_text(), func)
    ast.parse(src)  # syntactic self-check before writing
    return write_generated(out, src, source=numpy_py.name)


def _emit_cli(module: str, numpy_py: pathlib.Path, out_dir: pathlib.Path, extra: List[str]) -> str:
    cmd = [sys.executable, "-m", module, "emit", "--kernel", str(numpy_py), "--out", str(out_dir), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr.strip().splitlines() or ["unknown error"])[-1]
        return f"fail: {tail}"
    last = (r.stdout.strip().splitlines() or [""])[-1]
    return "override" if " override " in f" {last} " else "ok"


def _emit_target(target: str, numpy_py: pathlib.Path, kdir: pathlib.Path, bench_info: pathlib.Path) -> str:
    if target == "dace":
        return _emit_dace(numpy_py, bench_info, kdir / _file_for(numpy_py.stem.removesuffix("_numpy"), "dace"))
    if target == "jax":
        return _emit_jax(numpy_py, bench_info, kdir / _file_for(numpy_py.stem.removesuffix("_numpy"), "jax"))
    if target == "cupy":
        return _emit_cli("numpyto_cupy.cli", numpy_py, kdir, [])
    if target in ("numba_n", "numba_np"):
        suffix = target.split("_", 1)[1]
        return _emit_cli("numpyto_numba.cli", numpy_py, kdir, ["--bench-info", str(bench_info), "--suffix", suffix])
    if target == "pythran":
        return _emit_cli("numpyto_pythran.cli", numpy_py, kdir, ["--bench-info", str(bench_info)])
    raise ValueError(f"unknown auto-gen target {target!r}; known: {TARGETS}")


def emit_targets(spec, targets: Iterable[str]) -> Dict[str, str]:
    """Emit ``targets`` for one :class:`~optarena.spec.BenchSpec` to their
    canonical names (override-aware). Returns ``{target: status}``."""
    from optarena.emit_bridge import bench_info_tempfile
    kdir = paths.BENCHMARKS / spec.relative_path
    numpy_py = kdir / f"{spec.module_name}_numpy.py"
    if not numpy_py.exists():
        return {}
    out: Dict[str, str] = {}
    with bench_info_tempfile(spec) as bi:
        for t in targets:
            try:
                out[t] = _emit_target(t, numpy_py, kdir, bi)
            except Exception as exc:  # noqa: BLE001 - report, keep going
                out[t] = f"fail: {type(exc).__name__}: {exc}"
    return out


def ensure(short_name: str, targets: Iterable[str]) -> None:
    """Generate any of ``targets`` whose canonical file is MISSING for kernel
    ``short_name``. Best-effort: a failed emit is swallowed so the caller's
    own import raises the real, specific error. No-op when nothing is missing.
    """
    targets = list(targets)
    if not targets:
        return
    try:
        from optarena.spec import BenchSpec
        spec = BenchSpec.load(short_name)
    except Exception:
        return
    kdir = paths.BENCHMARKS / spec.relative_path
    missing = [t for t in targets if not (kdir / _file_for(spec.module_name, t)).exists()]
    if missing:
        try:
            emit_targets(spec, missing)
        except Exception:
            pass


# --- Native (C / C++ / Fortran) ---------------------------------------------
#
# Native siblings live in the kernel's ``cpp_backend/`` as precision-monomorphic
# sources ``<short>[_<sparse>]_<fptype>.<ext>`` (symbol == file stem), generated
# on demand from ``<short>_numpy.py`` and gitignored -- the repo commits none.
# A thin ``<module>_cpp.py`` wrapper (also generated) exposes one ``kernel_<fw>``
# per native framework via :func:`optarena.benchmarks.cpp_runtime.wrap_kernel`.

#: native framework -> source language (mirror of cpp_runtime.FRAMEWORK_LANG).
#: Polly/Pluto reuse the C++ source (flag presets), so they add a wrapper entry
#: but no new emitted source.
NATIVE_FRAMEWORKS = {"cc": "c", "llvm": "cpp", "fortran": "fortran", "polly": "cpp", "pluto": "cpp"}
#: language -> the numpyto ``--target`` that emits it (the C target writes BOTH
#: ``.c`` and ``.cpp`` in one run; fortran has its own target).
_LANG_TARGET = {"c": "c", "cpp": "c", "fortran": "fortran"}
#: precisions to materialise per native source (numpy dtype name -> empty = fp64).
_NATIVE_PRECISIONS = ("", "float32")


def _wrapper_path(spec) -> pathlib.Path:
    return paths.BENCHMARKS / spec.relative_path / f"{spec.module_name}_cpp.py"


def _native_targets(spec) -> List[tuple]:
    """``[(config_or_None, native_base)]`` -- one entry per emit-distinct layout.

    A dense kernel yields ``[(None, <short>)]``; a sparse kernel yields one
    ``(<config>, <short>_<config>)`` per configuration (the layout IS the
    sub-benchmark -- each is a full kernel with its own source / symbol / lib).
    Distributions sharing one configuration collapse to a single native source
    (they differ only in runtime data), so the list is deduped by base."""
    seen: set = set()
    out: List[tuple] = []
    for rb in spec.expand_layouts():
        if rb.config_key == "dense":
            cfg, base = None, spec.short_name
        else:
            cfg, base = rb.config_key, f"{spec.short_name}_{rb.config_key}"
        if base in seen:
            continue
        seen.add(base)
        out.append((cfg, base))
    return out


def _wrapper_src(spec) -> str:
    """Generate the ``<module>_cpp.py`` wrapper. Exposes ``kernel_<fw>`` per
    native framework; for a sparse kernel each configuration also gets a
    ``kernel_<fw>_<config>`` entry (each layout is independently runnable) and
    ``kernel_<fw>`` aliases the first configuration so the default run path
    resolves a sub-benchmark without naming a layout."""
    targets = _native_targets(spec)
    lines = ["from optarena.benchmarks.cpp_runtime import wrap_kernel", ""]
    for fw in NATIVE_FRAMEWORKS:
        default_attr = None
        for cfg, base in targets:
            if cfg is None:
                lines.append(f'kernel_{fw} = wrap_kernel(__file__, "{base}", "{fw}")')
            else:
                attr = f"kernel_{fw}_{cfg}"
                lines.append(f'{attr} = wrap_kernel(__file__, "{base}", "{fw}")')
                if default_attr is None:
                    default_attr = attr
        if default_attr is not None:
            lines.append(f"kernel_{fw} = {default_attr}")
    return "\n".join(lines) + "\n"


def emit_native(spec, langs: Iterable[str]) -> Dict[str, str]:
    """Emit the native sources for ``langs`` (both precisions, every layout) +
    the ``_cpp.py`` wrapper for one spec. Returns ``{tag: status}`` (best-effort).

    For a sparse kernel one source set is emitted per configuration (passed as
    ``--config`` so the emitter unpacks the logical array to that layout's member
    buffers); the file/symbol stem is ``<short>_<config>[_<fptype>]``."""
    from optarena.emit_bridge import emit_kernel
    from numpyto_common.emit_io import write_generated
    kdir = paths.BENCHMARKS / spec.relative_path
    numpy_py = kdir / f"{spec.module_name}_numpy.py"
    out: Dict[str, str] = {}
    if not numpy_py.exists():
        return out
    cppdir = kdir / "cpp_backend"
    for tgt in {_LANG_TARGET[l] for l in langs}:  # noqa: E741
        for cfg, base in _native_targets(spec):
            for prec in _NATIVE_PRECISIONS:
                key = f"{tgt}:{base}:{prec or 'fp64'}"
                try:
                    rc = emit_kernel(spec.short_name, numpy_py, cppdir, target=tgt, config=cfg, precision=prec)
                    out[key] = "ok" if rc == 0 else f"fail rc={rc}"
                except Exception as exc:  # noqa: BLE001
                    out[key] = f"fail: {type(exc).__name__}: {exc}"
    out["wrapper"] = write_generated(_wrapper_path(spec), _wrapper_src(spec), source=f"{spec.module_name}_numpy.py")
    return out


def ensure_native(short_name: str, lang: Optional[str] = None) -> None:
    """Generate the native sources (+ wrapper) for kernel ``short_name`` if
    missing. ``lang`` restricts to one language (else all of NATIVE_FRAMEWORKS).
    Best-effort: a failed emit is swallowed so the caller surfaces the real
    error at build/import time."""
    try:
        from optarena.spec import BenchSpec
        spec = BenchSpec.load(short_name)
    except Exception:
        return
    langs = [lang] if lang else sorted(set(NATIVE_FRAMEWORKS.values()))
    try:
        emit_native(spec, langs)
    except Exception:
        pass
