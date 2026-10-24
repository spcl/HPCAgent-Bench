# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Auto-generate framework sibling files (``<module>_<fw>.py``, ``fw`` in :data:`EMITTERS`) from the numpy reference,
on demand (:func:`ensure`). A present file WITHOUT the ``hpcagent_bench-autogen`` marker is a hand-written override
and is never overwritten. Native sources: :func:`ensure_native`."""

import ast
import fcntl
import importlib
import pathlib
import time
from collections.abc import Callable, Iterable

from hpcagent_bench import framework_cache, paths, phase_clock
from hpcagent_bench.categories import Language, Track
from hpcagent_bench.emit_bridge import bench_info_tempfile
from hpcagent_bench.frameworks.framework import native_column_languages
from hpcagent_bench.languages import LANG_TARGET
from hpcagent_bench.spec import BenchSpec

__all__ = [
    "EMITTERS",
    "NATIVE_FRAMEWORKS",
    "NATIVE_PRECISIONS",
    "NO_NUMBA_TRACKS",
    "Emitter",
    "emit_numba_sibling",
    "emit_native",
    "emit_targets",
    "emittable",
    "ensure",
    "ensure_locked",
    "ensure_native",
    "file_for",
    "has_numba",
    "wrapper_path",
]

#: ``(numpy_py, kernel_dir, bench_info) -> status`` (``ok`` / ``override`` / ``fail: ...``).
type Emitter = Callable[[pathlib.Path, pathlib.Path, pathlib.Path], str]


def file_for(module_name: str, target: str) -> str:
    return f"{module_name}_{target}.py"


def _emit_dace(numpy_py: pathlib.Path, kdir: pathlib.Path, bench_info: pathlib.Path) -> str:
    out = kdir / file_for(numpy_py.stem.removesuffix("_numpy"), "dace")
    from hpcagent_bench.translators.numpyto_c.dace_emit import emit_dace
    from hpcagent_bench.translators.numpyto_common.emit_io import write_generated
    from hpcagent_bench.translators.numpyto_common.frontend import emit_with_inline_fallback, parse_kernel

    def render() -> str:
        rendered = emit_dace(parse_kernel(numpy_py, bench_info, open_mesh_grids=False, library_eigh=True))
        ast.parse(rendered)  # syntactic self-check before writing
        return rendered

    # A level-3 kernel keeps its helpers; if that form fails to emit, the render is repeated with inlining forced.
    return write_generated(out, emit_with_inline_fallback(render), source=numpy_py.name)


def emit_numba_sibling(numpy_py: pathlib.Path, kdir: pathlib.Path, bench_info: pathlib.Path) -> str:
    """``<module>_numba.py`` beside the reference, in-process; a refusal is ``fail: ...``."""
    from hpcagent_bench.translators.numpyto_common.targets import EmitRequest, emit

    try:
        return emit("numba", EmitRequest(kernel=numpy_py, out=kdir, bench_info=bench_info))
    except Exception as exc:  # noqa: BLE001 -- the refusal is the status
        return f"fail: {type(exc).__name__}: {exc}"


#: Auto-generatable Python target -> its emitter, which writes ``<module>_<target>.py`` beside the reference. A new
#: target is one entry here plus a ``Framework.autogen_targets``.
EMITTERS: dict[str, Emitter] = {
    "dace": _emit_dace,
    "numba": emit_numba_sibling,
}


#: Tracks no numba sibling is generated for: a machine_learning kernel's only reference is PyTorch.
NO_NUMBA_TRACKS: frozenset[Track] = frozenset({Track.MACHINE_LEARNING})


def has_numba(spec: BenchSpec) -> bool:
    """Whether ``spec`` has a numba reference at all (its track is not in :data:`NO_NUMBA_TRACKS`)."""
    return spec.track not in NO_NUMBA_TRACKS


def emittable(spec: BenchSpec, targets: Iterable[str]) -> list[str]:
    """``targets`` without ``numba`` on a track that has none (:func:`has_numba`)."""
    return [t for t in targets if t != "numba" or has_numba(spec)]


def emit_target(target: str, numpy_py: pathlib.Path, kdir: pathlib.Path, bench_info: pathlib.Path) -> str:
    emitter = EMITTERS.get(target)
    if emitter is None:
        raise ValueError(f"unknown auto-gen target {target!r}; known: {tuple(EMITTERS)}")
    return emitter(numpy_py, kdir, bench_info)


def emit_targets(spec: BenchSpec, targets: Iterable[str]) -> dict[str, str]:
    """Emit ``targets`` for one spec to their canonical names (override-aware); returns ``{target: status}``."""
    kdir = paths.BENCHMARKS / spec.relative_path
    numpy_py = kdir / f"{spec.module_name}_numpy.py"
    if not numpy_py.exists():
        return {}
    out: dict[str, str] = {}
    with bench_info_tempfile(spec) as bi:
        for t in emittable(spec, targets):
            try:
                out[t] = emit_target(t, numpy_py, kdir, bi)
            except Exception as exc:  # noqa: BLE001 - report, keep going
                out[t] = f"fail: {type(exc).__name__}: {exc}"
    return out


def ensure(key: str, targets: Iterable[str]) -> None:
    """Generate any of ``targets`` whose canonical file is MISSING OR STALE for the kernel registered under ``key``.

    Freshness is a fingerprint over the reference and its bench_info (:mod:`hpcagent_bench.framework_cache`).
    ``key`` is a REGISTRY key, never ``spec.short_name``. A failed emit is reported as a status, not cached, and
    DELETES the stale canonical file, so the caller's import fails naming it instead of running stale code."""
    spec = BenchSpec.load(key)
    targets = emittable(spec, targets)
    if not targets:
        return
    kdir = paths.BENCHMARKS / spec.relative_path
    if not (kdir / f"{spec.module_name}_numpy.py").exists():
        return
    # A job's ranks share the tree: one generates while the others wait.
    with (framework_cache.kernel_cache_dir(kdir) / "autogen.lock").open("a") as lock:
        fcntl.lockf(lock, fcntl.LOCK_EX)
        ensure_locked(spec, kdir, targets)
    # A sibling written after the first directory listing is invisible to imports until the caches are invalidated.
    importlib.invalidate_caches()


def ensure_locked(spec: BenchSpec, kdir: pathlib.Path, targets: list[str]) -> None:
    """:func:`ensure` under the kernel's autogen lock."""
    from hpcagent_bench.translators.numpyto_common.emit_io import is_generated, is_override

    numpy_py = kdir / f"{spec.module_name}_numpy.py"
    with bench_info_tempfile(spec) as bi:
        fingerprint = framework_cache.source_fingerprint(numpy_py, bi.read_bytes())
    cache_dir = framework_cache.kernel_cache_dir(kdir)
    to_emit: list[str] = []
    for t in targets:
        canonical = kdir / file_for(spec.module_name, t)
        # A hand-written override always wins: never emitted, never cached.
        if is_override(canonical):
            continue
        if not framework_cache.load_generated(cache_dir, canonical, fingerprint):
            to_emit.append(t)
    for t in to_emit:
        # One target per emit, so each cached file records its own emit seconds (phase_clock).
        started = time.perf_counter()
        with phase_clock.phase(phase_clock.Phase.CODEGEN):
            status = emit_targets(spec, [t]).get(t, "")
        seconds = time.perf_counter() - started
        # Cache only a freshly generated file ("ok"), never an override or a failure.
        canonical = kdir / file_for(spec.module_name, t)
        if status == "ok" and is_generated(canonical):
            framework_cache.save_generated(cache_dir, canonical, fingerprint, seconds)
        elif status.startswith("fail") and is_generated(canonical):
            canonical.unlink()


# Native (C / C++ / Fortran): precision-monomorphic ``cpp_backend/<short>[_<sparse>]_<fptype>.<ext>`` sources
# (gitignored), plus a generated ``<module>_cpp.py`` wrapper exposing ``kernel_<fw>`` per native framework.

#: native framework -> the language its sources are emitted in (the column's ``emit_language``, else ``language``).
#: Pluto and PPCG add a wrapper entry only (they transform the C target's ``_pluto_input.c``).
NATIVE_FRAMEWORKS: dict[str, Language] = {
    name: Language.parse(languages[0]) for name, languages in native_column_languages().items()
}
#: Precisions to materialise per native source (numpy dtype name; empty = fp64).
NATIVE_PRECISIONS = ("", "float32")


def wrapper_path(spec: BenchSpec) -> pathlib.Path:
    return paths.BENCHMARKS / spec.relative_path / f"{spec.module_name}_cpp.py"


def native_targets(spec: BenchSpec) -> list[tuple]:
    """``[(config_or_None, native_base)]``: one entry per layout the framework baselines run (a sparse kernel's
    default layout only); stems from :meth:`BenchSpec.native_base`."""
    seen: set = set()
    out: list[tuple] = []
    for rb in spec.expand_layouts():
        if spec.sparse_layouts and rb.config_key != spec.default_layout:
            continue
        cfg = None if rb.config_key == "dense" else rb.config_key
        base = spec.native_base(rb.config_key)
        if base in seen:
            continue
        seen.add(base)
        out.append((cfg, base))
    return out


def wrapper_src(spec: BenchSpec) -> str:
    """Generate the ``<module>_cpp.py`` wrapper: ``kernel_<fw>`` per native framework, plus ``kernel_<fw>_<config>``
    per sparse configuration (``kernel_<fw>`` aliasing the first)."""
    targets = native_targets(spec)
    lines = ["from hpcagent_bench.frameworks.native_runtime import wrap_kernel", ""]
    for fw in NATIVE_FRAMEWORKS:
        default_attr = None
        for cfg, base in targets:
            if cfg is None:
                lines.append(f'kernel_{fw} = wrap_kernel(__file__, "{base}", "{fw}", "{spec.short_name}")')
            else:
                attr = f"kernel_{fw}_{cfg}"
                lines.append(f'{attr} = wrap_kernel(__file__, "{base}", "{fw}", "{spec.short_name}")')
                if default_attr is None:
                    default_attr = attr
        if default_attr is not None:
            lines.append(f"kernel_{fw} = {default_attr}")
    return "\n".join(lines) + "\n"


def emit_native(spec: BenchSpec, langs: Iterable[Language]) -> dict[str, str]:
    """Emit the native sources for ``langs`` (both precisions, every layout, ``--config`` per sparse configuration)
    and the ``_cpp.py`` wrapper; returns ``{tag: status}`` (best-effort)."""
    from hpcagent_bench.emit_bridge import emit_kernel
    from hpcagent_bench.translators.numpyto_common.emit_io import write_generated

    kdir = paths.BENCHMARKS / spec.relative_path
    numpy_py = kdir / f"{spec.module_name}_numpy.py"
    out: dict[str, str] = {}
    if not numpy_py.exists():
        return out
    cppdir = kdir / "cpp_backend"
    for tgt in {LANG_TARGET[l] for l in langs}:  # noqa: E741
        for cfg, base in native_targets(spec):
            for prec in NATIVE_PRECISIONS:
                key = f"{tgt}:{base}:{prec or 'fp64'}"
                try:
                    rc = emit_kernel(spec, numpy_py, cppdir, target=tgt, config=cfg, precision=prec)
                    out[key] = "ok" if rc == 0 else f"fail rc={rc}"
                except Exception as exc:  # noqa: BLE001
                    out[key] = f"fail: {type(exc).__name__}: {exc}"
    out["wrapper"] = write_generated(wrapper_path(spec), wrapper_src(spec), source=f"{spec.module_name}_numpy.py")
    return out


def ensure_native(key: str, lang: Language | None = None) -> None:
    """Generate the native sources (+ wrapper) for the kernel registered under ``key`` (``lang`` restricts to one).

    Raises ``RuntimeError`` naming the failed target(s): the wrapper is emitted regardless, so swallowing would
    surface later as an unrelated ctypes build error. ``key`` is a REGISTRY key (see :func:`ensure`)."""
    spec = BenchSpec.load(key)
    langs = [lang] if lang else sorted(set(NATIVE_FRAMEWORKS.values()), key=lambda one: one.text)
    failed = {t: s for t, s in emit_native(spec, langs).items() if s.startswith("fail")}
    if failed:
        detail = "; ".join(f"{t}: {s}" for t, s in sorted(failed.items()))
        raise RuntimeError(f"{key}: native emit failed for {sorted(failed)} -- {detail}")
