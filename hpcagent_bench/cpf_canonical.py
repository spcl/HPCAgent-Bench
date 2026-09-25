# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A kernel's canonical SDFG, produced once per (generated program, dace commit) and cached.

The key is taken before any parse: the text of the generated ``<module>_dace.py``, the dace HEAD
commit, precision, target, the ``DACE_*`` environment and this module's own bytes. A hit loads the
stored ``canonical.sdfgz``; a miss parses and canonicalizes, stores the SDFG or the raised error,
and loads the stored copy back, so a first render and a cached re-render read the same file.
Rendering code lives in :mod:`hpcagent_bench.cpf_bridge` and keys its forms on this entry, so an
edit there re-renders without re-canonicalizing.
"""

import hashlib
import importlib
import os
import pathlib
import subprocess
import tempfile
from collections.abc import Sequence
from types import ModuleType
from typing import TYPE_CHECKING

from hpcagent_bench.translators.numpyto_common.naming import fptype_tag

from hpcagent_bench import cpf_cache, paths
from hpcagent_bench.spec import BenchSpec

if TYPE_CHECKING:
    from dace import SDFG
    from dace.frontend.python.parser import DaceProgram

#: Postfixes a generated impl's stem carries over its ``@dace.program`` name, longest first:
#: sorting the other way would let the bare ``_dace`` suffix shadow a future ``_dace_x``.
IMPL_POSTFIXES = ("_dace_gpu", "_dace_cpu", "_dace")

#: ``DACE_*`` variables that name locations rather than behaviour, left out of every key.
DACE_PATH_VARIABLES = frozenset({"DACE_TREE", "DACE_default_build_folder", "DACE_BUILD_CACHE_DIR"})


def program_name(path: pathlib.Path) -> str:
    """The ``@dace.program`` name a generated impl file is expected to define."""
    for postfix in IMPL_POSTFIXES:
        if path.stem.endswith(postfix):
            return path.stem[: -len(postfix)]
    return path.stem


def resolve_program(module: ModuleType, path: pathlib.Path, entry: str = "") -> "DaceProgram | None":
    """The ``DaceProgram`` in ``module`` that is the kernel's ENTRY POINT, or ``None``.

    ``entry`` is the manifest's ``func_name``, and it is asked first because it is the only name
    that is DECLARED. The file stem answers next, for a caller that holds no spec. Both can miss: a
    kernel whose function is named for the algorithm rather than the file matches neither.

    A module holding several programs is the normal case, not the ambiguous one -- the emitter
    keeps each inlined helper as its own ``@dc.program`` -- so the last two readings work down from
    that: the helpers it generates are ``_``-prefixed, which leaves one public program, and a
    module with a single program of any name is that program.
    """
    programs = {name: value for name, value in vars(module).items() if type(value).__name__ == "DaceProgram"}
    chosen = entry_program_name(list(programs), path, entry)
    return None if chosen is None else programs[chosen]


def entry_program_name(names: Sequence[str], path: pathlib.Path, entry: str = "") -> str | None:
    """Which of a generated impl's program ``names`` is the entry, by :func:`resolve_program`'s rule."""
    for name in (entry, program_name(path)):
        if name in names:
            return name
    public = [name for name in names if not name.startswith("_")]
    if len(public) == 1:
        return public[0]
    return names[0] if len(names) == 1 else None


def failure(exc: BaseException) -> dict[str, str]:
    """A step that produced nothing, as a verdict: CPF refuses by NotImplementedError and names the construct."""
    if isinstance(exc, NotImplementedError):
        return {"verdict": "refused", "error": str(exc)[:400]}
    return {"verdict": "fail", "error": f"{type(exc).__name__}: {exc}"[:400]}


def dace_commit(dace_root: pathlib.Path) -> str:
    """The dace checkout's HEAD commit, which keys every cache entry; a tree with no commit is refused."""
    done = subprocess.run(
        ["git", "-C", str(dace_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    commit = done.stdout.strip()
    if done.returncode != 0 or not commit:
        raise RuntimeError(f"dace at {dace_root} is not a git checkout; the CPF cache keys on its commit")
    return commit


def dace_environment() -> dict[str, str]:
    """The ``DACE_*`` settings that change what dace produces."""
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith("DACE_") and k not in DACE_PATH_VARIABLES}


def producer_digest() -> str:
    """Hash of this module, the code between the generated program and the canonical SDFG."""
    return hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()


def canonical_key(program: pathlib.Path, commit: str, precision: str, target: str) -> str:
    """The canonical entry of ``program`` as the dace at ``commit`` canonicalizes it for ``target``."""
    options = {
        "precision": fptype_tag(precision),
        "target": target,
        "dace_env": dace_environment(),
        "producer": producer_digest(),
    }
    return cpf_cache.canonical_key(hashlib.sha256(program.read_bytes()).hexdigest(), commit, options)


def emit_program(spec: BenchSpec) -> "pathlib.Path | dict[str, str]":
    """Step 1: the generated ``<module>_dace.py``, or the ``noemit`` verdict that stopped it."""
    from hpcagent_bench import autogen

    status = autogen.emit_targets(spec, ["dace"]).get("dace", "")
    if status.startswith("fail"):
        return {"verdict": "noemit", "error": status}
    return paths.BENCHMARKS / spec.relative_path / f"{spec.module_name}_dace.py"


def bind_precision(precision: str) -> None:
    """Bind the precision globals every generated impl annotates; unbound, the import fails on ``None[...]``."""
    import dace

    from hpcagent_bench.frameworks import dace_framework
    from hpcagent_bench.precision import Precision, precision_from_datatype

    prec = precision_from_datatype(precision or None)
    dace_framework.dc_float = {
        Precision.FP64: dace.float64,
        Precision.FP32: dace.float32,
        Precision.FP16: dace.float16,
    }.get(prec, dace.float32)
    dace_framework.dc_complex_float = dace.complex128 if prec is Precision.FP64 else dace.complex64


def parse_program(spec: BenchSpec, impl: pathlib.Path, precision: str) -> "SDFG | dict[str, str]":
    """Step 2: the entry program's parsed SDFG, or the ``noprogram`` verdict."""
    bind_precision(precision)
    module = importlib.import_module(".".join(impl.relative_to(paths.ROOT).with_suffix("").parts))
    prog = resolve_program(module, impl, spec.func_name)
    if prog is None:
        return {"verdict": "noprogram"}
    return prog.to_sdfg(simplify=True)


def parse_kernel(spec: BenchSpec, numpy_py: pathlib.Path, precision: str) -> "SDFG | dict[str, str]":
    """Steps 1-2 for an uncached caller: the parsed SDFG, or the verdict (``noemit`` / ``noprogram``)."""
    del numpy_py  # the program is generated from the spec's own reference
    impl = emit_program(spec)
    return impl if isinstance(impl, dict) else parse_program(spec, impl, precision)


def canonicalize_for(sdfg: "SDFG", target: str) -> None:
    """Step 3, in place.

    canonicalize leaves every choice PARALLEL but decides no OpenMP region -- skip the tail and the
    unit renders correct but entirely SEQUENTIAL. GPU adds offload_to_gpu between the two calls;
    finalize REJECTS a graph never offloaded, so a wiring bug fails loudly, not silently.
    """
    from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
    from dace.transformation.passes.canonicalize.pipeline import canonicalize

    canonicalize(sdfg, validate=True, validate_all=False, target=target)
    if target == "gpu":
        offload_to_gpu(sdfg)
    finalize_for_target(sdfg, target, validate=True)


def produce(
    spec: BenchSpec, impl: pathlib.Path, key: str, cache_root: pathlib.Path, precision: str, target: str
) -> None:
    """Parse and canonicalize ``impl`` and publish the SDFG, or the error it raised, under ``key``.

    Only an ``Exception`` is published: a timeout or a kill ends this process before it gets here,
    so an interrupted canonicalize is retried rather than remembered.
    """
    manifest = {"kernel": spec.short_name, "program": impl.name, "target": target, "precision": fptype_tag(precision)}
    try:
        parsed = parse_program(spec, impl, precision)
        if isinstance(parsed, dict):
            cpf_cache.publish_canonical(cache_root, key, {**manifest, **parsed}, None)
            return
        canonicalize_for(parsed, target)
    except Exception as exc:  # noqa: BLE001 -- a raised error is a verdict worth remembering
        cpf_cache.publish_canonical(cache_root, key, {**manifest, **failure(exc)}, None)
        return
    cache_root.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(tempfile.mkdtemp(prefix=f".{key}.", dir=cache_root))
    try:
        stored = staging / cpf_cache.CANONICAL_SDFG_NAME
        parsed.save(str(stored), compress=True)
        cpf_cache.publish_canonical(cache_root, key, {**manifest, "verdict": "ok"}, stored)
    finally:
        for leftover in staging.glob("*"):
            leftover.unlink()
        staging.rmdir()


def canonical_sdfg(
    spec: BenchSpec, impl: pathlib.Path, key: str, cache_root: pathlib.Path, precision: str, target: str
) -> "tuple[SDFG | dict[str, str], bool]":
    """The canonical SDFG under ``key``, always loaded from the cache, and whether it was already there.

    A cached failure comes back as its verdict. A fresh entry is produced first and then read back
    like any hit, so every render starts from the stored file.
    """
    from dace import SDFG

    entry = cpf_cache.canonical_entry(cache_root, key)
    cached = entry is not None
    if entry is None:
        produce(spec, impl, key, cache_root, precision, target)
        entry = cpf_cache.canonical_entry(cache_root, key)
    if entry is None:
        raise RuntimeError(f"canonical entry {key} was not published")
    manifest, stored = entry
    if stored is None:
        return {"verdict": str(manifest.get("verdict", "fail")), "error": str(manifest.get("error", ""))}, cached
    return SDFG.from_file(str(stored)), cached
