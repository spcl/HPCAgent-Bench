# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""``hpcagent-bench dace emit KERNEL``: run one SDFG pipeline over a kernel and write the generated sources to
disk, produced without compiling anything (no GPU, no CUDA toolchain, no build folder). A ``cuda``/``hip`` target
writes the self-contained pair (:func:`device_units`): the host unit with the judge's C ABI entry, the device unit
and their build line; ``frame`` writes the units a ``dace_gpu_*`` column hands its compiler instead (they need
DaCe's runtime headers).

Dace is imported lazily, so the command degrades to a clean exit code on a harness install without it. The
module itself never imports dace: its naming and option plumbing is testable standalone."""

import contextlib
import copy
import dataclasses
import importlib
import os
import pathlib
import shlex
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Protocol

from hpcagent_bench.frameworks.dace_options import DacePipelineOptions, from_env

if TYPE_CHECKING:
    from dace import SDFG

    from hpcagent_bench.spec import BenchSpec

__all__ = [
    "DACE_EMIT_TARGETS",
    "DaceEmitError",
    "DaceEmitUnavailable",
    "Unit",
    "default_pipeline_for",
    "device_units",
    "emit_kernel_sources",
    "emit_program_sources",
    "write_code_objects",
]

#: The codegen targets ``dace emit`` names, and the DaCe ``compiler.cuda.backend`` each GPU one selects.
DACE_EMIT_TARGETS = ("cpu", "cuda", "hip")


class DaceEmitError(RuntimeError):
    """A kernel, pipeline or code-generation failure ``dace emit`` reports as exit 1."""


class DaceEmitUnavailable(ImportError):
    """Dace is not importable in this interpreter: ``dace emit`` skips (exit 2) instead of failing."""


class _CodeObject(Protocol):
    """The one slice of a dace ``CodeObject`` the writer reads: a name, its code, and the language, which is the
    file extension dace itself uses for the file (``cpp``, ``h``, ``cu``)."""

    name: str
    code: str
    language: str


@dataclasses.dataclass(slots=True)
class Unit:
    """One file of the self-contained form, in the shape :func:`write_code_objects` writes."""

    name: str
    code: str
    language: str


def default_pipeline_for(target: str) -> str:
    """The pipeline ``dace emit`` runs when none is named: the ``parallel`` pipeline for the target's device --
    the pipeline the ``dace_gpu_parallel`` / ``dace_cpu_parallel`` columns score."""
    return "parallel_cpu" if target == "cpu" else "parallel_gpu"


def write_code_objects(objects: Iterable[_CodeObject], out_dir: str | os.PathLike[str]) -> list[pathlib.Path]:
    """Write each code object as ``<name>.<language>`` under ``out_dir`` (created); a repeated (name, language)
    pair gets ``-2``, ``-3``, ... appended to its stem, so dace's frame and device units never overwrite each
    other. :returns: the written paths, in input order."""
    root = pathlib.Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    seen: dict[tuple[str, str], int] = {}
    written: list[pathlib.Path] = []
    for obj in objects:
        n = seen.get((obj.name, obj.language), 0) + 1
        seen[(obj.name, obj.language)] = n
        stem = obj.name if n == 1 else f"{obj.name}-{n}"
        path = root / f"{stem}.{obj.language}"
        path.write_text(obj.code)
        written.append(path)
    return written


@contextlib.contextmanager
def _pipeline_config(config: Iterable[tuple[tuple[str, ...], str | bool]]) -> Iterator[None]:
    """The pipeline's code-generator configuration, applied for the duration and restored afterwards (an
    inspection command must not leak process-global config). A key the ``DACE_*`` environment names keeps that
    value, exactly as :func:`hpcagent_bench.frameworks.dace_framework.apply_pipeline_config` reads it; a key this
    DaCe does not declare is skipped, as :func:`...pin_build_caching` tolerates."""
    import dace

    saved: list[tuple[tuple[str, ...], object]] = []
    try:
        for path, value in config:
            if "_".join(("DACE", *path)) in os.environ:
                continue
            try:
                saved.append((path, dace.Config.get(*path)))
            except KeyError:
                continue  # this DaCe declares no such key; codegen reads its default
            dace.Config.set(*path, value=value)
        yield
    finally:
        for path, previous in saved:
            dace.Config.set(*path, value=previous)


def emit_program_sources(
    sdfg: "SDFG",
    out_dir: str | os.PathLike[str],
    *,
    target: str = "cuda",
    pipeline: str | None = None,
    options: DacePipelineOptions | None = None,
    name: str | None = None,
    spec: "BenchSpec | None" = None,
) -> list[pathlib.Path]:
    """Run one SDFG pipeline over a deep copy of the parsed base ``sdfg`` and write the generated sources under
    ``out_dir``; :returns: the written paths. The GPU residuals ride in ``options`` (default: the environment's,
    so ``HPCAGENT_BENCH_DACE_*`` steers this command the way it steers a column); ``name`` renames the SDFG (so
    the file names carry the kernel and pipeline, not the parsed module path). With the kernel's ``spec`` the
    pipeline sees its declared symbols as the column does, and a GPU target writes :func:`device_units` in place
    of DaCe's frame.

    Code generation only: nothing is compiled, so this works on a GPU-less host once the backend is named --
    which this does for the duration, since dace's generator probes ``compiler.cuda.backend`` even for the CPU
    frame."""
    if target not in DACE_EMIT_TARGETS:
        raise DaceEmitError(f"unknown emit target {target!r}; expected one of {', '.join(DACE_EMIT_TARGETS)}")
    try:
        dace = importlib.import_module("dace")
    except (ImportError, OSError) as exc:
        raise DaceEmitUnavailable(f"dace is not importable ({type(exc).__name__}: {exc}); nothing emitted") from exc

    from hpcagent_bench.frameworks.dace_framework import PipelineContext, pipeline_named

    pipe = pipeline_named(pipeline or default_pipeline_for(target))
    opts = options or from_env()
    gpu = target in ("cuda", "hip")
    ctx = PipelineContext(
        device=dace.dtypes.DeviceType.GPU if gpu else dace.dtypes.DeviceType.CPU,
        maximal=opts.maximal,
        remove_transients=opts.remove_transients,
    )
    if spec is not None:
        from hpcagent_bench.cpf import canonical

        ctx = dataclasses.replace(
            ctx,
            small=canonical.small_symbols(spec),
            nonnegative=canonical.nonnegative_symbols(spec),
            positive=canonical.positive_symbols(spec),
        )
    sdfg = copy.deepcopy(sdfg)
    sdfg.name = name or f"{sdfg.name}_{pipe.name}"  # pyright: ignore[reportAttributeAccessIssue]  # dace types Property, not its str
    with _pipeline_config(pipe.config), _backend_config(target):
        try:
            pipe.transform(sdfg, ctx)
            if gpu:
                # The two steps build_sdfgs() applies to every GPU variant after the pipeline, so what lands on
                # disk is what the column would compile.
                from hpcagent_bench.frameworks.dace_framework import enforce_gpu_residency, pool_gpu_transients

                enforce_gpu_residency(sdfg)
                pool_gpu_transients(sdfg)
            objects = (
                device_units(spec, sdfg, target)
                if spec is not None and gpu
                else [Unit(str(obj.name), str(obj.code), str(obj.language)) for obj in sdfg.generate_code()]
            )
        except DaceEmitUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 -- every pipeline/codegen failure is reported, not raised past the CLI
            raise DaceEmitError(f"{pipe.name} on {sdfg.name}: {type(exc).__name__}: {exc}") from exc
    return write_code_objects(objects, out_dir)


def device_units(spec: "BenchSpec", sdfg: "SDFG", target: str) -> list[Unit]:
    """The offloaded ``sdfg`` as the self-contained pair DaCe renders (``dace.codegen.cpf``, no DaCe runtime): the
    host unit ``<name>.cpp``, whose entry is the judge's C ABI symbol for ``spec``, the device unit
    (``<name>.cu`` / ``<name>.hip``), and ``<name>.build``, the commands that build the two into one shared
    library. A construct the renderer cannot hold is refused by name (``NotImplementedError``)."""
    from hpcagent_bench import languages
    from hpcagent_bench.categories import Language
    from hpcagent_bench.cpf import bridge
    from hpcagent_bench.cpf.cache import DEFAULT_HINTS, Target

    name = str(sdfg.name)
    form = bridge.render_canonical(spec, name, sdfg, target, "", Target.GPU, DEFAULT_HINTS)
    language = Language.parse(target)
    (_, host), (_, device) = languages.source_units(language, name)
    try:
        compile_flags, link_flags = languages.library_build_flags(language, form.libraries)
        if target == "cuda":
            from dace.codegen.cpf import CUDA_BUILD_FLAGS

            # What DaCe builds a cuda unit with; the standard is the compiler block's own.
            compile_flags = (*compile_flags, *(flag for flag in CUDA_BUILD_FLAGS if not flag.startswith("-std=")))
        commands = languages.build_shared_lib_commands(
            language,
            pathlib.Path(device),
            pathlib.Path(f"lib{name}.so"),
            extra_compile=compile_flags,
            extra_link=link_flags,
            extra_sources=[pathlib.Path(host)],
        )
        build = "\n".join(shlex.join(argv) for argv in commands)
    except (KeyError, FileNotFoundError) as exc:
        build = f"# no {target} compiler is configured on this host: {exc}"
    header = (
        f"# {form.entry}: device pointers for the arrays, workspace bytes {form.workspace_bytes}, "
        f"libraries {', '.join(form.libraries) or 'none'}.\n"
        "# The offload architecture is the emitting host's; on another GPU, run `dace emit` there.\n"
    )
    return [
        Unit(name, form.code, pathlib.Path(host).suffix[1:]),
        Unit(name, form.device_code, pathlib.Path(device).suffix[1:]),
        Unit(name, header + build + "\n", "build"),
    ]


@contextlib.contextmanager
def _backend_config(target: str) -> Iterator[None]:
    """``compiler.cuda.backend`` pointed at the emit target for the duration (dace autodetects otherwise, and an
    autodetect on a GPU-less host refuses to run at all)."""
    import dace

    key = ("compiler", "cuda", "backend")
    saved = dace.Config.get(*key)
    dace.Config.set(*key, value="cuda" if target == "cpu" else target)
    try:
        yield
    finally:
        dace.Config.set(*key, value=saved)


def emit_kernel_sources(
    kernel: str,
    out_dir: str | os.PathLike[str] | None = None,
    *,
    target: str = "cuda",
    pipeline: str | None = None,
    datatype: str | None = None,
    options: DacePipelineOptions | None = None,
    frame: bool = False,
) -> list[pathlib.Path]:
    """The ``dace emit`` command's body: resolve the selector ``kernel`` to exactly one benchmark, parse its
    generated ``@dace.program`` fresh (the on-disk SDFG cache is neither read nor written), and
    :func:`emit_program_sources` it. ``out_dir`` defaults to the kernel's ``.cache/emit/<pipeline>/``. ``frame``
    writes DaCe's frame units for a GPU target too, not the self-contained pair."""
    from hpcagent_bench import array_facts, modules
    from hpcagent_bench.autogen import ensure as autogen_ensure
    from hpcagent_bench.spec import KERNELS

    try:
        names = KERNELS.select(kernel)
    except (KeyError, ValueError) as exc:
        raise DaceEmitError(str(exc)) from exc
    if len(names) != 1:
        raise DaceEmitError(f"selector {kernel!r} matches {len(names)} kernels; name exactly one of {names}")
    bench_key = names[0]
    pipe_name = pipeline or default_pipeline_for(target)
    standalone = target != "cpu" and not frame
    if standalone and datatype is not None:
        raise DaceEmitError(
            "the self-contained form's entry is the judge's C ABI, which has the manifest's precision; "
            "a datatype applies to the frame units (--frame)"
        )

    try:
        import dace  # noqa: F401 -- fail here, before anything is written, when dace is absent
    except (ImportError, OSError) as exc:
        raise DaceEmitUnavailable(f"dace is not importable ({type(exc).__name__}: {exc}); nothing emitted") from exc

    # The precision globals every generated impl annotates; unbound, the import fails on None[...].
    from hpcagent_bench import paths
    from hpcagent_bench.cpf.canonical import bind_precision
    from hpcagent_bench.frameworks.benchmark import Benchmark

    bind_precision(datatype or "")
    bench = Benchmark(bench_key)
    root = (
        pathlib.Path(out_dir)
        if out_dir is not None
        else paths.BENCHMARKS / bench.spec.relative_path / ".cache" / "emit" / pipe_name
    )
    autogen_ensure(bench.bname, ("dace",))
    try:
        module = importlib.import_module(bench.impl_module("dace"))
    except ImportError as exc:
        raise DaceEmitError(f"the generated dace module for {bench_key} did not import: {exc}") from exc
    from hpcagent_bench.frameworks.dace_framework import DaceProgram

    program = modules.member(module, bench.spec.func_name, DaceProgram)
    if not isinstance(program, DaceProgram):
        raise DaceEmitError(f"{bench.impl_module('dace')} names no @dace.program {bench.spec.func_name!r}")
    array_facts.apply_to_program(bench.spec, program)
    sdfg = program.to_sdfg(simplify=False)
    if standalone:
        from hpcagent_bench.cpf.canonical import fold_closure_arrays

        # An array the parse lifted out of a constant expression is no parameter of the C ABI.
        fold_closure_arrays(program, sdfg)
    return emit_program_sources(
        sdfg,
        root,
        target=target,
        pipeline=pipe_name,
        options=options,
        name=f"{bench.spec.module_name}_{pipe_name}",
        spec=bench.spec if standalone else None,
    )
