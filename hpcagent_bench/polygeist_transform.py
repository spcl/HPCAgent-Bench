# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Running Polygeist: the ONE place the ``polygeist_cpu`` / ``polygeist_gpu`` pipelines are spelled.

The columns compile what this module writes from the C target's scop (``_pluto_input.c``, the file polycc reads,
selected by :func:`hpcagent_bench.pluto_transform.scop_inputs`): ``cgeist`` lowers it to affine MLIR,
``polygeist-opt`` parallelizes it with MLIR's own passes, and the image's LLVM turns it into LLVM IR that the
column's clang compiles (``compilers.yaml``: ``clang-polygeist`` / ``clang-polygeist-gpu``).

* CPU: ``affine-parallelize`` marks the outermost parallel loop (``max-nested=1``: a nested one would open a
  parallel region per iteration), ``convert-scf-to-openmp`` makes it an OpenMP worksharing loop (see
  :data:`CPU_PASSES` for why Polymer does not run).
* GPU: ``affine-parallelize`` up to three levels, each parallel nest tiled 16x16x1 into blocks and threads,
  ``convert-parallel-loops-to-gpu`` launches it, ROCDL compiles the kernel for ``<gfx>:xnack+``. The arrays stay
  where the caller allocated them (``HSA_XNACK=1``, see :func:`run_env`), so there is no data-movement step.
  Polymer is not run here: its tiled bounds are ``min``/``max`` expressions the GPU mapping does not accept, and a
  nest that is not mapped would run on the host.

cgeist types a VLA parameter ``double (*A)[N]`` as ``memref<?x?xf64>``, which only MLIR's memref-descriptor ABI can
pass, so the MLIR function is renamed ``<fn>_polygeist`` and :func:`shim_source` writes the C function the column
calls: the scop's own name and argument order (the pluto binding's), building each descriptor from the binding's
shape.

Every failure declines (:class:`NotSupportedByFramework`) rather than compile anything else under this label.
"""

import pathlib
import re
import shutil
import subprocess

from hpcagent_bench import config, flags
from hpcagent_bench.frameworks.errors import NotSupportedByFramework
from hpcagent_bench.pluto_transform import run_bounded, scop_inputs
from hpcagent_bench.translators.numpyto_c.bindings import BindingArg, read_binding
from hpcagent_bench.translators.numpyto_common.dtypes import SCALAR_KINDS, info_for_kind

__all__ = [
    "CGEIST_ARGS",
    "CPU_PASSES",
    "FRAMEWORK",
    "GPU_LOWERING",
    "GPU_PASSES",
    "TOOLS",
    "run_env",
    "shim_source",
    "transform_scop",
    "transformed_sources",
]

#: The framework name this module transforms for -- used in every decline message.
FRAMEWORK = "polygeist"

#: cgeist, C scop -> affine MLIR. ``--raise-scf-to-affine`` is what makes the loops affine at all.
CGEIST_ARGS: tuple[str, ...] = ("-S", flags.OPT_LEVEL, "--raise-scf-to-affine")

#: polygeist-opt, CPU: outermost parallel loop to OpenMP, LLVM dialect with descriptor ABI. Not
#: ``--polyhedral-opt``: the image builds Polymer, but its Pluto schedule computed heat_3d wrong (max rel error
#: 3e16 against numpy) and ran gemm 4.7x slower than this pipeline at preset S.
CPU_PASSES: tuple[str, ...] = (
    "--affine-parallelize=max-nested=1",
    "--lower-affine",
    "--canonicalize",
    "--convert-scf-to-openmp",
    "--convert-polygeist-to-llvm=use-c-style-memref=0",
)

#: polygeist-opt, GPU: parallel nests to tiled ``gpu.launch_func`` kernels.
GPU_PASSES: tuple[str, ...] = (
    "--affine-parallelize=max-nested=3",
    "--lower-affine",
    "--canonicalize-polygeist",
    "--scf-parallel-loop-tiling=parallel-loop-tile-sizes=16,16,1 no-min-max-bounds=true",
    "--gpu-map-parallel-loops",
    "--convert-parallel-loops-to-gpu",
    "--gpu-kernel-outlining",
    "--lower-affine",
    "--canonicalize",
)

#: polygeist-opt, GPU: the kernels to ROCDL for ``<gfx>:xnack+`` (``{chip}`` = the node's gfx target), the host to the
#: LLVM dialect -- upstream passes only, run by polygeist-opt because only it parses what cgeist emits. The
#: kernels are then serialized by the image's mlir-opt (``gpu-module-to-binary``), which initializes the AMDGPU
#: backend that polygeist-opt does not.
GPU_LOWERING = (
    "builtin.module(gpu.module(strip-debuginfo,convert-scf-to-cf,"
    "convert-gpu-to-rocdl{{chipset={chip} index-bitwidth=64}},reconcile-unrealized-casts),"
    "rocdl-attach-target{{chip={chip} features=+xnack O=3}},gpu-to-llvm,convert-scf-to-cf,finalize-memref-to-llvm,"
    "convert-math-to-llvm,convert-arith-to-llvm,convert-index-to-llvm,convert-cf-to-llvm,convert-func-to-llvm,"
    "reconcile-unrealized-casts)"
)

#: Every executable a transform runs.
TOOLS: tuple[str, ...] = ("cgeist", "polygeist-opt", "mlir-opt", "mlir-translate", "clang")

#: The scop's own function: the last ``void <name>(`` before ``#pragma scop``.
FUNCTION_RE = re.compile(r"\bvoid\s+(\w+)\s*\(")


def run_env() -> dict[str, str]:
    """What a ``polygeist_gpu`` process must run under: its kernels read host memory, which faults
    (\"memory access fault by GPU\") unless XNACK is on when HSA initializes."""
    return {"HSA_XNACK": "1"}


def missing_tools() -> list[str]:
    """The :data:`TOOLS` not on PATH."""
    return [tool for tool in TOOLS if shutil.which(tool) is None]


def scop_function(scop_text: str) -> str:
    """The name of the function holding ``#pragma scop``."""
    names = FUNCTION_RE.findall(scop_text[: scop_text.index("#pragma scop")])
    if not names:
        raise ValueError("no function encloses #pragma scop")
    return names[-1]


def shim_source(function: str, args: list[BindingArg]) -> str:
    """The C function ``function(args...)`` that calls MLIR's ``_mlir_ciface_<function>_polygeist`` with one
    memref descriptor per array, its strides the suffix products of the binding's shape."""
    params: list[str] = []
    callee: list[str] = []
    setup: list[str] = []
    call: list[str] = []
    for arg in args:
        if arg.kind in SCALAR_KINDS:
            params.append(f"{info_for_kind(arg.kind).c} {arg.name}")
            callee.append(info_for_kind(arg.kind).c)
            call.append(arg.name)
            continue
        shape = arg.shape or ["1"]
        rank = len(shape)
        strides = [" * ".join(f"(int64_t)({extent})" for extent in shape[i + 1 :]) or "1" for i in range(rank)]
        params.append(f"void *{arg.name}")
        callee.append("void *")
        setup.append(
            f"  struct {{ void *allocated, *aligned; int64_t offset, sizes[{rank}], strides[{rank}]; }} "
            f"d_{arg.name} = {{{arg.name}, {arg.name}, 0, "
            f"{{{', '.join(f'(int64_t)({extent})' for extent in shape)}}}, {{{', '.join(strides)}}}}};"
        )
        call.append(f"&d_{arg.name}")
    return "\n".join(
        [
            "#include <stdbool.h>",
            "#include <stdint.h>",
            f"void _mlir_ciface_{function}_polygeist({', '.join(callee) or 'void'});",
            f"void {function}({', '.join(params)}) {{",
            *setup,
            f"  _mlir_ciface_{function}_polygeist({', '.join(call)});",
            "}",
            "",
        ]
    )


def entry_point(mlir: str, function: str) -> str:
    """``mlir`` with ``function`` renamed ``<function>_polygeist`` and given MLIR's C interface (the
    ``_mlir_ciface_`` wrapper :func:`shim_source` calls) -- that function only: the prelude's helpers are emitted
    into every precision's module, and a wrapper for each would be defined twice."""
    head, sep, rest = mlir.partition(f"func.func @{function}(")
    if not sep:
        raise ValueError(f"cgeist emitted no function {function}")
    signature, brace, body = rest.partition(" {\n")
    if " attributes {" in signature:
        signature = signature.replace(" attributes {", " attributes {llvm.emit_c_interface, ", 1)
    else:
        signature += " attributes {llvm.emit_c_interface}"
    return f"{head}func.func @{function}_polygeist({signature}{brace}{body}"


def run_tool(cmd: list[str], scop: pathlib.Path, base: str) -> subprocess.CompletedProcess:
    """One pipeline step under Pluto's polycc time bound; a failure declines with the tool's own stderr."""
    try:
        proc = run_bounded(cmd, cwd=str(scop.parent), timeout=config.current().oracle.polycc_timeout_s)
    except subprocess.TimeoutExpired:
        raise NotSupportedByFramework(FRAMEWORK, base, f"{pathlib.Path(cmd[0]).name} timed out on {scop.name}")
    if proc.returncode != 0:
        raise NotSupportedByFramework(
            FRAMEWORK, base, f"{pathlib.Path(cmd[0]).name} rejected {scop.name}: {proc.stderr.strip()[-500:]}"
        )
    return proc


def transform_scop(scop: pathlib.Path, base: str, device: str) -> list[pathlib.Path]:
    """``[<stem>_polygeist_<device>.ll, <stem>_polygeist_<device>_shim.c]`` for one scop."""
    function = scop_function(scop.read_text())
    precision = re.search(r"_(fp\d+)_", scop.name)
    binding_path = scop.parent / f"{base}_{precision.group(1) if precision else ''}_pluto_binding.json"
    if not binding_path.is_file():
        raise NotSupportedByFramework(FRAMEWORK, base, f"no pluto binding {binding_path.name} for {scop.name}")
    stem = scop.parent / f"{scop.stem}_polygeist_{device}"
    affine = stem.with_suffix(".affine.mlir")
    lowered = stem.with_suffix(".llvm.mlir")
    resource_dir = run_tool(["clang", "-print-resource-dir"], scop, base).stdout.strip()
    run_tool(
        [
            "cgeist",
            str(scop),
            *CGEIST_ARGS,
            f"--function={function}",
            f"-resource-dir={resource_dir}",
            "-o",
            str(affine),
        ],
        scop,
        base,
    )
    affine.write_text(entry_point(affine.read_text(), function))
    if device == "cpu":
        run_tool(["polygeist-opt", str(affine), *CPU_PASSES, "-o", str(lowered)], scop, base)
    else:
        mapped = stem.with_suffix(".gpu.mlir")
        run_tool(["polygeist-opt", str(affine), *GPU_PASSES, "-o", str(mapped)], scop, base)
        if "gpu.launch_func" not in mapped.read_text():
            raise NotSupportedByFramework(FRAMEWORK, base, f"no loop nest of {scop.name} maps to the GPU")
        hosted = stem.with_suffix(".host.mlir")
        pipeline = GPU_LOWERING.format(chip=flags.detect_gfx())
        run_tool(["polygeist-opt", str(mapped), f"--pass-pipeline={pipeline}", "-o", str(hosted)], scop, base)
        run_tool(["mlir-opt", str(hosted), "--gpu-module-to-binary", "-o", str(lowered)], scop, base)
    ll = stem.with_suffix(".ll")
    run_tool(["mlir-translate", "--mlir-to-llvmir", str(lowered), "-o", str(ll)], scop, base)
    shim = stem.parent / f"{stem.name}_shim.c"
    shim.write_text(shim_source(function, read_binding(binding_path).args))
    return [ll, shim]


def transformed_sources(cpp_backend: pathlib.Path, base: str, device: str) -> list[pathlib.Path]:
    """The LLVM IR and C shims the ``polygeist_<device>`` column compiles, regenerated on every build.

    Declines -- never returns the untransformed source -- when a tool is missing, when the translator emitted no
    scop, or when any step of the pipeline rejects one."""
    scops = scop_inputs(cpp_backend, base)
    if not scops:
        raise NotSupportedByFramework(FRAMEWORK, base, "the translator emitted no #pragma scop for this kernel")
    missing = missing_tools()
    if missing:
        raise NotSupportedByFramework(FRAMEWORK, base, f"not on PATH: {', '.join(missing)}")
    return [path for scop in scops for path in transform_scop(scop, base, device)]
