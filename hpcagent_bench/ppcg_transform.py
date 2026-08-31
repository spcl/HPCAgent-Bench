# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Running ``ppcg``: the ONE place the PPCG column's source-to-source step is spelled.

PPCG (Verdoolaege et al., TACO 2013, doi 10.1145/2400682.2400713) is the GPU sibling of Pluto:
same polyhedral front end (pet + isl), same ``#pragma scop`` input, CUDA out instead of tiled
OpenMP C. The two columns are the polyhedral pair -- Pluto on CPU, PPCG on GPU -- so the scop
selection and the affine guard are IMPORTED from :mod:`hpcagent_bench.pluto_transform` rather than
restated: a scop one column refuses is a scop the other cannot legally transform either.

PPCG emits TWO files per input (``<stem>_host.cu`` carrying the driver and ``<stem>_kernel.cu``
carrying the kernels) plus a ``<stem>_kernel.hu`` header they share. Both .cu files compile; the
header does not, so it is not returned.

PPCG HAS NO AMD TARGET -- ``--target`` takes ``c``, ``cuda`` or ``opencl`` and nothing else -- so on
a ROCm host what it wrote is translated by ROCm's own ``hipify-perl`` and built by hipcc. Same
polyhedral transform, same measurement; only the GPU it lands on differs, which is why the vendor is
DERIVED from the installed toolchain (:func:`hpcagent_bench.languages.gpu_backend`) instead of being
declared per column.
"""
import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import List, Optional, Sequence, Tuple

from hpcagent_bench.frameworks.errors import NotSupportedByFramework
from hpcagent_bench.languages import LANG_EXT, gpu_backend
from hpcagent_bench.pluto_transform import assert_affine, scop_inputs

#: The framework name this module transforms for -- used in every decline message.
FRAMEWORK = "ppcg"

#: How ``ppcg`` is invoked. ``--target=cuda`` is the whole point of the column; the two tile-size
#: knobs are ppcg's own defaults spelled out, so a host that changes them cannot silently change
#: what this column measures.
PPCG_ARGS: Tuple[str, ...] = ("--target=cuda", "--tile", "--tile-size=32")

#: ROCm's CUDA-to-HIP source translator, run on ppcg's output on an AMD host (see the module
#: docstring). Named once so the decline message and the call cannot disagree about the tool.
HIPIFY = "hipify-perl"


def ppcg_exe() -> Optional[str]:
    """``ppcg`` on PATH, or ``None`` when PPCG is not installed."""
    return shutil.which("ppcg")


def hipify_exe() -> Optional[str]:
    """``hipify-perl`` on PATH, or ``None`` when this host has no ROCm. Mirrors :func:`ppcg_exe`."""
    return shutil.which(HIPIFY)


def transformed_paths(scop: pathlib.Path) -> List[pathlib.Path]:
    """The two GPU sources ppcg's transform of ``scop`` compiles to, in compile order.

    ``.hip`` on a ROCm host rather than the ``.cu`` ppcg wrote: the extension is what makes hipcc
    build them for the local device instead of reading them as CUDA for an NVIDIA one.
    """
    stem, ext = scop.stem, LANG_EXT[gpu_backend()]
    return [scop.with_name(f"{stem}_host.{ext}"), scop.with_name(f"{stem}_kernel.{ext}")]


def hipify(scratch: pathlib.Path, stem: str) -> None:
    """Rewrite ppcg's CUDA output in ``scratch`` as HIP, before anything is moved next to the scop.

    The shared ``<stem>_kernel.hu`` is translated IN PLACE and keeps its name: ppcg has already
    written the ``#include`` that names it into both halves and offers no way to rename it.
    """
    exe = hipify_exe()
    if exe is None:  # transformed_sources already declines for this; here it keeps the argv typed
        raise NotSupportedByFramework(FRAMEWORK, stem, f"{HIPIFY} is not installed on this host")
    for half in ("host", "kernel"):
        cu = scratch / f"{stem}_{half}.cu"
        translated = subprocess.run([exe, str(cu)], capture_output=True, text=True, check=True)
        (scratch / f"{stem}_{half}.{LANG_EXT['hip']}").write_text(translated.stdout)
    subprocess.run([exe, "-inplace", str(scratch / f"{stem}_kernel.hu")], capture_output=True, check=True)


def run_ppcg(scop: pathlib.Path,
             args: Sequence[str] = PPCG_ARGS,
             timeout: Optional[float] = None) -> Tuple[List[str], subprocess.CompletedProcess]:
    """Transform one scop with ``ppcg``. Returns ``(argv, result)``.

    ppcg names its outputs after the INPUT and writes them into the current directory, with no
    ``-o`` for the pair, so it runs in a throwaway cwd and the results are moved next to ``scop``
    only on success -- a failed run leaves no half-written .cu for the build to pick up.
    """
    exe = ppcg_exe()
    argv = [str(exe), *args, str(scop)]
    with tempfile.TemporaryDirectory() as scratch:
        proc = subprocess.run(argv, cwd=scratch, capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0:
            if gpu_backend() == "hip":
                hipify(pathlib.Path(scratch), scop.stem)
            for produced in transformed_paths(scop):
                src = pathlib.Path(scratch) / produced.name
                if src.is_file():
                    os.replace(src, produced)
            header = pathlib.Path(scratch) / f"{scop.stem}_kernel.hu"
            if header.is_file():
                os.replace(header, scop.with_name(header.name))
    return argv, proc


def transformed_sources(cpp_backend: pathlib.Path, base: str) -> List[pathlib.Path]:
    """The ppcg-transformed CUDA the ``ppcg`` column compiles, generated on demand.

    Mirrors :func:`pluto_transform.transformed_sources`: regenerate when stale, reuse when fresh,
    and DECLINE rather than fall back to untransformed source -- a PPCG column built from the
    emitted C would be an nvcc column wearing PPCG's label.
    """
    scops = scop_inputs(cpp_backend, base)
    if not scops:
        raise NotSupportedByFramework(FRAMEWORK, base, "the translator emitted no #pragma scop for this kernel")
    if ppcg_exe() is None:
        raise NotSupportedByFramework(FRAMEWORK, base, "ppcg is not installed on this host")
    if gpu_backend() == "hip" and hipify_exe() is None:
        raise NotSupportedByFramework(FRAMEWORK, base,
                                      f"ppcg emits CUDA and this host builds HIP, but {HIPIFY} is not installed")
    out: List[pathlib.Path] = []
    for scop in scops:
        assert_affine(scop, base)
        produced = transformed_paths(scop)
        if any(not p.exists() or p.stat().st_mtime < scop.stat().st_mtime for p in produced):
            argv, proc = run_ppcg(scop)
            if proc.returncode != 0 or any(not p.is_file() for p in produced):
                raise NotSupportedByFramework(FRAMEWORK, base,
                                              f"ppcg rejected {scop.name}: {proc.stderr.strip()[-500:]}")
        out.extend(produced)
    return out
