"""Lazy wrapper around the dace work-depth + total-volume analyses.

The analysis API lives on the ``modernize-perf-analysis-and-memory-volume``
branch of github.com/spcl/dace and is not yet merged into main. Until it
lands we *try* to import it; if it isn't available, the helpers in here
return ``None`` and ``plot_roofline.py`` falls back to whatever flops /
bytes were declared statically in ``bench_info``. No crash either way.

Public entry point: :func:`get_flops_bytes(short_name, preset, datatype)`.
"""

from __future__ import annotations

import importlib
import pathlib
import sys
from typing import Optional, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _try_import_analysis():
    """Return (work_depth_module, total_volume_module) or (None, None)."""
    try:
        wd = importlib.import_module("dace.sdfg.performance_evaluation.work_depth")
        tv = importlib.import_module("dace.sdfg.performance_evaluation.total_volume")
        return wd, tv
    except ImportError:
        return None, None


def _load_bench_info(short_name: str) -> Optional[dict]:
    """Return the legacy ``benchmark`` dict for ``short_name`` from its
    co-located YAML manifest (the source of truth), or None if unknown.
    """
    from optarena.spec import BenchSpec
    from optarena.emit_bridge import legacy_bench_info_dict
    try:
        return legacy_bench_info_dict(BenchSpec.load(short_name))["benchmark"]
    except Exception:
        return None


def _ensure_dace_dtypes(datatype: str = "float64"):
    """Populate the dace_framework module-level ``dc_float`` /
    ``dc_complex_float`` globals.

    The dace kernel modules read these at import time (for the
    @dc.program type annotations); without a prior
    ``DaceFramework.set_datatype(...)`` call they are ``None`` and any
    annotation like ``dc_float[N]`` raises ``TypeError: 'NoneType' is
    not subscriptable``. The analysis path doesn't instantiate a
    framework so it needs to set them itself.
    """
    from optarena.infrastructure import dace_framework
    from dace import float32, float64, complex64, complex128
    if datatype == "float32":
        dace_framework.dc_float = float32
        dace_framework.dc_complex_float = complex64
    else:
        dace_framework.dc_float = float64
        dace_framework.dc_complex_float = complex128


def _load_dace_kernel(bench_info: dict, datatype: str = "float64"):
    """Import the corresponding ``*_dace.py`` module and return its
    ``@dc.program`` callable (the bench's ``func_name`` attribute).
    """
    _ensure_dace_dtypes(datatype)
    rel = bench_info["relative_path"].replace("/", ".")
    mod_name = bench_info["module_name"]
    full = f"optarena.benchmarks.{rel}.{mod_name}_dace"
    try:
        mod = importlib.import_module(full)
        # Force re-evaluation of the module's @dc.program annotations
        # in case dc_float was set differently last time around (a
        # process that flips between fp32 and fp64 within one run).
        if datatype != vars(mod).get("_optarena_analysis_dtype"):
            importlib.reload(mod)
            mod._optarena_analysis_dtype = datatype
    except Exception as e:
        print(f"[dace_analysis] cannot import {full}: {e}", file=sys.stderr)
        return None
    fn_name = bench_info["func_name"]
    return vars(mod).get(fn_name)


def _to_sdfg(prog) -> Optional["dace.SDFG"]:  # noqa: F821
    """Lower a ``@dc.program`` callable to a raw (un-optimised) SDFG."""
    if prog is None or "to_sdfg" not in dir(prog):  # method duck-type (dir, not hasattr)
        return None
    try:
        return prog.to_sdfg()
    except Exception as e:
        print(f"[dace_analysis] .to_sdfg() failed: {e}", file=sys.stderr)
        return None


def _auto_optimize_copy(sdfg, device: str = "cpu"):
    """Return a deep-copied SDFG with ``auto_optimize`` applied.

    Returns the original sdfg if the auto-opt path fails (older dace
    builds don't have it on this exact import path).
    """
    if sdfg is None:
        return None
    try:
        from dace.transformation.auto.auto_optimize import auto_optimize
        from dace import dtypes
        from copy import deepcopy
        opt = deepcopy(sdfg)
        device_enum = (dtypes.DeviceType.GPU if device == "gpu" else dtypes.DeviceType.CPU)
        auto_optimize(opt, device=device_enum)
        return opt
    except Exception as e:
        print(f"[dace_analysis] auto_optimize failed (using un-optimised "
              f"SDFG): {e}", file=sys.stderr)
        return sdfg


def _substitute_symbols(expr, preset_params: dict, datatype: str):
    """Substitute concrete preset values into a sympy expression and
    fold to int. Returns None on failure.

    ``datatype`` is used to interpret memory volumes that the
    total_volume pass reports in *bytes* — the analysis already counts
    bytes (not elements), so the only multiplication left to do is for
    fp32 vs fp64 size factors that the analysis may not have applied.
    For now we trust the value as-is.
    """
    try:
        # `expr` is a sympy expression with symbols whose names match
        # the bench's preset parameter keys.
        free = {str(s): s for s in expr.free_symbols}
        subs = {free[name]: val for name, val in preset_params.items() if name in free}
        value = expr.subs(subs)
        if value.free_symbols:
            # still has unresolved symbols — bail
            return None
        return int(value)
    except Exception as e:
        print(f"[dace_analysis] symbol substitution failed for {expr!r}: {e}", file=sys.stderr)
        return None


def get_flops_bytes(short_name: str,
                    preset: str,
                    datatype: str = "float64",
                    cache: Optional[dict] = None,
                    device: str = "cpu") -> Tuple[Optional[int], Optional[int]]:
    """Return (flops, bytes) for the named benchmark + preset, or
    (None, None) if the dace analysis branch isn't installed or the
    benchmark doesn't have a dace implementation.

    A ``cache`` dict, if supplied, is used as a process-local memo
    indexed by (short_name, preset, datatype).
    """
    if cache is not None and (short_name, preset, datatype) in cache:
        return cache[(short_name, preset, datatype)]

    wd, tv = _try_import_analysis()
    if wd is None or tv is None:
        if cache is not None:
            cache[(short_name, preset, datatype)] = (None, None)
        return None, None

    bench_info = _load_bench_info(short_name)
    if bench_info is None:
        return None, None

    presets = bench_info.get("parameters", {})
    if preset not in presets:
        return None, None
    preset_params = presets[preset]

    prog = _load_dace_kernel(bench_info, datatype=datatype)
    raw_sdfg = _to_sdfg(prog)
    if raw_sdfg is None:
        if cache is not None:
            cache[(short_name, preset, datatype)] = (None, None)
        return None, None
    # auto-opt copy for the bytes analysis (per user direction); the
    # work analysis sticks with the raw SDFG because auto_optimize
    # converts python tasklets to C++ which work_depth can't count.
    opt_sdfg = _auto_optimize_copy(raw_sdfg, device=device)

    work_expr = None
    try:
        from dace.sdfg.performance_evaluation import work_depth as wdmod
        tasklet_fn = vars(wdmod).get("get_tasklet_work_depth", vars(wdmod).get("get_tasklet_work"))
        if tasklet_fn is not None:
            # Expand library nodes (matmul, reduce, etc.) so the
            # per-tasklet work counter actually sees compute. Without
            # this an unexpanded MatMul LibraryNode reports 0 work.
            from copy import deepcopy
            sdfg_for_work = deepcopy(raw_sdfg)
            try:
                sdfg_for_work.expand_library_nodes()
            except Exception:
                pass
            # work_depth is allowed to fail on SDFGs with LoopRegions /
            # ConditionalBlocks — the upstream fix lives in a pending
            # PR. For roofline plotting we only need ``bytes``; FLOPs
            # remain a nice-to-have. Don't lower into the old-style
            # state machine just to get a number here.
            w_d_map = {}
            wdmod.analyze_sdfg(sdfg_for_work, w_d_map, tasklet_fn, assumptions=[])
            # The top-level entry (max depth = 0) holds the SDFG-wide
            # total. The branch uses get_uuid(sdfg) as the key.
            try:
                from dace.sdfg.utils import get_uuid
                top = w_d_map.get(get_uuid(sdfg_for_work))
            except Exception:
                top = None
            if top is None and w_d_map:
                # Best-effort fall-back: the maximum work value seen,
                # which is typically the SDFG-level total.
                vals = [v[0] if isinstance(v, tuple) else v for v in w_d_map.values()]
                if vals:
                    # Filter out symbols-only or zero entries; pick max
                    # by string length as a crude largest-expression
                    # proxy.
                    nonzero = [v for v in vals if v != 0]
                    if nonzero:
                        top = max(nonzero, key=lambda e: len(str(e)))
            if top is not None:
                work_expr = top[0] if isinstance(top, tuple) else top
    except Exception as e:
        print(f"[dace_analysis] work_depth failed for {short_name}: {e}", file=sys.stderr)

    bytes_expr = None
    try:
        result = tv.analyze_sdfg(opt_sdfg)
        if isinstance(result, tuple) and len(result) == 2:
            read_v, write_v = result
            bytes_expr = read_v + write_v
        else:
            bytes_expr = result
    except Exception as e:
        print(f"[dace_analysis] total_volume failed for {short_name}: {e}", file=sys.stderr)

    flops = _substitute_symbols(work_expr, preset_params, datatype) \
        if work_expr is not None else None
    byts = _substitute_symbols(bytes_expr, preset_params, datatype) \
        if bytes_expr is not None else None

    result = (flops, byts)
    if cache is not None:
        cache[(short_name, preset, datatype)] = result
    return result


def is_available() -> bool:
    """Quick probe used by callers that want to print "skip dace" if
    the analysis branch isn't installed."""
    wd, tv = _try_import_analysis()
    return wd is not None and tv is not None
