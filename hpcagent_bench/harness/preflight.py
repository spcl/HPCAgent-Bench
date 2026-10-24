# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a batch job must check BEFORE it spends an allocation: runnable columns, the dace fork, the polyhedral
toolchains and autopar evidence. A vacuous autopar column is reported, never fatal."""

from collections.abc import Sequence

from hpcagent_bench import flags, languages
from hpcagent_bench.polyhedral import pluto as pluto_transform
from hpcagent_bench.polyhedral import ppcg as ppcg_transform
from hpcagent_bench.columns import FRAMEWORKS
from hpcagent_bench.flags import AutoparVerdict, Mode
from hpcagent_bench.vocabulary import DaceStage, Transform

__all__ = [
    "AUTOPAR_PROBES",
    "DETERMINISTIC_FRAMEWORKS",
    "check_autopar",
    "check_dace_pipeline",
    "check_deterministic",
    "check_polycc",
    "check_ppcg",
    "missing_tools",
    "needs_canonicalize",
    "needs_polycc",
    "needs_ppcg",
    "run",
    "thread_env",
]

#: Columns a deterministic (unjudged) sweep may run, from the registered columns' ``sweep_deterministic`` flag.
DETERMINISTIC_FRAMEWORKS: tuple[str, ...] = tuple(
    name for name, column in FRAMEWORKS.entries.items() if column.sweep_deterministic
)

#: Autopar column -> the capability probe that decides whether it is one in fact as well as name.
AUTOPAR_PROBES = {
    "polly": flags.polly_capability,
    "cc_autopar": flags.gcc_autopar_capability,
    "fortran_autopar": flags.gcc_autopar_capability,
}


def check_deterministic(frameworks: Sequence[str]) -> list[str]:
    """The entries of ``frameworks`` that a deterministic sweep cannot run."""
    return [name for name in frameworks if name not in DETERMINISTIC_FRAMEWORKS]


def needs_canonicalize(frameworks: Sequence[str]) -> list[str]:
    """The requested columns whose SDFG pipelines include ``canonicalize`` (they need the dace fork)."""
    out: list[str] = []
    for name in frameworks:
        column = FRAMEWORKS.entries.get(name)
        if column is None or column.base != "dace":
            continue
        if column.stage is DaceStage.CANON:
            out.append(name)
    return out


def needs_polycc(frameworks: Sequence[str]) -> list[str]:
    """The requested columns whose TIMED build runs ``polycc`` (without it they decline every kernel)."""
    return [
        name
        for name in frameworks
        if (column := FRAMEWORKS.entries.get(name)) is not None and column.transform is Transform.PLUTO
    ]


def check_polycc() -> str:
    """``""`` when ``polycc`` is on PATH (the lookup the build uses), else why not."""
    if pluto_transform.polycc_exe() is None:
        return (
            "polycc is not on PATH; the pluto column compiles polycc's output and has nothing to "
            "build without it (Pluto is built from source -- see containers/images/lib/build-pluto.sh)"
        )
    return ""


def needs_ppcg(frameworks: Sequence[str]) -> list[str]:
    """The requested columns whose TIMED build runs ``ppcg`` (without it they decline every kernel)."""
    return [
        name
        for name in frameworks
        if (column := FRAMEWORKS.entries.get(name)) is not None and column.transform is Transform.PPCG
    ]


def check_ppcg(frameworks: Sequence[str]) -> str:
    """``""`` when every requested ppcg column's toolchain is here (checked per column: vendors differ), else why."""
    from hpcagent_bench.frameworks.native_runtime import FRAMEWORK_LANG

    for name in frameworks:
        problem = ppcg_transform.missing_tool(FRAMEWORK_LANG[name])
        if problem:
            return problem
    return ""


def missing_tools(frameworks: Sequence[str]) -> list[tuple[str, str]]:
    """``(column, why)`` for every requested column whose own compiler this host lacks (else silent declines)."""
    out: list[tuple[str, str]] = []
    for name in needs_polycc(frameworks):
        problem = check_polycc()
        if problem:
            out.append((name, problem))
    ppcg_columns = needs_ppcg(frameworks)
    if ppcg_columns:
        problem = check_ppcg(ppcg_columns)
        if problem:
            out.extend((name, problem) for name in ppcg_columns)
    return out


def check_dace_pipeline() -> str:
    """``""`` when the installed dace carries the fork's canonicalize pipeline, else why not.

    Only columns that ask for canonicalize are gated; ``dace_cpu_autoopt`` is meant to run on stock DaCe."""
    try:
        __import__("dace.transformation.passes.canonicalize.pipeline", fromlist=["canonicalize"])
    except ImportError as exc:
        return f"installed dace has no canonicalize pipeline ({exc}); HPCAgent-Bench needs spcl/dace@extended"
    return ""


def check_autopar(frameworks: Sequence[str]) -> list[tuple[str, AutoparVerdict, str]]:
    """``(framework, verdict, detail)`` for each requested autopar column, measured on THIS node."""
    out: list[tuple[str, AutoparVerdict, str]] = []
    for name in frameworks:
        probe = AUTOPAR_PROBES.get(name)
        if probe is None:
            continue
        result = probe()
        out.append((name, result.verdict, result.detail))
    return out


def thread_env(mode: Mode = Mode.MULTI_CORE, ranks_per_node: int = 1) -> dict[str, str]:
    """The thread-count environment a timed run needs; ``ranks_per_node`` > 1 splits the node's cores (floor 1)."""
    if ranks_per_node <= 1:
        return flags.cpu_env(mode)
    # SINGLE_CORE means one thread per rank however many ranks share the node.
    share = 1 if mode is Mode.SINGLE_CORE else max(1, flags.ncores() // ranks_per_node)
    return flags.cpu_env(mode, threads=share)


def run(
    frameworks: Sequence[str], print_env: bool = False, ranks_per_node: int = 1, tools_only: bool = False
) -> tuple[int, list[str], list[str]]:
    """Every preflight check, as ``(exit_code, report_lines, env_lines)``.

    ``env_lines`` are separate because a caller EVALS them. Non-zero only for a fatal finding; a vacuous autopar
    probe only warns."""
    report: list[str] = []
    if tools_only:
        #: Tool presence only, for a runner that has already settled which columns it may run.
        absent = missing_tools(frameworks)
        for name, problem in absent:
            report.append(f"preflight: FATAL -- {name} cannot run here: {problem}")
        if absent:
            return 1, report, []
        report.append(f"preflight: the toolchain of {', '.join(frameworks)} is installed on this node")
        return 0, report, []
    unknown = check_deterministic(frameworks)
    if unknown:
        report.append(f"preflight: FATAL -- not deterministic optimizers: {', '.join(unknown)}")
        return 1, report, []
    fork_columns = needs_canonicalize(frameworks)
    if fork_columns:
        problem = check_dace_pipeline()
        if problem:
            report.append(f"preflight: FATAL -- {problem} (needed by {', '.join(fork_columns)})")
            return 1, report, []
        report.append(f"preflight: dace canonicalize pipeline present (needed by {', '.join(fork_columns)})")
    pluto_columns = needs_polycc(frameworks)
    if pluto_columns:
        problem = check_polycc()
        if problem:
            report.append(f"preflight: FATAL -- {problem} (needed by {', '.join(pluto_columns)})")
            return 1, report, []
        report.append(f"preflight: polycc present (needed by {', '.join(pluto_columns)})")
    ppcg_columns = needs_ppcg(frameworks)
    if ppcg_columns:
        problem = check_ppcg(ppcg_columns)
        if problem:
            report.append(f"preflight: FATAL -- {problem} (needed by {', '.join(ppcg_columns)})")
            return 1, report, []
        report.append(f"preflight: ppcg present (needed by {', '.join(ppcg_columns)})")
    for name, verdict, detail in check_autopar(frameworks):
        if verdict is AutoparVerdict.OK:
            report.append(f"preflight: {name} PARALLELIZES on this node ({detail})")
        else:
            report.append(
                f"preflight: WARNING -- {name} is {verdict.text}: {detail}; "
                "this column is a serial baseline wearing an autopar label"
            )
    env = (
        [f"export {name}={value}" for name, value in thread_env(ranks_per_node=ranks_per_node).items()]
        if print_env
        else []
    )
    return 0, report, env
