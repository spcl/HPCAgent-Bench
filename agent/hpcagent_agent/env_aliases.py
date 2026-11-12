# Copyright 2026 the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deprecated environment names of the operator knobs that moved into configuration (F10 alpha).

Each entry of :data:`DEPRECATED` maps a short name the tree read before the env cleanup to the
``HPCAGENT_BENCH_<SECTION>_<KEY>`` name its configuration key resolves from today (``agent.effort``,
``timeouts.judge_s``, ...). The old names keep working for one release: :func:`value` prefers the
current name, falls back to the deprecated one with ONE :class:`DeprecationWarning` per name, and
:func:`resolve` adds the reader's default. Setting the current name is silent in both states.

Standard library only: the agent image carries no hpcagent_bench, and the driver, its harnesses and
the judge tools run there (hpcagent_bench imports this module for its own table, never the reverse).
"""

import os
import warnings
from collections.abc import Mapping

__all__ = ["DEPRECATED", "WARNED", "current_name", "resolve", "value", "warn_once"]

#: Deprecated name -> the current ``HPCAGENT_BENCH_*`` name of the configuration key it used to set.
DEPRECATED: dict[str, str] = {
    "AGENT_EFFORT": "HPCAGENT_BENCH_AGENT_EFFORT",
    "AGENT_EFFORT_POLICY": "HPCAGENT_BENCH_AGENT_EFFORT_POLICY",
    "EFFORT_LADDER": "HPCAGENT_BENCH_AGENT_EFFORT_LADDER",
    "HARNESS": "HPCAGENT_BENCH_AGENT_HARNESS",
    "CLAUDE_MODEL": "HPCAGENT_BENCH_AGENT_CLAUDE_MODEL",
    "CLAUDE_MAX_TURNS": "HPCAGENT_BENCH_AGENT_CLAUDE_MAX_TURNS",
    "JUDGE_TIMEOUT_SECONDS": "HPCAGENT_BENCH_TIMEOUTS_JUDGE_S",
    "HPCAGENT_BENCH_NCORES": "HPCAGENT_BENCH_LIMITS_CORES",
    "JUDGE_INPUT_MODE": "HPCAGENT_BENCH_SERVICE_INPUT_MODE",
}

#: Deprecated names already warned about, so ``value`` warns ONCE per name and process.
WARNED: set[str] = set()


def warn_once(old: str, current: str) -> None:
    """Emit the deprecation of ``old`` the first time this process reads it; silent on every later read."""
    if old not in WARNED:
        WARNED.add(old)
        warnings.warn(
            f"{old} is deprecated; set {current} (the environment name of its configuration key)",
            DeprecationWarning,
            stacklevel=3,
        )


def current_name(old: str) -> str | None:
    """The current ``HPCAGENT_BENCH_*`` name ``old`` maps to; None when ``old`` is not deprecated."""
    return DEPRECATED.get(old)


def value(old: str, environ: Mapping[str, str] = os.environ) -> str | None:
    """The knob ``old`` used to set: its current name when the environment names it (silently), else
    ``old`` with the one-time warning, else None. A value set but empty is returned as ``""``: the
    short names spell "unset" as absence, and an empty effort rung is a real value."""
    current = DEPRECATED.get(old, old)
    if environ.get(current) is not None:
        return environ[current]
    if old != current and old in environ:
        warn_once(old, current)
        return environ[old]
    return None


def resolve(old: str, default: str, environ: Mapping[str, str] = os.environ) -> str:
    """:func:`value` with the reader's default: what ``environ.get(old, default)`` read before the rename."""
    text = value(old, environ)
    return default if text is None else text
