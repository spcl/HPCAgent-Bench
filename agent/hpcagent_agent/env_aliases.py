# Copyright 2026 the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deprecated environment names of the operator knobs that moved into configuration (F10 alpha).

Each entry of :data:`DEPRECATED` maps a short name the tree read before the env cleanup to the
``HPCAGENT_BENCH_<SECTION>_<KEY>`` name its configuration key resolves from today (``agent.effort``,
``timeouts.judge_s``, ...). Every reader reads the current name only. The old names keep working for one release
through ONE translation at process entry, :func:`translate`: each deprecated name the environment sets, whose current
name it does not, is copied to the current name with a :class:`FutureWarning`. It warns once per name: after the copy
the current name is set, so a second call in the same process, and the same call in every child that inherits the
environment, has nothing left to translate. Setting the current name is silent, and it wins when both are set.

Where it runs: ``hpcagent_agent/__init__.py`` and ``hpcagent_bench/__init__.py`` call it when imported, so every
Python process of either package translates before any of its modules reads a knob; ``run_cluster.sh`` evaluates
``python -m hpcagent_agent.env_aliases`` (:func:`exports`) after it sources the setup ``.env``, so the batch shell and
every step it starts hold the current names too.

Standard library only: the agent image carries no hpcagent_bench, and the driver, its harnesses and
the judge tools run there (hpcagent_bench imports this module for its own table, never the reverse).
"""

import os
import shlex
import sys
import warnings
from collections.abc import Mapping, MutableMapping

__all__ = ["DEPRECATED", "exports", "main", "translate"]

#: Deprecated name -> the current ``HPCAGENT_BENCH_*`` name of the configuration key it used to set.
DEPRECATED: Mapping[str, str] = {
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


def translate(environ: MutableMapping[str, str] = os.environ) -> dict[str, str]:
    """Copy every deprecated name ``environ`` sets to its current name where that is unset, warning once per name;
    returns the current names it set, with their values. An empty value is copied as it is: the short names spell
    "unset" as absence, and an empty effort rung is a real value."""
    copied: dict[str, str] = {}
    for old, current in DEPRECATED.items():
        if old in environ and current not in environ:
            environ[current] = copied[current] = environ[old]
            warnings.warn(
                f"{old} is deprecated; set {current} (the environment name of its configuration key)",
                FutureWarning,
                stacklevel=2,
            )
    return copied


def exports(environ: MutableMapping[str, str]) -> list[str]:
    """``export`` lines that give a shell the current name of every deprecated name ``environ`` sets, at the value
    :func:`translate` leaves it (a current name the shell set already is exported unchanged)."""
    translate(environ)
    return [
        f"export {current}={shlex.quote(environ[current])}" for old, current in DEPRECATED.items() if old in environ
    ]


def main() -> int:
    """Print :func:`exports` of this process's environment, for a shell to ``eval``; warnings go to stderr."""
    lines = exports(os.environ)
    if lines:
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
