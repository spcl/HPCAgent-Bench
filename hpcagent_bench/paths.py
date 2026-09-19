# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Single source for repo-relative paths.

Previously the path math :code:`__file__.parent.absolute() / ".." / ".."`
was triplicated across :mod:`hpcagent_bench.frameworks.benchmark`,
:mod:`hpcagent_bench.frameworks.framework`, and the top-level
``run_*.py`` drivers. Consolidate here so a layout change touches one file."""

import os
import pathlib

#: Repository root (the directory containing ``pyproject.toml``).
ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]

#: Root of the per-kernel implementation tree.
BENCHMARKS: pathlib.Path = ROOT / "hpcagent_bench" / "benchmarks"

#: Everything a run produces -- results DB and its shards, JSONL logs, job output. Never the repo
#: root. Relative, so it follows ``record.db_path``'s ROOT anchoring and a job's own ``--output``.
RESULTS_DIR: str = "results"

#: Figures, beside the results they were rendered from.
PLOTS_DIR: str = RESULTS_DIR + "/plots"


def repo_root() -> pathlib.Path:
    """This checkout's root: ``$HPCAGENT_BENCH_REPO`` if a caller already resolved one (every shell
    entry point does -- ``experiments/env.sh`` exports it before sourcing anything else), else
    this file's own location.

    The one fallback TARGET for every script that needs a durable root and has no ``$SCRATCH`` --
    a container, CI runner, or a laptop clone -- instead of each guessing its own (``~/.cache``,
    ``__file__``'s grandparent, the checkout's parent directory: three actual, independently wrong
    spellings this replaced)."""
    repo = os.environ.get("HPCAGENT_BENCH_REPO")
    return pathlib.Path(repo) if repo else ROOT


def scratch_or_repo() -> pathlib.Path:
    """``$SCRATCH`` if set, else :func:`repo_root`. For a caller that wants the scratch root
    ITSELF (a directory to glob campaign output under) rather than one rebuildable subtree under
    it -- see :func:`scratch_root` for that case."""
    scratch = os.environ.get("SCRATCH")
    return pathlib.Path(scratch) if scratch else repo_root()


def scratch_root(name: str) -> pathlib.Path:
    """Where a REBUILDABLE tree named ``name`` belongs: ``$SCRATCH/<name>``, else
    ``<repo_root>/.cache/<name>``.

    Three of these grew independently -- the numeric oracle's DaCe builds, the size-extrapolation
    workdir, the preset sweep -- and all three defaulted under ``~/.cache``. That is the wrong
    default twice over on a cluster: an HPC home is typically quota'd on INODE COUNT rather than
    bytes, and a corpus of C++ build trees is precisely tens of thousands of tiny files (measured
    here: 26k inodes, 195 MB, from one gate). ``/tmp`` is not the alternative either -- it is tmpfs
    on these nodes, so a build competes with the run for RAM. Scratch is the one filesystem that is
    on disk, large, and expected to be purged.

    ``repo_root()`` stays the fallback rather than an error so a laptop or a container with no
    ``$SCRATCH`` still works, and rather than ``~/.cache`` because a checkout is a real, durable
    location THIS process already proved exists, where the previous default was a guess.
    """
    scratch = os.environ.get("SCRATCH")
    base = pathlib.Path(scratch) if scratch else repo_root() / ".cache"
    return base / name
