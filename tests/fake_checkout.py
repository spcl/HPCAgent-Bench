# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A test's stand-in checkout gets the real ``scripts/repo_env.sh`` and ``scripts/repo_python``.

Every script sources ``<checkout>/scripts/repo_env.sh`` for its import path, so a temp tree a test
hands a script as its checkout needs that file like any other sibling it reads. The real one, not a
stub: it derives the checkout from its own location, which is exactly the temp tree.
"""

import pathlib
import shutil

REPO = pathlib.Path(__file__).resolve().parents[1]


def install_repo_env(root: pathlib.Path) -> None:
    """Copy ``repo_env.sh`` and ``repo_python`` into ``root/scripts``."""
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    for name in ("repo_env.sh", "repo_python"):
        shutil.copy2(REPO / "scripts" / name, root / "scripts" / name)
