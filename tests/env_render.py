# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A layered env file flattened the way a submitter stages it (experiments/env_layers.sh render)."""

import pathlib
import subprocess

LAYERS = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "env_layers.sh"


def rendered(path: pathlib.Path) -> str:
    """``path`` flattened through its ``# extends:`` parents: one ``KEY=VALUE`` line per key."""
    return subprocess.run(["bash", str(LAYERS), "render", str(path)], capture_output=True, text=True, check=True).stdout
