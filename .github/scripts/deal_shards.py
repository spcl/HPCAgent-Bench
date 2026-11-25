# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deal the unit sweep's test files to its shards by measured runtime, so the shards finish together.

    python .github/scripts/deal_shards.py SHARD SHARDS

prints SHARD's files (space-separated). The files are every ``tests/test_*.py`` that
``.github/dedicated_tests.txt`` leaves to the sweep; each weighs its seconds in ``.github/unit_durations.json`` (the
summed ``--durations`` of a CI run, tests of 5 s or more), a file it does not name weighs DEFAULT_S. Heaviest first,
each file goes to the lightest shard (LPT), ties to the lower index, so every shard computes the same partition."""

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
DURATIONS = REPO / ".github" / "unit_durations.json"
DEDICATED = REPO / ".github" / "dedicated_tests.txt"
#: A file with no test of 5 s or more.
DEFAULT_S = 2.0


def sweep_files() -> list[str]:
    """The files the sweep runs: every top-level test module no dedicated phase claims."""
    dedicated = {
        line.strip()
        for line in DEDICATED.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return sorted(
        str(p.relative_to(REPO))
        for p in (REPO / "tests").glob("test_*.py")
        if str(p.relative_to(REPO)) not in dedicated
    )


def deal(files: list[str], durations: dict[str, float], shards: int) -> list[list[str]]:
    """``files`` in ``shards`` lists, heaviest file first onto the lightest shard."""
    bins: list[list[str]] = [[] for unused in range(shards)]
    loads = [0.0] * shards
    for name in sorted(files, key=lambda f: (-durations.get(f, DEFAULT_S), f)):
        lightest = min(range(shards), key=lambda i: (loads[i], i))
        bins[lightest].append(name)
        loads[lightest] += durations.get(name, DEFAULT_S)
    return [sorted(b) for b in bins]


def main(argv: list[str]) -> int:
    shard, shards = int(argv[0]), int(argv[1])
    dealt = deal(sweep_files(), json.loads(DURATIONS.read_text()), shards)[shard]
    if not dealt:
        print(f"shard {shard} of {shards} dealt no files", file=sys.stderr)
        return 1
    print(" ".join(dealt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
