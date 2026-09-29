# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Write the table that names every recorded arm by its configuration,
``<tag>-<model>-<lang>[-<packet>]`` (``hpcagent_bench/envs/arm_renames.yaml``).

An arm is one configuration: model, language, device, packet, harness and prompt setup. Arms that
recorded the same configuration under several names -- ``X`` and ``X-clean``, the v9/v10/v11/v11w2
waves, ``llrblind`` and ``llrblind-cmp``, ``scicomp-dc`` and ``scicomp-perf-playbook`` -- fold into
one; the latest run per kernel already picks within it. A group is folded only when every member's
recorded identity (``arms``: model, language, device, harness, and packet where recorded) is the
same. A group that is not stays split: its members that agree fold together, and every part but
the current one keeps the campaign it ran under as a suffix. Those groups are printed for review.

    python scripts/arm_renames.py hpcagent-bench-v1.db > hpcagent_bench/envs/arm_renames.yaml

The migration (``scripts/migrate_db.py``) applies the committed table; readers resolve an old name
through it (:func:`hpcagent_bench.experiment_tags.aliased_arm`).
"""

import collections
import contextlib
import dataclasses
import sqlite3
import sys

#: Legacy arm-name prefix -> (tag, the campaign variant it names, the language it implies), longest first.
FAMILIES: tuple[tuple[str, str, str, str], ...] = (
    ("harness-focus20-smoke-mi200-", "harness20", "smoke", "c"),
    ("llr-focus40-mi200-smoke-", "llr40", "smoke", "c"),
    ("scicomp-perf-playbook-fortran-", "scicomp40", "", "fortran"),
    ("scicomp-perf-playbook-gpu-", "scicomp40", "", ""),
    ("scicomp-perf-playbook-", "scicomp40", "", "c"),
    ("scicomp-dc-fortran-", "scicomp40", "dc", "fortran"),
    ("scicomp-dc-gpu-", "scicomp40", "dc", ""),
    ("scicomp-dc-cpp-", "scicomp40", "dc", "cpp"),
    ("scicomp-dc-", "scicomp40", "dc", "c"),
    ("gpu-llr-focus40-", "llr40", "", ""),
    ("llrblind-cmp-", "llr40", "cmp", ""),
    ("llrblind-", "llr40", "", ""),
    ("llr-focus40-", "llr40", "", ""),
    ("llr40v9-", "llr40", "v9", ""),
    ("llr40v10-", "llr40", "v10", ""),
    ("llr40v11-", "llr40", "v11", ""),
    ("v11w2-", "llr40", "v11w2", ""),
    ("gpuv2-llr40-", "llr40", "gpuv2", ""),
    ("gpuv4-llr40-", "llr40", "gpuv4", ""),
    ("git-scicomp-", "gitscicomp10", "", "c"),
    ("harness-focus20-", "harness20", "focus20", "c"),
    ("harness20-", "harness20", "", "c"),
    ("mlscale-part2-", "mlscale20", "part2", ""),
    ("mlscale-smoke-", "mlscale20", "smoke", ""),
    ("mlscale-", "mlscale20", "", ""),
)
#: Language tokens an arm name spells, longest first, and the spellings that name another.
LANGUAGES: tuple[str, ...] = ("c-openmp-device", "c-openmp", "triton-device", "triton", "pytriton", "fortran", "cpp")
LANGUAGES += ("hip", "omp", "c")
LANGUAGE_SPELLINGS: dict[str, str] = {"pytriton": "triton", "omp": "c-openmp"}
#: Name tokens that say nothing the configuration does not: the default harness and the empty packet.
SILENT_TOKENS: frozenset[str] = frozenset({"claude", "plain", "kernel"})
#: The suffix a no-score-tool arm carries.
BLIND = "blind"
#: The variant a smoke run keeps whatever else it shares: it is a different configuration.
SMOKE = "smoke"


@dataclasses.dataclass(frozen=True, slots=True)
class Parsed:
    """An old arm name read as the configuration name it records, and what it ran under."""

    name: str
    variant: str
    clean: bool


def parse(arm: str) -> Parsed:
    """``arm``'s configuration name ``<tag>-<model>-<lang>[-<packet>]`` (``-smoke`` for a smoke run)."""
    prefix, tag, variant, language = next(family for family in FAMILIES if arm.startswith(family[0]))
    rest = arm.removeprefix(prefix)
    clean = rest.endswith("-clean")
    rest = rest.removesuffix("-clean")
    if rest.endswith(f"-{SMOKE}"):
        rest, variant = rest.removesuffix(f"-{SMOKE}"), SMOKE
    model, _, tail = rest.partition("-")
    if not language:
        language = next((one for one in LANGUAGES if tail == one or tail.startswith(f"{one}-")), "")
    if tail == language or tail.startswith(f"{language}-"):
        tail = tail[len(language) :].strip("-")
    tokens = [token for token in tail.split("-") if token and token not in SILENT_TOKENS]
    if prefix.startswith("llrblind"):
        tokens.append(BLIND)
    if variant == SMOKE:
        tokens.append(SMOKE)
    name = "-".join([tag, model, LANGUAGE_SPELLINGS.get(language, language), *tokens])
    return Parsed(name, variant, clean)


def identity(row: sqlite3.Row) -> tuple[str, ...]:
    """What an arm recorded about its configuration, but its packet (checked apart: often blank)."""
    return (str(row["model"]), str(row["language"]), str(row["device"]), str(row["harness"]))


def fold(arms: dict[str, sqlite3.Row]) -> tuple[dict[str, str], list[str]]:
    """Every arm -> its new name, and the split groups described for review."""
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for arm in sorted(arms):
        groups[parse(arm).name].append(arm)
    renames: dict[str, str] = {}
    splits: list[str] = []
    for name, members in groups.items():
        parts: dict[tuple[str, ...], list[str]] = collections.defaultdict(list)
        for arm in members:
            packet = str(arms[arm]["packet"] or "")
            parts[(*identity(arms[arm]), packet)].append(arm)
        merged = merge_blank_packets(parts)
        current = max(merged, key=lambda key: current_rank(merged[key]))
        for key, part in merged.items():
            suffix = "" if key == current else part_suffix(part)
            renames.update({arm: f"{name}-{suffix}" if suffix else name for arm in part})
        if len(merged) > 1:
            listed = "; ".join(f"{renames[part[0]]} <- {', '.join(part)} {key}" for key, part in merged.items())
            splits.append(f"{name}: {listed}")
    return renames, splits


def merge_blank_packets(parts: dict[tuple[str, ...], list[str]]) -> dict[tuple[str, ...], list[str]]:
    """``parts`` with an arm that recorded no packet folded into the one part of its identity that
    did (a blank is an unrecorded packet, not another one)."""
    merged: dict[tuple[str, ...], list[str]] = {}
    for key in sorted(parts, key=lambda key: key[-1] == ""):
        recorded = [other for other in merged if other[:-1] == key[:-1]]
        target = recorded[0] if key[-1] == "" and len(recorded) == 1 else key
        merged.setdefault(target, []).extend(parts[key])
    return merged


def current_rank(part: list[str]) -> tuple[int, int]:
    """How current a part is: the campaign without a variant, then the clean reruns, then size."""
    parsed = [parse(arm) for arm in part]
    return (sum(not one.variant for one in parsed), sum(one.clean for one in parsed) * 100 + len(part))


def part_suffix(part: list[str]) -> str:
    """The suffix a split part keeps: the campaigns it ran under, else what sets it apart."""
    variants = sorted({parse(arm).variant for arm in part} - {""})
    if variants:
        return "-".join(variants)
    return "clean" if all(parse(arm).clean for arm in part) else "early"


def main(argv: list[str]) -> int:
    with contextlib.closing(sqlite3.connect(f"file:{argv[1]}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        arms = {str(row["arm"]): row for row in conn.execute("SELECT * FROM arms")}
    renames, splits = fold(arms)
    print("# Every recorded arm -> the arm it is, named by its configuration <tag>-<model>-<lang>[-<packet>].")
    print("# Generated by scripts/arm_renames.py from the migrated database; scripts/migrate_db.py applies it")
    print("# and experiment_tags.aliased_arm reads an old name through it. DATA: do not edit by hand.")
    for old in sorted(renames):
        print(f"{old}: {renames[old]}")
    for line in splits:
        print(f"split {line}", file=sys.stderr)
    folded = len(set(renames.values()))
    print(f"{len(renames)} arms -> {folded} arms, {len(splits)} groups split", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
