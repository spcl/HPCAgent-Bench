#!/usr/bin/env python
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-area, per-rule ruff finding counts -- the hand-off report for a fixer pass.

Regenerate on demand (the report is never committed as a snapshot -- it goes stale the moment a
file changes, and the ratchet baselines are the durable record):

    python scripts/lint_area_report.py               # table to stdout
    python scripts/lint_area_report.py --json out.json

Groups by the same AREAS the ratchets and per-file-ignores use (hpcagent_bench/benchmarks and
hpcagent_bench/numpy_translators split out from the rest of hpcagent_bench, since they carry the
numpy-kernel-naming per-file-ignores and nothing else does), then tags each rule SAFE (ruff --fix
applies it with no behavior change -- modernization, cosmetic) or MANUAL (a fix can change
behavior, or the rule needs judgment) from :data:`RULE_DISPOSITION`. A code absent from that table
prints as "review" (the safe list is an allowlist, not a denylist -- an unclassified rule is never
silently assumed safe).
"""

import argparse
import collections
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
ROOTS = ("hpcagent_bench", "tests", "scripts", "experiments", "tools")

#: Rules ruff's own --fix applies with NO behavior change (import cleanup, pyupgrade syntax
#: rewrites, cosmetic dict/list-literal forms) vs rules where the mechanical fix can change what
#: the program DOES and a person must read the diff. Absence from both = "review" (never assume
#: safe by default). Built from the counts actually observed in this repo
#: (scripts/lint_area_report.py --json against pyproject.toml's curated select) plus the ruff
#: docs' own fix-safety marker ([*] in `ruff check --statistics`).
SAFE_AUTOFIX = frozenset(
    {
        "UP006",
        "UP007",
        "UP035",
        "UP034",
        "UP045",
        "UP040",
        "UP042",
        "UP047",
        "UP046",
        "UP037",
        "UP032",
        "UP031",
        "UP033",
        "UP030",
        "UP022",
        "UP015",
        "UP004",  # pyupgrade syntax modernization -- typing.X -> builtin, Optional -> |, etc.
        "C408",
        "C401",
        "C402",
        "C416",
        "C419",
        "C420",  # comprehension / literal-call rewrites, semantically identical
        "RUF100",  # drop an unused suppression comment -- it silenced nothing
        "RUF022",  # sort __all__
        "PIE808",  # range(0, n) -> range(n)
        "PIE790",  # remove unnecessary pass
        "PIE810",  # combine multiple isinstance calls into one, semantically identical
        "SIM905",  # split() on a string literal -> a list literal, semantically identical
        "SIM101",
        "SIM102",
        "SIM103",
        "SIM105",
        "SIM108",
        "SIM109",
        "SIM112",
        "SIM113",
        "SIM114",
        "SIM117",
        "SIM118",
        "SIM212",
        "SIM222",
        "SIM300",
        "SIM401",  # control-flow simplifications ruff proves equivalent before offering the fix
        "E731",  # lambda assignment -> def (ruff does not autofix this one, but the rewrite is mechanical)
        "W291",  # trailing whitespace
        "E402",  # module level import not at top (only autofixable when ruff can prove no side effect moved)
    }
)

#: Rules whose mechanical fix (or the underlying finding) can change runtime behavior, or that
#: name a real code smell needing a judgment call rather than a syntax rewrite. Each needs a human
#: (or a fixer agent with the specific behavioral note) reading the diff, not a blind --fix.
MANUAL_REVIEW = frozenset(
    {
        "B905",  # zip(..., strict=?) -- adding strict=True RAISES on unequal-length inputs; a real behavior change
        "PLW1510",  # subprocess.run without check= -- silently-ignored failures may be the CURRENT behavior
        "F841",  # unused local -- may be dead code, or a bug (the computed value was meant to be used)
        "B006",
        "B008",  # mutable / call-expression default argument -- the shared-default bug this rule exists to catch
        "PLC0415",  # import not at top level -- often a deliberate cycle-break or heavy-import deferral; case by case
        "PLR0913",
        "PLR0917",  # too many (positional) arguments -- outside the exempted kernel dirs, a real signature smell
        "PLR0912",
        "PLR0915",
        "PLR0911",  # too many branches/statements/returns -- the CC<=20 house rule; needs a real decomposition
        "PLR2004",  # magic value -- outside tests/, naming the constant is a judgment call about what it means
        "PLW2901",  # loop variable overwritten -- almost always a real bug (the original value is lost)
        "PLW0603",  # global statement -- a design smell, not a syntax fix
        "B023",  # function defined in a loop captures the loop variable by reference -- a real closure bug
        "B007",  # loop variable never used in the loop body -- rename to `_name` needs a read, not a blind rename
        "RUF005",  # collection concat -> unpacking; SAFE for list/tuple, but ruff also flags some str cases to check
        "RUF059",  # unpacked variable never used -- same judgment as F841
        "N803",
        "N806",
        "N802",
        "N812",
        "N815",
        # naming -- outside the exempted kernel/translator/port-oracle dirs, a real rename with call-site fanout
        "N818",
        "E741",  # ambiguous name (l/O/I) -- a rename, not a mechanical rewrite
        "PERF401",
        "PERF403",  # loop -> comprehension -- correct only when the loop has no other side effect; read it
        "PLR0402",  # import aliasing -- mechanical but touches every call site in the file
        "PLR1714",
        "PLR2044",
        "PLR5501",
        "PLR1730",
        "PLR1711",
        "PLR1704",
        "PLR0133",
        "PLR0124",
        "PLC0207",
        "PLW0108",
        "PLW3301",
        "PLW1509",
        "PLW0128",
        "PLC0105",
        "PLC0206",
        "PLC0208",
        "PLC3002",
        "RUF012",  # mutable class attribute needs ClassVar -- a real typing fix, check the actual mutation contract
        "RUF013",
        "RUF015",
        "RUF019",
        "RUF021",
        "RUF023",
        "RUF028",
        "RUF034",
        "RUF036",
        "RUF043",
        "RUF046",
        "RUF007",
        "N801",
        "N805",
        "N814",
        "N817",
        "B002",
        "B011",
        "B018",
        "B024",
        "B028",
        "B904",
        "E712",
        "E714",
        "F401",
        "F402",
        "F403",
        "F405",
        "F541",
        "F601",
        "E501",  # line too long -- ruff has no safe automatic line-splitter; the wrap is a judgment call
        "SIM115",  # open() without a context manager -- a real resource-lifetime change to verify
        "PLW1641",  # __eq__ without __hash__ -- a real correctness note (object becomes unhashable), not cosmetic
    }
)


def area(rel: str) -> str:
    if rel.startswith("hpcagent_bench/benchmarks/"):
        return "hpcagent_bench/benchmarks"
    if rel.startswith("hpcagent_bench/numpy_translators/"):
        return "hpcagent_bench/numpy_translators"
    for top in ROOTS:
        if rel.startswith(f"{top}/"):
            return top
    return "other"


def disposition(code: str) -> str:
    if code in SAFE_AUTOFIX:
        return "safe-autofix"
    if code in MANUAL_REVIEW:
        return "manual-review"
    return "review (unclassified)"


def collect() -> list[dict]:
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--output-format", "json", "--no-cache", *ROOTS],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,  # ruff exits 1 on findings, not on failure
    )
    if not proc.stdout.strip():
        raise RuntimeError(f"ruff produced no output (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
    findings = []
    for item in json.loads(proc.stdout):
        rel = str(pathlib.Path(item["filename"]).resolve().relative_to(REPO))
        findings.append({"path": rel, "area": area(rel), "code": item["code"]})
    return findings


def build_table(findings: list[dict]) -> dict[str, dict[str, int]]:
    table: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for item in findings:
        table[item["area"]][item["code"]] += 1
    return {a: dict(c.most_common()) for a, c in sorted(table.items())}


def print_table(table: dict[str, dict[str, int]]) -> None:
    grand_total = 0
    for area_name, counts in table.items():
        area_total = sum(counts.values())
        grand_total += area_total
        print(f"\n=== {area_name}  ({area_total} findings) ===")
        print(f"{'rule':10s} {'count':>6s}  disposition")
        for code, n in counts.items():
            print(f"{code:10s} {n:6d}  {disposition(code)}")
    print(f"\n=== TOTAL: {grand_total} findings ===")
    print("\nUnclassified rules (not in SAFE_AUTOFIX or MANUAL_REVIEW -- treat as manual until triaged):")
    seen_codes = {code for counts in table.values() for code in counts}
    unclassified = sorted(c for c in seen_codes if c not in SAFE_AUTOFIX and c not in MANUAL_REVIEW)
    print("  " + (", ".join(unclassified) if unclassified else "(none)"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=pathlib.Path, default=None, help="write the full table as JSON here too")
    args = parser.parse_args(argv)

    findings = collect()
    table = build_table(findings)
    print_table(table)
    if args.json is not None:
        args.json.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
