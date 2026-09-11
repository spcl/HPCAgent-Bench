"""Audit arm ``.env`` files for numbers that do not add up. Read-only.

Every check here exists because its absence cost a campaign:

duplicate keys      ``run_cluster.sh`` sources under ``set -a``, so a repeated key is right by
                    last-wins accident. ``arm_nodes.sh`` does not: it greps a key with ``-oP`` and
                    feeds the result to ``$(( ))``, so the day a duplicated key is one IT reads,
                    the submit dies on a syntax error instead of a wrong number.
wave arithmetic     an arm's wall has to cover ``waves x AGENT_TIMEOUT_SECONDS``, and waves is
                    ``ceil(problems / (AGENT_NODES x AGENTS_PER_NODE))``. The git arms ran 30
                    problems at 20 per node -- two waves of 6 h inside a 12 h wall -- and the job
                    hit TIMEOUT with the second wave still running.
budget regime       rows POOL across completion waves (``CAMPAIGN_ARM`` deliberately keeps the
                    base name), so two waves at different budgets put one arm's kernels under two
                    regimes. Worse, a completion wave re-runs exactly the kernels that did not
                    land, so a later raise lands systematically on the HARD ones.
missing problems    ``PROBLEMS_FILE`` naming a file that is not there materializes zero kernels
                    and leaves the arm running over nothing.
"""

import argparse
import collections
import pathlib
import re

#: ``KEY=value`` at the start of a line. Anything else in these files is a comment or a
#: continuation of a quoted value, neither of which is a key.
ASSIGN = re.compile(r"^([A-Z_][A-Z0-9_]*)=(.*)$")


def read_env(path: pathlib.Path) -> tuple[dict[str, str], collections.Counter]:
    """The file's effective values (last-wins, as ``set -a`` would take them) and its repeat counts."""
    values: dict[str, str] = {}
    repeats: collections.Counter = collections.Counter()
    for line in path.read_text().splitlines():
        found = ASSIGN.match(line)
        if found is None:
            continue
        key, value = found.group(1), found.group(2)
        if key in values:
            repeats[key] += 1
        values[key] = value
    return values, repeats


def problem_count(here: pathlib.Path, values: dict[str, str]) -> int:
    """How many problems this arm runs -- the JSONL line count, or the KERNELS word count."""
    named = values.get("PROBLEMS_FILE", "").strip()
    if named:
        path = here / named
        if not path.exists():
            return -1
        return sum(1 for line in path.read_text().splitlines() if line.strip())
    return len(values.get("KERNELS", "").split())


def audit(here: pathlib.Path, pattern: str) -> int:
    """Print one row per arm env and a findings block. Returns the number of findings."""
    rows = []
    for path in sorted(here.glob(f".env.{pattern}")):
        values, repeats = read_env(path)
        if "CAMPAIGN_ARM" not in values:
            continue
        # The launcher's own defaults, for an arm that leaves one unset -- same numbers arm_nodes.sh uses.
        nodes = {k: int(values.get(f"{k}_NODES", d)) for k, d in (("INFERENCE", 2), ("AGENT", 1), ("JUDGE", 1))}
        slots = nodes["AGENT"] * int(values.get("AGENTS_PER_NODE", 1))
        problems = problem_count(here, values)
        timeout = int(values.get("AGENT_TIMEOUT_SECONDS", 0))
        waves = -1 if problems < 0 else (problems + slots - 1) // slots
        rows.append(
            {
                "env": path.name[len(".env.") :],
                "nodes": sum(nodes.values()),
                "slots": slots,
                "problems": problems,
                "waves": waves,
                "hours": waves * timeout / 3600 if waves > 0 else 0.0,
                "timeout_h": timeout / 3600,
                "tokens_m": int(values.get("AGENT_MAX_TOKENS", 0)) / 1e6,
                "repeats": sorted(repeats),
            }
        )
    if not rows:
        print(f"no arm envs match .env.{pattern}")
        return 0

    head = f"{'env':46s} {'nodes':>5s} {'slots':>5s} {'prob':>4s} {'wave':>4s} {'agent-h':>7s} {'need-h':>6s} {'tok-M':>5s}"
    print(head)
    print("-" * len(head))
    for row in rows:
        print(
            f"{row['env'][:46]:46s} {row['nodes']:5d} {row['slots']:5d} {row['problems']:4d} "
            f"{row['waves']:4d} {row['timeout_h']:7.1f} {row['hours']:6.1f} {row['tokens_m']:5.0f}"
        )

    findings = 0
    for row in rows:
        if row["repeats"]:
            print(f"DUPLICATE KEY  {row['env']}: {', '.join(row['repeats'])}")
            findings += 1
        if row["problems"] < 0:
            print(f"MISSING PROBLEMS FILE  {row['env']}")
            findings += 1
        elif row["waves"] > 1:
            print(f"MULTI-WAVE  {row['env']}: {row['waves']} waves, wall must exceed {row['hours']:.1f} h")
            findings += 1
    # A campaign's arms pool their rows, so a budget that differs BETWEEN them is a confound, not a
    # setting. Reported as a spread rather than per-row: no single row is wrong on its own.
    budgets = collections.Counter((row["timeout_h"], row["tokens_m"]) for row in rows)
    if len(budgets) > 1:
        spread = ", ".join(f"{h:.1f}h/{t:.0f}M x{n}" for (h, t), n in sorted(budgets.items()))
        print(f"MIXED BUDGET REGIME  {len(budgets)} distinct: {spread}")
        findings += 1
    print(f"\n{len(rows)} arm envs, {findings} findings")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pattern", nargs="?", default="*", help="glob after '.env.' (default: every arm env)")
    args = parser.parse_args()
    audit(pathlib.Path(__file__).resolve().parent, args.pattern)


if __name__ == "__main__":
    main()
