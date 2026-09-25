"""Add one version per distinct skills tree the paper's runs staged, read from the benchmark at the
exact commit each run recorded. Read-only on the mirror and the benchmark checkout.

    python3 snapshot_runs.py --runs <mirror>/hpcagent-bench-runs --bench <hpcagent-bench checkout>

Every run's .agent-launch/<job>/.env records HPCAGENT_BENCH_RECORD_COMMIT, _PACKET and _EXPERIMENT.
Commits whose hpcagent_bench/skills tree is identical share one version, so a version is a change in
what an agent could read, not a commit. Versions continue the numbering of the language-packet
history (after v11) in the order their first run's commit was made.
"""

import argparse
import collections
import io
import pathlib
import re
import subprocess
import tarfile

HERE = pathlib.Path(__file__).resolve().parent
SKILLS = "hpcagent_bench/skills"
#: Experiments the paper reports; smoke and mi200 runs are not part of it.
PAPER_EXPERIMENTS = {"llr-focus40", "llr-focus40-blind", "git-scicomp", "scicomp-focus40", "harness20", "mlscale"}
FIRST_VERSION = 12


def git(bench: pathlib.Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(bench), *args], check=True, capture_output=True, text=True).stdout.strip()


def run_records(runs: pathlib.Path) -> list[dict[str, str]]:
    """(commit, packet, experiment, job) of every paper run under the mirror."""
    records = []
    for env in sorted(runs.glob("*/.agent-launch/*/.env")):
        fields = dict(re.findall(r"^HPCAGENT_BENCH_RECORD_(COMMIT|PACKET|EXPERIMENT)=(.*)$", env.read_text(), re.M))
        if fields.get("EXPERIMENT") in PAPER_EXPERIMENTS and fields.get("COMMIT"):
            records.append({**fields, "JOB": env.parent.name, "ROOT": env.parts[-4]})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", required=True, type=pathlib.Path)
    parser.add_argument("--bench", required=True, type=pathlib.Path)
    args = parser.parse_args()
    by_commit = collections.defaultdict(list)
    for record in run_records(args.runs):
        by_commit[record["COMMIT"]].append(record)
    trees: dict[str, list[str]] = collections.defaultdict(list)
    dated, records_of = {}, collections.defaultdict(list)
    for commit, records in by_commit.items():
        full = git(args.bench, "rev-parse", commit)
        records_of[full] += records
        if full not in dated:
            trees[git(args.bench, "rev-parse", f"{full}:{SKILLS}")].append(full)
            dated[full] = git(args.bench, "show", "-s", "--format=%cs", full)
    ordered = sorted(trees.items(), key=lambda item: min(dated[c] for c in item[1]))
    rows = []
    for number, (tree, commits) in enumerate(ordered, start=FIRST_VERSION):
        out = HERE / f"v{number:02d}"
        out.mkdir(exist_ok=True)
        archive = subprocess.run(
            ["git", "-C", str(args.bench), "archive", commits[0], SKILLS], check=True, capture_output=True
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            for member in tar.getmembers():
                if member.isfile():
                    target = out / pathlib.Path(member.name).relative_to(SKILLS)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(tar.extractfile(member).read())
        uses = collections.Counter()
        for commit in commits:
            for record in records_of[commit]:
                uses[(record["EXPERIMENT"], record["PACKET"] or "none")] += 1
        first = min(dated[c] for c in commits)
        lines = [
            f"# v{number:02d} (as run)",
            "",
            f"Skills tree `{tree[:12]}`, first run {first}.",
            "",
            "Commits: " + ", ".join(f"`{c[:9]}`" for c in sorted(commits, key=dated.get)),
            "",
            "| Experiment | Packet | Jobs |",
            "|---|---|---|",
        ]
        lines += [f"| {e} | {p} | {n} |" for (e, p), n in sorted(uses.items())]
        (out / "INDEX.md").write_text("\n".join(lines) + "\n")
        rows.append((number, first, len(commits), sum(uses.values())))
    for number, first, commits, jobs in rows:
        print(f"v{number:02d}  first run {first}  {commits} commits  {jobs} jobs")


if __name__ == "__main__":
    main()
