# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The patterns of ``scripts/checks/check_repo_rules.py`` on synthetic text: each must catch the
offence it names and pass its lookalikes, or the hook is green for nothing."""

import types

from hpcagent_bench import paths
from tests.fresh_module import module_at


def rules() -> types.ModuleType:
    return module_at(paths.ROOT / "scripts" / "checks" / "check_repo_rules.py", "check_repo_rules")


def test_the_import_path_pattern_tells_an_edit_from_a_read() -> None:
    edit = rules().IMPORT_PATH_EDIT
    for text in (
        'sys.path.insert(0, "x")',
        "sys.path[:] = saved",
        'export PYTHONPATH="${x}"',
        'env["PYTHONPATH"] = x',
        'env = {"PYTHONPATH": x}',
        'env.update(PYTHONPATH="x")',
        "monkeypatch.syspath_prepend(d)",
    ):
        assert edit.search(text), text
    for text in ('env.get("PYTHONPATH", "")', 'value = env["PYTHONPATH"]', "if x == sys.path:", '"PYTHONPATH",'):
        assert not edit.search(text), text


def test_the_site_scan_catches_every_kind_of_hit() -> None:
    module = rules()
    files = {
        "bad.sh": (
            "#!/usr/bin/env bash\n"
            "#SBATCH --partition=gpu1\n"
            "#SBATCH -A proj\n"
            "# this comment mentions /users/ybudanaz and must NOT be flagged as a home directory\n"
            'CE_EDF="/users/ybudanaz/x86_64/.edf/agent.toml"\n'
            'FAST_SCRATCH="/iopsstor/scratch/cscs/${USER}"\n'
            'SCRATCH="/ritom/scratch/cscs/$(id -un)"\n'
            'STORE="/capstor/store/cscs/project"\n'
            'OLD_HOME="/home/alice/runs"\n'
            "sbatch --account=a-g34 --partition=gpu1 --exclude=nid[001,002] bad.sh\n"
            "srun -p debug -N 1 true\n"
            'SBATCH_PARTITION="${SBATCH_PARTITION:-normal}"\n'
            "ssh -J beverin nid002664\n"
            'RUNS="${SCRATCH}/hpcagent-bench-runs/llr40-20260916/639344"\n'
            'OUT="${SCRATCH}/canon-648131"\n'
            "IMAGE=jfrog.svc.cscs.ch/hpcagent/judge:latest\n"
        ),
        "bad.py": (
            f'"""A note naming the {module.OLD_NAME} rename. Not a violation."""\n'
            'REPO_DEFAULT = "/capstor/scratch/cscs/someone/hpcagent-bench"\n'
            'ACCOUNT = "a-g200"\n'
            'CONTACT = "someone@inf.ethz.ch"\n'
            'CMD = "srun -p mi300 true"\n'
        ),
        "bad.md": f"# Notes\n\nDo not reintroduce {module.OLD_NAME} anywhere in the docs.\n",
        "bad.tsv": "job\tdb\n1\t/capstor/scratch/cscs/x/hpcagent-bench-runs/wave/1/rank-0.db\n",
        "site.env": 'FAST_SCRATCH="${FAST_SCRATCH:-/iopsstor/scratch/cscs/${USER}}"\n',
        "good.sh": (
            "#!/usr/bin/env bash\n"
            "#SBATCH --nodes=1\n"
            '. "${HPCAGENT_BENCH_REPO}/scripts/site_env.sh"\n'
            'FAST_SCRATCH="${FAST_SCRATCH:-${SCRATCH}}"\n'
            'sbatch ${part:+--partition="${part}"} --partition="${PARTITION}" job.sbatch\n'
            'HOST_HOME="/users/someone"\n'
            'RUNS="${HPCAGENT_BENCH_RUNS_ROOT}/canon/${tag}-${stamp}"\n'
            "EDF=hpcagent-bench-agent-mi300-latest\n"
            "sbatch services.sbatch\n"
        ),
    }
    hits = {name: [hit[1] for hit in module.site_hits(name, text)] for name, text in files.items()}

    def hit(name: str, text: str) -> bool:
        return any(text in matched for matched in hits[name])

    for text in (
        "/users/ybudanaz",
        "/iopsstor",
        "/ritom",
        "/capstor/",
        "/home/alice",
        "a-g34",
        "--partition=gpu1",
        "-A proj",
        "nid[",
        "-p debug",
        "PARTITION:-normal",
        "beverin",
        "nid002664",
        "hpcagent-bench-runs/llr40-20260916",
        "canon-648131",
        "jfrog.svc.cscs.ch",
    ):
        assert hit("bad.sh", text), (text, hits["bad.sh"])
    for text in ("/capstor/", "a-g200", "ethz.ch", "-p mi300"):
        assert hit("bad.py", text), (text, hits["bad.py"])
    assert hit("bad.md", module.OLD_NAME), hits["bad.md"]
    assert hit("bad.tsv", "/capstor/"), hits["bad.tsv"]
    assert hit("site.env", "/iopsstor"), hits["site.env"]
    assert not hit("bad.py", module.OLD_NAME), "a docstring may name the old project"
    assert hits["good.sh"] == []


def test_the_storage_pattern_matches_only_real_mounts() -> None:
    """Fires on the mounts even behind a $USER/${VAR} suffix; quiet on lookalikes."""
    pattern = next(p for label, p in rules().SITE_PATTERNS.items() if label.startswith("literal storage mount"))
    for text in (
        'SCRATCH="/ritom/scratch/cscs/someone/$(uname -m)"',
        'FAST_SCRATCH="${FAST_SCRATCH:-/iopsstor/scratch/cscs/${USER}}"',
        '    echo "/capstor/scratch/cscs" >&2',
        '"/ritom:/ritom"',
        "BASE=/capstor/store/cscs/cscs/public",
    ):
        assert pattern.search(text), text
    for text in (
        'FAST_SCRATCH="${FAST_SCRATCH:-${SCRATCH}}"',
        'ALT="/iopsstorbackup/old"',
        'NESTED="something/ritom/x"',
        'FIXTURE="/scratchfs/runs/1"',
    ):
        assert not pattern.search(text), text


if __name__ == "__main__":
    test_the_import_path_pattern_tells_an_edit_from_a_read()
    test_the_site_scan_catches_every_kind_of_hit()
    test_the_storage_pattern_matches_only_real_mounts()
