# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The patterns of ``helpers/scripts/checks/check_repo_rules.py`` on synthetic text: each must catch the
offence it names and pass its lookalikes, or the hook is green for nothing."""

import types

from hpcagent_bench import paths
from tests.fresh_module import module_at


def rules() -> types.ModuleType:
    return module_at(paths.ROOT / "helpers" / "scripts" / "checks" / "check_repo_rules.py", "check_repo_rules")


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
            "# this comment mentions /users/jdoe and must NOT be flagged as a home directory\n"
            'CE_EDF="/users/jdoe/x86_64/.edf/agent.toml"\n'
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
            '. "${HPCAGENT_BENCH_REPO}/helpers/scripts/site_env.sh"\n'
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
        "/users/jdoe",
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


def test_the_user_pattern_matches_a_name_only_as_a_path_component_or_address() -> None:
    """``user_name_pattern("jdoe")`` fires on ``/…/jdoe``, ``~jdoe`` and ``jdoe@``; quiet on the bare word and on
    longer names."""
    pattern = rules().user_name_pattern("jdoe")
    for text in ("/scratch/cscs/jdoe/x", "cd /users/jdoe", "~jdoe/.edf", "ssh jdoe@login"):
        assert pattern.search(text), text
    for text in ("ask jdoe about it", "/scratch/jdoex/y", "/scratch/jdoe.old", "xjdoe@login"):
        assert not pattern.search(text), text


def test_the_running_user_pattern_is_off_under_a_generic_account() -> None:
    """Under ``root``/``runner`` a ``/root/`` or ``/home/runner`` path is generic: no running-user pattern."""
    module = rules()
    assert module.GENERIC_ACCOUNTS >= {"root", "runner"}
    labels = list(module.USER_PATTERNS)
    assert labels == ([] if module.RUNNING_USER in module.GENERIC_ACCOUNTS else [labels[0]])


def test_the_private_address_and_prose_node_patterns() -> None:
    """A private IPv4 host address is flagged outside tests; a CIDR block, a dashed or assigned version and a
    documentation address (RFC 5737) are not; a lockfile is not scanned for them. Markdown naming a node is
    flagged; a placeholder is not."""
    module = rules()
    (pattern,) = module.NON_TEST_PATTERNS.values()
    for text in ("http://172.28.9.16:30000/v1", "host 10.1.2.3 up", "192.168.0.7"):
        assert pattern.search(text), text
    for text in ("172.28.0.0/16", "pkg-10.4.0.35", "v=10.4.0.35", "http://192.0.2.16:30000"):
        assert not pattern.search(text), text
    md = "base URL: http://nid002968:8000/v1\nbase URL: http://<node>:8000/v1\n"
    assert [hit[1] for hit in module.site_hits("docs/x.md", md)] == ["nid002968"]
    assert module.site_hits("tests/x.md", md) == []
    assert module.site_hits("uv.lock", 'version = "10.4.0.35"\n') == []


def test_the_package_literal_scan_names_layout_joins_and_absolute_prefixes() -> None:
    """``x / "judge"``, a whole ``"results.db"`` and a ``"/opt/rocm"``/``"/dev/shm/…"`` string are hits in a
    package file; a docstring, a message naming them mid-sentence and an unrelated join are not."""
    module = rules()
    text = (
        '"""Writes judge/ and results.db under /dev/shm/x."""\n'
        'a = run / "judge"\n'
        'b = "results.db"\n'
        'c = "/opt/rocm"\n'
        'd = f"/dev/shm/{user}"\n'
        'e = run / "logs"\n'
        'f = "put /opt/rocm/bin on PATH"\n'
    )
    hits = module.package_literals("hpcagent_bench/x.py", text)
    assert [hit.split(":")[1] for hit in hits] == ["2", "3", "4", "5"], hits


def test_the_root_climb_and_shell_patterns() -> None:
    """``Path(__file__)`` climbing above its directory, a shell ``/../..`` and a spelled EDF directory match; a
    sibling file, ``parent`` alone and ``EDF_PATH`` do not."""
    module = rules()
    for text in (
        "ROOT = pathlib.Path(__file__).resolve().parents[1]",
        "ROOT = Path(__file__).parent.parent",
        "x = pathlib.Path(__file__).parents[2]",
    ):
        assert module.FILE_ROOT_CLIMB.search(text), text
    for text in ('pathlib.Path(__file__).with_name("a.yaml")', 'pathlib.Path(__file__).parent / "x"'):
        assert not module.FILE_ROOT_CLIMB.search(text), text
    assert module.SHELL_ROOT_CLIMB.search('"$(dirname "$0")/../.."')
    for text in ('"${HOME}/.edf/x.toml"', "$HOME/.edf", "~/.edf/a.toml"):
        assert module.EDF_HOME.search(text), text
    assert not module.EDF_HOME.search('"${EDF_PATH%%:*}/x.toml"')


if __name__ == "__main__":
    test_the_import_path_pattern_tells_an_edit_from_a_read()
    test_the_site_scan_catches_every_kind_of_hit()
    test_the_storage_pattern_matches_only_real_mounts()
    test_the_user_pattern_matches_a_name_only_as_a_path_component_or_address()
    test_the_running_user_pattern_is_off_under_a_generic_account()
    test_the_private_address_and_prose_node_patterns()
    test_the_package_literal_scan_names_layout_joins_and_absolute_prefixes()
    test_the_root_climb_and_shell_patterns()
