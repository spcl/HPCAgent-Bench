# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""experiments/submit-canon-llr40.sh's KERNELS_FILE support.

Every other family submitter narrows its roster with KERNELS_FILE (submit_common.sh's
kernels_file_list); this compiler-baseline launcher read the whole ${TAG} roster off roster_for and
had no KERNELS_FILE branch at all -- an operator handing it the same owed-kernels file that narrows
every agent arm was silently ignored, running the FULL roster instead of the subset. Fixed by giving
it the same KERNELS_FILE contract (one kernel name per line, comments/blanks dropped, unknown name
refused) as every other submit-*.sh.

Runs from a temp copy of the launcher's inputs, SUBMIT unset (prepare-only): no sbatch is ever
reached (COLUMNS defaults to seven, each such call is a distinct assertion point, so the test never
lets one through).
"""

import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"

#: Two real scientific_computing kernels (shared with the git-scicomp/scicomp-dc tests), so
#: KERNELS_FILE's unknown-name check resolves them without a fabricated manifest.
ROSTER_KERNELS = ("kmp", "dfa")


def stub(directory: pathlib.Path, name: str, body: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(f"#!/usr/bin/env bash\n{body}\n")
    path.chmod(0o755)


def submit_tree(root: pathlib.Path) -> pathlib.Path:
    """A temp experiments/submit-canon-llr40.sh + roster.sh; account_env.sh resolves off a stub
    sacctmgr (one association), the same way test_submit_file_isolation.py's stub_account does.

    The launcher sources account_env.sh by a path RELATIVE to its own location
    (``$(dirname BASH_SOURCE)/../scripts/cscs/account_env.sh``), not through OPT/HPCAGENT_BENCH_REPO
    like every other family submitter -- so the temp tree needs a real copy one level above
    experiments/, not just the stub sacctmgr on PATH."""
    (root / "experiments").mkdir(parents=True)
    for name in ("submit-canon-llr40.sh", "roster.sh"):
        shutil.copy2(EXPERIMENTS / name, root / "experiments" / name)
    (root / "scripts" / "cscs").mkdir(parents=True)
    shutil.copy2(REPO / "scripts" / "cscs" / "account_env.sh", root / "scripts" / "cscs" / "account_env.sh")
    shutil.copy2(REPO / "scripts" / "site_env.sh", root / "scripts" / "site_env.sh")
    stub(root / "bin", "sacctmgr", "printf 'project-a\n'")
    stub(root / "bin", "sbatch", 'touch "${STUB_MARKERS}/sbatch-called"; exit 1')
    return root


def run_submit(root: pathlib.Path, **knobs: str) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": f"{root / 'bin'}:/usr/bin:/bin",
        "USER": "tester",
        "SCRATCH": str(root / "scratch"),
        # roster.sh/the KERNELS_FILE validator import hpcagent_bench off OPT directly (not ambient
        # PYTHONPATH), so OPT must be the real checkout even though the script itself runs from copy.
        "OPT": str(REPO),
        "PY": sys.executable,
        "SUBMIT": "0",
        "STUB_MARKERS": str(root),
        **knobs,
    }
    return subprocess.run(
        ["bash", str(root / "experiments" / "submit-canon-llr40.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_kernels_file_narrows_the_roster_instead_of_the_whole_tag(tmp_path: pathlib.Path) -> None:
    """The bug this closes: KERNELS_FILE used to be read by every OTHER submitter and silently
    dropped here, so an owed-kernels rerun submitted via this launcher ran the entire tag again."""
    root = submit_tree(tmp_path)
    kf = root / "experiments" / "owed.txt"
    kf.write_text("dfa\nkmp  # rerun\n")
    result = run_submit(root, KERNELS_FILE=str(kf))
    assert result.returncode == 0, result.stderr
    assert "roster: 2 kernels" in result.stdout, result.stdout
    assert not (root / "sbatch-called").exists()


def test_kernels_file_kernels_are_exact_and_in_deterministic_order(tmp_path: pathlib.Path) -> None:
    """KERNELS is comma-joined straight into canon_column.sh's argv (not printed on the SUBMIT=0
    preview line, only the column name is) -- a scrambled input file must not scramble the column's
    own kernel loop, so this reads it back off the real --wrap string a recording sbatch stub
    captures, and checks it is sorted the same way roster_for's own output is."""
    root = submit_tree(tmp_path)
    stub(
        root / "bin",
        "sbatch",
        'printf \'%s\\n\' "$@" > "${STUB_MARKERS}/sbatch-argv.txt"; echo 999999; exit 0',
    )
    kf = root / "experiments" / "owed.txt"
    kf.write_text("kmp\n# a comment line\ndfa\n\n")
    result = run_submit(root, KERNELS_FILE=str(kf), COLUMNS="numba", SUBMIT="1")
    assert result.returncode == 0, result.stderr
    argv = (root / "sbatch-argv.txt").read_text().splitlines()
    wrap = argv[argv.index("--wrap") + 1]
    # bash <path>/canon_column.sh outer <col> <out_root> <kernels> <preset> <opt>
    kernels = wrap.split()[5].split(",")
    assert kernels == sorted(ROSTER_KERNELS)


def test_an_unknown_kernel_name_is_refused_not_silently_dropped(tmp_path: pathlib.Path) -> None:
    root = submit_tree(tmp_path)
    kf = root / "experiments" / "owed.txt"
    kf.write_text("kmp\nnosuchkernel123\n")
    result = run_submit(root, KERNELS_FILE=str(kf))
    assert result.returncode == 2
    assert "nosuchkernel123" in result.stderr
    assert "would submit" not in result.stdout
    assert not (root / "sbatch-called").exists()


def test_a_missing_kernels_file_is_refused(tmp_path: pathlib.Path) -> None:
    root = submit_tree(tmp_path)
    result = run_submit(root, KERNELS_FILE=str(root / "experiments" / "does-not-exist.txt"))
    assert result.returncode == 2
    assert "missing or empty" in result.stderr


def test_nice_is_passed_through_when_set_and_defaults_to_the_site_nice(tmp_path: pathlib.Path) -> None:
    """NICE=300 (a gap-filling canon run sits behind the priority LLR/cpfsrc waves but ahead of a
    background scicomp sweep) reaches sbatch as ``--nice=300``; unset, the job still starts nicely,
    at the site layer's HPCAGENT_BENCH_NICE (scripts/site_env.sh)."""
    root = submit_tree(tmp_path)
    stub(
        root / "bin",
        "sbatch",
        'printf \'%s\\n\' "$@" > "${STUB_MARKERS}/sbatch-argv.txt"; echo 999999; exit 0',
    )
    result = run_submit(root, TAG="llr-focus40", COLUMNS="numba", SUBMIT="1", NICE="300")
    assert result.returncode == 0, result.stderr
    argv = (root / "sbatch-argv.txt").read_text().splitlines()
    assert "--nice=300" in argv

    stub(
        root / "bin",
        "sbatch",
        'printf \'%s\\n\' "$@" > "${STUB_MARKERS}/sbatch-argv-no-nice.txt"; echo 999999; exit 0',
    )
    result = run_submit(root, TAG="llr-focus40", COLUMNS="numba", SUBMIT="1", HPCAGENT_BENCH_NICE="77")
    assert result.returncode == 0, result.stderr
    argv = (root / "sbatch-argv-no-nice.txt").read_text().splitlines()
    assert [flag for flag in argv if flag.startswith("--nice")] == ["--nice=77"]

    result = run_submit(root, TAG="llr-focus40", COLUMNS="numba", SUBMIT="1")
    assert result.returncode == 0, result.stderr
    argv = (root / "sbatch-argv-no-nice.txt").read_text().splitlines()
    assert [flag for flag in argv if flag.startswith("--nice")] == ["--nice=100"]


def test_kernels_file_unset_still_uses_the_whole_tag_roster(tmp_path: pathlib.Path) -> None:
    """The pre-existing behaviour, untouched: no KERNELS_FILE, no KERNELS -- roster_for(TAG)."""
    root = submit_tree(tmp_path)
    result = run_submit(root, TAG="llr-focus40")
    assert result.returncode == 0, result.stderr
    assert "roster: 40 kernels" in result.stdout, result.stdout
