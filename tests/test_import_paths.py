# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Code gets its import path from ONE place, so no tracked file edits ``sys.path`` or ``PYTHONPATH``.

The package is installable (``pip install -e .``). A checkout used without installing it takes its
path from ``scripts/repo_env.sh`` (shells) or ``scripts/repo_python`` (a command that starts inside
a container), and the suite from pyproject's pytest ``pythonpath``. Every other edit is either on
:data:`ALLOWED` below, with the reason it has to exist, or a failure here. Markdown is scanned too,
so an instruction to export PYTHONPATH cannot come back into the docs.
"""

import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]

#: An edit of the import path: a ``sys.path`` insert/append/extend or assignment, pytest's
#: ``syspath_prepend``, ``site.addsitedir``, or ``PYTHONPATH`` set in a shell, an env dict or a
#: keyword argument. Reading ``PYTHONPATH`` or naming it in a comment is not an edit.
EDIT = re.compile(
    r"""sys\.path\.(insert|append|extend)\("""
    r"""|sys\.path(\[[^\]]*\])?\s*=(?!=)"""
    r"""|syspath_prepend\("""
    r"""|addsitedir\("""
    r"""|\bPYTHONPATH\s*=(?!=)"""
    r"""|\[\s*["']PYTHONPATH["']\s*\]\s*=(?!=)"""
    r"""|["']PYTHONPATH["']\s*:"""
    r"""|export\s+PYTHONPATH\b"""
)

#: Repo-relative path -> why that file may edit the import path.
ALLOWED: dict[str, str] = {
    # The one mechanism.
    "scripts/repo_env.sh": "the one place a shell puts the checkout (and DACE_TREE) on PYTHONPATH",
    # Scripts that run inside the agent/judge images, beside sibling modules they import by bare
    # name. The images set PYTHONSAFEPATH=1, which drops the script's own directory from sys.path.
    "containers/agent/harness/run_miniswe.py": "image script: own-directory insert (PYTHONSAFEPATH=1)",
    "containers/agent/harness/run_openhands.py": "image script: own-directory insert (PYTHONSAFEPATH=1)",
    "containers/agent/tools/hpcagent_bench_tool.py": "image script: own-directory insert (PYTHONSAFEPATH=1)",
    "containers/agent/tools/mcp_server.py": "image script: own-directory insert (PYTHONSAFEPATH=1)",
    "experiments/agent_driver.py": "agent-image script: own-directory insert for its staged siblings",
    "experiments/harnesses.py": "agent-image module: own-directory insert; optimas_env builds the "
    "optimas runner's PYTHONPATH (mounted checkout + vendored SDK) inside the judge image",
    "experiments/mpi/smoke_gang_rccl.py": "judge-image smoke: own-directory insert (PYTHONSAFEPATH=1)",
    "containers/images/selfcontained_check.py": "REMOVES its own directory from sys.path to "
    "prove the image imports without the checkout",
    # Third-party runtimes, not this repository's code.
    "containers/images/judge-agent-amd/Dockerfile": "rocprof-compute's wrapper names its "
    "own install dir, which PYTHONSAFEPATH=1 would otherwise hide",
    "containers/inference/tune-moe-int4-mi300a.sbatch": "vendored deps (pydeps) of "
    "the MoE tuning script",
    # Forwarding or resetting the value repo_env.sh built.
    "scripts/release_smoke_mi200.sbatch": "forwards repo_env.sh's PYTHONPATH into the harbor verifier "
    "container, which mounts the checkout",
    # Tests: a child process or a temp module, given its own path.
    "tests/test_dace_helper_programs.py": "temp module written under tmp_path",
    "tests/test_disk_cache.py": "child processes racing the store import the checkout",
    "tests/test_fork_openmp_safety.py": "child interpreter imports the checkout",
    "tests/test_forked.py": "child forker script imports the checkout",
    "tests/test_fused_owed_wave.py": "child owed_wave.py imports the checkout",
    "tests/test_harness_runners.py": "child mimics the image: runner/tool dir on the path, PYTHONSAFEPATH=1",
    "tests/test_integration_sweep.py": "child CLI run from a tmp cwd imports the checkout",
    "tests/test_judge_upstream_supervisor.py": "child probe imports the checkout",
    "tests/test_make_problems_select.py": "child make_problems.py with a minimal env",
    "tests/test_materialize_shared.py": "child materialize_shared.sh imports the checkout",
    "tests/test_numba_compile_cache.py": "fresh child interpreter imports the checkout",
    "tests/test_optimas_tools.py": "vendored openai-agents SDK, auto-reverted by monkeypatch",
    "tests/test_packaging.py": "child imports the installed wheel and nothing else",
    "tests/test_packet_wiring.py": "child mimics the image: mcp_server dir on the path, PYTHONSAFEPATH=1",
    "tests/test_perf_reports.py": "temp numba module under tmp_path, auto-reverted by monkeypatch",
    "tests/test_prepare_job_generated_cache.py": "child with a minimal env imports the checkout",
    "tests/test_record_tag_version.py": "child script imports the checkout",
    "tests/test_reporting_e2e.py": "child CLI run from a tmp cwd imports the checkout",
    "tests/test_skill_isolation_matrix.py": "child materialize_shared.sh imports the checkout",
    "tests/test_submit_git_scicomp.py": "child submit script imports the checkout",
    "tests/test_submit_gpu_llr40_clean_dryrun.py": "child submit script imports the checkout",
    "tests/test_submit_llrblind.py": "child submit script imports the checkout",
    "tests/test_submit_scicomp_dc_cpfsrc.py": "child submit script imports the checkout",
    "tests/test_submit_scicomp_perf_playbook.py": "child submit script imports the checkout",
    "tests/test_submit_scicomp_perf_playbook_clean_dryrun.py": "child submit script imports the checkout",
    "tests/test_import_paths.py": "this file spells the patterns it searches for",
    # Documents.
    "docs/extending/agent-harness.md": "quotes a harness runner's own-directory insert (PYTHONSAFEPATH=1)",
}


def tracked_code() -> list[pathlib.Path]:
    """Every tracked regular file (symlinks are scanned at their target)."""
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True)
    names = [name for name in listed.stdout.split("\0") if name]
    return [REPO / name for name in names if (REPO / name).is_file() and not (REPO / name).is_symlink()]


def edits(path: pathlib.Path) -> list[int]:
    """Line numbers of ``path`` that edit the import path; comment lines do not count."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    return [
        number
        for number, line in enumerate(text.splitlines(), 1)
        if not line.lstrip().startswith("#") and EDIT.search(line)
    ]


def test_no_file_edits_the_import_path_outside_the_allowlist() -> None:
    offenders = []
    for path in tracked_code():
        rel = path.relative_to(REPO).as_posix()
        if rel not in ALLOWED:
            offenders.extend(f"{rel}:{number}" for number in edits(path))
    assert not offenders, (
        "import-path edits outside tests/test_import_paths.py's ALLOWED (source scripts/repo_env.sh, run "
        "scripts/repo_python, or rely on pytest's pythonpath instead):\n" + "\n".join(offenders)
    )


def test_every_allowlisted_file_still_edits_the_import_path() -> None:
    """An entry whose file no longer edits the path (or no longer exists) is removed, not kept."""
    stale = [rel for rel in ALLOWED if not (REPO / rel).is_file() or not edits(REPO / rel)]
    assert not stale, f"ALLOWED entries with no import-path edit left: {stale}"


def test_the_pattern_tells_an_edit_from_a_read() -> None:
    for edit in (
        'sys.path.insert(0, "x")',
        "sys.path[:] = saved",
        'export PYTHONPATH="${x}"',
        'env["PYTHONPATH"] = x',
        'env = {"PYTHONPATH": x}',
        'env.update(PYTHONPATH="x")',
        "monkeypatch.syspath_prepend(d)",
    ):
        assert EDIT.search(edit), edit
    for read in ('env.get("PYTHONPATH", "")', 'value = env["PYTHONPATH"]', "if x == sys.path:", '"PYTHONPATH",'):
        assert not EDIT.search(read), read
