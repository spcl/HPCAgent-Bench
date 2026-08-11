# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every test file runs somewhere in CI.

This exists because the opposite was true and nothing said so: 144 test files, 44 named anywhere
in the workflow, 94 that never executed -- including guards written for regressions they were
meant to catch. A hand-written file list drifts in one direction only, because a new test is inert
by default and inertness is silent.
"""
import pathlib
import re
from typing import Set

REPO = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "tests.yml"
DEDICATED = REPO / ".github" / "dedicated_tests.txt"


def dedicated_files() -> Set[str]:
    """Paths the exclusion file claims, comments and blanks dropped -- the same parse tests.yml does."""
    out: Set[str] = set()
    for line in DEDICATED.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(line)
    return out


def all_test_files() -> Set[str]:
    return {f"tests/{p.name}" for p in sorted((REPO / "tests").glob("test_*.py"))}


def test_every_test_file_runs_somewhere() -> None:
    """The invariant: a file is swept, or claimed by a dedicated phase. There is no third state."""
    claimed = dedicated_files()
    swept = all_test_files() - claimed
    assert swept, "the unit sweep would select nothing"
    orphaned = claimed - all_test_files()
    assert not orphaned, (f"dedicated_tests.txt names files that do not exist: {sorted(orphaned)}. "
                          "A stale entry silently shrinks the sweep.")


def test_a_dedicated_file_is_actually_run_by_some_phase() -> None:
    """Excluding a file from the sweep is only legitimate when another phase runs it.

    Without this, dedicated_tests.txt becomes the new silent-inertness mechanism -- the exact
    failure it was introduced to end, one indirection later."""
    workflow = WORKFLOW.read_text()
    missing = [name for name in sorted(dedicated_files()) if name not in workflow]
    assert not missing, (f"excluded from the sweep but named by no phase, so they run NOWHERE: {missing}. "
                         "Either give the file a phase, or quarantine it with a written reason.")


def test_the_sweep_is_discovered_not_enumerated() -> None:
    """The workflow must derive its file list from the filesystem, not carry one."""
    workflow = WORKFLOW.read_text()
    assert "dedicated_tests.txt" in workflow, "the sweep no longer reads the exclusion file"
    assert "ls tests/test_*.py" in workflow, "the sweep no longer discovers files with ls"


def test_ci_never_asks_for_a_billed_runner() -> None:
    """Standard GitHub-hosted runners are free on a public repo; LARGER runners bill per minute
    even here. Self-hosted is our own hardware and bills nothing.

    A guard rather than a review habit: `runs-on: ubuntu-latest-8-cores` is one plausible edit away
    from `ubuntu-latest`, reads as a harmless speedup, and the cost of getting it wrong arrives on
    an invoice rather than in a test run."""
    standard = {"ubuntu-latest", "ubuntu-24.04", "ubuntu-22.04", "windows-latest", "macos-latest"}
    offenders = []
    for workflow in sorted((REPO / ".github" / "workflows").glob("*.y*ml")):
        for line in workflow.read_text().splitlines():
            match = re.match(r"\s*runs-on:\s*(.+?)\s*$", line)
            if not match:
                continue
            value = match.group(1)
            if value.startswith("["):  # a label list -- self-hosted, i.e. our own machine
                if "self-hosted" not in value:
                    offenders.append(f"{workflow.name}: {value}")
            elif value not in standard:
                offenders.append(f"{workflow.name}: {value}")
    assert not offenders, (f"non-standard, billed-per-minute runners requested: {offenders}. "
                           f"Free on a public repo are {sorted(standard)}, plus self-hosted labels.")


def test_no_workflow_declares_the_same_key_twice() -> None:
    """A duplicate mapping key makes GitHub reject the WHOLE workflow at startup -- zero jobs, zero
    logs, and a run that reports "failed because of a workflow file issue" with nothing to read.

    PyYAML does not help: it silently keeps the last value, so a duplicate parses locally, passes
    every yaml-based check, and only fails once pushed. That is exactly how a second job-level
    ``env:`` shipped in frameworks-pluto and mpi -- the coverage variable landed in a new block
    beside the existing one and quietly discarded ``PLUTO_COMMIT`` and the four ``OMPI_MCA_*``
    settings on the way. This loader refuses instead of keeping the last one.
    """
    import yaml

    class NoDuplicates(yaml.SafeLoader):
        pass

    def strict_mapping(loader, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in seen:
                raise AssertionError(f"duplicate key {key!r} at line {key_node.start_mark.line + 1}")
            seen.add(key)
        return yaml.SafeLoader.construct_mapping(loader, node, deep)

    NoDuplicates.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, strict_mapping)
    for path in sorted((REPO / ".github" / "workflows").glob("*.yml")):
        yaml.load(path.read_text(), Loader=NoDuplicates)  # raises AssertionError naming the key
    yaml.load((REPO / ".github" / "actions" / "setup" / "action.yml").read_text(), Loader=NoDuplicates)


def test_every_pytest_plugin_the_workflow_asks_for_is_installed() -> None:
    """A plugin flag in PYTEST_ADDOPTS is a HARD dependency: pytest fails at argument parsing, so
    every phase of every job dies before collecting a single test.

    That is not hypothetical. Setting ``PYTEST_ADDOPTS: --cov=hpcagent_bench`` job-wide without
    adding pytest-cov to the setup action turned six green jobs red at once, each with
    ``ERROR: usage: python -m pytest [options]`` and nothing else -- a failure that looks like a
    test failure and is not one. The flags and the install list have to agree.
    """
    import re

    setup = (REPO / ".github" / "actions" / "setup" / "action.yml").read_text()
    text = WORKFLOW.read_text()
    # option prefix -> the distribution that provides it
    plugins = {"--cov": "pytest-cov", "--timeout": "pytest-timeout", "-n ": "pytest-xdist", "--dist": "pytest-xdist"}
    asked = {dist for opt, dist in plugins.items() if re.search(rf"PYTEST_ADDOPTS:.*{re.escape(opt.strip())}", text)}
    missing = sorted(d for d in asked if d not in setup)
    assert not missing, (f"PYTEST_ADDOPTS asks for {missing}, which .github/actions/setup/action.yml "
                         f"does not install -- every pytest call in CI would fail on an unrecognized argument")


def test_asking_for_skip_reasons_does_not_hide_the_failures() -> None:
    """``-r`` REPLACES the report set, it does not add to it. pytest's default is ``-rfE``, so a
    phase that asks for skip reasons with a bare ``-rs`` prints its skips and stops naming which
    tests FAILED -- the summary still says ``5 failed`` and no longer says which five.

    That is measured, not theorised: on the same three-test file, ``-rs`` prints only the SKIPPED
    line while ``-rfEs`` prints ``FAILED ...::test_fail`` above it. A CI job whose whole purpose is
    to say what broke must keep the f and E.
    """
    # Only pytest lines, and only a STANDALONE -r<letters> token: --no-install-recommends is not one.
    offenders = [
        i + 1 for i, line in enumerate(WORKFLOW.read_text().splitlines()) if "pytest" in line
        for token in re.findall(r"(?<![\w-])-r[a-zA-Z]+\b", line) if "s" in token and "f" not in token
    ]
    assert not offenders, (f"tests.yml lines {offenders} ask for skip reasons without keeping failures in the "
                           "report set; use -rfEs so a failing test is still named in the short summary")


def test_the_combined_total_is_built_from_every_job_not_one_of_them() -> None:
    """Seven jobs each upload their coverage data as a file literally named ``.coverage``.
    ``merge-multiple: true`` flattens them into ONE directory, so seven artifacts race for one
    path: six are discarded and whichever wins becomes the published "total".

    That is measured, not theorised. Two consecutive GREEN runs reported ``Combined 1 file`` and a
    total of 59.96% and 13.44% -- the same repo, the swing being purely which job won. The defect
    only ever announced itself when two extractions interleaved and left a torn SQLite file, which
    surfaced as ``database disk image is malformed`` against the repo-root path (coverage's
    ATTACH-based combine misattributes the error to the main db, so the message names the wrong
    file). A wrong total that stays green is the worse half of this bug.

    Two things have to hold: artifacts land in per-artifact subdirectories, and the combine
    REFUSES a partial merge rather than reporting a plausible fraction of the project.
    """
    text = WORKFLOW.read_text()
    combine = [i + 1 for i, line in enumerate(text.splitlines()) if "coverage combine" in line]
    assert combine, "no `coverage combine` step -- the combined total is not being built at all"
    assert "merge-multiple: true" not in text, (
        "an artifact download uses merge-multiple: true; every job's data file is named `.coverage`, "
        "so flattening them makes six of seven silently disappear into one contested path")
    assert "coverage-data/*/.coverage*" in text, (
        "the combine glob must reach into the per-artifact subdirectories that dropping "
        "merge-multiple creates, or it finds nothing at all")
    assert 'Combined ${#files[@]} file' in text, (
        "nothing checks that combine consumed every uploaded file; a partial combine prints a "
        "perfectly plausible percentage and stays green, which is how this went unnoticed")


def test_the_corpus_reference_phase_is_not_instrumented() -> None:
    """Phase 2c runs ``hpcagent_bench/benchmarks/``. Every file it measures is inside the
    ``[tool.coverage.run] omit`` pattern, so instrumenting it produces no report data at all --
    it is pure cost.

    And the cost is not small. ``omit`` stops LINE tracing, not the per-call dispatch: sys.settrace
    fires on every call event even for a file it will never record. This phase is call-dominated
    (one cloudsc test makes 4.4M calls), so it pays that dispatch millions of times to discard the
    result. Measured: 8.28 s bare against >1500 s instrumented (killed, not finished -- >181x), and in
    CI the same 745 tests went
    183.57 s -> 736 s when coverage landed, which is what pushed the heaviest test past
    ``--timeout=600`` and made the job red for three consecutive runs.

    ``COVERAGE_CORE=sysmon`` is not an escape: coverage refuses it while ``branch = true`` on
    Python < 3.14 and again for ``concurrency=``, warns, and falls back to the C tracer -- so it
    looks like a fix and changes nothing.
    """
    text = WORKFLOW.read_text()
    phase = text.index("Phase 2c -- benchmark reference validation")
    nxt = text.index("- name: ", phase)
    step = text[phase:nxt]
    assert 'PYTEST_ADDOPTS: ""' in step, (
        "Phase 2c must clear PYTEST_ADDOPTS: it runs only corpus files, every one of which the "
        "coverage config omits, so instrumenting it costs the job and yields nothing")


def test_the_coverage_omit_list_and_the_uninstrumented_phase_agree() -> None:
    """The phase above is only safe to leave uninstrumented BECAUSE its tree is omitted. If the
    omit pattern is ever narrowed, that phase silently starts being the one place a real library
    path went unmeasured -- so pin the two together rather than leaving the link in a comment.
    """
    import tomllib

    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    omit = pyproject["tool"]["coverage"]["run"]["omit"]
    assert any(
        pattern.startswith("hpcagent_bench/benchmarks")
        for pattern in omit), ("coverage no longer omits hpcagent_bench/benchmarks/, but Phase 2c still runs that tree "
                               "with coverage disabled -- either re-instrument the phase or restore the omit")


def test_ci_installs_the_tools_that_fail_silently_when_absent() -> None:
    """ninja and ccache do not error when missing -- the build just gets slower, which reads as
    "CI is sluggish" rather than as a defect, so nothing surfaces it.

    ninja is the sharper of the two: DaCe chooses its CMake generator with
    ``shutil.which('ninja')`` and replays recorded compile commands ONLY when it picked Ninja, so
    without the package ``compiler.command_cache`` still reports True while every SDFG pays a full
    CMake configure. A config that reads enabled and does nothing is the same failure shape as a
    guard that checks one direction of a two-directional error.
    """
    setup = (REPO / ".github" / "actions" / "setup" / "action.yml").read_text()
    # The INSTALL lines, not the whole file: the comment block right above them explains why each
    # tool is there and names both, so a substring search over the file passes on its own prose
    # after the package is dropped -- the same silent-absence failure this test exists to catch.
    joined = re.sub(r"\\\n\s*", " ", setup)  # the package list wraps with a backslash continuation
    installs = [line for line in joined.splitlines() if "apt-get install" in line]
    installed = " ".join(installs)
    assert installs, "no apt-get install line in .github/actions/setup/action.yml"
    for tool in ("ninja-build", "ccache"):
        assert tool in installed, (f"{tool} is not installed by .github/actions/setup/action.yml; without it the "
                                   f"build silently loses its cache instead of failing")
