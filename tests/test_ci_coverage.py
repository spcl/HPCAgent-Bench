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
from typing import List, Set

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
    assert not orphaned, (
        f"dedicated_tests.txt names files that do not exist: {sorted(orphaned)}. "
        "A stale entry silently shrinks the sweep."
    )


def test_a_dedicated_file_is_actually_run_by_some_phase() -> None:
    """Excluding a file from the sweep is only legitimate when another phase runs it.

    Without this, dedicated_tests.txt becomes the new silent-inertness mechanism -- the exact
    failure it was introduced to end, one indirection later."""
    workflow = WORKFLOW.read_text()
    missing = [name for name in sorted(dedicated_files()) if name not in workflow]
    assert not missing, (
        f"excluded from the sweep but named by no phase, so they run NOWHERE: {missing}. "
        "Either give the file a phase, or quarantine it with a written reason."
    )


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
    assert not offenders, (
        f"non-standard, billed-per-minute runners requested: {offenders}. "
        f"Free on a public repo are {sorted(standard)}, plus self-hosted labels."
    )


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
    adding pytest-cov to what CI installs turned six green jobs red at once, each with
    ``ERROR: usage: python -m pytest [options]`` and nothing else -- a failure that looks like a
    test failure and is not one. The flags and the install list have to agree.

    Read from the ``testing`` dependency group in pyproject.toml, which is what the setup action
    installs (``pip install --group testing``). Grepping the action's own text was the same check
    while the plugin names were written out there; once they moved into the project metadata that
    grep could only ever answer "missing", and the invariant lives wherever the names now are.
    """
    import re
    import tomllib

    groups = tomllib.loads((REPO / "pyproject.toml").read_text())["dependency-groups"]
    installed = " ".join(str(entry) for entries in groups.values() for entry in entries)
    text = WORKFLOW.read_text()
    # option prefix -> the distribution that provides it
    plugins = {"--cov": "pytest-cov", "--timeout": "pytest-timeout", "-n ": "pytest-xdist", "--dist": "pytest-xdist"}
    asked = {dist for opt, dist in plugins.items() if re.search(rf"PYTEST_ADDOPTS:.*{re.escape(opt.strip())}", text)}
    missing = sorted(d for d in asked if d not in installed)
    assert not missing, (
        f"PYTEST_ADDOPTS asks for {missing}, which no pyproject.toml dependency group "
        f"installs -- every pytest call in CI would fail on an unrecognized argument"
    )


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
        i + 1
        for i, line in enumerate(WORKFLOW.read_text().splitlines())
        if "pytest" in line
        for token in re.findall(r"(?<![\w-])-r[a-zA-Z]+\b", line)
        if "s" in token and "f" not in token
    ]
    assert not offenders, (
        f"tests.yml lines {offenders} ask for skip reasons without keeping failures in the "
        "report set; use -rfEs so a failing test is still named in the short summary"
    )


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
        "so flattening them makes six of seven silently disappear into one contested path"
    )
    assert "coverage-data/*/.coverage*" in text, (
        "the combine glob must reach into the per-artifact subdirectories that dropping "
        "merge-multiple creates, or it finds nothing at all"
    )
    assert "Combined ${#files[@]} file" in text, (
        "nothing checks that combine consumed every uploaded file; a partial combine prints a "
        "perfectly plausible percentage and stays green, which is how this went unnoticed"
    )


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
        "coverage config omits, so instrumenting it costs the job and yields nothing"
    )


def test_the_coverage_omit_list_and_the_uninstrumented_phase_agree() -> None:
    """The phase above is only safe to leave uninstrumented BECAUSE its tree is omitted. If the
    omit pattern is ever narrowed, that phase silently starts being the one place a real library
    path went unmeasured -- so pin the two together rather than leaving the link in a comment.
    """
    import tomllib

    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    omit = pyproject["tool"]["coverage"]["run"]["omit"]
    assert any(pattern.startswith("hpcagent_bench/benchmarks") for pattern in omit), (
        "coverage no longer omits hpcagent_bench/benchmarks/, but Phase 2c still runs that tree "
        "with coverage disabled -- either re-instrument the phase or restore the omit"
    )


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
        assert tool in installed, (
            f"{tool} is not installed by .github/actions/setup/action.yml; without it the "
            f"build silently loses its cache instead of failing"
        )


def grouped_test_files() -> Set[str]:
    """Test files that pin themselves to one xdist worker with an ``xdist_group`` marker.

    Matched by an ESCAPED regex for the applied marker rather than by a plain substring, because
    the obvious spellings of this check are self-matching: any file searching for the marker's name
    contains that name, so this file reports ITSELF as grouped and the guard fails on its own text.
    The backslashes keep the literal out of this source while matching it everywhere else.
    """
    marker = re.compile(r"pytest\.mark\.xdist_group\s*\(")
    return {f"tests/{p.name}" for p in sorted((REPO / "tests").glob("test_*.py")) if marker.search(p.read_text())}


def pytest_invocations() -> List[str]:
    """Every ``python -m pytest`` command in the workflow, backslash continuations folded first
    so a command wrapped over three lines is matched as the one command it is."""
    joined = re.sub(r"\\\n\s*", " ", WORKFLOW.read_text())
    return [line.strip() for line in joined.splitlines() if "python -m pytest" in line]


def test_an_xdist_group_marker_is_never_a_no_op() -> None:
    """A file carrying ``xdist_group`` must be swept WITH ``--dist loadgroup``.

    The marker is inert under any other distribution -- pytest-xdist reads it only in loadgroup
    mode -- so the flag and the marker are one mechanism written in two files, and dropping either
    half silently restores the behaviour the marker was added to stop. Nothing fails; the suite
    just quietly costs more again, which is why this has to be asserted rather than noticed.

    What it costs when it lapses, measured on tests/test_generated_references.py: its module-scoped
    fixture is 726 emits that each spawn a ``numpyto_common.cli`` subprocess, and under the default
    per-test distribution every worker that draws one of its 8 tests rebuilds the whole thing. At
    -n16 a narrow selection scattered all 8 and paid 8 rebuilds -- 5808 spawns instead of 726, with
    eight copies resident at once -- while a full-sweep run happened to land them together and paid
    1. The lapse is therefore not reliably visible in a green sweep, which is the other half of why
    it is asserted here.
    """
    grouped = grouped_test_files()
    assert grouped, "no test file declares an xdist_group marker; this guard has lost its subject"
    claimed = dedicated_files()
    offenders = []
    for cmd in pytest_invocations():
        workers = re.search(r"-n\s+(\S+)", cmd)
        # -n0/-n1 is one process, where a module-scoped fixture is built once whatever the
        # distribution is, so there is nothing for the marker to do and nothing to assert.
        if workers is None or workers.group(1) in ("0", "1") or "--dist loadgroup" in cmd:
            continue
        # A command carries a grouped file either by naming it or by sweeping $files, which is
        # every test file no dedicated phase claims.
        carried = {f for f in grouped if f in cmd}
        if "$files" in cmd:
            carried |= grouped - claimed
        if carried:
            offenders.append(f"{sorted(carried)} run by: {cmd[:70]}...")
    assert not offenders, (
        "these xdist runs carry a file with an xdist_group marker but no --dist loadgroup, "
        "so the marker does nothing: " + "; ".join(offenders)
    )


TRANSLATOR_TESTS = REPO / "hpcagent_bench" / "numpy_translators" / "tests"


def translator_legs() -> List[dict]:
    """The ``integration_translators`` matrix, as YAML rather than as text."""
    import yaml

    return list(yaml.safe_load(WORKFLOW.read_text())["jobs"]["integration_translators"]["strategy"]["matrix"]["leg"])


def test_the_translator_integration_legs_partition_the_tree() -> None:
    """The tree's ``-m integration`` selection is split over containers by naming two files on
    their own legs and ``--ignore``-ing exactly those two on the leg that sweeps the directory.

    Both halves of that have to agree or a whole file stops running while every leg goes green: an
    --ignore whose path no longer resolves silently ignores nothing (the file runs twice), and a
    named file the sweeping leg forgot to ignore is a corpus-wide lowering pass paid twice.
    """
    named: Set[str] = set()
    ignored: Set[str] = set()
    roots: Set[str] = set()
    for leg in translator_legs():
        for token in str(leg["select"]).split():
            if token.startswith("--ignore="):
                ignored.add(token.split("=", 1)[1])
            elif token.endswith("/"):
                roots.add(token)
            else:
                named.add(token)
    assert roots == {"hpcagent_bench/numpy_translators/tests/"}, (
        f"the legs sweep {sorted(roots)}; exactly one of them has to name the whole tree, or the "
        "files no leg names are the ones nothing runs"
    )
    assert named == ignored, (
        f"legs name {sorted(named)} but the sweeping leg ignores {sorted(ignored)}. "
        "A file on both sides runs twice; a file on neither runs once per leg or not at all."
    )
    missing = [path for path in sorted(named | ignored) if not (REPO / path).is_file()]
    assert not missing, (
        f"the matrix names files that do not exist: {missing} -- an --ignore that misses ignores nothing"
    )


def test_a_sharded_leg_runs_every_slice_it_splits_into() -> None:
    """A leg that names ``0/2`` and no ``1/2`` runs half the registry and reports green.

    The shard is invisible in the leg's own result -- the sweep asserts its findings are EMPTY, and
    half a registry produces emptier findings than a whole one -- so the only place this can be
    caught is here, against the matrix.
    """
    slices: dict = {}
    for leg in translator_legs():
        index, _, count = str(leg["shard"]).partition("/")
        slices.setdefault((str(leg["select"]), int(count)), set()).add(int(index))
    for (select, count), indices in sorted(slices.items()):
        assert indices == set(range(count)), (
            f"leg {select.split()[0]} runs shards {sorted(indices)} of {count}; "
            f"the missing ones are registry nothing sweeps"
        )


def test_every_integration_marked_translator_file_reaches_a_leg() -> None:
    """The other direction: a NEW ``-m integration`` file under the tree must land on some leg.

    It does, by construction -- the sweeping leg names the directory -- so what this actually pins
    is that nobody 'fixes' a slow new file by adding a fourth ``--ignore`` and no leg to match.
    """
    ignored = {
        token.split("=", 1)[1]
        for leg in translator_legs()
        for token in str(leg["select"]).split()
        if token.startswith("--ignore=")
    }
    named = {token for leg in translator_legs() for token in str(leg["select"]).split() if token.endswith(".py")}
    marked = {
        f"hpcagent_bench/numpy_translators/tests/{p.name}"
        for p in sorted(TRANSLATOR_TESTS.glob("test_*.py"))
        if "pytest.mark.integration" in p.read_text()
    }
    orphaned = sorted((ignored & marked) - named)
    assert not orphaned, f"ignored by the sweeping leg and run by no other leg: {orphaned}"


#: The standing per-container budget, in minutes. Not a suggestion: a job over it becomes the run's
#: critical path, and the whole shape of tests.yml -- four matrix jobs over a slice knob, the trees
#: split apart, hf-export lifted off mpi -- exists to hold it. Raising this number is a decision
#: somebody makes here, once, instead of one job at a time in a comment nobody reads.
CONTAINER_BUDGET_MINUTES = 45


def workflow_jobs() -> dict:
    import yaml

    return dict(yaml.safe_load(WORKFLOW.read_text())["jobs"])


def test_no_job_budgets_itself_past_the_container_ceiling() -> None:
    """``timeout-minutes`` is where the budget is enforced, so it is also where it can be dodged.

    A job that quietly raises its own cap is the only way back to a 78-minute container, and it
    reads as a one-line diff. Disabled jobs (``if: false``) are exempt -- no container runs them --
    but they say so in the workflow rather than here.
    """
    over = {
        name: job["timeout-minutes"]
        for name, job in workflow_jobs().items()
        if job.get("if") is not False and int(job.get("timeout-minutes", 10**6)) > CONTAINER_BUDGET_MINUTES
    }
    assert not over, (
        f"these jobs budget past {CONTAINER_BUDGET_MINUTES} minutes: {over}. "
        "Split the work across containers -- never deselect it -- or move the ceiling here."
    )


def test_every_job_sets_a_timeout_at_all() -> None:
    """A job with no ``timeout-minutes`` inherits GitHub's 360, which is the budget not existing."""
    missing = sorted(name for name, job in workflow_jobs().items() if "timeout-minutes" not in job)
    assert not missing, f"no timeout-minutes on {missing}; the default is 6 hours"


def test_the_unit_sweep_matrix_runs_every_slice_it_deals_into() -> None:
    """The discovery sweep is dealt round-robin over the file list by ``awk 'NR % N == I'``, so the
    matrix has to list every I in [0, N). A missing index is test FILES nothing runs, and the
    remaining shards go green -- the same silent hole the discovery mechanism exists to prevent,
    reintroduced one level up.
    """
    job = workflow_jobs()["unit"]
    indices = {int(s) for s in job["strategy"]["matrix"]["shard"]}
    deals = set(re.findall(r"awk 'NR % (\d+) == \$\{\{ matrix\.shard \}\}'", WORKFLOW.read_text()))
    assert len(deals) == 1, f"the unit sweep deals {deals or 'nothing'}; it has to deal exactly one modulus"
    count = int(deals.pop())
    assert indices == set(range(count)), (
        f"unit runs shards {sorted(indices)} of {count}; the missing ones are test files nothing sweeps"
    )
