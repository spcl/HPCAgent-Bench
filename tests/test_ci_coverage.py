# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every test file runs somewhere in CI, and .github/workflows/tests.yml keeps the properties its
jobs rely on. A new test file is inert by default and inertness is silent, so these are asserted."""

import ast
import pathlib
import re
import tomllib
from collections.abc import Hashable
from typing import Any

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "tests.yml"
DEDICATED = REPO / ".github" / "dedicated_tests.txt"
ACTIONS = sorted((REPO / ".github" / "actions").glob("*/action.yml"))
TRANSLATOR_TESTS = REPO / "tests" / "translators"

#: The per-container budget in minutes; a job over it becomes the run's critical path.
CONTAINER_BUDGET_MINUTES = 45


def dedicated_files() -> set[str]:
    """Paths the exclusion file claims, comments and blanks dropped -- the same parse tests.yml does."""
    return {line.strip() for line in DEDICATED.read_text().splitlines() if line.strip() and not line.startswith("#")}


def all_test_files() -> set[str]:
    return {f"tests/{p.name}" for p in sorted((REPO / "tests").glob("test_*.py"))}


def workflow_jobs() -> dict:
    return dict(yaml.safe_load(WORKFLOW.read_text())["jobs"])


def pyproject() -> dict:
    return tomllib.loads((REPO / "pyproject.toml").read_text())


def test_every_test_file_runs_somewhere() -> None:
    """A file is swept, or claimed by a dedicated phase. There is no third state."""
    claimed = dedicated_files()
    assert all_test_files() - claimed, "the unit sweep would select nothing"
    orphaned = claimed - all_test_files()
    assert not orphaned, f"dedicated_tests.txt names files that do not exist: {sorted(orphaned)}"


def test_a_dedicated_file_is_actually_run_by_some_phase() -> None:
    """Excluding a file from the sweep is only legitimate when another phase names it."""
    workflow = WORKFLOW.read_text()
    missing = [name for name in sorted(dedicated_files()) if name not in workflow]
    assert not missing, f"excluded from the sweep but named by no phase, so they run NOWHERE: {missing}"


def test_the_sweep_is_discovered_not_enumerated() -> None:
    workflow = WORKFLOW.read_text()
    assert "dedicated_tests.txt" in workflow, "the sweep no longer reads the exclusion file"
    assert "ls tests/test_*.py" in workflow, "the sweep no longer discovers files with ls"


def test_ci_never_asks_for_a_billed_runner() -> None:
    """Larger hosted runners bill per minute even on a public repo; self-hosted labels are ours."""
    standard = {"ubuntu-latest", "ubuntu-24.04", "ubuntu-22.04", "windows-latest", "macos-latest"}
    offenders = []
    for workflow in sorted((REPO / ".github" / "workflows").glob("*.y*ml")):
        for line in workflow.read_text().splitlines():
            match = re.match(r"\s*runs-on:\s*(.+?)\s*$", line)
            if not match:
                continue
            value = match.group(1)
            if (value.startswith("[") and "self-hosted" not in value) or (
                not value.startswith("[") and value not in standard
            ):
                offenders.append(f"{workflow.name}: {value}")
    assert not offenders, f"non-standard, billed-per-minute runners requested: {offenders}"


def test_no_workflow_declares_the_same_key_twice() -> None:
    """GitHub rejects a workflow with a duplicate mapping key, while PyYAML keeps the last value."""

    class NoDuplicates(yaml.SafeLoader):
        pass

    def strict_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Hashable, Any]:
        seen = set()
        for key_node, _ in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in seen:
                raise AssertionError(f"duplicate key {key!r} at line {key_node.start_mark.line + 1}")
            seen.add(key)
        return yaml.SafeLoader.construct_mapping(loader, node, deep)

    NoDuplicates.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, strict_mapping)
    for path in [*sorted((REPO / ".github" / "workflows").glob("*.yml")), *ACTIONS]:
        yaml.load(path.read_text(), Loader=NoDuplicates)  # raises AssertionError naming the key


def test_every_pytest_plugin_the_workflow_asks_for_is_installed() -> None:
    """A plugin flag in PYTEST_ADDOPTS the install lacks fails every pytest call at argument parsing."""
    installed = " ".join(pyproject()["project"]["optional-dependencies"]["dev"])
    text = WORKFLOW.read_text()
    plugins = {"--cov": "pytest-cov", "--timeout": "pytest-timeout", "-n ": "pytest-xdist", "--dist": "pytest-xdist"}
    asked = {dist for opt, dist in plugins.items() if re.search(rf"PYTEST_ADDOPTS:.*{re.escape(opt.strip())}", text)}
    missing = sorted(d for d in asked if d not in installed)
    assert not missing, f"PYTEST_ADDOPTS asks for {missing}, which pyproject.toml's dev extra does not install"


def test_asking_for_skip_reasons_does_not_hide_the_failures() -> None:
    """``-r`` replaces pytest's default ``fE`` report set: a bare ``-rs`` stops naming failed tests."""
    offenders = [
        i + 1
        for i, line in enumerate(WORKFLOW.read_text().splitlines())
        if "pytest" in line
        for token in re.findall(r"(?<![\w-])-r[a-zA-Z]+\b", line)
        if "s" in token and "f" not in token
    ]
    assert not offenders, f"tests.yml lines {offenders} ask for skip reasons without failures; use -rfEs"


def test_the_combined_total_is_built_from_every_job_not_one_of_them() -> None:
    """Every artifact holds a file named ``.coverage``: merge-multiple keeps one of them, and a
    partial combine prints a plausible percentage. Per-artifact directories, and a combine that
    accounts for every file (combined or skipped as a duplicate, in both of coverage.py's formats)."""
    text = WORKFLOW.read_text()
    assert "coverage combine" in text, "no `coverage combine` step"
    assert "merge-multiple: true" not in text
    assert "coverage-data/*/.coverage*" in text
    assert "combine accounted for" in text
    assert "Skipping duplicate data " in text and "Combined (\\d+) files?" in text


def test_the_corpus_reference_phase_is_not_instrumented() -> None:
    """Phase 2c runs only files coverage omits, and the tracer still pays per call (>180x slower)."""
    text = WORKFLOW.read_text()
    phase = text.index("Phase 2c -- benchmark reference validation")
    step = text[phase : text.index("- name: ", phase)]
    assert 'PYTEST_ADDOPTS: ""' in step, "Phase 2c must clear PYTEST_ADDOPTS"


def test_the_coverage_omit_list_and_the_uninstrumented_phase_agree() -> None:
    """Phase 2c is only safe uninstrumented while its tree is omitted from coverage."""
    omit = pyproject()["tool"]["coverage"]["run"]["omit"]
    assert any(pattern.startswith("hpcagent_bench/benchmarks") for pattern in omit)


def test_ci_installs_the_tools_that_fail_silently_when_absent() -> None:
    """Without ninja dace silently skips its compile-command cache; without ccache the build just
    slows down. Checked on the install lines, not the prose around them."""
    setup = (REPO / ".github" / "actions" / "setup" / "action.yml").read_text()
    joined = re.sub(r"\\\n\s*", " ", setup)
    installs = [line for line in joined.splitlines() if "apt-get install" in line]
    assert installs, "no apt-get install line in .github/actions/setup/action.yml"
    for tool in ("ninja-build", "ccache"):
        assert tool in " ".join(installs), f"{tool} is not installed by .github/actions/setup/action.yml"


def grouped_test_files() -> set[str]:
    """Test files carrying an ``xdist_group`` marker (escaped regex: this file must not match itself)."""
    marker = re.compile(r"pytest\.mark\.xdist_group\s*\(")
    return {f"tests/{p.name}" for p in sorted((REPO / "tests").glob("test_*.py")) if marker.search(p.read_text())}


def pytest_invocations() -> list[str]:
    """Every ``python -m pytest`` command, backslash continuations folded."""
    joined = re.sub(r"\\\n\s*", " ", WORKFLOW.read_text())
    return [line.strip() for line in joined.splitlines() if "python -m pytest" in line]


def test_an_xdist_group_marker_is_never_a_no_op() -> None:
    """``xdist_group`` only acts under ``--dist loadgroup``; without it every worker rebuilds the
    file's module fixture (8x the emits for test_generated_references.py at -n16)."""
    grouped = grouped_test_files()
    assert grouped, "no test file declares an xdist_group marker; this guard has lost its subject"
    claimed = dedicated_files()
    offenders = []
    for cmd in pytest_invocations():
        workers = re.search(r"-n\s+(\S+)", cmd)
        if workers is None or workers.group(1) in ("0", "1") or "--dist loadgroup" in cmd:
            continue
        carried = {f for f in grouped if f in cmd}
        if "$files" in cmd:
            carried |= grouped - claimed
        if carried:
            offenders.append(f"{sorted(carried)} run by: {cmd[:70]}...")
    assert not offenders, "xdist runs of xdist_group files without --dist loadgroup: " + "; ".join(offenders)


def translator_legs() -> list[dict]:
    """The ``translators`` job's leg matrix."""
    return list(workflow_jobs()["translators"]["strategy"]["matrix"]["leg"])


def test_the_translator_integration_legs_partition_the_tree() -> None:
    """One leg names the tree; any file another leg names, the sweeping leg ``--ignore``s."""
    named: set[str] = set()
    ignored: set[str] = set()
    roots: set[str] = set()
    for leg in translator_legs():
        for token in str(leg["select"]).split():
            if token.startswith("--ignore="):
                ignored.add(token.split("=", 1)[1])
            elif token.endswith("/"):
                roots.add(token)
            else:
                named.add(token)
    assert roots == {"tests/translators/"}, f"the legs sweep {sorted(roots)}"
    assert named == ignored, f"legs name {sorted(named)} but the sweeping leg ignores {sorted(ignored)}"
    missing = [path for path in sorted(named | ignored) if not (REPO / path).is_file()]
    assert not missing, f"the matrix names files that do not exist: {missing}"


def test_a_sharded_leg_runs_every_slice_it_splits_into() -> None:
    """A leg naming ``0/2`` with no ``1/2`` sweeps half the registry and still reports green."""
    slices: dict = {}
    for leg in translator_legs():
        index, _, count = str(leg["shard"]).partition("/")
        slices.setdefault((str(leg["select"]), int(count)), set()).add(int(index))
    for (select, count), indices in sorted(slices.items()):
        assert indices == set(range(count)), f"leg {select.split()[0]} runs shards {sorted(indices)} of {count}"


def test_every_integration_marked_translator_file_reaches_a_leg() -> None:
    ignored = {
        token.split("=", 1)[1]
        for leg in translator_legs()
        for token in str(leg["select"]).split()
        if token.startswith("--ignore=")
    }
    named = {token for leg in translator_legs() for token in str(leg["select"]).split() if token.endswith(".py")}
    marked = {
        f"tests/translators/{p.name}"
        for p in sorted(TRANSLATOR_TESTS.glob("test_*.py"))
        if "pytest.mark.integration" in p.read_text()
    }
    orphaned = sorted((ignored & marked) - named)
    assert not orphaned, f"ignored by the sweeping leg and run by no other leg: {orphaned}"


def test_no_job_budgets_itself_past_the_container_ceiling() -> None:
    over = {
        name: job["timeout-minutes"]
        for name, job in workflow_jobs().items()
        if job.get("if") is not False and int(job.get("timeout-minutes", 10**6)) > CONTAINER_BUDGET_MINUTES
    }
    assert not over, f"these jobs budget past {CONTAINER_BUDGET_MINUTES} minutes: {over}; split the work instead"


def test_every_job_sets_a_timeout_at_all() -> None:
    """A job with no ``timeout-minutes`` inherits GitHub's 360."""
    missing = sorted(name for name, job in workflow_jobs().items() if "timeout-minutes" not in job)
    assert not missing, f"no timeout-minutes on {missing}"


def test_the_unit_sweep_matrix_runs_every_slice_it_deals_into() -> None:
    """``awk 'NR % N == I'`` deals the file list; a missing I is test files nothing runs."""
    legs = workflow_jobs()["unit"]["strategy"]["matrix"]["include"]
    indices = {int(leg["shard"]) for leg in legs if str(leg["shard"]).isdigit()}
    deals = set(re.findall(r"awk 'NR % (\d+) == \$\{\{ matrix\.shard \}\}'", WORKFLOW.read_text()))
    assert len(deals) == 1, f"the unit sweep deals {deals or 'nothing'}; it has to deal exactly one modulus"
    count = int(deals.pop())
    assert indices == set(range(count)), f"unit runs shards {sorted(indices)} of {count}"


def test_the_unit_sweep_runs_the_python_floor_and_the_default() -> None:
    """The full sweep's shards run on the interpreter every other job uses; the ``floor`` leg runs on
    ``requires-python``'s floor (lint, import every module, collect every test)."""
    floor = re.search(r">=\s*(\d+\.\d+)", pyproject()["project"]["requires-python"])
    assert floor, "requires-python has no >= floor"
    default = str(yaml.safe_load(WORKFLOW.read_text())["env"]["PYTHON_VERSION"])
    legs = workflow_jobs()["unit"]["strategy"]["matrix"]["include"]
    sharded = {str(leg["python"]) for leg in legs if str(leg["shard"]).isdigit()}
    floors = {str(leg["python"]) for leg in legs if leg["shard"] == "floor"}
    assert sharded == {default}, f"the sharded sweep runs {sorted(sharded)}, not {default}"
    assert floors == {floor.group(1)}, f"the floor leg runs {sorted(floors)}, not {floor.group(1)}"


def test_ruff_and_pyright_target_the_python_floor() -> None:
    floor = re.search(r">=\s*(\d+)\.(\d+)", pyproject()["project"]["requires-python"])
    assert floor
    major, minor = floor.groups()
    assert pyproject()["tool"]["ruff"]["target-version"] == f"py{major}{minor}"
    assert pyproject()["tool"]["pyright"]["pythonVersion"] == f"{major}.{minor}"


#: Distributions in ``[project.optional-dependencies]`` whose module name differs from the PyPI name.
EXTRA_IMPORT_NAMES = {
    "apache-tvm": "tvm",
    "cupy-cuda13x": "cupy",
    "py-cpuinfo": "cpuinfo",
    "z3-solver": "z3",
}

#: The setup action installs dace from a checkout in every job, so importing it cannot abort collection.
ALWAYS_INSTALLED_EXTRAS = frozenset({"dace"})


def optional_extra_modules() -> set[str]:
    """Module names provided only by a hardware extra, i.e. not installed unless a job asks. The dev
    extra (pytest and the formatters) is part of every install."""
    out: set[str] = set()
    extras = pyproject()["project"]["optional-dependencies"]
    for requirements in (entries for name, entries in extras.items() if name != "dev"):
        for requirement in requirements:
            dist = re.split(r"[<>=!\[ ;@]", requirement.strip())[0]
            if not dist.startswith("hpcagent_bench"):
                out.add(EXTRA_IMPORT_NAMES.get(dist, dist.replace("-", "_")))
    return out - ALWAYS_INSTALLED_EXTRAS


def module_level_imports(path: pathlib.Path) -> list[tuple[int, str]]:
    """``(lineno, top-level package)`` for every import in the module body (not in functions)."""
    found: list[tuple[int, str]] = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name.split(".")[0]) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.lineno, node.module.split(".")[0]))
    return found


def test_no_test_module_imports_an_optional_extra_at_module_scope() -> None:
    """A module-level import of an extra some job lacks aborts that job's whole collection. Reach it
    through ``tests.optional_imports.import_or_skip`` inside the test instead."""
    extras = optional_extra_modules()
    offenders = [
        f"{path.relative_to(REPO)}:{lineno}: {name}"
        for path in sorted((REPO / "tests").rglob("*.py"))
        for lineno, name in module_level_imports(path)
        if name in extras
    ]
    assert not offenders, "module-level import of an optional extra under tests/:\n  " + "\n  ".join(offenders)
