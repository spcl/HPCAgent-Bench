# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Toolchain markers select through ``-m``: deselected unless named, and a named-but-absent
toolchain is an error, never a skip. The logic is tests/toolchains.py; these drive it through an
inner pytest session, and hold the CI workflows to the markers each job provisions."""

import pathlib
import re
from collections.abc import Callable

import pytest
import yaml

from tests import toolchains

pytest_plugins = ("pytester",)

REPO = pathlib.Path(__file__).resolve().parents[1]
WORKFLOWS = REPO / ".github" / "workflows"

MARKED = """
import pytest


@pytest.mark.mpi
def test_needs_mpi():
    pass


def test_plain():
    pass
"""

SKIPPING = """
import pytest


def test_skipped():
    pytest.skip("this host lacks it")


@pytest.mark.xfail(strict=True, reason="a known gap")
def test_known_gap():
    assert False
"""


def answer(diagnosis: str) -> Callable[[], str]:
    return lambda: diagnosis


def fake_mpi(monkeypatch: pytest.MonkeyPatch, c: str, launcher: str) -> None:
    """Replace the mpi probes with fixed answers, so the outcome does not depend on this host."""
    requirements = {"c": answer(c), "mpi4py": answer(launcher)}
    monkeypatch.setitem(toolchains.TOOLCHAINS, "mpi", toolchains.Toolchain("fake. CI: mpi.", requirements))


def run(pytester: pytest.Pytester, *args: str) -> pytest.RunResult:
    return pytester.runpytest_inprocess("-p", "tests.toolchains", "-p", "no:cacheprovider", *args)


def test_a_run_that_does_not_name_the_marker_deselects_instead_of_skipping(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default run on a host without MPI must not report a green skip for every MPI test."""
    fake_mpi(monkeypatch, c="mpicc is not on PATH", launcher="")
    pytester.makepyfile(MARKED)
    run(pytester).assert_outcomes(passed=1, deselected=1)


def test_naming_the_marker_where_the_toolchain_is_absent_fails_with_its_diagnosis(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job that asked for MPI and does not have it is a broken job, and must say why."""
    fake_mpi(monkeypatch, c="mpicc is not on PATH", launcher="")
    pytester.makepyfile(MARKED)
    result = run(pytester, "-m", "mpi or not mpi")
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*-m selected mpi; its requirement 'c' is not met: mpicc is not on PATH*"])


def test_naming_the_marker_where_the_toolchain_is_present_runs_it(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_mpi(monkeypatch, c="", launcher="")
    pytester.makepyfile(MARKED)
    run(pytester, "-m", "mpi").assert_outcomes(passed=1, deselected=1)


def test_only_the_named_requirement_is_probed(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    """An mpi4py-only test must not fail on a host whose C wrapper is missing."""
    fake_mpi(monkeypatch, c="mpicc is not on PATH", launcher="")
    pytester.makepyfile('import pytest\n\n\n@pytest.mark.mpi("mpi4py")\ndef test_python_delivery():\n    pass\n')
    run(pytester, "-m", "mpi").assert_outcomes(passed=1)


def test_a_requirement_the_family_does_not_have_is_a_usage_error(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A misspelled requirement would otherwise probe nothing and pass."""
    fake_mpi(monkeypatch, c="", launcher="")
    pytester.makepyfile('import pytest\n\n\n@pytest.mark.mpi("mpich")\ndef test_typo():\n    pass\n')
    result = run(pytester, "-m", "mpi")
    assert result.ret == pytest.ExitCode.USAGE_ERROR, result.stdout.str()
    assert "has no requirement ['mpich']" in result.stdout.str() + result.stderr.str()


def test_a_marker_named_only_to_exclude_it_is_deselected_once(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``-m "not mpi"`` names mpi, so pytest's own filter drops the test and ours must not count it again."""
    fake_mpi(monkeypatch, c="", launcher="")
    pytester.makepyfile(MARKED)
    run(pytester, "-m", "not mpi").assert_outcomes(passed=1, deselected=1)


@pytest.mark.parametrize(
    "expression,named",
    [
        ("", ()),
        ("not integration", ("integration",)),
        ("mpi or not mpi", ("mpi",)),
        ("mpich", ("mpich",)),
        ("integration and (oneapi or distro or not (oneapi or distro))", ("integration", "oneapi", "distro")),
    ],
)
def test_the_expression_names_exactly_its_marker_words(expression: str, named: tuple[str, ...]) -> None:
    assert toolchains.named_markers(expression) == frozenset(named)


def test_under_no_skip_a_runtime_skip_fails_the_session_and_names_the_test(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(toolchains.NO_SKIP_ENV, "1")
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    pytester.makepyfile(SKIPPING)
    result = run(pytester)
    result.assert_outcomes(skipped=1, xfailed=1)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(["*HPCAGENT_BENCH_NO_SKIP=1: 1 skipped test(s) fail the session*", "*::test_skipped"])
    assert "::test_known_gap" not in result.stdout.str(), "a strict xfail is a stated gap, not a skip"


def test_without_no_skip_a_skip_leaves_the_session_green(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(toolchains.NO_SKIP_ENV, raising=False)
    pytester.makepyfile(SKIPPING)
    result = run(pytester)
    result.assert_outcomes(skipped=1, xfailed=1)
    assert result.ret == pytest.ExitCode.OK


def workflow_jobs() -> dict[str, dict[str, object]]:
    jobs: dict[str, dict[str, object]] = {}
    for path in sorted(WORKFLOWS.glob("*.yml")):
        jobs.update(yaml.safe_load(path.read_text())["jobs"])
    return jobs


def selections(job: dict[str, object]) -> list[str]:
    """Every ``-m "..."`` expression, and every ``marks="..."`` a step assembles one from."""
    steps = job.get("steps", [])
    assert isinstance(steps, list)
    scripts = "\n".join(str(step.get("run", "")) for step in steps)
    return [re.sub(r"\$\w+", " ", expression) for expression in re.findall(r'(?:-m |marks=)"([^"]+)"', scripts)]


@pytest.mark.parametrize("name", sorted(toolchains.TOOLCHAINS))
def test_the_jobs_a_toolchain_names_are_the_jobs_that_select_it(name: str) -> None:
    """A job that installs a toolchain and does not name its marker deselects the very tests it
    installed it for, and a job naming a marker it never provisions fails every one of them."""
    match = re.search(r"CI: ([^.]+)\.$", toolchains.TOOLCHAINS[name].description)
    assert match, f"{name}: the description must end with 'CI: <job>, <job>.' or 'CI: none.'"
    jobs = workflow_jobs()
    claimed = set() if match.group(1) == "none" else {job.strip() for job in match.group(1).split(",")}
    assert claimed <= set(jobs), f"{name}: no workflow job named {sorted(claimed - set(jobs))}"
    selecting = {
        job for job, spec in jobs.items() if any(name in toolchains.named_markers(e) for e in selections(spec))
    }
    assert selecting == claimed, (
        f"{name}: the description claims {sorted(claimed)}, the workflows select it in {sorted(selecting)}"
    )


def test_every_word_a_workflow_selects_by_is_a_registered_marker() -> None:
    """A misspelled marker in ``-m`` matches nothing, which deselects instead of failing."""
    registered = set(toolchains.TOOLCHAINS)
    for conftest in (
        REPO / "tests" / "conftest.py",
        REPO / "hpcagent_bench" / "numpy_translators" / "tests" / "conftest.py",
    ):
        registered |= set(re.findall(r'"markers",\s*"(\w+)[:(]', conftest.read_text()))
    used = {word for spec in workflow_jobs().values() for e in selections(spec) for word in toolchains.named_markers(e)}
    assert used <= registered, f"selected but never registered: {sorted(used - registered)}"
