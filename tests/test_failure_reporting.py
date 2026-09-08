# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A failure must name its reason before the session ends, because the session often does not end.

pytest writes every traceback in the FAILURES section, which ``pytest_terminal_summary`` emits
after the run is over. A job cap is a SIGKILL and an xdist INTERNALERROR aborts outright, so
neither reaches that section -- and this suite hits both. ``tests/conftest.py`` prints the reason
from ``pytest_runtest_logreport`` instead; these pin that it lands EARLY rather than merely that it
lands, which is the whole property.
"""

import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

MARKER = "the reason this test failed"

FAILING_TEST = f'''
def test_that_fails():
    assert 1 == 2, "{MARKER}"
'''


def run_probe(tmp_path, *extra):
    """Run one deliberately failing test with the repo's conftest loaded, and return its output."""
    probe = tmp_path / "test_probe.py"
    probe.write_text(FAILING_TEST)
    finished = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "tests.conftest", "-rfEs", str(probe), *extra],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return finished.stdout + finished.stderr


@pytest.mark.parametrize("extra", [(), ("-n", "2", "--dist", "loadgroup")], ids=["serial", "xdist"])
def test_a_failure_names_its_reason_before_the_summary_section(tmp_path, extra):
    """Cut the log where a SIGKILL or an INTERNALERROR would cut it; the reason must already be there.

    Asserting only that the reason appears somewhere is the assertion that passed all along --
    pytest's own FAILURES section satisfies it, and that section is exactly what a killed session
    never prints. The `xdist` case is the one that matters most: both reds in run 34221523664 were
    reported by a worker to a controller that then died.
    """
    output = run_probe(tmp_path, *extra)
    assert MARKER in output, f"the probe did not fail as intended:\n{output}"

    head, marker, _ = output.partition("=== FAILURES ===")
    assert marker, f"pytest printed no FAILURES section, so this test is measuring nothing:\n{output}"
    assert MARKER in head, (
        "the failure reason appears only in the end-of-run summary, so a killed session reports a "
        f"bare F and names nothing:\n{output}"
    )


def test_the_early_report_carries_the_failing_test_id(tmp_path):
    """A reason with no test id attached cannot be acted on when several tests are in flight."""
    output = run_probe(tmp_path)
    head, _, _ = output.partition("=== FAILURES ===")
    assert "test_probe.py::test_that_fails" in head, (
        f"the early failure report does not name which test it belongs to:\n{head}"
    )
