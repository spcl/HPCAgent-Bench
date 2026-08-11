# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Direct tests for the pure / host-independent logic in ``hpcagent_bench.harness.discover_tools``.

No test file named this module directly. Most of it is a real host probe (``ldconfig``, package
config, ``/etc/os-release``) that this suite deliberately does not fake out -- those paths are a
thin wrapper over whatever toolchain happens to be installed, and pinning them would either mock
away the thing under test or be flaky across machines. What IS pure and load-bearing is covered
here instead: ``missing_for_target``'s per-target filter (drives the CLI's ``--require`` exit
code), ``_as_list``'s scalar/list normalization, and the "absolutely nothing found" degrade path
shared by every detector -- exercised with a name that cannot exist rather than mocked, so it stays
true to what the real function does.
"""
import types

import pytest

from hpcagent_bench.harness import discover_tools

_MISSING_NAME = "hpcagent_bench_test_definitely_absent_tool_9f3c1a"


# --- _as_list: scalar/list normalization, shared by every pkgconfig/soname/header spec field ------
def test_as_list_wraps_a_bare_scalar():
    assert discover_tools._as_list("libfoo.so") == ["libfoo.so"]


def test_as_list_passes_a_list_through_unchanged():
    assert discover_tools._as_list(["a", "b"]) == ["a", "b"]


def test_as_list_of_empty_list_stays_empty():
    assert discover_tools._as_list([]) == []


# --- _run_version: tries each version arg in order, stops at the first regex match ----------------
def test_run_version_returns_none_when_no_version_args_are_given():
    assert discover_tools._run_version("anything", None) is None
    assert discover_tools._run_version("anything", []) is None


def test_run_version_tries_args_in_order_and_stops_at_the_first_match(monkeypatch):
    calls = []

    def fake_run(cmd, capture_output, text, timeout):
        calls.append(cmd)
        stdout = "tool version 3.14.1\n" if cmd[-1] == "--version" else ""
        return types.SimpleNamespace(stdout=stdout, stderr="")

    # Rebind the NAME the module looks up, not subprocess.run itself -- subprocess is a shared,
    # process-wide module, and patching its attribute would leak into every other running test.
    monkeypatch.setattr(discover_tools, "subprocess", types.SimpleNamespace(run=fake_run))
    result = discover_tools._run_version("tool", ["-v", "--version", "-V"])
    assert result == "3.14.1"
    # "-V" is never tried -- the loop stopped at the first arg that yielded a version.
    assert calls == [["tool", "-v"], ["tool", "--version"]]


def test_run_version_returns_none_when_no_arg_yields_a_version(monkeypatch):

    def fake_run(cmd, capture_output, text, timeout):
        return types.SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(discover_tools, "subprocess", types.SimpleNamespace(run=fake_run))
    assert discover_tools._run_version("tool", ["--help"]) is None


# --- detect_binary: absent-name degrade path (no mocking -- the name genuinely cannot resolve) ----
def test_detect_binary_reports_not_found_for_a_name_that_cannot_exist():
    assert discover_tools.detect_binary({"names": [_MISSING_NAME]}) == {"found": False}


def test_detect_binary_found_path_picks_the_first_matching_name_and_lists_all_variants(monkeypatch):

    def fake_which(name):
        return {"gcc-13": "/usr/bin/gcc-13", "gcc": "/usr/bin/gcc"}.get(name)

    def fake_run_version(cmd, args):
        return "13.2.0"

    # Same rebind-the-name reasoning: shutil is shared process-wide, discover_tools's own
    # module-level names (shutil, _run_version) are not.
    monkeypatch.setattr(discover_tools, "shutil", types.SimpleNamespace(which=fake_which))
    monkeypatch.setattr(discover_tools, "_run_version", fake_run_version)
    result = discover_tools.detect_binary({"names": ["gcc-13", "gcc"], "version_arg": ["--version"]})
    assert result == {
        "found": True,
        "path": "/usr/bin/gcc-13",
        "version": "13.2.0",
        "variants": ["gcc-13", "gcc"],
    }


# --- detect_library / detect_header: nothing declared, or nothing resolvable, is "not found" ------
def test_detect_library_with_no_criteria_reports_not_found():
    assert discover_tools.detect_library({}) == {"found": False}


def test_detect_header_delegates_to_detect_library_on_the_header_field():
    result = discover_tools.detect_header({"header": [f"{_MISSING_NAME}.h"]})
    assert result == {"found": False}


# --- missing_for_target: the pure filter behind the CLI's --require exit code ----------------------
def _report(**tools):
    return {"categories": {"compilers": tools}}


def test_missing_for_target_lists_a_required_and_absent_tool():
    report = _report(gcc={"found": True, "required_on": ["cpu"]}, nvcc={"found": False, "required_on": ["nvidia"]})
    assert discover_tools.missing_for_target(report, "nvidia") == ["nvcc"]


def test_missing_for_target_excludes_a_found_tool_even_if_required():
    report = _report(gcc={"found": True, "required_on": ["cpu"]})
    assert discover_tools.missing_for_target(report, "cpu") == []


def test_missing_for_target_excludes_an_absent_but_optional_tool():
    report = _report(clang={"found": False, "required_on": []})
    assert discover_tools.missing_for_target(report, "cpu") == []


def test_missing_for_target_only_reports_tools_required_on_the_queried_target():
    # Required on nvidia only -- must not show up when checking cpu or amd.
    report = _report(nvcc={"found": False, "required_on": ["nvidia"]})
    assert discover_tools.missing_for_target(report, "cpu") == []
    assert discover_tools.missing_for_target(report, "amd") == []
    assert discover_tools.missing_for_target(report, "nvidia") == ["nvcc"]


def test_missing_for_target_a_tool_required_on_several_targets_counts_for_each():
    report = _report(cudnn={"found": False, "required_on": ["nvidia", "amd"]})
    assert discover_tools.missing_for_target(report, "nvidia") == ["cudnn"]
    assert discover_tools.missing_for_target(report, "amd") == ["cudnn"]


def test_missing_for_target_on_an_empty_report_is_empty():
    assert discover_tools.missing_for_target({"categories": {}}, "cpu") == []


@pytest.mark.parametrize("target", ["cpu", "nvidia", "amd"])
def test_missing_for_target_scans_every_category_not_just_the_first(target):
    report = {
        "categories": {
            "compilers": {
                "gcc": {
                    "found": True,
                    "required_on": ["cpu"]
                }
            },
            "numeric_libs": {
                "cudnn": {
                    "found": False,
                    "required_on": [target]
                }
            },
        },
    }
    assert discover_tools.missing_for_target(report, target) == ["cudnn"]
