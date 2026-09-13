# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The CPU / amd / nvidia test groups: unmarked tests run everywhere, CI included; a hardware test
runs only when ``-m`` names its group, and then fails on a host that lacks the hardware."""

import pathlib
import types

import pytest

from tests import conftest


class FakeItem:
    """A collected test carrying exactly ``markers``: what the hooks read from a real item."""

    def __init__(self, name: str, *markers: str) -> None:
        self.name = name
        self.markers = frozenset(markers)

    def get_closest_marker(self, name: str) -> object | None:
        return object() if name in self.markers else None


def fake_config(markexpr: str, deselected: list[list[FakeItem]]) -> types.SimpleNamespace:
    """A config whose ``-m`` is ``markexpr`` and whose deselect hook records what it is handed."""
    return types.SimpleNamespace(
        getoption=lambda name: markexpr,
        hook=types.SimpleNamespace(pytest_deselected=lambda items: deselected.append(list(items))),
    )


@pytest.mark.parametrize(
    "markexpr, named",
    [
        ("", frozenset()),
        ("amd", frozenset({"amd"})),
        ("nvidia and not slow", frozenset({"nvidia"})),
        ("amd or nvidia", frozenset({"amd", "nvidia"})),
        ("amdx", frozenset()),
        ("my_amd_helper", frozenset()),
    ],
)
def test_a_group_is_named_only_by_its_whole_word(markexpr: str, named: frozenset[str]) -> None:
    """A marker that merely contains 'amd' must not pull real-GPU tests into a CPU run."""
    assert conftest.named_groups(markexpr) == named


@pytest.mark.parametrize(
    "markexpr, kept",
    [
        ("", ["cpu"]),
        ("amd", ["cpu", "amd"]),
        ("nvidia", ["cpu", "nvidia"]),
        ("amd or nvidia", ["cpu", "amd", "nvidia"]),
    ],
)
def test_a_hardware_test_is_deselected_unless_its_group_is_named(markexpr: str, kept: list[str]) -> None:
    """A CI run with no -m keeps every CPU test and reports each hardware test as deselected."""
    items = [FakeItem("cpu"), FakeItem("amd", "amd"), FakeItem("nvidia", "nvidia")]
    deselected: list[list[FakeItem]] = []
    conftest.pytest_collection_modifyitems(fake_config(markexpr, deselected), items)
    assert [item.name for item in items] == kept
    reported = [item.name for batch in deselected for item in batch]
    assert sorted(reported) == sorted({"amd", "nvidia"} - set(kept)), reported


def test_nothing_is_reported_deselected_when_no_hardware_test_was_collected() -> None:
    items = [FakeItem("cpu-a"), FakeItem("cpu-b")]
    deselected: list[list[FakeItem]] = []
    conftest.pytest_collection_modifyitems(fake_config("", deselected), items)
    assert [item.name for item in items] == ["cpu-a", "cpu-b"]
    assert deselected == []


def test_a_host_with_the_device_and_every_tool_lacks_nothing(tmp_path: pathlib.Path) -> None:
    device = tmp_path / "kfd"
    device.write_text("")
    groups = {"amd": (device, ("rocminfo", "rocprofv3"))}
    assert conftest.hardware_missing("amd", groups, which=lambda tool: f"/opt/rocm/bin/{tool}") == ""


def test_a_host_without_the_device_names_the_device_and_each_missing_tool(tmp_path: pathlib.Path) -> None:
    device = tmp_path / "nvidiactl"
    groups = {"nvidia": (device, ("nsys", "ncu"))}
    missing = conftest.hardware_missing(
        "nvidia", groups, which=lambda tool: "/usr/bin/nsys" if tool == "nsys" else None
    )
    assert missing == f"{device}, ncu"


def test_a_selected_hardware_test_fails_at_setup_on_a_host_without_the_hardware(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skipping here would report a green hardware run that measured nothing."""
    absent = tmp_path / "kfd"
    monkeypatch.setattr(conftest, "HARDWARE_GROUPS", {"amd": (absent, ())})
    with pytest.raises(pytest.fail.Exception, match=f"amd group, but this host lacks: {absent}"):
        conftest.pytest_runtest_setup(FakeItem("amd", "amd"))


def test_a_cpu_test_is_never_failed_by_the_hardware_check(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(conftest, "HARDWARE_GROUPS", {"amd": (tmp_path / "kfd", ("rocprofv3",))})
    conftest.pytest_runtest_setup(FakeItem("cpu"))


@pytest.mark.parametrize("group", sorted(conftest.HARDWARE_GROUPS))
def test_every_hardware_group_is_a_registered_marker(pytestconfig: pytest.Config, group: str) -> None:
    """An unregistered marker is a warning, and warnings are errors in this suite."""
    registered = {line.split(":", 1)[0] for line in pytestconfig.getini("markers")}
    assert group in registered, sorted(registered)
