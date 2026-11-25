# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Which DaCe the benchmark accepts (dace-extended, or a developer's checkout alone) and which commit it reports."""

import json
import pathlib
import re
import tomllib
from importlib import metadata

import pytest

from hpcagent_bench import dace_guard
from hpcagent_bench.cpf import canonical
from hpcagent_bench.paths import ROOT

SHA = "0123456789abcdef0123456789abcdef01234567"
CHECKOUT = {"url": "file:///src/dace", "dir_info": {"editable": True}}
GIT = {"url": "https://github.com/spcl/dace.git", "vcs_info": {"vcs": "git", "commit_id": SHA}}


class FakeDist:
    """An installed distribution: its version, its ``direct_url.json`` (None: installed from an index), its files."""

    def __init__(self, version: str, origin: dict[str, object] | None, root: pathlib.Path | None = None) -> None:
        self.version = version
        self.origin = origin
        self.root = root

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        return None if self.origin is None else json.dumps(self.origin)

    def locate_file(self, path: str) -> pathlib.Path:
        return (self.root or pathlib.Path("/nonexistent")) / path


def installs(monkeypatch: pytest.MonkeyPatch, dists: dict[str, FakeDist]) -> None:
    """Make ``metadata.distribution`` see exactly ``dists``."""

    def distribution(name: str) -> FakeDist:
        if name not in dists:
            raise metadata.PackageNotFoundError(name)
        return dists[name]

    monkeypatch.setattr(metadata, "distribution", distribution)


@pytest.mark.parametrize(
    ("dists", "verdict"),
    [
        ({"dace-extended": FakeDist("1.0.0", None)}, ""),
        ({"dace": FakeDist("2.0.0b2", CHECKOUT)}, ""),
        ({"dace": FakeDist("2.0.0b2", GIT)}, ""),
        ({}, "not installed"),
        ({"dace": FakeDist("1.0.2", None)}, "upstream DaCe 1.0.2"),
        ({"dace": FakeDist("1.0.2", None), "dace-extended": FakeDist("1.0.0", None)}, "upstream DaCe 1.0.2"),
        ({"dace": FakeDist("2.0.0b2", CHECKOUT), "dace-extended": FakeDist("1.0.0", None)}, "both installed"),
    ],
)
def test_the_guard_accepts_dace_extended_or_a_checkout_alone(
    monkeypatch: pytest.MonkeyPatch, dists: dict[str, FakeDist], verdict: str
) -> None:
    """Upstream PyPI dace shares the dace/ folder with dace-extended, so it is refused beside it or in its place."""
    installs(monkeypatch, dists)
    problem = dace_guard.dace_problem()
    assert (verdict in problem) if verdict else problem == ""


def test_the_console_script_refuses_every_verb_while_upstream_dace_is_installed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`hpcagent-bench <verb>` runs nothing and names the fix while upstream dace replaces DaCe extended."""
    installs(monkeypatch, {"dace": FakeDist("1.0.2", None)})
    monkeypatch.setattr("sys.argv", ["hpcagent-bench", "--version"])
    assert dace_guard.console_main() == 2
    assert "pip install dace-extended" in capsys.readouterr().err


def test_a_release_reports_the_spcl_dace_commit_it_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A PyPI install has no direct_url.json: the commit is the one the release writes as dace.__commit__."""
    (tmp_path / "dace").mkdir()
    (tmp_path / "dace" / "version.py").write_text(f'__version__ = "1.0.0"\n__commit__ = "{SHA}"\n')
    installs(monkeypatch, {"dace-extended": FakeDist("1.0.0", None, tmp_path)})
    assert canonical.dace_commit() == SHA


def test_a_git_install_reports_its_commit_and_an_index_install_without_one_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a recorded commit the CPF cache has no key, so a DaCe naming none is refused, not keyed on ""."""
    installs(monkeypatch, {"dace": FakeDist("2.0.0b2", GIT)})
    assert canonical.dace_commit() == SHA
    installs(monkeypatch, {"dace-extended": FakeDist("1.0.0", None)})
    with pytest.raises(RuntimeError, match="no spcl/dace commit"):
        canonical.dace_commit()


def test_pyproject_depends_on_dace_extended_from_pypi_and_never_on_upstream_dace() -> None:
    """pip and uv install the same release: no git source for dace-extended, and upstream dace is overridden away."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert any(re.fullmatch(r"dace-extended>=\d+\.\d+\.\d+", dep) for dep in project["project"]["dependencies"])
    assert not any(re.match(r"dace\b(?!-)", dep) for dep in project["project"]["dependencies"])
    sources = project["tool"]["uv"]["sources"]
    assert "dace" not in sources and "dace-extended" not in sources and "dace-relayout" not in sources
    assert "dace ; sys_platform == 'never'" in project["tool"]["uv"]["override-dependencies"]
