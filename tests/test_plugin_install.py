# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A plugin's ``[install]`` command installs its package into the interpreter the judge runs in: ``{python}`` and
``{root}`` are filled in, it runs in ``root``, a plugin without one is left alone, and a failure names the plugin."""

import functools
import pathlib
import sys

import pytest
from hpcagent_agent import plugins as agent_plugins

from hpcagent_bench import plugins


def plugin_folder(repo: pathlib.Path, name: str, install: str) -> pathlib.Path:
    """``repo/plugin/<name>``, a plugin whose manifest carries ``install`` (TOML, empty for none)."""
    folder = repo / "plugin" / name
    folder.mkdir(parents=True)
    (folder / "plugin.toml").write_text(f'name = "{name}"\n{install}', encoding="utf-8")
    return folder


def test_the_install_command_runs_in_its_root_with_the_interpreter_and_root_filled_in(tmp_path: pathlib.Path) -> None:
    folder = plugin_folder(tmp_path, "probe", "")

    record = "import os, sys, pathlib; pathlib.Path(sys.argv[2]).write_text(sys.executable + '|' + sys.argv[1] + '|' + os.getcwd())"
    out = tmp_path / "installed.txt"
    folder.joinpath("plugin.toml").write_text(
        f'name = "probe"\n[install]\nroot = "../.."\ncommand = ["{{python}}", "-c", "{record}", "{{root}}", "{out}"]\n'
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'probe'\n")

    sut = plugins.install(agent_plugins.find([folder]), sys.executable)

    assert sut == ["probe"]
    python, root, cwd = out.read_text().split("|")
    assert python == sys.executable and root == cwd, "the command runs in the root it is given"
    assert pathlib.Path(root).name == tmp_path.name and pathlib.Path(root) != tmp_path, "a writable copy of the root"


def test_the_fingerprint_follows_the_plugin_sources(tmp_path: pathlib.Path) -> None:
    folder = plugin_folder(tmp_path, "probe", '\n[install]\nroot = "../.."\ncommand = ["{python}", "-V"]\n')
    found = agent_plugins.find([folder])
    before = plugins.fingerprint(found)

    (tmp_path / "module.py").write_text("changed = True\n")

    assert plugins.fingerprint(found) != before
    assert plugins.fingerprint(
        agent_plugins.find([plugin_folder(tmp_path / "other", "plain", "")])
    ) == plugins.fingerprint(())


def test_a_plugin_without_an_install_command_is_left_alone(tmp_path: pathlib.Path) -> None:
    folder = plugin_folder(tmp_path, "plain", "")

    assert plugins.install(agent_plugins.find([folder])) == []


def test_a_failing_install_command_names_its_plugin(tmp_path: pathlib.Path) -> None:
    folder = plugin_folder(tmp_path, "broken", '\n[install]\ncommand = ["{python}", "-c", "raise SystemExit(3)"]\n')

    with pytest.raises(RuntimeError, match="plugin 'broken'"):
        plugins.install(agent_plugins.find([folder]))


def test_an_install_block_without_a_command_is_refused_by_its_file(tmp_path: pathlib.Path) -> None:
    folder = plugin_folder(tmp_path, "empty", "\n[install]\ncommand = []\n")

    with pytest.raises(ValueError, match="plugin.toml"):
        agent_plugins.find([folder])


def test_a_subpackage_named_build_is_installed_but_a_top_level_build_folder_is_not(tmp_path: pathlib.Path) -> None:
    folder = plugin_folder(tmp_path, "probe", "")
    (tmp_path / "pkg" / "build").mkdir(parents=True)
    (tmp_path / "pkg" / "build" / "arena.py").write_text("")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "stale.o").write_text("")
    listing = tmp_path / "listing.txt"
    record = "import pathlib, sys; pathlib.Path(sys.argv[2]).write_text(' '.join(sorted(str(p.relative_to(sys.argv[1])) for p in pathlib.Path(sys.argv[1]).rglob('*'))))"
    folder.joinpath("plugin.toml").write_text(
        f'name = "probe"\n[install]\nroot = "../.."\ncommand = ["{{python}}", "-c", "{record}", "{{root}}", "{listing}"]\n'
    )

    plugins.install(agent_plugins.find([folder]))

    copied = listing.read_text().split()
    assert "pkg/build/arena.py" in copied and "build/stale.o" not in copied


def visible_as(monkeypatch: pytest.MonkeyPatch, folder: pathlib.Path, installed: pathlib.Path) -> None:
    """``folder`` through the user plugin variable, and ``installed`` as an installed package's entry point."""
    monkeypatch.delenv(agent_plugins.PLUGIN_DIRS_ENV, raising=False)
    monkeypatch.setenv(agent_plugins.USER_PLUGIN_DIRS_ENV, str(folder))
    monkeypatch.setattr(agent_plugins, "entry_point_dirs", lambda: (installed,))
    # A fresh cache for this test only: the module's own stays as other tests left it.
    monkeypatch.setattr(agent_plugins, "roots_of", functools.lru_cache(maxsize=8)(agent_plugins.roots_of.__wrapped__))


def test_a_plugin_that_installs_itself_is_found_once(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install = '\n[install]\nroot = "../.."\ncommand = ["{python}", "-V"]\n'
    folder = plugin_folder(tmp_path / "checkout", "selfie", install)
    visible_as(monkeypatch, folder, plugin_folder(tmp_path / "site-packages", "selfie", install))

    sut = agent_plugins.find(plugins.roots())

    assert [(plugin.name, plugin.root) for plugin in sut if plugin.name == "selfie"] == [("selfie", folder)]


def test_an_installed_plugin_of_a_folders_name_is_refused_when_the_folder_installs_nothing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = plugin_folder(tmp_path / "checkout", "twin", "")
    visible_as(monkeypatch, folder, plugin_folder(tmp_path / "site-packages", "twin", ""))

    with pytest.raises(ValueError, match="plugin 'twin' is both"):
        agent_plugins.find(plugins.roots())
