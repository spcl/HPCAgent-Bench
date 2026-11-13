# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""helpers/scripts/host_python.sh accepts only an interpreter that imports the sourcing checkout's hpcagent_bench:
another checkout's would render that checkout's experiments.toml under this one's commit."""

import os
import pathlib
import shutil
import subprocess
import sys

from hpcagent_bench import paths

SCRIPT = pathlib.Path("helpers/scripts/host_python.sh")


def source(checkout: pathlib.Path) -> subprocess.CompletedProcess[str]:
    """Source ``checkout``'s host_python.sh with this test's interpreter, which imports the repository's package."""
    env = {**os.environ, "HPCAGENT_BENCH_HOST_PYTHON": sys.executable}
    env.pop("VIRTUAL_ENV", None)
    script = f'. "{checkout / SCRIPT}" && echo "$HPCAGENT_BENCH_HOST_PYTHON"'
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, check=False)


def test_the_checkouts_own_package_is_accepted() -> None:
    done = source(paths.ROOT)
    assert done.returncode == 0, done.stderr


def test_another_checkouts_package_is_refused_and_a_scripts_only_tree_is_not(tmp_path: pathlib.Path) -> None:
    """A checkout with its own package served by another checkout's interpreter is refused; a tree of the scripts
    alone (no package of its own, as the cluster tests build) is served by the installed one."""
    (tmp_path / SCRIPT).parent.mkdir(parents=True)
    shutil.copy2(paths.ROOT / SCRIPT, tmp_path / SCRIPT)
    assert source(tmp_path).returncode == 0
    (tmp_path / "hpcagent_bench").mkdir()
    (tmp_path / "hpcagent_bench" / "__init__.py").write_text("")
    done = source(tmp_path)
    assert done.returncode == 1
    assert f"imports hpcagent_bench from {paths.ROOT.resolve()}, not this checkout ({tmp_path})" in done.stderr


if __name__ == "__main__":
    import tempfile

    test_the_checkouts_own_package_is_accepted()
    test_another_checkouts_package_is_refused_and_a_scripts_only_tree_is_not(pathlib.Path(tempfile.mkdtemp()))
