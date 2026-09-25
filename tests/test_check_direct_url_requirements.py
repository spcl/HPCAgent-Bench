# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The release guard (scripts/do_release.sh) flags a published direct-URL requirement and nothing
else: a comment telling the reader to ``pip install "dace @ git+..."`` is not a requirement.
"""

import importlib.util
import pathlib
import sys

from hpcagent_bench import paths

SPEC = importlib.util.spec_from_file_location(
    "check_direct_url_requirements", paths.ROOT / "scripts" / "check_direct_url_requirements.py"
)
check_direct_url_requirements = importlib.util.module_from_spec(SPEC)
sys.modules["check_direct_url_requirements"] = check_direct_url_requirements
SPEC.loader.exec_module(check_direct_url_requirements)

DACE_URL = "dace @ git+https://github.com/spcl/dace.git@extended"


def write(tmp_path: pathlib.Path, body: str) -> pathlib.Path:
    path = tmp_path / "pyproject.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_a_comment_spelling_a_url_install_is_not_a_requirement(tmp_path: pathlib.Path) -> None:
    """The false positive that motivated the parse: only the TOML values are requirements."""
    pyproject = write(
        tmp_path,
        f'[project]\nname = "demo"\n# pip install "{DACE_URL}"\ndependencies = ["numpy>=2,<3"]\n'
        "[project.optional-dependencies]\ncpu = [\"scipy; python_version >= '3.12'\"]\n",
    )
    assert check_direct_url_requirements.direct_url_requirements(pyproject) == []
    assert check_direct_url_requirements.main([str(pyproject)]) == 0


def test_a_direct_url_in_an_extra_or_the_dependencies_is_flagged(tmp_path: pathlib.Path) -> None:
    """Both published requirement lists are checked; a dependency group never reaches the wheel."""
    pyproject = write(
        tmp_path,
        f'[project]\nname = "demo"\ndependencies = ["tool[x] @ file:///opt/tool"]\n'
        f'[project.optional-dependencies]\ndace = ["{DACE_URL}"]\n'
        f'[dependency-groups]\ntesting = ["{DACE_URL}"]\n',
    )
    assert check_direct_url_requirements.direct_url_requirements(pyproject) == [
        "tool[x] @ file:///opt/tool",
        DACE_URL,
    ]
    assert check_direct_url_requirements.main([str(pyproject)]) == 1


def test_the_repo_pyproject_publishes_no_direct_url() -> None:
    """The release gate holds on the committed metadata: dace stays a separate install, not an extra."""
    assert check_direct_url_requirements.direct_url_requirements(paths.ROOT / "pyproject.toml") == []
