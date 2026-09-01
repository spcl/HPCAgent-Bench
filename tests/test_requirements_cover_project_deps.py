# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""pyproject's ``[project] dependencies`` must be a subset of every container requirements file.

The judge runs from a repo MOUNTED into the CE image, so nothing installs the package's own
declared deps at container-build time -- only the ``-r requirements/<hw>.txt`` install does
(see containers/cluster/ce-images/amd/Dockerfile and .../nvidia/Dockerfile, each installing
exactly one requirements/<hw>.txt and never the project itself). A declared dependency that is
missing from a hardware requirements file is a ModuleNotFoundError waiting inside the judge
container. Stdlib only: pyproject is read with tomllib, never through setuptools.
"""
import pathlib
import re
import tomllib
from typing import FrozenSet, List

from hpcagent_bench import paths

REQUIREMENTS_DIR = paths.ROOT / "requirements"

#: Declared names that are intentionally never installed from a requirements file. dace is
#: git-cloned editable in the Dockerfile (see requirements/*.txt's own comment on this), so it
#: has no PyPI-style entry to match; exempted here, not silently dropped, so the exemption is
#: visible and a REAL new dependency named "dace" would still need a second look.
DACE_EXEMPT = frozenset({"dace"})


def normalize(name: str) -> str:
    """PEP 503 normalization: lowercase, runs of ``-_.`` collapse to one ``-``."""
    return re.sub(r"[-_.]+", "-", name).lower()


def package_name(requirement: str) -> str:
    """The bare distribution name out of a requirement spec, dropping extras/version/markers.

    ``jax[cuda12]``, ``ordered-set >= 4.0.0`` and ``numpy>=2,<3`` all yield just the leading
    name token; a line with no name (blank after stripping) yields "".
    """
    requirement = requirement.split(";", 1)[0]  # drop the environment marker, if any
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement.strip())
    return match.group(0) if match else ""


def project_dependencies(pyproject: pathlib.Path) -> List[str]:
    """``[project] dependencies`` out of pyproject.toml.

    tomllib, not setuptools: this test asserts what the file DECLARES, and building the metadata
    would let a backend default or a plugin supply a name the file itself does not.
    """
    return list(tomllib.loads(pyproject.read_text())["project"]["dependencies"])


def parse_requirements(path: pathlib.Path) -> FrozenSet[str]:
    """Normalized distribution names declared in a requirements.txt.

    Skips blank lines, ``#`` comments, and pip options (``--pre``, ``--no-binary=...``) -- none
    of those are a distribution name.
    """
    names = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        name = package_name(line)
        if name:
            names.add(normalize(name))
    return frozenset(names)


PYPROJECT = paths.ROOT / "pyproject.toml"
DEPENDENCIES = project_dependencies(PYPROJECT)


def test_dependencies_parse_to_the_known_deps():
    """Sanity check on the ast parse itself, independent of the subset assertion below --
    if this list came back empty or truncated, the coverage test would pass for the wrong
    reason (nothing left to check)."""
    normalized = {normalize(package_name(r)) for r in DEPENDENCIES}
    assert {"numpy", "scipy", "jinja2", "sqlmodel", "cffi"} <= normalized
    assert len(DEPENDENCIES) >= 8


def test_package_name_strips_specifiers_extras_and_markers():
    cases = {
        "numpy>=2,<3": "numpy",
        "jax[cuda12]": "jax",
        "ordered-set >= 4.0.0": "ordered-set",
        "blake3": "blake3",
        'foo>=1; python_version<"3.11"': "foo",
    }
    for spec, expected in cases.items():
        assert package_name(spec) == expected


def test_normalize_follows_pep_503():
    assert normalize("ml_dtypes") == normalize("ml-dtypes") == "ml-dtypes"
    assert normalize("PyYAML") == "pyyaml"
    assert normalize("tree_sitter.language--pack") == "tree-sitter-language-pack"


def test_parse_requirements_skips_comments_and_pip_options(tmp_path):
    path = tmp_path / "req.txt"
    path.write_text("# a comment\n--pre\nnumpy>=2,<3\n\n--no-binary=mpi4py\nmpi4py\n")
    assert parse_requirements(path) == frozenset({"numpy", "mpi4py"})


def test_dependencies_are_a_subset_of_amd_and_nvidia_requirements():
    """The real gate: every runtime import pyproject declares must be preinstalled in the
    hardware requirements file the CE image actually installs (amd.txt / nvidia.txt), since
    the judge never installs the project inside the mounted-repo container."""
    declared = {normalize(package_name(r)) for r in DEPENDENCIES} - DACE_EXEMPT
    for filename in ("amd.txt", "nvidia.txt"):
        req_names = parse_requirements(REQUIREMENTS_DIR / filename)
        missing = sorted(declared - req_names)
        assert not missing, (f"[project] dependencies missing from requirements/{filename}: {missing} "
                             f"-- the judge mounts the repo into the CE image and never installs the project, "
                             f"so a name absent here is a ModuleNotFoundError inside that container")
