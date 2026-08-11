# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""setup.py's ``install_requires`` must be a subset of every container requirements file.

The judge runs from a repo MOUNTED into the CE image, so nothing installs the package's own
declared deps at container-build time -- only ``pip install -r requirements/<hw>.txt`` does
(see containers/cluster/ce-images/amd/Dockerfile and .../nvidia/Dockerfile, each installing
exactly one requirements/<hw>.txt with no setup.py step). A name in install_requires that is
missing from a hardware requirements file is a ModuleNotFoundError waiting inside the judge
container. Stdlib only: setup.py is parsed with ast, not imported or run through setuptools.
"""
import ast
import pathlib
import re
from typing import FrozenSet, List

from hpcagent_bench import paths

REQUIREMENTS_DIR = paths.ROOT / "requirements"

#: setup.py names that are intentionally never pip-installed from a requirements file. dace is
#: git-cloned editable in the Dockerfile (see requirements/*.txt's own comment on this), so it
#: has no PyPI-style entry to match; exempted here, not silently dropped, so the exemption is
#: visible and setup.py adding a REAL new dep named "dace" would still need a second look.
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


def parse_install_requires(setup_py: pathlib.Path) -> List[str]:
    """The ``install_requires=[...]`` list literal out of setup.py, via ast (no import/exec).

    setup.py's list is plain string literals (no f-strings, no computed entries), so the
    keyword's value node is ast.literal_eval-able directly once found.
    """
    tree = ast.parse(setup_py.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        func_name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if func_name != "setup":
            continue
        for keyword in node.keywords:
            if keyword.arg == "install_requires":
                return list(ast.literal_eval(keyword.value))
    raise ValueError(f"no install_requires= found in {setup_py}")


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


SETUP_PY = paths.ROOT / "setup.py"
INSTALL_REQUIRES = parse_install_requires(SETUP_PY)


def test_install_requires_parses_the_known_deps():
    """Sanity check on the ast parse itself, independent of the subset assertion below --
    if this list came back empty or truncated, the coverage test would pass for the wrong
    reason (nothing left to check)."""
    normalized = {normalize(package_name(r)) for r in INSTALL_REQUIRES}
    assert {"numpy", "scipy", "jinja2", "sqlmodel", "cffi"} <= normalized
    assert len(INSTALL_REQUIRES) >= 8


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


def test_install_requires_is_a_subset_of_amd_and_nvidia_requirements():
    """The real gate: every runtime import setup.py declares must be preinstalled in the
    hardware requirements file the CE image actually installs (amd.txt / nvidia.txt), since
    the judge never runs ``pip install -e .`` inside the mounted-repo container."""
    setup_names = {normalize(package_name(r)) for r in INSTALL_REQUIRES} - DACE_EXEMPT
    for filename in ("amd.txt", "nvidia.txt"):
        req_names = parse_requirements(REQUIREMENTS_DIR / filename)
        missing = sorted(setup_names - req_names)
        assert not missing, (f"setup.py install_requires names missing from requirements/{filename}: {missing} "
                             f"-- the judge mounts the repo into the CE image and never runs `pip install -e .`, "
                             f"so a name absent here is a ModuleNotFoundError inside that container")
