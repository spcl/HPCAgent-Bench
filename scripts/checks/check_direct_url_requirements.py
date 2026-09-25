#!/usr/bin/env python
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Release guard: list every direct-URL requirement (``name @ <url>``, PEP 508) the package would publish.

PyPI rejects an upload whose Requires-Dist names a URL. The published requirements are
``[project] dependencies`` plus every ``[project.optional-dependencies]`` extra; dependency groups
never reach the wheel's metadata and are not checked. The TOML is parsed, never grepped, so a
comment that spells an install command (``pip install "dace @ git+..."``) is not a requirement.

    python scripts/checks/check_direct_url_requirements.py [pyproject.toml]

Exit status: 0 when none is found, 1 when one or more are (each is printed).
"""

import pathlib
import sys
import tomllib


def direct_url_requirements(pyproject: pathlib.Path) -> list[str]:
    """The published requirement strings in ``pyproject`` that carry a direct reference. A version
    specifier and an extras list never contain ``@``, so its presence before the marker is one."""
    with pyproject.open("rb") as handle:
        project = tomllib.load(handle).get("project", {})
    published = list(project.get("dependencies", []))
    for requirements in project.get("optional-dependencies", {}).values():
        published.extend(requirements)
    return [req for req in published if "@" in req.split(";", 1)[0]]


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    pyproject = pathlib.Path(args[0] if args else "pyproject.toml")
    found = direct_url_requirements(pyproject)
    for req in found:
        print(f"{pyproject}: direct-URL requirement PyPI rejects: {req}", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
