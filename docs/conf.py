# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Sphinx configuration: the Markdown files in this directory, rendered with MyST and Furo."""

import re
from pathlib import Path

from sphinx.application import Sphinx

project = "HPCAgent-Bench"
copyright = "2026 ETH Zurich and the HPCAgent-Bench authors"  # noqa: A001
author = "SPCL @ ETH Zurich"

extensions = ["myst_parser"]
myst_heading_anchors = 4
exclude_patterns = ["_build"]
html_theme = "furo"

REPO_URL = "https://github.com/spcl/HPCAgent-Bench/blob/main"
LINK = re.compile(r"\]\((?!https?:|mailto:|#)([^)\s]+)\)")


def link_repo_files(app: Sphinx, docname: str, source: list[str]) -> None:
    """Point a relative link that leaves docs/ (a README, a source file) at the file on GitHub."""
    docs = Path(app.srcdir).resolve()
    here = (docs / docname).parent

    def rewrite(match: re.Match[str]) -> str:
        path, hash_, anchor = match.group(1).partition("#")
        target = (here / path).resolve()
        if target.is_relative_to(docs):
            return match.group(0)
        return f"]({REPO_URL}/{target.relative_to(docs.parent)}{hash_}{anchor})"

    source[0] = LINK.sub(rewrite, source[0])


def setup(app: Sphinx) -> None:
    app.connect("source-read", link_repo_files)
