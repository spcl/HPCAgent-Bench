# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every image recipe installs the TIP of spcl/dace@extended, and says which tip it got.

`git clone --branch extended` READS as "always the tip" and is not: a Docker layer caches on the
command string, which never changes, so the second build reuses the first clone and the image ages
into a pin that nothing records. That is not hypothetical -- the venv this repo tests with reached
10,748 commits behind extended and reported frontend refusals the branch had already fixed, which
cost a day of hunting a regression that did not exist.

So a Dockerfile takes the commit as a build-arg (the builder resolves the tip, and the sha is what
busts the layer), and every recipe writes the resolved sha to /opt/dace.commit so a run can say
which dace graded it. A .def file has no layer cache and may clone the branch, but it still has to
record what it got.
"""
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Recipes whose dace layer is CACHED, so the commit has to enter through a build-arg.
DOCKERFILES = [
    "containers/hpcagent_bench.Dockerfile",
    "containers/cluster/ce-images/amd/Dockerfile",
    "containers/cluster/ce-images/nvidia/Dockerfile",
]

#: Recipes with no layer cache: the branch is fine, the record is still required.
DEFINITIONS = ["containers/cpu.def"]

#: Wrappers that must resolve the tip themselves and hand it to the build.
BUILDERS = [
    "containers/cluster/ce-images/amd/build_sqsh.sh",
    "containers/cluster/ce-images/nvidia/build_sqsh.sh",
]


@pytest.mark.parametrize("recipe", DOCKERFILES)
def test_a_cached_recipe_takes_the_dace_commit_as_a_build_arg(recipe: str) -> None:
    text = (ROOT / recipe).read_text()
    assert "ARG DACE_COMMIT" in text, (f"{recipe} does not declare ARG DACE_COMMIT; without it the dace layer "
                                       "caches forever and the image pins itself to its first build")
    assert "--branch extended https://github.com/spcl/dace.git" not in text, (
        f"{recipe} clones the branch directly -- that layer never invalidates. Fetch $DACE_COMMIT instead.")


@pytest.mark.parametrize("recipe", DOCKERFILES + DEFINITIONS)
def test_every_recipe_records_the_dace_commit_it_installed(recipe: str) -> None:
    text = (ROOT / recipe).read_text()
    assert "/opt/dace.commit" in text, (f"{recipe} does not write /opt/dace.commit; an image that cannot say which "
                                        "dace it carries cannot be told from a stale one")


@pytest.mark.parametrize("builder", BUILDERS)
def test_the_builder_resolves_the_tip_and_passes_it_in(builder: str) -> None:
    """The resolve belongs OUTSIDE the build: inside, the layer cache would eat it."""
    text = (ROOT / builder).read_text()
    assert "git ls-remote https://github.com/spcl/dace.git refs/heads/extended" in text, (
        f"{builder} does not resolve extended's tip")
    assert "--build-arg \"DACE_COMMIT=${DACE_COMMIT}\"" in text, f"{builder} resolves the tip but never passes it"
