# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""CE_IMAGE_FLAVOR=native moves an experiment's agent and judge EDFs to the -native ones a native build
renders; the serving EDF stays -latest, and the default flavor changes nothing. CE_IMAGE_RELEASE then names the frozen
release of every role's EDF."""

import pathlib
import subprocess
from collections.abc import Callable

import pytest

from hpcagent_bench.paths import ROOT

SUBMIT_COMMON = ROOT / "hpcagent_bench" / "cluster" / "submit_common.sh"
SETUP_ENV = (
    "AMD_CE_ENV=hpcagent-bench-agent-mi300-latest\n"
    "JUDGE_CE_ENV=hpcagent-bench-judge-mi300-mlscale-latest\n"
    "INFERENCE_CE_ENV=hpcagent-bench-vllm-mi300-latest\n"
)


def apply_flavor(tmp_path: pathlib.Path, flavor: str) -> tuple[subprocess.CompletedProcess[str], str]:
    env_file = tmp_path / "setup.env"
    env_file.write_text(SETUP_ENV, encoding="utf-8")
    done = subprocess.run(
        ["bash", "-c", 'source "$1" && apply_flavor "$2"', "bash", str(SUBMIT_COMMON), str(env_file)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "CE_IMAGE_FLAVOR": flavor},
    )
    return done, env_file.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("flavor", "expected"),
    [
        pytest.param(
            "native",
            "AMD_CE_ENV=hpcagent-bench-agent-mi300-native\n"
            "JUDGE_CE_ENV=hpcagent-bench-judge-mi300-mlscale-native\n"
            "INFERENCE_CE_ENV=hpcagent-bench-vllm-mi300-latest\n",
            id="native_moves_agent_and_judge_but_not_serving",
        ),
        pytest.param("latest", SETUP_ENV, id="latest_changes_nothing"),
    ],
)
def test_a_flavor_rewrites_the_agent_and_judge_edfs_and_leaves_serving_alone(
    tmp_path: pathlib.Path, flavor: str, expected: str
) -> None:
    done, env = apply_flavor(tmp_path, flavor)
    assert done.returncode == 0, done.stderr
    assert env == expected


def apply_release(tmp_path: pathlib.Path, release: str) -> tuple[subprocess.CompletedProcess[str], str]:
    env_file = tmp_path / "setup.env"
    env_file.write_text(SETUP_ENV, encoding="utf-8")
    script = 'source "$1" && apply_flavor "$2" && apply_release "$2"'
    done = subprocess.run(
        ["bash", "-c", script, "bash", str(SUBMIT_COMMON), str(env_file)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "CE_IMAGE_FLAVOR": "native", "CE_IMAGE_RELEASE": release},
    )
    return done, env_file.read_text(encoding="utf-8")


def test_a_release_names_the_frozen_edf_of_every_role_serving_included(tmp_path: pathlib.Path) -> None:
    """CE_IMAGE_RELEASE=v10: the EDFs `CE_IMAGE_RELEASE=v10 install_edfs.sh` renders, after the flavor."""
    done, env = apply_release(tmp_path, "v10")
    assert done.returncode == 0, done.stderr
    assert env == (
        "AMD_CE_ENV=hpcagent-bench-agent-mi300-native-v10\n"
        "JUDGE_CE_ENV=hpcagent-bench-judge-mi300-mlscale-native-v10\n"
        "INFERENCE_CE_ENV=hpcagent-bench-vllm-mi300-latest-v10\n"
    )


@pytest.mark.parametrize(
    ("apply", "value", "message"),
    [
        (apply_flavor, "zen4", "latest or native"),
        (apply_flavor, "Native", "latest or native"),
        (apply_release, "latest", "CE_IMAGE_RELEASE is v<N>"),
    ],
    ids=["unknown-flavor", "flavor-spelling", "release-that-is-no-version"],
)
def test_an_unknown_flavor_is_refused(
    tmp_path: pathlib.Path,
    apply: Callable[[pathlib.Path, str], tuple[subprocess.CompletedProcess[str], str]],
    value: str,
    message: str,
) -> None:
    """A flavor or a release the setup does not know is refused at staging and leaves the setup env as it was."""
    done, env = apply(tmp_path, value)
    assert done.returncode == 2
    assert message in done.stderr
    if apply is apply_flavor:
        assert env == SETUP_ENV
    else:
        assert "-v" not in env.replace("-vllm-", "-")
