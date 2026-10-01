# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""CE_IMAGE_FLAVOR=native moves an experiment's agent and judge EDFs to the -native ones a native build
renders; the serving EDF stays -latest, and the default flavor changes nothing."""

import pathlib
import subprocess

import pytest

from hpcagent_bench.paths import ROOT

SUBMIT_COMMON = ROOT / "hpcagent_bench" / "cluster" / "submit_common.sh"
SETUP_ENV = (
    "AMD_CE_ENV=hpcagent-bench-agent-mi300-latest\n"
    "JUDGE_CE_ENV=hpcagent-bench-judge-mi300-mlscale-latest\n"
    "INFERENCE_CE_ENV=hpcagent-bench-vllm-mi300-latest\n"
)


def apply_flavor(tmp_path: pathlib.Path, flavor: str) -> tuple[subprocess.CompletedProcess[str], str]:
    env_file = tmp_path / "arm.env"
    env_file.write_text(SETUP_ENV, encoding="utf-8")
    done = subprocess.run(
        ["bash", "-c", 'source "$1" && apply_flavor "$2"', "bash", str(SUBMIT_COMMON), str(env_file)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "CE_IMAGE_FLAVOR": flavor},
    )
    return done, env_file.read_text(encoding="utf-8")


def test_native_moves_agent_and_judge_edfs_but_not_serving(tmp_path: pathlib.Path) -> None:
    done, env = apply_flavor(tmp_path, "native")
    assert done.returncode == 0, done.stderr
    assert env == (
        "AMD_CE_ENV=hpcagent-bench-agent-mi300-native\n"
        "JUDGE_CE_ENV=hpcagent-bench-judge-mi300-mlscale-native\n"
        "INFERENCE_CE_ENV=hpcagent-bench-vllm-mi300-latest\n"
    )


def test_latest_changes_nothing(tmp_path: pathlib.Path) -> None:
    done, env = apply_flavor(tmp_path, "latest")
    assert done.returncode == 0, done.stderr
    assert env == SETUP_ENV


@pytest.mark.parametrize("flavor", ["zen4", "Native"])
def test_an_unknown_flavor_is_refused(tmp_path: pathlib.Path, flavor: str) -> None:
    done, env = apply_flavor(tmp_path, flavor)
    assert done.returncode == 2
    assert "latest or native" in done.stderr
    assert env == SETUP_ENV
