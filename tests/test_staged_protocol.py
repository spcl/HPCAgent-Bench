# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""submit.sh refuses a staged setup env whose run protocol (env_spec.PROTOCOL_KEYS) is not what experiments.toml
renders for its experiment and model, the setup's packet's own keys aside (the blind packet's submission mode): one
place sets the protocol, and nothing on the way to the job changes it."""

import datetime
import pathlib

from hpcagent_bench.cluster import env_spec
from tests.test_submit import submit, tree


def test_the_rendered_protocol_and_a_packets_mode_pass_and_any_other_value_is_refused() -> None:
    env = env_spec.render("llrblind:qwen38")
    assert env_spec.staged_refusal(env, "llrblind:qwen38", {}) == ""
    blind = {**env, "AGENT_SUBMISSION_MODE": "blind"}
    assert env_spec.staged_refusal(blind, "llrblind:qwen38", {"AGENT_SUBMISSION_MODE": "blind"}) == ""
    assert "AGENT_SUBMISSION_MODE=blind" in env_spec.staged_refusal(blind, "llrblind:qwen38", {})
    raised = {**env, "AGENT_MAX_TOKENS": "240000000"}
    assert "AGENT_MAX_TOKENS=240000000, experiments.toml 50000000" in env_spec.staged_refusal(
        raised, "llrblind:qwen38", {}
    )
    # a packet (a plugin's included) sets the submission mode only, never a limit
    assert "AGENT_MAX_TOKENS=240000000, experiments.toml 50000000" in env_spec.staged_refusal(
        raised, "llrblind:qwen38", {"AGENT_MAX_TOKENS": "240000000"}
    )


def test_a_deadline_that_cuts_the_wall_clock_is_refused_at_staging(tmp_path: pathlib.Path) -> None:
    root = tree(tmp_path)
    deadline = (datetime.datetime.now() + datetime.timedelta(hours=10)).strftime("%Y-%m-%d %H:%M")
    done = submit(root, "--account", "project", KERNELS_FILE="subset.txt", LANGUAGES="c", DEADLINE=deadline)
    assert done.returncode != 0
    assert "the staged run protocol differs: AGENT_TIMEOUT_SECONDS=" in done.stderr
    assert not list((root / "experiments").glob(".env.wave-*"))


if __name__ == "__main__":
    import tempfile

    test_the_rendered_protocol_and_a_packets_mode_pass_and_any_other_value_is_refused()
    test_a_deadline_that_cuts_the_wall_clock_is_refused_at_staging(pathlib.Path(tempfile.mkdtemp()))
