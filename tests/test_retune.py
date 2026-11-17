# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``hpcagent-bench job retune``: a pending job's read-only env is changed through a backup and an atomic replace, only
the requested lines differ afterwards, and the ledger holds what changed and why; a running job, a limit, the identity
of the setup, a secret, an unknown key and a serving-rule breach are refused with the file untouched."""

import json
import os
import pathlib
import stat
import tempfile

import pytest

from hpcagent_bench.cluster import env_spec, retune
from tests.bash_stub import bash_stub

ENV = """SETUP=exp-kimi-c
AGENT_TIMEOUT_SECONDS=57600
MOE_CHECKPOINT=int4-wna16
SGLANG_EXTRA_ARGS="--trust-remote-code --mem-fraction-static 0.48"
INFERENCE_NODES=4
"""


def queued(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, state: str = "PENDING") -> pathlib.Path:
    """A job 7 in ``state`` whose snapshot env is read-only, as ``snapshot_env`` leaves it."""
    rendered = tmp_path / ".rendered"
    rendered.mkdir()
    env = rendered / "exp-kimi-c-20261117T000000Z-abc.env"
    env.write_text(ENV)
    env.chmod(0o400)
    line = (
        f"JobId=7 JobName=exp-kimi-c JobState={state} Command=/repo/hpcagent_bench/cluster/services.sbatch {env} "
        "WorkDir=/repo/experiments"
    )
    bash_stub(tmp_path / "bin", "scontrol", f"echo '{line}'")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")
    return env


def test_a_pending_jobs_env_is_changed_backed_up_and_recorded(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = queued(tmp_path, monkeypatch)
    entries = retune.retune(
        "7",
        {"SGLANG_EXTRA_ARGS": '"--trust-remote-code --mem-fraction-static 0.52"', "INFERENCE_NODES": "8"},
        "0.48 left no KV pool",
    )
    assert [(entry["key"], entry["old"]) for entry in entries] == [
        ("SGLANG_EXTRA_ARGS", '"--trust-remote-code --mem-fraction-static 0.48"'),
        ("INFERENCE_NODES", "4"),
    ]
    after = env_spec.assignments(env)
    assert after["SGLANG_EXTRA_ARGS"].endswith('0.52"') and after["INFERENCE_NODES"] == "8"
    assert after["SETUP"] == "exp-kimi-c" and after["AGENT_TIMEOUT_SECONDS"] == "57600"
    assert stat.S_IMODE(env.stat().st_mode) == 0o400, "the snapshot stays read-only"
    backup = env.with_name(f"{env.name}.retune-1.bak")
    assert backup.read_text() == ENV and stat.S_IMODE(backup.stat().st_mode) == 0o400
    ledger = [json.loads(line) for line in (tmp_path / ".rendered" / "retune.jsonl").read_text().splitlines()]
    assert [(line["key"], line["new"], line["reason"], line["job"]) for line in ledger][1] == (
        "INFERENCE_NODES",
        "8",
        "0.48 left no KV pool",
        "7",
    )
    retune.retune("7", {"INFERENCE_NODES": "6"}, "more nodes")
    assert env.with_name(f"{env.name}.retune-2.bak").read_text().count("INFERENCE_NODES=8") == 1
    assert [entry["new"] for entry in retune.history("7")] == [line["new"] for line in ledger] + ["6"]


@pytest.mark.parametrize(
    ("edits", "match"),
    [
        ({"AGENT_TIMEOUT_SECONDS": "100"}, "run protocol"),
        ({"SETUP": "other"}, "identity"),
        ({"HPCAGENT_BENCH_RECORD_MODEL": "x"}, "identity"),
        ({"LITELLM_API_KEY": "x"}, "secret"),
        ({"NEW_SETTING": "1"}, "not set"),
        ({"SGLANG_EXTRA_ARGS": '"--ep-size 4"'}, "moe-runner-backend triton"),
    ],
)
def test_a_refused_change_leaves_the_env_and_the_ledger_untouched(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, edits: dict[str, str], match: str
) -> None:
    env = queued(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match=match):
        retune.retune("7", edits, "because")
    assert env.read_text() == ENV and sorted(path.name for path in env.parent.iterdir()) == [env.name]


def test_a_running_job_and_a_missing_reason_are_refused_and_add_sets_a_new_key(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = queued(tmp_path, monkeypatch, "RUNNING")
    with pytest.raises(SystemExit, match="RUNNING: it has read its env"):
        retune.retune("7", {"INFERENCE_NODES": "8"}, "why")
    queued_again = tmp_path / "again"
    queued_again.mkdir()
    env = queued(queued_again, monkeypatch)
    with pytest.raises(SystemExit, match="--reason is required"):
        retune.retune("7", {"INFERENCE_NODES": "8"}, " ")
    retune.retune("7", {"KV_GUARD": "warn"}, "serve a small pool anyway", adding=True)
    assert env_spec.assignments(env)["KV_GUARD"] == "warn"
    assert (
        retune.retune("7", {"KV_GUARD": "fail"}, "x", dry_run=True) and env_spec.assignments(env)["KV_GUARD"] == "warn"
    )


if __name__ == "__main__":

    def scratch() -> pathlib.Path:
        return pathlib.Path(tempfile.mkdtemp())

    with pytest.MonkeyPatch.context() as patch:
        test_a_pending_jobs_env_is_changed_backed_up_and_recorded(scratch(), patch)
    for edits, match in (
        ({"AGENT_TIMEOUT_SECONDS": "100"}, "run protocol"),
        ({"SETUP": "other"}, "identity"),
        ({"HPCAGENT_BENCH_RECORD_MODEL": "x"}, "identity"),
        ({"LITELLM_API_KEY": "x"}, "secret"),
        ({"NEW_SETTING": "1"}, "not set"),
        ({"SGLANG_EXTRA_ARGS": '"--ep-size 4"'}, "moe-runner-backend triton"),
    ):
        with pytest.MonkeyPatch.context() as patch:
            test_a_refused_change_leaves_the_env_and_the_ledger_untouched(scratch(), patch, edits, match)
    with pytest.MonkeyPatch.context() as patch:
        test_a_running_job_and_a_missing_reason_are_refused_and_add_sets_a_new_key(scratch(), patch)
