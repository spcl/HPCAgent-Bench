# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""stream_idle_timeout.py: the byte-stream idle timeout, derived instead of copied.

qwen38 lost 9/9 attempts of job 641738 and 16/40 episodes of 641748 (2026-09-19) to "API Error: The
operation timed out." with CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS already pinned to the CLI's 30-minute
ceiling -- so raising it further is not possible, only checking that the ceiling is really what the
installed CLI enforces, and that the arithmetic which justifies sitting at it is right.
"""

import importlib.util
import pathlib
import subprocess
import sys
import types

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "stream_idle_timeout.py"


@pytest.fixture(name="module", scope="module")
def module_fixture() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("stream_idle_timeout", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_no_arm_ever_gets_a_value_the_cli_would_silently_clamp(module: types.ModuleType) -> None:
    """``Math.min(Math.max(n, ViS), KiS)`` in the installed 2.1.224 bundle -- outside it is not a
    bigger or smaller timeout, it is FLOOR_MS or CEILING_MS with a wasted round trip."""
    for requested in (-1, 0, 1, module.FLOOR_MS - 1, module.FLOOR_MS, module.CEILING_MS, module.CEILING_MS + 1, 1e9):
        assert module.FLOOR_MS <= module.clamp_ms(requested) <= module.CEILING_MS


def test_an_arm_naming_neither_var_gets_the_same_default_as_before_this_module_existed(
    module: types.ModuleType,
) -> None:
    assert module.derive_ms(0, 0) == module.CEILING_MS
    assert module.derive_ms(262144, 0) == module.CEILING_MS
    assert module.derive_ms(0, 40) == module.CEILING_MS


def test_qwen38s_own_perf_playbook_config_already_needs_the_full_ceiling(module: types.ModuleType) -> None:
    """262144-token context (models.py SERVED_CONTEXT['qwen38']) at AGENTS_PER_NODE=40: the derived,
    margined worst case is itself far past 30 minutes, so the ceiling is not generous -- it is the
    most patience the CLI will ever grant, and still short of the theoretical worst case."""
    pre_clamp_s = 262144 / (module.MEASURED_NODE_PROMPT_TOK_S / 40) * module.SAFETY_MARGIN
    assert pre_clamp_s * 1000 > module.CEILING_MS
    assert module.derive_ms(262144, 40) == module.CEILING_MS


def test_a_small_low_concurrency_arm_derives_under_the_ceiling(module: types.ModuleType) -> None:
    """git-scicomp's qwen38 arms cap AGENTS_PER_NODE at 30, not 40 -- still not enough headroom at
    the full served context, but a SHORT context on a lightly-loaded node derives well under the
    ceiling, which is the case this module exists to speed up: a dead stream on that arm is now
    caught in less than the blanket 30 minutes every arm used to sit at."""
    assert module.FLOOR_MS < module.derive_ms(8192, 4) < module.CEILING_MS


def test_the_launcher_reads_the_same_value_off_the_environment(module: types.ModuleType) -> None:
    """run_cluster.sh shells out to this file, so the CLI and the import must not drift apart."""
    done = subprocess.run(
        [sys.executable, str(SCRIPT)],
        env={"CONTEXT_LENGTH": "262144", "AGENTS_PER_NODE": "40", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert int(done.stdout.strip()) == module.derive_ms(262144, 40)


def test_a_blank_or_garbage_env_value_is_zero_not_a_crash(module: types.ModuleType) -> None:
    assert module.positive_int("") == 0
    assert module.positive_int("not-a-number") == 0
    assert module.positive_int("-5") == 0
    assert module.positive_int("40") == 40
