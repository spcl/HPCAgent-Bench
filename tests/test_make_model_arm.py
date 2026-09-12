# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""make_model_arm.py derives one arm's env for a different model from a sibling arm's env.

Its substitution used to demand exactly one existing line per model key, which breaks the day a
source arm drops a vestigial key the target model still needs (e.g. an SGLang arm pruned of its
inert VLLM_* lines). models.py is the one source of per-model serving config those keys come from;
a copy hardcoded here or in a generator is how a context-window fix lands in one and stays stale in
the other.
"""

import importlib.util
import pathlib
import sys
from types import ModuleType

import pytest

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "experiments"


@pytest.fixture(name="make_model_arm")
def make_model_arm_fixture() -> ModuleType:
    if str(EXAMPLE) not in sys.path:
        sys.path.insert(0, str(EXAMPLE))
    spec = importlib.util.spec_from_file_location("make_model_arm", EXAMPLE / "make_model_arm.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_derive_appends_a_model_key_the_source_never_had(make_model_arm: ModuleType, tmp_path: pathlib.Path) -> None:
    """A source arm that dropped a vestigial key (e.g. VLLM_EXTRA_ARGS on an SGLang-only arm)
    must still gain it, rather than the derivation refusing to run at all."""
    src = tmp_path / ".env.gpu-llr40-qwen38-hip"
    src.write_text("CAMPAIGN_ARM=gpu-llr40-qwen38-hip\nLANGUAGE=hip\n")
    text = make_model_arm.derive(src, "oss120b", "qwen38")
    assert text.count("VLLM_EXTRA_ARGS=") == 1


def test_derive_raises_on_a_source_carrying_a_key_twice(make_model_arm: ModuleType, tmp_path: pathlib.Path) -> None:
    """Two lines for one model key is ambiguous -- which one is the arm's real value -- so
    derive() must refuse rather than silently pick one."""
    src = tmp_path / ".env.gpu-llr40-qwen38-hip"
    src.write_text("CAMPAIGN_ARM=gpu-llr40-qwen38-hip\nVLLM_MODEL=a\nVLLM_MODEL=b\n")
    with pytest.raises(SystemExit):
        make_model_arm.derive(src, "oss120b", "qwen38")


def test_oss120b_model_table_matches_its_base_env(make_model_arm: ModuleType) -> None:
    """models.py is the one source oss120b's serving block comes from; a base env edited without
    this table is exactly the drift this generator exists to prevent."""
    base_text = (EXAMPLE / ".env.base-oss120b").read_text()
    for key, value in make_model_arm.MODELS["oss120b"].items():
        assert f"{key}={value}" in base_text, f"{key} in models.py no longer matches .env.base-oss120b"
