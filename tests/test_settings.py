# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The typed config singleton.

Pins that the dataclass defaults agree with ``config.yaml`` (the file is the source a user
edits permanently), that assigning to a section changes the config for this process only,
and that the precedence override > env > file survives the typed layer.
"""
import dataclasses

import pytest

from optarena import config
from optarena.config import AttemptSettings, PromptSettings, Section, Settings, settings


@pytest.fixture(autouse=True)
def clean_settings():
    """Every test starts from the file, and leaves no override behind for the next one."""
    config.reload()
    yield
    config.reload()


def test_settings_is_a_singleton():
    assert settings() is settings()


def test_sections_load_from_the_yaml_file():
    """"Matches config.yaml by default" -- the loaded value is the file's, not the
    dataclass default, wherever the two could differ."""
    assert settings().prompt.template == config.get("prompt.template")
    assert settings().attempts.max_rounds == config.get("attempts.max_rounds")


@pytest.mark.parametrize("section", [PromptSettings, AttemptSettings])
def test_declared_defaults_match_the_yaml(section):
    """A field whose declared default disagrees with config.yaml is a drift bug: the two
    would then depend on which layer answered."""
    mismatched = {}
    for f in dataclasses.fields(section):
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
        in_file = config.get(f"{section.prefix}.{f.name}", default)
        # The YAML spells a sequence as a list; the field is a tuple for hashability.
        if isinstance(default, tuple):
            in_file = tuple(in_file or ())
        if in_file != default:
            mismatched[f.name] = (default, in_file)
    assert not mismatched, f"{section.__name__} defaults disagree with config.yaml: {mismatched}"


def test_every_declared_field_exists_in_the_yaml():
    """A typed field with no key in the file is unreachable for a user editing the file."""
    for section in (PromptSettings, AttemptSettings):
        missing = [f.name for f in dataclasses.fields(section) if config.get(f"{section.prefix}.{f.name}", "?") == "?"]
        assert not missing, f"{section.__name__} fields absent from config.yaml: {missing}"


def test_prompt_settings_mirrors_prompt_config():
    """PromptConfig resolves the same prompt.* keys per call; a field added to one and not
    the other means the singleton cannot reach it."""
    from optarena.harness.prompts import PromptConfig
    assert {f.name for f in dataclasses.fields(PromptSettings)} == {f.name for f in dataclasses.fields(PromptConfig)}


def test_assignment_changes_the_config_for_this_process():
    settings().prompt.debug = True
    assert config.get("prompt.debug") is True


def test_assignment_reaches_the_consumer_not_just_the_singleton():
    """The point of the singleton: a component that resolves its own config sees the change."""
    from optarena.harness.prompts import PromptConfig
    assert PromptConfig.from_config().inline_kernel is False
    settings().prompt.inline_kernel = True
    assert PromptConfig.from_config().inline_kernel is True


def test_assignment_beats_an_env_var(monkeypatch):
    """Precedence is override > env > file, and the singleton is the override layer."""
    monkeypatch.setenv("OPTARENA_ATTEMPTS_MAX_ROUNDS", "9")
    assert config.get("attempts.max_rounds") == 9
    settings().attempts.max_rounds = 3
    assert config.get("attempts.max_rounds") == 3


def test_env_still_resolves_per_call_not_at_load(monkeypatch):
    """Env must NOT be snapshotted into the singleton: tests (and callers) set OPTARENA_*
    after the config has already been read."""
    settings()  # force the load first
    monkeypatch.setenv("OPTARENA_ATTEMPTS_MAX_ROUNDS", "7")
    assert config.get("attempts.max_rounds") == 7


def test_loading_registers_no_overrides(monkeypatch):
    """A plain load must not pin its values as overrides -- if it did, the env layer below
    would be permanently masked for every field the singleton touched."""
    config.reload()
    settings()  # load, without assigning anything
    monkeypatch.setenv("OPTARENA_PROMPT_STRATEGY", "loopnest")
    assert config.get("prompt.strategy") == "loopnest"


def test_reload_drops_runtime_changes():
    settings().prompt.strategy = "loopnest"
    assert config.get("prompt.strategy") == "loopnest"
    config.reload()
    assert config.get("prompt.strategy") == "default"


def test_non_field_attribute_does_not_register_an_override():
    """Only declared config fields are config; scratch attributes are not."""
    settings().prompt.scratch = 1
    assert config.get("prompt.scratch") is None


def test_untyped_blocks_are_still_reachable():
    """Sections are typed incrementally, so a block with no dataclass must still resolve."""
    assert config.get("seeds.fuzz") is not None


def test_section_subclass_needs_only_a_prefix_and_fields():
    """Adding a block is declaring it -- no loader/registry edit."""

    @dataclasses.dataclass
    class SeedSettings(Section):
        prefix = "seeds"
        fuzz: int = 0

    assert SeedSettings.load().fuzz == config.get("seeds.fuzz")


def test_settings_exposes_the_typed_sections():
    assert isinstance(settings(), Settings)
    assert isinstance(settings().prompt, PromptSettings)
    assert isinstance(settings().attempts, AttemptSettings)
