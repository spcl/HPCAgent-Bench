# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Docs must not drift from the implementation.

Prose goes stale silently: a knob is renamed, a template moves, an endpoint changes shape,
and the doc keeps confidently describing the old world. These check the claims that are
mechanically checkable, so drift fails a test instead of misleading a reader.
"""
import dataclasses
import pathlib
import re

import pytest

from optarena import config
from optarena.config import AttemptSettings, PromptSettings
from optarena.harness.prompts import PROMPT_VARIANTS, PromptConfig

PROMPTS_DIR = pathlib.Path("optarena/harness/prompts")
DOCS = [
    pathlib.Path("README.md"),
    pathlib.Path("docs/PROMPTS.md"),
    pathlib.Path("docs/PROMPT_WALKTHROUGH.md"),
    pathlib.Path("docs/AGENTS_AND_TOOL_ACCESS.md"),
    pathlib.Path("docs/WRITING_AN_AGENT.md"),
    pathlib.Path("optarena/harness/README.md"),
]

#: `prompt.*` names that are config keys but NOT PromptConfig fields.
NON_FIELD_PROMPT_KEYS = {"variants", "variant"}


def doc_text():
    return [(p, p.read_text()) for p in DOCS if p.exists()]


def test_every_prompt_config_field_has_a_config_key():
    """A field with no config.yaml key cannot be set by a user editing the file."""
    missing = [f.name for f in dataclasses.fields(PromptConfig) if config.get(f"prompt.{f.name}", "?") == "?"]
    assert not missing, f"PromptConfig fields absent from config.yaml: {missing}"


def test_prompt_config_and_settings_stay_in_lockstep():
    assert {f.name for f in dataclasses.fields(PromptConfig)} == {f.name for f in dataclasses.fields(PromptSettings)}


def test_docs_name_no_prompt_key_that_does_not_exist():
    """Catches a knob that was removed or renamed but is still documented."""
    fields = {f.name for f in dataclasses.fields(PromptConfig)} | NON_FIELD_PROMPT_KEYS
    stale = []
    for path, text in doc_text():
        for m in re.finditer(r"`prompt\.([a-z_]+)`", text):
            if m.group(1) not in fields:
                stale.append(f"{path}: prompt.{m.group(1)}")
    assert not stale, f"documented prompt.* keys that no longer exist: {stale}"


def test_docs_name_no_attempts_key_that_does_not_exist():
    fields = {f.name for f in dataclasses.fields(AttemptSettings)}
    stale = []
    for path, text in doc_text():
        for m in re.finditer(r"`attempts\.([a-z_]+)`", text):
            if m.group(1) not in fields:
                stale.append(f"{path}: attempts.{m.group(1)}")
    assert not stale, f"documented attempts.* keys that no longer exist: {stale}"


def test_docs_name_no_variant_that_is_not_registered():
    """A `--variant X` in the docs must resolve: a built-in, a `task_var<N>.j2` discovery, or
    one the same doc declares in its own `prompt.variants` example."""
    stale = []
    for path, text in doc_text():
        # Names the doc itself declares under `variants:` are legitimate examples.
        declared = set(re.findall(r"^\s{4}([a-z_]\w*):\s*\{", text, re.M))
        for m in re.finditer(r"--variant\s+([a-z_]+)", text):
            name = m.group(1)
            if name not in PROMPT_VARIANTS and name not in declared and not name.startswith("var"):
                stale.append(f"{path}: --variant {name}")
    assert not stale, f"documented variants that are not registered: {stale}"


def test_docs_name_no_template_that_does_not_exist():
    """A doc naming a template that was renamed/removed sends the reader to nothing."""
    known = {p.name for p in PROMPTS_DIR.rglob("*.j2")}
    stale = []
    for path, text in doc_text():
        for m in re.finditer(r"`([a-z_]+\.j2)`", text):
            name = m.group(1)
            if name not in known and "_var" not in name:
                stale.append(f"{path}: {name}")
    assert not stale, f"documented templates that do not exist: {stale}"


def test_every_internal_doc_link_resolves():
    """A renamed heading silently breaks inbound anchors from other docs."""

    def anchors(p: pathlib.Path):
        out = set()
        for line in p.read_text().splitlines():
            m = re.match(r"^#+\s+(.*)", line)
            if m:
                a = re.sub(r"[^\w\s-]", "", m.group(1).lower()).strip().replace(" ", "-")
                out.add(a)
        return out

    broken = []
    for path, text in doc_text():
        for m in re.finditer(r"\]\(([^)]+)\)", text):
            target = m.group(1)
            if target.startswith(("http", "mailto")):
                continue
            rel, _, frag = target.partition("#")
            dest = (path.parent / rel).resolve() if rel else path
            if rel and not dest.exists():
                broken.append(f"{path}: missing file {target}")
            elif frag and dest.suffix == ".md" and dest.exists() and frag not in anchors(dest):
                broken.append(f"{path}: dead anchor {target}")
    assert not broken, "broken internal doc links: " + "; ".join(broken)


@pytest.mark.parametrize("forbidden", ["disclose_public_seed", "PUBLIC seed"])
def test_the_removed_seed_disclosure_is_gone_everywhere(forbidden):
    """The seed is never disclosed: the prompt states the RANGE only. Docs must not promise
    otherwise, and no template may reference the removed context keys."""
    hits = [str(p) for p, text in doc_text() if forbidden in text]
    hits += [str(p) for p in PROMPTS_DIR.rglob("*.j2") if forbidden in p.read_text()]
    assert not hits, f"{forbidden!r} still referenced in: {hits}"
