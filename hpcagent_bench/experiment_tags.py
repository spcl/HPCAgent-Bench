# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""How an entity this repo plots is SPELLED in a figure.

An arm is named for the machine that routes it -- ``llr40v11`` says track, roster size and campaign
version in eight characters, which is right for a filename and wrong for a figure title. A reader
who has not spent a week in this repo cannot expand it, and a title they cannot expand says
nothing.

THE NAMES ARE DATA, in ``envs/display_names.yaml``, not literals in this module and emphatically
not literals at each call site. They are edited by whoever is writing the paper, and a name spelled
in three plotting scripts is a name that will disagree with itself -- which it already did once,
with one figure saying ``qwen38`` where its neighbour said ``Qwen3.8-27B`` for the same arm.

The mapping cannot be derived from the strings either: ``llr40`` and ``v11w2`` are the same
experiment run in two waves, and no rule over the tag would say so.

Every lookup here FALLS BACK to the tag unchanged rather than raising. A new campaign must not
break a figure; it just gets a plain label until someone names it. ``tests/test_display_names.py``
is what stops that fallback from going unnoticed.
"""

from __future__ import annotations

import functools
import pathlib

import yaml

REGISTRY = pathlib.Path(__file__).resolve().parent / "envs" / "display_names.yaml"


@functools.lru_cache(maxsize=1, typed=True)
def registry() -> dict:
    """The parsed registry. Cached: every label on every figure goes through here."""
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8")) or {}


def display_name(tag: str) -> str:
    """The name to put on a figure for a campaign tag. Falls back to the tag itself."""
    if not tag:
        return ""
    names = registry().get("experiments") or {}
    if tag in names:
        return names[tag]
    # An arm rather than a campaign ("llr40v11-qwen38-c-skills"): title it by its campaign.
    return names.get(tag.split("-", 1)[0], tag)


def model_name(model: str) -> str:
    """The display spelling of a model. Unknown ones pass through unchanged."""
    entry = (registry().get("models") or {}).get(str(model).lower())
    return entry["name"] if entry else str(model)


def model_checkpoint(model: str) -> str:
    """The checkpoint a model tag is expected to serve, or "" if the registry does not say.

    Recorded so ``tests/test_display_names.py`` can check the label against what the arms really
    ran: a campaign that swaps a checkpoint must not silently keep the old name on its axis.
    """
    entry = (registry().get("models") or {}).get(str(model).lower())
    return entry.get("serves", "") if entry else ""


def packet_name(packet: str) -> str:
    """The display spelling of a skill packet. "" is the control, which every figure needs a word
    for. A combination is spelled as its parts joined by " + ", so an unregistered pairing of two
    registered packets still reads."""
    names = registry().get("packets") or {}
    key = str(packet)
    if key in names:
        return names[key]
    parts = [p for p in key.split("+") if p]
    return " + ".join(names.get(p, p) for p in parts) if parts else names.get("", "No Skill Packet")


def language_name(language: str) -> str:
    """The display spelling of a language. Unknown ones pass through unchanged."""
    names = registry().get("languages") or {}
    return names.get(str(language).lower(), str(language))
