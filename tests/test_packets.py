# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""PACKETS: resolving a spec into skills/env/method, the canonical identity, the label, the colour.

Covers every predefined packet (cpf, cpfsrc, lang, lang-skills, profiling bundle, repo,
no-score-tool, autokernel, all-in), an implicit single-skill packet, an ad-hoc ``;``-separated list,
the error paths, and the identity/colour round trips that ``runs.packet`` already depends on.
"""

from __future__ import annotations

import dataclasses

import pytest

from hpcagent_bench import experiment_tags as tags
from hpcagent_bench import packets
from hpcagent_bench.stats import palette
from tests.test_palette import PUBLISHED_PACKET_COLORS


def test_resolve_control_is_empty() -> None:
    resolved = packets.resolve("", "c")
    assert resolved.key == ""
    assert resolved.label == "No Skill Packet"
    assert resolved.skills == ()
    assert resolved.env == ()
    assert resolved.method == ""
    assert resolved.parts == ()


def test_resolve_cpf_stages_the_page_and_the_dir_env() -> None:
    resolved = packets.resolve("cpf", "c", environ={"CPF_VIEW": "/views/cpf"})
    assert resolved.key == "cpf"
    assert resolved.label == "Canonical Parallel Form Page"
    assert resolved.skills == ("canonical-parallel-form",)
    assert resolved.env == (("HPCAGENT_BENCH_SERVICE_CANONICAL_PARALLEL_FORM_DIR", "/views/cpf"),)
    assert resolved.method == ""


def test_resolve_cpfsrc_sets_the_dropin_env_with_no_page() -> None:
    resolved = packets.resolve("cpfsrc", "c", environ={"CPF_VIEW": "/views/dropin"})
    assert resolved.key == "cpfsrc"
    assert resolved.skills == ()
    assert resolved.env == (("CPF_DROPIN_DIR", "/views/dropin"),)


def test_resolve_lang_expands_language_and_openmp_pages_for_c() -> None:
    resolved = packets.resolve("lang", "c")
    assert resolved.key == "lang"
    assert resolved.skills == ("lang-c", "openmp-c")


def test_resolve_lang_has_no_openmp_page_for_cuda() -> None:
    resolved = packets.resolve("lang", "cuda")
    assert resolved.skills == ("lang-cuda",)


def test_resolve_lang_skills_stages_every_shipped_page() -> None:
    resolved = packets.resolve("lang-skills", "c")
    assert resolved.key == "lang-skills"
    assert resolved.label == "All Skill Pages"
    shipped = sorted(p.name for p in packets.SKILLS_DIR.iterdir() if p.is_dir())
    assert list(resolved.skills) == shipped
    assert resolved.env == (("AGENT_HINTS_FILE", "hints-and-triggers.md"),)


def test_resolve_profiling_is_the_bundle() -> None:
    resolved = packets.resolve("profiling", "c")
    assert resolved.key == "profiling"
    assert resolved.skills == ("nsys", "opt-reports", "profiling", "rocprof")
    assert resolved.env == ()


def test_resolve_repo_sets_the_layout_env() -> None:
    resolved = packets.resolve("repo", "c", environ={"REPO_LAYOUT_PYTHON": "/venv/bin/python"})
    assert resolved.key == "repo"
    assert resolved.env == (
        ("AGENT_PROMPT_FILE", "prompt-repo.md"),
        ("REPO_LAYOUT", "1"),
        ("REPO_LAYOUT_LANGUAGE", "c"),
        ("REPO_LAYOUT_PYTHON", "/venv/bin/python"),
    )


def test_resolve_no_score_tool_sets_both_disable_switches() -> None:
    resolved = packets.resolve("no-score-tool", "c")
    assert resolved.key == "no-score-tool"
    assert resolved.env == (
        ("AGENT_SCORE_TOOL", "0"),
        ("AGENT_SUBMISSION_POLICY_FILE", "submission-blind.md"),
        ("HPCAGENT_BENCH_SERVICE_SCORE_ENABLED", "0"),
    )


def test_resolve_autokernel_is_a_method_packet() -> None:
    resolved = packets.resolve("autokernel", "c")
    assert resolved.key == "autokernel"
    assert resolved.method == "autokernel"
    assert resolved.env == (("AGENT_PACKET", "autokernel"),)
    assert resolved.skills == ()


def test_resolve_all_in_composes_cpfsrc_dc_profiling_and_lang() -> None:
    resolved = packets.resolve("all-in", "c", environ={"CPF_VIEW": "/views/dropin"})
    assert resolved.key == "all-in"
    assert resolved.label == "All-in"
    assert resolved.skills == (
        "divide-and-conquer",
        "lang-c",
        "nsys",
        "openmp-c",
        "opt-reports",
        "profiling",
        "rocprof",
    )
    assert resolved.env == (("CPF_DROPIN_DIR", "/views/dropin"),)
    assert resolved.method == ""


def test_resolve_an_implicit_single_skill_packet() -> None:
    resolved = packets.resolve("solver", "c")
    assert resolved.key == "solver"
    assert resolved.label == "solver"
    assert resolved.skills == ("solver",)
    assert resolved.env == ()
    assert resolved.parts == ("solver",)


def test_resolve_an_ad_hoc_semicolon_list() -> None:
    resolved = packets.resolve("rocprof;nsys", "c")
    assert resolved.skills == ("nsys", "rocprof")
    assert resolved.key == "nsys+rocprof"
    assert resolved.label == "ROCm Profiler + Nsight Systems"


def test_resolve_ignores_whitespace_and_duplicate_tokens() -> None:
    resolved = packets.resolve(" rocprof ; rocprof ; nsys ", "c")
    assert resolved.skills == ("nsys", "rocprof")
    assert resolved.parts == ("nsys", "rocprof")


def test_resolve_an_unknown_token_raises_naming_it() -> None:
    with pytest.raises(ValueError, match="nosuchpacket"):
        packets.resolve("nosuchpacket", "c")


def test_resolve_a_missing_placeholder_raises_naming_the_var() -> None:
    with pytest.raises(ValueError, match="CPF_VIEW"):
        packets.resolve("cpf", "c", environ={})


def test_resolve_conflicting_env_between_two_packets_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    real = tags.registry()
    conflicting = dict(real.packet_defs)
    conflicting["fake-a"] = tags.PacketDef(name="Fake A", skills=(), packets=(), env=(("X", "1"),), method="", color="")
    conflicting["fake-b"] = tags.PacketDef(name="Fake B", skills=(), packets=(), env=(("X", "2"),), method="", color="")
    fake_registry = dataclasses.replace(real, packet_defs=conflicting)
    monkeypatch.setattr(tags, "registry", lambda: fake_registry)
    with pytest.raises(ValueError, match="X"):
        packets.resolve("fake-a;fake-b", "c")


def test_canonical_of_control_is_empty() -> None:
    assert packets.canonical("") == ""


@pytest.mark.parametrize(
    "spec",
    ["lang-skills", "no-score-tool", "cpf", "cpfsrc", "repo", "lang"],
)
def test_canonical_of_a_registered_key_is_itself(spec: str) -> None:
    assert packets.canonical(spec) == spec


def test_canonical_of_an_already_canonical_combination_round_trips() -> None:
    assert packets.canonical("lang-skills+no-score-tool") == "lang-skills+no-score-tool"
    assert (
        packets.canonical("divide-and-conquer+nsys+opt-reports+profiling+rocprof")
        == "divide-and-conquer+nsys+opt-reports+profiling+rocprof"
    )
    assert (
        packets.canonical("cpf+divide-and-conquer+nsys+opt-reports+profiling+rocprof")
        == "cpf+divide-and-conquer+nsys+opt-reports+profiling+rocprof"
    )


def test_canonical_recognises_all_in_from_its_parts() -> None:
    assert packets.canonical("cpfsrc;divide-and-conquer;profiling;lang") == "all-in"
    assert packets.canonical("lang;profiling;divide-and-conquer;cpfsrc") == "all-in"


def test_canonical_does_not_collapse_profiling_when_its_parts_are_spelled_out() -> None:
    """The bundle's members plus its own name is five tokens, not the three ``profiling``
    composes from -- it must NOT collapse to ``profiling`` and drop a token silently."""
    assert (
        packets.canonical("divide-and-conquer;nsys;opt-reports;profiling;rocprof")
        == "divide-and-conquer+nsys+opt-reports+profiling+rocprof"
    )


def test_label_of_a_registered_key_is_its_display_name() -> None:
    assert packets.label("cpf") == "Canonical Parallel Form Page"
    assert packets.label("all-in") == "All-in"
    assert packets.label("lang") == "Language Pages"


def test_label_of_an_ad_hoc_combination_joins_the_parts() -> None:
    assert packets.label("divide-and-conquer;profiling") == "Divide and Conquer + Profiling Tools"


@pytest.mark.parametrize("key", tags.order("packets"))
def test_packet_color_matches_palette_for_every_registered_key(key: str) -> None:
    assert packets.packet_color(key) == palette.color(key)


@pytest.mark.parametrize("spec,expected", sorted(PUBLISHED_PACKET_COLORS.items()))
def test_packet_color_matches_the_published_colours(spec: str, expected: str) -> None:
    assert packets.packet_color(spec) == expected
    assert packets.packet_color(spec) == palette.color(spec)


def test_packet_color_of_an_ad_hoc_combination_is_deterministic() -> None:
    first = packets.packet_color("mystery-tool;another-mystery")
    second = packets.packet_color("another-mystery;mystery-tool")
    assert first == second == palette.color("mystery-tool;another-mystery")


def test_an_unfilled_resolve_keeps_the_placeholder_templates_the_db_records() -> None:
    """fill=False is the packet's definition, not one launch: no environment is needed and every
    ${VAR} survives verbatim, including through a composition."""
    assert packets.resolve("cpfsrc", "c", environ={}, fill=False).env == (("CPF_DROPIN_DIR", "${CPF_VIEW}"),)
    assert (
        dict(packets.resolve("repo", "c", environ={}, fill=False).env)["REPO_LAYOUT_PYTHON"] == "${REPO_LAYOUT_PYTHON}"
    )
    all_in = packets.resolve("all-in", "c", environ={}, fill=False)
    assert dict(all_in.env) == {"CPF_DROPIN_DIR": "${CPF_VIEW}"}
    assert {"lang-c", "openmp-c", "divide-and-conquer", "profiling", "rocprof", "nsys", "opt-reports"} <= set(
        all_in.skills
    )
