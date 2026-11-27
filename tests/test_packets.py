# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Packet specs resolve into skills/env/tools with a canonical identity, label and colour, for every predefined,
implicit and ad-hoc packet, plus refusals and error paths."""

import dataclasses

import pytest

from hpcagent_bench import packets
from hpcagent_bench import study_tags as tags
from hpcagent_bench.stats import palette


def test_resolve_control_is_empty() -> None:
    resolved = packets.resolve("", "c")
    assert resolved.key == ""
    assert resolved.label == "No Skill Packet"
    assert resolved.skills == ()
    assert resolved.env == ()
    assert resolved.parts == ()

    blind = packets.resolve("no-score-tool", "c")
    assert blind.key == "no-score-tool"
    assert blind.env == (("AGENT_SUBMISSION_MODE", "blind"),)

    autokernel = packets.resolve("autokernel", "c")
    assert autokernel.key == "autokernel"
    assert autokernel.tools == ("experiment",)
    assert autokernel.env == ()
    assert autokernel.skills == ()
    assert packets.prompt_layers("autokernel") == ("autokernel:prompt/overlay.yaml",)


@pytest.mark.parametrize(
    ("spec", "language", "key", "label"),
    [
        ("cpf-tool-cpu", "c", "cpf-tool-cpu", "Canonical Parallel Form Tool, CPU"),
        ("cpf-tool-gpu", "hip", "cpf-tool-gpu", "Canonical Parallel Form Tool, GPU"),
    ],
)
def test_resolve_cpf_stages_the_page_and_the_dir_env(spec: str, language: str, key: str, label: str) -> None:
    """A tool packet names its device: there is no device-neutral spelling."""
    resolved = packets.resolve(spec, language, environ={"HPCAGENT_BENCH_CPF_CACHE": "/views/cpf"})
    assert resolved.key == key
    assert resolved.label == label
    assert resolved.skills == (key,)
    assert resolved.env == (
        ("HPCAGENT_BENCH_CPF_VERIFIED_ONLY", "true"),
        ("HPCAGENT_BENCH_SERVICE_CANONICAL_PARALLEL_FORM_DIR", "/views/cpf"),
    )


@pytest.mark.parametrize(
    ("spec", "language", "key"),
    [
        ("cpf-cpu", "c", "cpf-cpu"),
        ("cpf-gpu", "hip", "cpf-gpu"),
        ("cpf-cpu-lean", "c", "cpf-cpu-lean"),
        ("cpf-gpu-lean", "hip", "cpf-gpu-lean"),
    ],
)
def test_resolve_cpf_src_stages_its_own_page_and_sets_the_dropin_env(spec: str, language: str, key: str) -> None:
    """A CPF-as-source packet stages exactly its own page -- the drop-in's comment reference -- alongside
    the drop-in directory env; the drop-in file itself is staged by materialize_shared.sh, not by a
    skill page, so the packet's only page is the one explaining what sits in that file."""
    resolved = packets.resolve(spec, language, environ={"HPCAGENT_BENCH_CPF_CACHE": "/views/dropin"})
    assert resolved.key == key
    assert resolved.skills == ((key,) if key != "cpf-cpu-lean" else ())
    lean = (("HPCAGENT_BENCH_CPF_FORM", "lean"),) if key.endswith("-lean") else ()
    assert resolved.env == (
        ("CPF_DROPIN_DIR", "/views/dropin"),
        *lean,
        ("HPCAGENT_BENCH_CPF_VERIFIED_ONLY", "true"),
    )


def test_resolve_lang_expands_language_and_openmp_pages_for_c() -> None:
    resolved = packets.resolve("lang", "c")
    assert resolved.key == "lang"
    assert resolved.skills == ("lang-c", "openmp-c")

    # cuda ships no ``openmp-cuda`` page; the setup gets the host-language page instead.
    cuda = packets.resolve("lang", "cuda")
    assert "openmp-cuda" not in cuda.skills
    assert cuda.skills == ("lang-cpp", "lang-cuda")


@pytest.mark.parametrize(
    ("language", "companion"), [("hip", "lang-cpp"), ("cuda", "lang-cpp"), ("triton", "lang-python")]
)
def test_resolve_lang_stages_the_page_its_own_page_sends_the_agent_to(language: str, companion: str) -> None:
    """``lang-hip`` opens "read this page first, together with lang-cpp, which governs the host half
    of the same file" -- a trigger naming a page the setup did not stage points at
    ``/skills/lang-cpp.md``, which is not there. ``*`` picked the companion up all along
    (the companion's own ``applies.languages`` names hip); the ``lang`` token did not, so ``lang``,
    ``all-in-amd`` and ``all-in-nvidia`` shipped half the language packet."""
    resolved = packets.resolve("lang", language)
    assert f"lang-{language}" in resolved.skills
    assert companion in resolved.skills


def test_resolve_lang_skills_stages_every_shipped_page_but_a_packet_tools_own_or_an_explicit_one() -> None:
    """``*`` is every page except the manual for a tool only one packet's setups are served (staging
    that page here would hand the skills setup instructions for a tool it does not have) and except a
    page marked ``explicit: true`` (caveman, the CPF-as-source pages): those are treatments of their own, reachable
    only by naming them, never picked up as part of the whole-library packet. With no language,
    image or topology named, nothing else narrows it (packets.applies_to)."""
    resolved = packets.resolve("lang-skills", "", multinode=True)
    assert resolved.key == "lang-skills"
    assert resolved.label == "Language Skill Packet"
    shipped = sorted(p.name for p in packets.SKILLS_DIR.iterdir() if p.is_dir())
    applicable = [page for page in shipped if packets.applies_to(page, "", None, True)]
    assert list(resolved.skills) == [page for page in applicable if page not in packets.tool_pages()]
    assert set(shipped) - set(resolved.skills) == packets.tool_pages() | {
        "caveman",
        "cpf-cpu",
        "cpf-gpu",
        "cpf-cpu-bare",
        "cpf-gpu-bare",
        "cpf-cpu-lean",
        "cpf-gpu-lean",
    }
    assert resolved.env == (), "no skill content may ride in the main prompt: the packet sets no hints file"


@pytest.mark.parametrize(
    ("spec", "language", "tracer"),
    [
        ("perf-playbook-cpu", "c", ()),
        ("perf-playbook-amd", "hip", ("rocprof",)),
        ("perf-playbook-nvidia", "cuda", ("nsys",)),
    ],
)
def test_a_perf_playbook_stages_the_cpu_pages_and_only_its_own_device_tracer(
    spec: str, language: str, tracer: tuple[str, ...]
) -> None:
    """The other vendor's tracer page can never run on the device, so it is rent with no payoff."""
    resolved = packets.resolve(spec, language)
    assert resolved.pages == ("divide-and-conquer", "profiling", *tracer, "opt-reports")
    assert resolved.env == ()


@pytest.mark.parametrize(
    ("spec", "language"),
    [
        ("perf-playbook-amd", "c"),
        ("perf-playbook-amd", "cuda"),
        ("perf-playbook-nvidia", "hip"),
        ("perf-playbook-cpu", "hip"),
        ("perf-playbook-cpu", "cuda"),
        ("all-in-amd", "c"),
        ("all-in-nvidia", "fortran"),
        ("all-in-cpu", "cuda"),
    ],
)
def test_a_device_packet_refuses_a_language_its_device_does_not_run(spec: str, language: str) -> None:
    with pytest.raises(ValueError, match="packet 'perf-playbook-"):
        packets.resolve(spec, language, environ={"HPCAGENT_BENCH_CPF_CACHE": "/views/dropin"})


@pytest.mark.parametrize(
    ("device", "language", "cpf"),
    [("cpu", "c", "cpf-cpu"), ("amd", "hip", "cpf-gpu"), ("nvidia", "cuda", "cpf-gpu")],
)
def test_all_in_for_a_device_is_cpf_src_its_perf_playbook_and_the_language_pages(
    device: str, language: str, cpf: str
) -> None:
    all_in = packets.resolve(f"all-in-{device}", language, environ={"HPCAGENT_BENCH_CPF_CACHE": "/views/dropin"})
    playbook = packets.resolve(f"perf-playbook-{device}", language)
    assert set(all_in.skills) == (set(playbook.skills) | set(packets.resolve("lang", language).skills) | {cpf})
    assert all_in.env == (("CPF_DROPIN_DIR", "/views/dropin"), ("HPCAGENT_BENCH_CPF_VERIFIED_ONLY", "true"))
    assert packets.canonical(f"lang;perf-playbook-{device};{cpf}") == f"all-in-{device}"


def test_canonical_names_a_composite_only_when_the_spec_stages_the_same_pages() -> None:
    """``profiling`` as a token is the frozen bundle, rocprof and nsys included: spelling a playbook's
    pages with it stages two tracer pages the playbook does not carry, so it is not that playbook."""
    assert packets.canonical("divide-and-conquer;profiling;opt-reports") == "divide-and-conquer+opt-reports+profiling"
    # A composite is its skills AND its sub-packets: without the profiling page the tracers alone are not the
    # profiling bundle, and the bundle's members plus its own name do not collapse to ``profiling``.
    assert packets.canonical("opt-reports;rocprof") == "opt-reports+rocprof"
    assert packets.canonical("nsys;opt-reports;rocprof") == "nsys+opt-reports+rocprof"
    assert (
        packets.canonical("divide-and-conquer;nsys;opt-reports;profiling;rocprof")
        == "divide-and-conquer+nsys+opt-reports+profiling+rocprof"
    )


@pytest.mark.parametrize("spec", ["profiling", "all-in", "divide-and-conquer;profiling", "lang;all-in"])
def test_a_spec_reaching_a_frozen_packet_takes_no_new_submission(spec: str) -> None:
    with pytest.raises(ValueError, match=r"takes no new submissions.*profiling"):
        packets.refuse_frozen(spec)


def test_a_frozen_packet_still_resolves_for_the_records_that_hold_it() -> None:
    packets.refuse_frozen("perf-playbook-cpu;lang")
    assert packets.resolve("all-in", "c", environ={}, fill=False).key == "all-in"


def test_resolve_an_ad_hoc_semicolon_list() -> None:
    resolved = packets.resolve("rocprof;nsys", "c")
    assert resolved.skills == ("nsys", "rocprof")
    assert resolved.key == "nsys+rocprof"
    assert resolved.label == "ROCm Profiler + Nsight Systems"
    spaced = packets.resolve(" rocprof ; rocprof ; nsys ", "c")
    assert spaced.skills == ("nsys", "rocprof")
    assert spaced.parts == ("nsys", "rocprof")
    assert packets.label("divide-and-conquer;profiling") == "Divide and Conquer + Profiling Tools"


def test_resolve_an_unknown_token_raises_naming_it() -> None:
    with pytest.raises(ValueError, match="nosuchpacket"):
        packets.resolve("nosuchpacket", "c")
    with pytest.raises(ValueError, match="HPCAGENT_BENCH_CPF_CACHE"):
        packets.resolve("cpf-tool-cpu", "c", environ={})


def test_resolve_conflicting_env_between_two_packets_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    real = tags.registry()
    conflicting = dict(real.packet_defs)
    conflicting["fake-a"] = tags.PacketDef(name="Fake A", skills=(), packets=(), env=(("X", "1"),))
    conflicting["fake-b"] = tags.PacketDef(name="Fake B", skills=(), packets=(), env=(("X", "2"),))
    fake_registry = dataclasses.replace(real, packet_defs=conflicting)
    monkeypatch.setattr(tags, "registry", lambda: fake_registry)
    with pytest.raises(ValueError, match="X"):
        packets.resolve("fake-a;fake-b", "c")


@pytest.mark.parametrize(
    "spec",
    ["lang-skills", "no-score-tool", "cpf-tool-cpu", "cpf-gpu", "repo", "lang", "perf-playbook-cpu", "all-in-amd"],
)
def test_canonical_of_a_registered_key_is_itself(spec: str) -> None:
    assert packets.canonical(spec) == spec


@pytest.mark.parametrize(("old", "key"), sorted(tags.OLD_PACKET_KEYS.items()))
def test_a_retired_cpf_key_reads_as_its_current_one_but_no_launch_takes_it(old: str, key: str) -> None:
    """A database row keeps its series under the renamed key; a new launch must name the current key."""
    assert packets.canonical(old) == key
    assert packets.canonical(f"{old}+lang-skills") == packets.canonical(f"{key}+lang-skills")
    assert key in tags.registry().packet_defs
    with pytest.raises(ValueError, match=f"was renamed {key!r}"):
        packets.resolve(old, "c", environ={"HPCAGENT_BENCH_CPF_CACHE": "/views/cpf"})


def test_has_part_matches_a_bare_packet() -> None:
    assert packets.has_part("lang-skills", "lang-skills")
    assert not packets.has_part("cpf-tool-cpu", "lang-skills")
    assert not packets.has_part("", "lang-skills")
    # ``llrsingle`` records ``lang-skills+no-score-tool``: a reader finds a part inside the composite.
    assert packets.has_part("lang-skills+no-score-tool", "lang-skills")
    assert packets.has_part("lang-skills+no-score-tool", "no-score-tool")
    assert not packets.has_part("lang-skills+no-score-tool", "cpf-tool-cpu")
    assert not packets.has_part("no-score-tool", "lang-skills")


@pytest.mark.parametrize("key", tags.order("packets"))
def test_every_registered_key_is_coloured_by_the_part_this_module_leads_it_with(key: str) -> None:
    """This module owns the IDENTITY half of the colour rule -- which part a spec is led by -- and
    the palette owns the table. The two halves have to agree on every registered key."""
    parts = packets.spec_parts(key)
    expected = palette.control_color() if not parts else palette.color(packets.lead(parts))
    assert palette.color(key) == expected


def test_the_colour_of_an_ad_hoc_combination_is_deterministic() -> None:
    first = palette.color("mystery-tool;another-mystery")
    second = palette.color("another-mystery;mystery-tool")
    assert first == second


def test_an_unfilled_resolve_keeps_the_placeholder_templates_the_db_records() -> None:
    """fill=False is the packet's definition, not one launch: no environment is needed and every
    ${VAR} survives verbatim, including through a composition."""
    assert packets.resolve("cpf-cpu", "c", environ={}, fill=False).env == (
        ("CPF_DROPIN_DIR", "${HPCAGENT_BENCH_CPF_CACHE}"),
        ("HPCAGENT_BENCH_CPF_VERIFIED_ONLY", "true"),
    )
    all_in = packets.resolve("all-in", "c", environ={}, fill=False)
    assert dict(all_in.env) == {
        "CPF_DROPIN_DIR": "${HPCAGENT_BENCH_CPF_CACHE}",
        "HPCAGENT_BENCH_CPF_VERIFIED_ONLY": "true",
    }
    assert {"lang-c", "openmp-c", "divide-and-conquer", "profiling", "rocprof", "nsys", "opt-reports"} <= set(
        all_in.skills
    )
