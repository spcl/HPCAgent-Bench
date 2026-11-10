# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A knowledge base (``agent.knowledge``): a directory whose manifest states name, version, licence, source and the
hash of its documents, searched and read through two tools so a large reference never enters the prompt, and named
with its hash in every episode record so two setups that differ in their documents are told apart."""

import json
import pathlib
from types import ModuleType

import pytest
from hpcagent_agent import knowledge
from hpcagent_agent.tools import knowledge_read, knowledge_search, mcp_server, registry

from hpcagent_bench import study_tags
from hpcagent_bench.harness import episodes
from tests.fresh_module import fresh

#: A three-document sample written for this test, in the shape of an instruction-set reference.
ARITHMETIC = """\
# Integer arithmetic

## add
add.s32 d, a, b
Adds a and b and stores the sum in d. Operands are 32-bit signed integers.

## mad.lo.s32
mad.lo.s32 d, a, b, c
Multiplies a and b, keeps the low 32 bits of the product and adds c. One instruction replaces a
mul.lo followed by an add when indexing a row-major array.
"""
CONVERSION = """\
# Conversion

## cvt.rn.f32.f64
cvt.rn.f32.f64 d, a
Converts a 64-bit float to a 32-bit float, rounding to nearest even.
"""
LONG = "\n".join(f"line {number}: filler about register allocation" for number in range(1, 301)) + "\n"

KB_TOOLS = {registry.PACKET_TOOLS_ENV: "knowledge_search,knowledge_read"}


def write_base(root: pathlib.Path, **fields: object) -> pathlib.Path:
    """A valid base under ``root``; ``fields`` replace or (``None``) remove manifest keys."""
    (root / "isa").mkdir(parents=True)
    (root / "isa" / "arithmetic.md").write_text(ARITHMETIC, encoding="utf-8")
    (root / "isa" / "conversion.md").write_text(CONVERSION, encoding="utf-8")
    (root / "notes.txt").write_text(LONG, encoding="utf-8")
    (root / "figure.bin").write_bytes(b"\xff\xfe\x00binary")
    manifest: dict[str, object] = {
        "name": "sample-isa",
        "version": "1.0",
        "licence": "CC0-1.0",
        "source": "written for the test suite",
        "content_hash": knowledge.content_hash(root),
        "description": "A sample instruction reference.",
        **fields,
    }
    lines = [f"{key} = {json.dumps(value)}" for key, value in manifest.items() if value is not None]
    (root / knowledge.MANIFEST_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


@pytest.fixture(name="base_dir")
def base_dir_fixture(tmp_path: pathlib.Path) -> pathlib.Path:
    knowledge.load.cache_clear()
    return write_base(tmp_path / "sample-isa")


def test_a_valid_base_loads_with_its_manifest_and_its_text_documents(base_dir: pathlib.Path) -> None:
    base = knowledge.load(base_dir)
    assert base.manifest.name == "sample-isa"
    assert base.manifest.licence == "CC0-1.0"
    assert base.manifest.redistributable is False, "publishing documents is opt-in"
    # The binary file is hashed but not searchable or readable; the manifest is neither.
    assert base.readable == {"isa/arithmetic.md", "isa/conversion.md", "notes.txt"}


@pytest.mark.parametrize("missing", ["name", "version", "licence", "source", "content_hash"])
def test_a_manifest_missing_a_required_field_is_refused(tmp_path: pathlib.Path, missing: str) -> None:
    root = write_base(tmp_path / "kb", **{missing: None})
    with pytest.raises(knowledge.KnowledgeError, match=missing):
        knowledge.load(root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "Sample ISA"),
        ("version", ""),
        ("licence", ""),
        ("content_hash", "sha256:abc"),
        ("content_hash", "md5:" + "0" * 64),
        ("author", "someone"),
    ],
)
def test_a_malformed_or_unknown_manifest_field_is_refused(tmp_path: pathlib.Path, field: str, value: str) -> None:
    root = write_base(tmp_path / "kb", **{field: value})
    with pytest.raises(knowledge.KnowledgeError, match=field):
        knowledge.load(root)


def test_a_directory_without_a_manifest_is_not_a_knowledge_base(tmp_path: pathlib.Path) -> None:
    (tmp_path / "doc.md").write_text("text\n", encoding="utf-8")
    with pytest.raises(knowledge.KnowledgeError, match=knowledge.MANIFEST_NAME):
        knowledge.load(tmp_path)


def test_the_hash_names_the_documents_and_not_the_manifest(base_dir: pathlib.Path) -> None:
    """Editing, renaming, adding or removing a document changes the hash; rewording the manifest does not."""
    original = knowledge.content_hash(base_dir)
    assert (original[:7], len(original)) == ("sha256:", 7 + 64)
    manifest = base_dir / knowledge.MANIFEST_NAME
    manifest.write_text(manifest.read_text(encoding="utf-8") + "# a comment\n", encoding="utf-8")
    assert knowledge.content_hash(base_dir) == original

    page = base_dir / "isa" / "conversion.md"
    page.write_text(CONVERSION + "\n", encoding="utf-8")
    edited = knowledge.content_hash(base_dir)
    assert edited != original
    page.write_text(CONVERSION, encoding="utf-8")
    assert knowledge.content_hash(base_dir) == original

    page.rename(base_dir / "isa" / "convert.md")
    assert knowledge.content_hash(base_dir) not in (original, edited)
    (base_dir / "isa" / "convert.md").rename(page)
    (base_dir / "extra.md").write_text("", encoding="utf-8")
    assert knowledge.content_hash(base_dir) != original, "an added empty file is still a different base"


def test_moving_bytes_between_two_files_changes_the_hash(tmp_path: pathlib.Path) -> None:
    """Each file's size is hashed with it, so the same bytes split differently are a different base."""
    for name, (first, second) in {"a": ("ab", "c"), "b": ("a", "bc")}.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "1.md").write_text(first, encoding="utf-8")
        (tmp_path / name / "2.md").write_text(second, encoding="utf-8")
    assert knowledge.content_hash(tmp_path / "a") != knowledge.content_hash(tmp_path / "b")


def test_a_base_whose_documents_changed_under_its_manifest_is_refused(base_dir: pathlib.Path) -> None:
    (base_dir / "isa" / "arithmetic.md").write_text(ARITHMETIC + "edited\n", encoding="utf-8")
    with pytest.raises(knowledge.KnowledgeError, match="update version and content_hash together"):
        knowledge.load(base_dir)


def test_a_symbolic_link_is_refused_because_the_hash_would_not_cover_its_target(
    base_dir: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    outside = tmp_path / "outside.md"
    outside.write_text("not part of the base\n", encoding="utf-8")
    (base_dir / "isa" / "linked.md").symlink_to(outside)
    with pytest.raises(knowledge.KnowledgeError, match="symbolic link"):
        knowledge.content_hash(base_dir)


def test_search_finds_an_instruction_by_its_full_spelling_and_by_a_part(base_dir: pathlib.Path) -> None:
    base = knowledge.load(base_dir)
    full = knowledge.search(base, "mad.lo.s32")
    assert full[0].passage.document == "isa/arithmetic.md"
    assert "mad.lo.s32" in full[0].passage.text
    assert knowledge.search(base, "MAD")[0].passage.document == "isa/arithmetic.md"
    convert = knowledge.search(base, "convert float rounding nearest")
    assert convert[0].passage.document == "isa/conversion.md"
    assert (convert[0].passage.first_line, convert[0].passage.last_line) == (1, 5)
    assert knowledge.search(base, "zzz_no_such_term") == []


def test_search_is_bounded_and_deterministic(base_dir: pathlib.Path) -> None:
    base = knowledge.load(base_dir)
    many = knowledge.search(base, "register allocation", limit=1000)
    assert len(many) == knowledge.MAX_RESULTS
    assert [hit.passage for hit in many] == [hit.passage for hit in knowledge.search(base, "register allocation", 1000)]
    assert all(len(hit.to_json()["snippet"]) <= knowledge.SNIPPET_CHARS + len(" ...") for hit in many)
    assert len(knowledge.search(base, "register allocation", limit=2)) == 2


def test_a_long_document_is_cut_into_overlapping_passages_that_cover_every_line() -> None:
    cut = knowledge.passages("notes.txt", LONG)
    assert (cut[0].first_line, cut[0].last_line) == (1, knowledge.PASSAGE_LINES)
    assert cut[1].first_line == knowledge.PASSAGE_STEP + 1
    assert cut[-1].last_line == 300
    covered = {line for passage in cut for line in range(passage.first_line, passage.last_line + 1)}
    assert covered == set(range(1, 301))


def test_read_returns_a_bounded_range_and_says_where_it_sits(base_dir: pathlib.Path) -> None:
    base = knowledge.load(base_dir)
    page = knowledge.read(base, "notes.txt", first_line=10, lines=3)
    assert page == {
        "document": "notes.txt",
        "first_line": 10,
        "last_line": 12,
        "total_lines": 300,
        "text": "\n".join(f"line {number}: filler about register allocation" for number in (10, 11, 12)),
    }
    assert knowledge.read(base, "notes.txt", lines=10_000)["last_line"] == knowledge.MAX_READ_LINES


@pytest.mark.parametrize("document", ["../outside.md", "/etc/passwd", knowledge.MANIFEST_NAME, "figure.bin", "isa"])
def test_read_opens_only_the_bases_own_text_documents(base_dir: pathlib.Path, document: str) -> None:
    (base_dir.parent / "outside.md").write_text("outside\n", encoding="utf-8")
    with pytest.raises(knowledge.KnowledgeError, match="no document"):
        knowledge.read(knowledge.load(base_dir), document)


def test_the_tools_answer_from_the_setups_base_and_log_what_they_returned(
    base_dir: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    usage = tmp_path / "knowledge-usage.jsonl"
    monkeypatch.setenv(knowledge.KEY, str(base_dir))
    monkeypatch.setenv(knowledge.USAGE_ENV, str(usage))
    found = knowledge_search.run({"query": "cvt.rn.f32.f64", "limit": 1})
    assert found["ok"] is True
    assert (found["knowledge_base"], found["version"]) == ("sample-isa", "1.0")
    hit = found["hits"][0]
    assert hit["document"] == "isa/conversion.md"
    page = knowledge_read.run({"document": hit["document"], "first_line": hit["first_line"], "lines": 5})
    assert page["ok"] is True
    assert "rounding to nearest even" in page["text"]
    refused = knowledge_read.run({"document": "../outside.md"})
    assert refused["ok"] is False

    totals = knowledge.usage_totals(usage)
    assert totals == {"calls": 2, "returned_chars": len(hit["snippet"]) + len(page["text"])}
    record = knowledge.episode_record(knowledge.load(base_dir), usage)
    assert record == {
        "name": "sample-isa",
        "version": "1.0",
        "licence": "CC0-1.0",
        "content_hash": knowledge.content_hash(base_dir),
        **totals,
    }
    assert episodes.KnowledgeRecord.model_validate(record).content_hash == record["content_hash"]


def test_the_tools_say_so_when_the_run_has_no_base_or_a_broken_one(
    base_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(knowledge.KEY, raising=False)
    assert "no knowledge base" in knowledge_search.run({"query": "add"})["error"]
    assert "no knowledge base" in knowledge_read.run({"document": "notes.txt"})["error"]
    (base_dir / "notes.txt").write_text("changed\n", encoding="utf-8")
    monkeypatch.setenv(knowledge.KEY, str(base_dir))
    assert "content_hash" in knowledge_search.run({"query": "add"})["error"]
    assert "content_hash" in knowledge_read.run({"document": "notes.txt"})["error"]


def test_the_knowledge_packet_serves_both_tools_under_every_network_policy(base_dir: pathlib.Path) -> None:
    """The tools are local and say so (``open_world: false``), so the default isolating policy keeps them; the
    search tool's description names the setup's base, and no setup without the packet sees either tool."""
    assert set(study_tags.registry().packet_defs["knowledge"].tools) == {"knowledge_search", "knowledge_read"}
    bare = mcp_server.describe({})
    assert not {"knowledge_search", "knowledge_read"} & set(bare["served_tools"])
    assert "knowledge_search" not in bare["prompt"]
    for policy in ("none", "allowlist", "open"):
        environ = {**KB_TOOLS, knowledge.KEY: str(base_dir), "HPCAGENT_BENCH_AGENT_NETWORK": policy}
        described = mcp_server.describe(environ)
        assert {"knowledge_search", "knowledge_read"} <= set(described["served_tools"]), policy
        assert described["withheld_tools"] == [], policy
        assert "`knowledge_search`" in described["prompt"]
    served = mcp_server.tools({**KB_TOOLS, knowledge.KEY: str(base_dir)})
    assert "`sample-isa` (version 1.0): A sample instruction reference." in served["knowledge_search"].description
    assert served["knowledge_search"].hints.annotations() == {
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    }


@pytest.fixture(name="driver")
def driver_fixture(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    for name in (knowledge.KEY, registry.PACKET_TOOLS_ENV, "RUN_DIR", "HPCAGENT_BENCH_AGENT_NETWORK"):
        monkeypatch.delenv(name, raising=False)
    return fresh("agent_driver")


def test_the_driver_takes_a_base_and_its_packet_together_or_not_at_all(
    driver: ModuleType, base_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A base no tool serves, or tools with no base, would put an intervention in the record the agent never had."""
    assert driver.setup_knowledge() is None
    monkeypatch.setenv(knowledge.KEY, str(base_dir))
    with pytest.raises(SystemExit, match="lacks `knowledge`"):
        driver.setup_knowledge()
    monkeypatch.setenv(registry.PACKET_TOOLS_ENV, KB_TOOLS[registry.PACKET_TOOLS_ENV])
    driver.tool_registry.cache_clear()
    assert driver.setup_knowledge().manifest.name == "sample-isa"
    monkeypatch.delenv(knowledge.KEY)
    with pytest.raises(SystemExit, match=r"agent\.knowledge"):
        driver.setup_knowledge()
    (base_dir / "notes.txt").write_text("changed\n", encoding="utf-8")
    knowledge.load.cache_clear()
    monkeypatch.setenv(knowledge.KEY, str(base_dir))
    with pytest.raises(SystemExit, match="content_hash"):
        driver.setup_knowledge()
    driver.tool_registry.cache_clear()


def test_the_episode_record_names_the_base_and_the_ingest_model_reads_it(
    driver: ModuleType, base_dir: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(knowledge.KEY, str(base_dir))
    workdir = tmp_path / "worker"
    workdir.mkdir()
    declared = driver.mcp_setup_env(workdir)
    assert declared[knowledge.KEY] == str(base_dir)
    assert declared[knowledge.USAGE_ENV] == str(workdir / driver.KNOWLEDGE_USAGE_NAME)
    knowledge.log_usage("knowledge_search", 120, declared)
    network = driver.open_worker_network(workdir, {"judge": "http://j0:8800"})
    driver.write_cost_record(
        workdir / "tokens.json",
        {"id": 1, "kernel": "k"},
        0,
        0,
        10,
        1,
        "success",
        interventions=driver.episode_interventions(network, knowledge.load(base_dir), workdir),
    )
    record = episodes.read_record(workdir / "tokens.json")
    assert record is not None, "the ingest model refuses the record the driver writes"
    assert record.knowledge == episodes.KnowledgeRecord(
        name="sample-isa",
        version="1.0",
        licence="CC0-1.0",
        content_hash=knowledge.content_hash(base_dir),
        calls=1,
        returned_chars=120,
    )
    assert driver.KNOWLEDGE_USAGE_NAME in driver.RELAUNCH_KEEPS, "a relaunch must not forget the calls made so far"


def test_the_command_line_prints_the_hash_and_checks_a_base(
    base_dir: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert knowledge.main(["hash", str(base_dir)]) == 0
    assert capsys.readouterr().out.strip() == knowledge.content_hash(base_dir)
    assert knowledge.main(["check", str(base_dir)]) == 0
    assert "sample-isa 1.0 (CC0-1.0)" in capsys.readouterr().out
    (base_dir / "notes.txt").write_text("changed\n", encoding="utf-8")
    knowledge.load.cache_clear()
    assert knowledge.main(["check", str(base_dir)]) == 1
    assert knowledge.main(["stage", str(base_dir)]) == 2
