# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Docs must not drift from the implementation: mechanically checkable claims are checked."""

import argparse
import dataclasses
import pathlib
import re
import subprocess

import pytest

from hpcagent_bench import cli, config, modules, paths
from hpcagent_bench.harness.prompts import PROMPT_VARIANTS, PromptConfig

PROMPTS_DIR = pathlib.Path("hpcagent_bench/harness/prompts")
#: Every contributor doc: the README, docs/, experiments/, containers/ and the package's own docs. Skill pages
#: and kernel notes are agent prompt data, not docs.
DOCS = sorted(
    {
        pathlib.Path("README.md"),
        pathlib.Path("CONTRIBUTING.md"),
        pathlib.Path("hpcagent_bench/harness/README.md"),
        *pathlib.Path("docs").rglob("*.md"),
        *pathlib.Path("experiments").rglob("*.md"),
        *pathlib.Path("containers").rglob("*.md"),
        *pathlib.Path("hpcagent_bench/docs").glob("*.md"),
        *pathlib.Path("statistics").glob("*.md"),
    }
)

#: `prompt.*` names that are config keys but NOT PromptConfig fields.
NON_FIELD_PROMPT_KEYS = {"variants", "variant"}


def doc_text() -> list[tuple[pathlib.Path, str]]:
    return [(p, p.read_text()) for p in DOCS if p.exists()]


def test_prompt_config_fields_are_the_prompt_keys() -> None:
    """A PromptConfig field with no ``prompt.*`` key cannot be configured, and a key with no field is never read."""
    fields = {f.name for f in dataclasses.fields(PromptConfig)}
    assert fields | NON_FIELD_PROMPT_KEYS == set(config.Prompt.model_fields)


def test_docs_name_no_prompt_key_that_does_not_exist() -> None:
    """Catches a knob that was removed or renamed but is still documented."""
    fields = {f.name for f in dataclasses.fields(PromptConfig)} | NON_FIELD_PROMPT_KEYS
    stale = []
    for path, text in doc_text():
        # `prompt.md`, `prompt.txt` are files, not keys.
        stale.extend(
            f"{path}: prompt.{m.group(1)}"
            for m in re.finditer(r"`prompt\.(?!(?:md|txt)`)([a-z_]+)`", text)
            if m.group(1) not in fields
        )
    assert not stale, f"documented prompt.* keys that no longer exist: {stale}"


def test_docs_name_no_attempts_key_that_does_not_exist() -> None:
    fields = set(config.Attempts.model_fields)
    stale = []
    for path, text in doc_text():
        stale.extend(
            f"{path}: attempts.{m.group(1)}"
            for m in re.finditer(r"`attempts\.(?!jsonl`)([a-z_]+)`", text)
            if m.group(1) not in fields
        )
    assert not stale, f"documented attempts.* keys that no longer exist: {stale}"


def test_docs_name_no_variant_that_is_not_registered() -> None:
    """A `--variant X` in the docs must resolve: a built-in, a `task_var<N>.j2` discovery, or
    one the same doc declares in its own `prompt.variants` example."""
    stale = []
    for path, text in doc_text():
        # Names the doc itself declares under `variants:` are legitimate examples.
        declared = set(re.findall(r"^\s{4}([a-z_]\w*):\s*\{", text, re.MULTILINE))
        for m in re.finditer(r"--variant\s+([a-z_]+)", text):
            name = m.group(1)
            if name not in PROMPT_VARIANTS and name not in declared and not name.startswith("var"):
                stale.append(f"{path}: --variant {name}")
    assert not stale, f"documented variants that are not registered: {stale}"


def test_docs_name_no_template_that_does_not_exist() -> None:
    """A doc naming a template that was renamed/removed sends the reader to nothing."""
    known = {p.name for p in PROMPTS_DIR.rglob("*.j2")}
    stale = []
    for path, text in doc_text():
        for m in re.finditer(r"`([a-z_]+\.j2)`", text):
            name = m.group(1)
            if name not in known and "_var" not in name:
                stale.append(f"{path}: {name}")
    assert not stale, f"documented templates that do not exist: {stale}"


def test_every_internal_doc_link_resolves() -> None:
    """A renamed heading silently breaks inbound anchors from other docs."""

    def anchors(p: pathlib.Path) -> set[str]:
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


def test_docs_name_no_cli_command_that_does_not_exist() -> None:
    """`hpcagent-bench <command>` in a doc is a command the CLI has (`job submit` runs on the login node)."""
    (sub,) = (a for a in cli.build_parser()._actions if isinstance(a, argparse._SubParsersAction))
    stale = [
        f"{path}: {m.group(0)}"
        for path, text in doc_text()
        for m in re.finditer(r"\bhpcagent-bench ([a-z][a-z-]+)", text)
        if m.group(1) not in sub.choices
    ]
    assert not stale, f"documented commands the CLI does not have: {stale}"


def test_every_docs_page_has_an_inbound_link() -> None:
    """A page only the Sphinx toctree names is invisible on GitHub, where the README is the way in."""
    linked = {
        (path.parent / m.group(1)).resolve()
        for path, text in doc_text()
        for m in re.finditer(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text)
        if (path.parent / m.group(1)).resolve() != path.resolve()
    }
    orphans = [str(p) for p in pathlib.Path("docs").rglob("*.md") if p.name != "index.md" and p.resolve() not in linked]
    assert not orphans, f"docs pages no other doc links to: {orphans}"


def test_the_readme_has_a_students_section_naming_every_shipped_optimizer() -> None:
    """The student entry point sits near the top of the README and names what `hpcagent-bench agent` registers."""
    from hpcagent_bench.columns import FRAMEWORKS
    from hpcagent_bench.harness.optimizers import optimizer_registry

    readme = pathlib.Path("README.md").read_text()
    section = readme.partition("\n## Getting started\n")[2].partition("\n## ")[0]
    assert section, "README.md has no '## Getting started' section"
    assert readme.index("## Getting started") < readme.index("## Concepts")
    for target in ("docs/extending/optimizer.md", "docs/triton_authoring.md", "docs/index.md"):
        assert f"]({target}" in section, f"the students section does not link {target}"
    shipped = set(optimizer_registry()) - set(FRAMEWORKS.entries)
    missing = sorted(name for name in shipped if f"`{name}`" not in section)
    assert not missing, f"optimizers the students section does not name: {missing}"


def test_no_doc_raises_the_stack_to_unlimited() -> None:
    """`check_launch_env` wants the soft stack limit AT the hard one; `ulimit -s unlimited` fails for a non-root
    shell whose hard limit is finite, and aborts a `set -e` job script copied from the docs."""
    sources = [*doc_text(), *((p, p.read_text()) for p in pathlib.Path("hpcagent_bench/cluster").glob("*.sh"))]
    stale = [str(path) for path, text in sources if "ulimit -s unlimited" in text]
    assert not stale, f"`ulimit -s unlimited` instead of the hard limit in: {stale}"


def test_docs_never_pass_the_agent_name_as_an_option() -> None:
    """`agent` takes the optimizer as a positional; `--agent <name>` is no option of it."""
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["agent", "--agent", "noop"])
    python = [*pathlib.Path("hpcagent_bench").rglob("*.py"), *pathlib.Path("tests").glob("*.py")]
    sources = [*doc_text(), *((p, p.read_text()) for p in python if p.resolve() != pathlib.Path(__file__).resolve())]
    stale = [str(path) for path, text in sources if "hpcagent-bench agent --agent" in text]
    assert not stale, f"`hpcagent-bench agent --agent` in: {stale}"


def test_docs_name_only_tags_subcommands_that_exist() -> None:
    """`python -m hpcagent_bench.tags <command>` in a doc is a command the module defines."""
    known = {"resolve", "version", "sample"}
    stale = [
        f"{path}: {m.group(0)}"
        for path, text in doc_text()
        for m in re.finditer(r"hpcagent_bench\.tags (\S+)", text)
        if m.group(1).strip("`") not in known
    ]
    assert not stale, f"documented tags commands the module does not have: {stale}"


def test_docs_name_only_autogen_registries_that_exist() -> None:
    """An upper-case name a paragraph places in `hpcagent_bench/autogen.py` is one of its attributes."""
    import hpcagent_bench.autogen

    stale = [
        f"{path}: {name}"
        for path, text in doc_text()
        for paragraph in text.split("\n\n")
        if "hpcagent_bench/autogen.py" in paragraph
        for name in re.findall(r"`([A-Z][A-Z_]+)`", paragraph)
        if not modules.defines(hpcagent_bench.autogen, name)
    ]
    assert not stale, f"documented autogen names that do not exist: {stale}"


def test_no_doc_promises_a_skip_row_for_an_unsupported_precision() -> None:
    """`run-framework` never reads a column's `precisions`, so nothing writes a `skip` row for one (a `skip:<why>`
    status of another cause is not this claim)."""
    sources = [*doc_text(), *((p, p.read_text()) for p in pathlib.Path("hpcagent_bench").glob("*.py"))]
    stale = [str(path) for path, text in sources if re.search(r"records `?skip`?(?![:\w])|status=\"skip\"", text)]
    assert not stale, f"a `skip` row for an unsupported precision is promised in: {stale}"


#: An ``HPCAGENT_BENCH_*`` name as prose writes it; a trailing ``_*`` or ``_<...>`` makes it a prefix.
ENV_NAME = re.compile(r"(HPCAGENT_BENCH_[A-Z0-9_]*[A-Z0-9])(_\*|_<)?")


def test_every_documented_env_name_has_a_reader() -> None:
    """A documented ``HPCAGENT_BENCH_*`` variable is a config key's env name, a listed non-config name, or a
    name some tracked non-doc file uses; a documented prefix (``HPCAGENT_BENCH_MPI_GANG_*``) prefixes one. A
    renamed or removed variable otherwise leaves the docs telling an operator to set a name nothing reads."""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=paths.ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    used: set[str] = set()
    for relative in tracked:
        path = paths.ROOT / relative
        if relative.endswith(".md") or relative.startswith("third_party/") or not path.is_file():
            continue
        try:
            used |= {m[1] for m in ENV_NAME.finditer(path.read_text(encoding="utf-8"))}
        except UnicodeDecodeError:
            continue
    known = used | set(config.ENV_KEYS) | config.NON_CONFIG_ENV
    stale = sorted(
        {
            f"{doc}: {m[0]}"
            for doc in DOCS
            if (paths.ROOT / doc).exists()
            for m in ENV_NAME.finditer((paths.ROOT / doc).read_text(encoding="utf-8"))
            if not (
                any(name.startswith(f"{m[1]}_") for name in known)
                if m[2]
                else m[1] in known or m[1].startswith(config.NON_CONFIG_ENV_PREFIXES)
            )
        }
    )
    assert not stale, f"documented env names nothing reads: {stale}"
