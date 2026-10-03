# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Replace or disable any built-in prompt section from config or the environment.

Each template under ``harness/prompts/`` (and each ``tools/<tool>.md`` fragment) is a section with a key:
the path without its extension, ``sections/`` dropped and every other separator turned into ``_``
(``sections/build_flags.j2`` is ``build_flags``, ``lang/cpp.j2`` is ``lang_cpp``, ``tools/web-search.md``
is ``tools_web_search``). A section's value is ``off`` to render nothing, or a template name on the
search path, or a file path, to render that instead. ``prompt.sections.<key>`` in ``config.yaml`` and
``HPCAGENT_BENCH_PROMPT_SECTIONS_<KEY>`` in the environment set it, the environment winning."""

import functools
import pathlib
import re
from collections.abc import Mapping
from types import MappingProxyType

from hpcagent_bench import config
from hpcagent_bench.spec import as_block

__all__ = [
    "ENV_PREFIX",
    "OFF_WORDS",
    "ON_WORDS",
    "SectionValue",
    "aliases",
    "env_name",
    "key_of",
    "parse_value",
    "pick_sections",
    "section_templates",
]

PROMPTS_DIR = pathlib.Path(__file__).parent / "prompts"
#: Where ``tools/<tool>.md`` ships; the loader's last root, so an include of ``tools/x.md`` resolves here.
PACKAGE_DIR = pathlib.Path(__file__).parent.parent

#: The two top-level templates are chosen by ``prompt.template``, not one of the sections they include.
TOP_LEVEL = frozenset({"task.j2", "service_task.j2"})

#: ``off`` for a section: nothing renders and no blank line is left behind.
OFF_WORDS = frozenset({"off", "false", "no", "none", "disabled", "0"})
#: Spellings that keep the built-in section, so an environment can undo what ``config.yaml`` set.
ON_WORDS = frozenset({"on", "true", "yes", "default", "1", ""})

ENV_PREFIX = "HPCAGENT_BENCH_PROMPT_SECTIONS_"

#: ``False`` turns the section off, a string replaces it with that template (a name or a file path).
SectionValue = str | bool


def key_of(template: str) -> str:
    """The section key of template ``template`` (``sections/build_flags.j2`` -> ``build_flags``)."""
    stem = template.rsplit(".", 1)[0].removeprefix("sections/")
    return re.sub(r"[^a-z0-9]+", "_", stem.lower())


@functools.lru_cache(maxsize=1, typed=True)
def section_templates() -> Mapping[str, str]:
    """Section key -> the template name its ``{% include %}`` uses, for every built-in section."""
    names = [path.relative_to(PROMPTS_DIR).as_posix() for path in sorted(PROMPTS_DIR.rglob("*.j2"))]
    names = [name for name in names if name not in TOP_LEVEL]
    names += [f"tools/{path.name}" for path in sorted((PACKAGE_DIR / "tools").glob("*.md"))]
    return MappingProxyType({key_of(name): name for name in names})


def env_name(section: str) -> str:
    """The environment variable that sets ``section``."""
    return ENV_PREFIX + section.upper()


def parse_value(raw: object) -> SectionValue | None:
    """``raw`` as a section value, or None when it keeps the built-in section."""
    if raw is None or raw is True:
        return None
    if raw is False:
        return False
    text = str(raw).strip()
    if text.lower() in OFF_WORDS:
        return False
    if text.lower() in ON_WORDS:
        return None
    return text


def pick_sections(given: Mapping[str, SectionValue] | None = None) -> tuple[tuple[str, SectionValue], ...]:
    """Every section that is not the built-in one, as sorted ``(key, value)`` pairs.

    Precedence, strongest first: ``given`` (a caller or a variant), the section's environment variable,
    then ``prompt.sections`` (the config block, which a ``HPCAGENT_BENCH_PROMPT_SECTIONS`` JSON object or
    a runtime override replaces whole). An unknown key is an error that lists the known ones."""
    known = section_templates()
    block = as_block(config.get("prompt.sections", {}))
    chosen: dict[str, SectionValue] = {}
    for name in sorted({*known, *block, *(given or {})}):
        if given is not None and name in given:
            raw: object = given[name]
        else:
            env = config.env_value(env_name(name))
            raw = config.coerce(env) if env is not None else block.get(name)
        value = parse_value(raw)
        if value is not None:
            chosen[name] = value
    unknown = sorted(set(chosen) - set(known))
    if unknown:
        raise ValueError(f"unknown prompt section(s) {unknown}; known: {', '.join(sorted(known))}")
    return tuple(chosen.items())


def aliases(sections: tuple[tuple[str, SectionValue], ...]) -> dict[str, str | None]:
    """Template name -> its replacement (a template name or a file path), or None where it is off."""
    templates = section_templates()
    return {templates[key]: None if value is False else str(value) for key, value in sections}
