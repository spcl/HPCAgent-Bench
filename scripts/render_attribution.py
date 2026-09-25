#!/usr/bin/env python
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render CONTRIBUTORS.md and NOTICE from the kernels' provenance lines.

Every kernel manifest carries one provenance line right below its header, a YAML flow mapping in a
comment (the manifest schema has no provenance field)::

    # provenance: {kind: derived, upstream: pyfai, via: [npbench]}

``third_party/upstreams.yaml`` describes every project, paper and contributor those lines name, and
documents the vocabulary. The two rendered files are derived data: edit a manifest or the registry,
then run ``--write``. Without it the script only checks, which is what the test suite does.

    python scripts/render_attribution.py            # exit 1 on a bad line or a stale file
    python scripts/render_attribution.py --write    # rewrite CONTRIBUTORS.md and NOTICE
"""

import argparse
import dataclasses
import pathlib
import sys
from collections.abc import Iterable
from enum import StrEnum

import yaml

from hpcagent_bench import paths

REGISTRY = paths.ROOT / "third_party" / "upstreams.yaml"
LICENSE_TEXTS = paths.ROOT / "third_party" / "licenses"
CONTRIBUTORS = paths.ROOT / "CONTRIBUTORS.md"
NOTICE = paths.ROOT / "NOTICE"

#: The comment prefix that marks the provenance line. Matched with the brace so that prose
#: comments starting with the word "provenance" never count.
MARKER = "# provenance: {"
KEYS = frozenset({"kind", "upstream", "via", "contributor", "detail"})
#: Past this many kernels an upstream's NOTICE section gives the count, not the list.
NOTICE_LIST_LIMIT = 40
RULE = "=" * 100


class Kind(StrEnum):
    """How a kernel relates to its upstream."""

    DERIVED = "derived"
    ALGORITHM = "algorithm"
    ORIGINAL = "original"


@dataclasses.dataclass(frozen=True, slots=True)
class Provenance:
    """One manifest's provenance line, parsed."""

    kernel: str
    kind: Kind
    upstreams: tuple[str, ...]
    via: tuple[str, ...]
    contributor: str | None
    detail: str | None


def as_keys(value: object) -> tuple[str, ...]:
    """A registry key or a list of them, as a tuple (absent is empty)."""
    if value is None:
        return ()
    items = value if isinstance(value, list) else [value]
    return tuple(str(item) for item in items)


def parse_line(line: str, kernel: str) -> Provenance:
    """Parse one provenance line; ``ValueError`` names the kernel and the problem."""
    raw = yaml.safe_load(line.removeprefix("# provenance: "))
    if not isinstance(raw, dict):
        raise ValueError(f"{kernel}: provenance is not a mapping")
    unknown = set(raw) - KEYS
    if unknown:
        raise ValueError(f"{kernel}: unknown provenance key(s) {sorted(unknown)}")
    kind = Kind(raw.get("kind"))
    detail = raw.get("detail")
    return Provenance(
        kernel=kernel,
        kind=kind,
        upstreams=as_keys(raw.get("upstream")),
        via=as_keys(raw.get("via")),
        contributor=None if raw.get("contributor") is None else str(raw["contributor"]),
        detail=None if detail is None else str(detail),
    )


def read_manifest(path: pathlib.Path) -> Provenance:
    """The provenance of the manifest at ``path``; exactly one line must carry it."""
    lines = [line for line in path.read_text().splitlines() if line.startswith(MARKER)]
    if len(lines) != 1:
        raise ValueError(f"{path.stem}: expected one provenance line, found {len(lines)}")
    return parse_line(lines[0], path.stem)


def manifests(root: pathlib.Path) -> list[pathlib.Path]:
    """Every kernel manifest under ``root``, the same predicate the kernel registry walks with."""
    return sorted((p for p in root.rglob("*.yaml") if not p.stem.startswith("_")), key=lambda p: p.stem)


def load_registry(path: pathlib.Path = REGISTRY) -> dict[str, dict[str, dict[str, str]]]:
    """The parsed registry: ``contributors``, ``other`` and ``upstreams``, each keyed by name."""
    return yaml.safe_load(path.read_text())


def problems(entries: list[Provenance], registry: dict[str, dict[str, dict[str, str]]]) -> list[str]:
    """Every inconsistency between the provenance lines and the registry."""
    upstreams, contributors = registry["upstreams"], registry["contributors"]
    out: list[str] = []
    used: set[str] = set()
    for entry in entries:
        named = (*entry.upstreams, *entry.via)
        used.update(named)
        out.extend(f"{entry.kernel}: unknown upstream {key!r}" for key in named if key not in upstreams)
        if entry.contributor is not None and entry.contributor not in contributors:
            out.append(f"{entry.kernel}: unknown contributor {entry.contributor!r}")
        if (entry.kind is Kind.ORIGINAL) == bool(entry.upstreams):
            out.append(
                f"{entry.kernel}: kind {entry.kind} {'must not' if entry.upstreams else 'must'} name an upstream"
            )
    out.extend(f"registry: upstream {key!r} is named by no kernel" for key in sorted(set(upstreams) - used))
    return out


def by_upstream(entries: Iterable[Provenance], kind: Kind, include_via: bool) -> dict[str, list[Provenance]]:
    """Kernels of ``kind`` grouped under every upstream they name (and intermediary, if asked)."""
    groups: dict[str, list[Provenance]] = {}
    for entry in entries:
        if entry.kind is not kind:
            continue
        for key in (*entry.upstreams, *(entry.via if include_via else ())):
            groups.setdefault(key, []).append(entry)
    return groups


def kernel_label(entry: Provenance, upstream: str, registry: dict[str, dict[str, dict[str, str]]]) -> str:
    """``kernel`` plus what it came from: the detail, and the intermediaries under its origin."""
    parts = [entry.detail] if entry.detail else []
    if entry.via and upstream in entry.upstreams:
        parts.append("via " + ", ".join(registry["upstreams"][key]["name"] for key in entry.via))
    elif upstream in entry.via:
        parts.append("from " + ", ".join(registry["upstreams"][key]["name"] for key in entry.upstreams))
    return f"`{entry.kernel}`" + (f" ({'; '.join(parts)})" if parts else "")


def upstream_block(key: str, meta: dict[str, str], kernels: list[Provenance], registry: dict) -> list[str]:
    """One CONTRIBUTORS.md section: the project, its license and citation, and its kernels."""
    lines = [f"### {meta['name']}", "", f"- Project: <{meta['url']}>", f"- License: {meta['license']}"]
    lines += [
        f"- {label}: {meta[field]}" for field, label in (("citation", "Cite"), ("note", "Note")) if meta.get(field)
    ]
    labels = ", ".join(kernel_label(entry, key, registry) for entry in kernels)
    return [*lines, f"- Kernels ({len(kernels)}): {labels}", ""]


def render_contributors(entries: list[Provenance], registry: dict) -> str:
    """CONTRIBUTORS.md: who contributed, then every kernel under the upstream it credits."""
    upstreams = registry["upstreams"]
    out = [
        "# Contributors",
        "",
        "<!-- Generated by scripts/render_attribution.py from the kernel manifests' provenance lines",
        "     and third_party/upstreams.yaml; do not edit by hand. -->",
        "",
        "HPCAgent-Bench is developed by [SPCL @ ETH Zurich](https://spcl.inf.ethz.ch/) and grew out of",
        "[NPBench](https://github.com/spcl/npbench). License notices for derived code are in [NOTICE](NOTICE).",
        "",
        "## Contributed kernels",
        "",
    ]
    for key, meta in registry["contributors"].items():
        mine = [f"`{e.kernel}`" for e in entries if e.contributor == key]
        out.append(f"- **{meta['name']}** {meta['note']} Kernels ({len(mine)}): {', '.join(mine)}")
    sections = (
        (
            Kind.DERIVED,
            "Kernels derived from upstream code",
            "Ported, transcribed or adapted from the project's source.",
        ),
        (
            Kind.ALGORITHM,
            "Kernels written from a published algorithm",
            "No upstream code is included; the source is cited.",
        ),
    )
    for kind, title, blurb in sections:
        groups = by_upstream(entries, kind, include_via=kind is Kind.DERIVED)
        out += ["", f"## {title}", "", blurb, "", "| Upstream | License | Kernels |", "|---|---|---:|"]
        order = sorted(groups, key=lambda key: upstreams[key]["name"].lower())
        out += [f"| {upstreams[key]['name']} | {upstreams[key]['license']} | {len(groups[key])} |" for key in order]
        out.append("")
        for key in order:
            out += upstream_block(key, upstreams[key], groups[key], registry)
    unclear = [
        f"- `{e.kernel}` ({', '.join(upstreams[key]['name'] for key in keys)})"
        for e in entries
        if e.kind is Kind.DERIVED
        and (keys := [key for key in (*e.upstreams, *e.via) if upstreams[key]["license"] == "NOASSERTION"])
    ]
    out += [
        "## License unclear",
        "",
        "These kernels derive from code whose upstream states no license.",
        "",
        *unclear,
        "",
    ]
    originals = [f"`{e.kernel}`" for e in entries if e.kind is Kind.ORIGINAL]
    out += ["## Original kernels", "", f"Written for HPCAgent-Bench ({len(originals)}): {', '.join(originals)}", ""]
    return "\n".join(line.rstrip() for line in out).rstrip("\n") + "\n"


def license_ids(license_expr: str) -> list[str]:
    """The SPDX ids of a ``" AND "``-joined license expression."""
    return [part.strip() for part in license_expr.split(" AND ")]


def notice_section(meta: dict[str, str], covered: str) -> list[str]:
    """One NOTICE section: project, license, copyright and what it covers."""
    lines = [RULE, meta["name"], RULE, ""]
    lines += [
        f"{label}: {meta[field]}" for field, label in (("url", "Project"), ("license", "License")) if field in meta
    ]
    if meta.get("copyright"):
        lines.append(meta["copyright"])
    if meta["license"] == "NOASSERTION":
        lines.append("No license is stated upstream.")
    lines += [
        f"License text: https://spdx.org/licenses/{spdx}.html"
        for spdx in license_ids(meta["license"])
        if spdx != "NOASSERTION" and not (LICENSE_TEXTS / f"{spdx}.txt").is_file()
    ]
    lines.append(covered)
    lines += [meta[field] for field in ("citation", "note") if meta.get(field)]
    return [*lines, ""]


def covered_kernels(kernels: list[Provenance]) -> str:
    """The kernel list a NOTICE section covers (the count alone past NOTICE_LIST_LIMIT)."""
    if len(kernels) > NOTICE_LIST_LIMIT:
        return f"Covers {len(kernels)} kernels, listed in CONTRIBUTORS.md."
    return "Covers: " + ", ".join(entry.kernel for entry in kernels)


def render_notice(entries: list[Provenance], registry: dict) -> str:
    """NOTICE: the project's own notice, every upstream whose code is included, and the license
    texts those upstreams require to travel with it."""
    upstreams = registry["upstreams"]
    derived = by_upstream(entries, Kind.DERIVED, include_via=True)
    out = [
        "HPCAgent-Bench",
        "Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.",
        "",
        "This product is licensed under the GNU General Public License v3.0 or later (see LICENSE).",
        "",
        "Generated by scripts/render_attribution.py from the kernel manifests and",
        "third_party/upstreams.yaml; do not edit by hand. CONTRIBUTORS.md lists every kernel by source.",
        "",
        "The kernels below contain code ported, transcribed or adapted from the named projects. Each",
        "copyright line applies with the license text of the same id at the end of this file; code",
        "under a GPL-family license is conveyed as part of this GPL-3.0 work.",
        "",
    ]
    used: dict[str, list[str]] = {}
    for key in sorted(derived, key=lambda k: upstreams[k]["name"].lower()):
        meta = upstreams[key]
        out += notice_section(meta, covered_kernels(derived[key]))
        for spdx in license_ids(meta["license"]):
            used.setdefault(spdx, []).append(meta["name"])
    for meta in registry["other"].values():
        out += notice_section(meta, f"Covers: {meta['files']}")
        for spdx in license_ids(meta["license"]):
            used.setdefault(spdx, []).append(meta["name"])
    out += [RULE, "License texts", RULE, ""]
    for spdx in sorted(used):
        text = LICENSE_TEXTS / f"{spdx}.txt"
        if text.is_file():
            out += [f"--- {spdx} (applies to: {', '.join(used[spdx])})", "", text.read_text().rstrip("\n"), ""]
    return "\n".join(line.rstrip() for line in out).rstrip("\n") + "\n"


def collect(root: pathlib.Path = paths.BENCHMARKS) -> tuple[list[Provenance], list[str]]:
    """Every manifest's provenance and every problem found reading them."""
    entries: list[Provenance] = []
    errors: list[str] = []
    for path in manifests(root):
        try:
            entries.append(read_manifest(path))
        except ValueError as exc:
            errors.append(str(exc))
    return entries, errors


def main(argv: list[str] | None = None) -> int:
    """Check (default) or rewrite the rendered attribution files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="rewrite CONTRIBUTORS.md and NOTICE")
    args = parser.parse_args(argv)
    registry = load_registry()
    entries, errors = collect()
    errors += problems(entries, registry)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    rendered = {CONTRIBUTORS: render_contributors(entries, registry), NOTICE: render_notice(entries, registry)}
    stale = [path for path, text in rendered.items() if not path.is_file() or path.read_text() != text]
    if args.write:
        for path in stale:
            path.write_text(rendered[path])
        return 0
    for path in stale:
        print(f"{path.name} is stale: run python scripts/render_attribution.py --write", file=sys.stderr)
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
