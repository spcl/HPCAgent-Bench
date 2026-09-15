# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-loop-nest parallelism classification and the parallel-nest-fraction metric, from TEXT.

Pure source-text analysis: no dace, no SDFG, no subprocess. Works verbatim on baseline
sequential C, raw DaCe ``dace_cpu`` codegen and rendered CPF forms, in C or C++, since all
three emit the same OpenMP pragma vocabulary and brace-delimited loop bodies.

Sibling to :mod:`hpcagent_bench.metrics.parallelism`, which measures the same question on the
SDFG directly (map/reduce/scan/contract buckets, dace's own predicates). That module needs a
built SDFG; this one needs only a rendered file, so it is what reads a CPF view or a baseline
``.c`` reference on disk. The two can disagree on a kernel -- a loop this module sees as
CONTRACT_GUARDED from the emitted ``if``/``else`` shape is the same construct the SDFG module's
``guarded_fallback_loop_set`` names -- and a disagreement is reported, never silently resolved
by preferring one over the other.

Definition of a "loop nest": a maximal ``for`` statement not itself nested inside another
``for`` in the SAME lexical scope (a nest's own body may contain further nested ``for``
loops, which count toward its depth, not as separate nests). A call to a helper function
that itself loops is a SEPARATE nest at the call site's own scope -- this module does not
follow calls, so a parallel-for that dispatches into a sequential helper is reported as two
independent nests, not one. That is a known, documented simplification.

A nest is CONTRACT_GUARDED when it sits inside one branch of an ``if``/``else if``/``else``
chain whose branches disagree on whether their loop nests are parallel -- the
"if (contract holds) parallel else sequential" shape. The guarded branch keeps its own
parallel pragma info (reduction, simd, ...); only its ``kind`` changes.

The recommended headline metric, ``parallel_nest_fraction``, counts ONLY unconditionally
parallel nests in the numerator. Contract-guarded and simd-only nests are reported as
separate columns, never silently folded into "parallel" or "sequential".
"""

import bisect
import dataclasses
import enum
import re
from collections.abc import Sequence

PRAGMA_LINE_RE = re.compile(r"^\s*#pragma\s+omp\b")
IF_CLAUSE_RE = re.compile(r"\bif\s*\(")
WORKSHARING_FOR_RE = re.compile(r"#pragma\s+omp\s+for\b")

#: Leading words DaCe's canonicalization comment uses to classify a loop's dependence. Only
#: a corroborating signal on CPF forms -- never required, since raw DaCe/baseline C carry none.
DACE_MARKER_PREFIXES = ("parallel", "sequential", "undecided", "unclassified", "wavefront", "reduction over", "scan:")

LANGUAGE_BY_SUFFIX = {".c": "c", ".cpp": "c++", ".cc": "c++", ".cxx": "c++", ".hip": "hip"}


class NestKind(enum.Enum):
    """One loop nest's parallel status, as textually emitted."""

    PARALLEL = "parallel"
    CONTRACT_GUARDED = "contract_guarded"
    SIMD_ONLY = "simd_only"
    SEQUENTIAL = "sequential"


@dataclasses.dataclass(frozen=True, slots=True)
class LoopNest:
    """One classified loop nest."""

    kind: NestKind
    header_line: int
    depth: int
    has_reduction: bool
    has_scan: bool
    has_if_clause: bool
    has_atomic: bool
    worksharing_only: bool
    nested_parallel_region: bool
    dace_comment: str


@dataclasses.dataclass(frozen=True, slots=True)
class ParallelismBreakdown:
    """Per-kernel counts, rollable straight into one CSV row."""

    kernel: str
    language: str
    total_nests: int
    parallel_nests: int
    contract_guarded_nests: int
    simd_only_nests: int
    sequential_nests: int
    reduction_nests: int
    scan_nests: int
    if_clause_nests: int
    atomic_nests: int
    nested_parallel_region_nests: int
    max_depth: int
    parallel_nest_fraction: float

    @staticmethod
    def from_nests(kernel: str, language: str, nests: Sequence[LoopNest]) -> "ParallelismBreakdown":
        total = len(nests)
        parallel = sum(1 for n in nests if n.kind == NestKind.PARALLEL)
        guarded = sum(1 for n in nests if n.kind == NestKind.CONTRACT_GUARDED)
        simd_only = sum(1 for n in nests if n.kind == NestKind.SIMD_ONLY)
        sequential = sum(1 for n in nests if n.kind == NestKind.SEQUENTIAL)
        return ParallelismBreakdown(
            kernel=kernel,
            language=language,
            total_nests=total,
            parallel_nests=parallel,
            contract_guarded_nests=guarded,
            simd_only_nests=simd_only,
            sequential_nests=sequential,
            reduction_nests=sum(1 for n in nests if n.has_reduction),
            scan_nests=sum(1 for n in nests if n.has_scan),
            if_clause_nests=sum(1 for n in nests if n.has_if_clause),
            atomic_nests=sum(1 for n in nests if n.has_atomic),
            nested_parallel_region_nests=sum(1 for n in nests if n.nested_parallel_region),
            max_depth=max((n.depth for n in nests), default=0),
            parallel_nest_fraction=(parallel / total) if total else 0.0,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ForSpan:
    """One ``for`` occurrence: character offsets ``[start, end)`` spanning header and body."""

    start: int
    end: int
    header_line: int


@dataclasses.dataclass(frozen=True, slots=True)
class IfChain:
    """One ``if``/``else if``/``else`` chain: each branch's body span."""

    branches: tuple[tuple[int, int], ...]


def blank_run(chars: list[str], start: int, end: int) -> None:
    """Replace ``chars[start:end]`` with spaces, keeping real newlines so line numbers hold."""
    for k in range(start, end):
        if chars[k] != "\n":
            chars[k] = " "


def blank_line_comment(chars: list[str], text: str, start: int) -> int:
    j = start
    while j < len(text) and text[j] != "\n":
        j += 1
    blank_run(chars, start, j)
    return j


def blank_block_comment(chars: list[str], text: str, start: int) -> int:
    n = len(text)
    j = start + 2
    while j < n - 1 and text[j : j + 2] != "*/":
        j += 1
    end = min(j + 2, n)
    blank_run(chars, start, end)
    return end


def blank_literal(chars: list[str], text: str, start: int, quote: str) -> int:
    n = len(text)
    j = start + 1
    while j < n and text[j] != quote:
        j += 2 if text[j] == "\\" and j + 1 < n else 1
    end = min(j + 1, n)
    blank_run(chars, start, end)
    return end


def mask_comments_and_strings(text: str) -> str:
    """Same length and line count as ``text``, with comments and string/char literals blanked.

    Structural scanning (brace and paren matching, keyword search) runs on this, so a brace
    or a ``for`` spelled inside a comment or a string never confuses it.
    """
    chars = list(text)
    n = len(text)
    i = 0
    while i < n:
        two = text[i : i + 2]
        if two == "//":
            i = blank_line_comment(chars, text, i)
        elif two == "/*":
            i = blank_block_comment(chars, text, i)
        elif text[i] in "\"'":
            i = blank_literal(chars, text, i, text[i])
        else:
            i += 1
    return "".join(chars)


PRAGMA_START_RE = re.compile(r"^[ \t]*#pragma\b.*$", re.MULTILINE)


def blank_pragma_lines(text: str) -> str:
    """Blank ``#pragma`` lines for the FOR/IF structural scan.

    An OpenMP clause like ``if(cond)`` on a ``#pragma omp parallel for if(cond)`` line is not
    C control flow; matching it as one derails the if-chain walk (its "body" is read starting
    mid-pragma) and corrupts every span after it. The pragma text itself is read separately,
    from the untouched source lines, by :func:`own_pragma_lines` and the per-nest tag scan.
    """
    chars = list(text)
    for m in PRAGMA_START_RE.finditer(text):
        blank_run(chars, m.start(), m.end())
    return "".join(chars)


def matching_bracket(text: str, open_pos: int) -> int:
    """Index of the bracket that closes ``text[open_pos]``, on comment/string-masked text."""
    pairs = {"(": ")", "{": "}", "[": "]"}
    open_ch = text[open_pos]
    close_ch = pairs[open_ch]
    depth = 0
    i = open_pos
    n = len(text)
    while i < n:
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError(f"unbalanced {open_ch!r} starting at offset {open_pos}")


def find_stmt_end(text: str, start: int) -> int:
    """Offset just past the ``;`` that ends the brace-free single statement starting at ``start``."""
    depth = 0
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == ";" and depth == 0:
            return i + 1
        i += 1
    raise ValueError(f"unterminated statement starting at offset {start}")


def skip_ws(text: str, pos: int) -> int:
    n = len(text)
    while pos < n and text[pos].isspace():
        pos += 1
    return pos


def body_span(text: str, start: int) -> int:
    """End offset of the statement/block starting at ``start`` (after leading whitespace)."""
    at = skip_ws(text, start)
    if at < len(text) and text[at] == "{":
        return matching_bracket(text, at) + 1
    return find_stmt_end(text, at)


def preceding_word(text: str, pos: int) -> str:
    """The identifier/keyword immediately before ``pos``, skipping whitespace, or ``""``."""
    j = pos
    while j > 0 and text[j - 1].isspace():
        j -= 1
    end = j
    while j > 0 and (text[j - 1].isalnum() or text[j - 1] == "_"):
        j -= 1
    return text[j:end]


def build_newline_offsets(text: str) -> list[int]:
    return [i for i, ch in enumerate(text) if ch == "\n"]


def line_of(newline_offsets: list[int], pos: int) -> int:
    """0-indexed line number of ``pos``.

    ``bisect_left``, not ``bisect_right``: a position that lands exactly ON a line's own
    trailing newline (the common case for a span end returned by ``find_stmt_end``, which
    stops right after a statement's ``;`` with nothing but the newline following it) must
    still count as being on that line, not the next one.
    """
    return bisect.bisect_left(newline_offsets, pos)


def find_for_spans(masked: str, newline_offsets: list[int]) -> list[ForSpan]:
    spans: list[ForSpan] = []
    for m in re.finditer(r"\bfor\b", masked):
        start = m.start()
        paren_open = masked.find("(", m.end())
        if paren_open == -1 or masked[m.end() : paren_open].strip():
            continue  # not a for-statement header
        paren_close = matching_bracket(masked, paren_open)
        end = body_span(masked, paren_close + 1)
        spans.append(ForSpan(start=start, end=end, header_line=line_of(newline_offsets, start)))
    return spans


def word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def starts_with_word(text: str, pos: int, word: str) -> bool:
    """``text[pos:]`` starts with ``word`` as a whole word (not a longer identifier)."""
    end = pos + len(word)
    return text[pos:end] == word and not word_char(text[end : end + 1] or " ")


def walk_if_chain(masked: str, if_pos: int) -> IfChain:
    branches: list[tuple[int, int]] = []
    pos = if_pos
    while True:
        paren_open = masked.find("(", pos)
        paren_close = matching_bracket(masked, paren_open)
        branch_start = skip_ws(masked, paren_close + 1)
        branch_end = body_span(masked, branch_start)
        branches.append((branch_start, branch_end))
        after = skip_ws(masked, branch_end)
        if not starts_with_word(masked, after, "else"):
            break
        after_else = skip_ws(masked, after + 4)
        if starts_with_word(masked, after_else, "if"):
            pos = after_else
            continue
        branches.append((after_else, body_span(masked, after_else)))
        break
    return IfChain(branches=tuple(branches))


def find_if_chains(masked: str) -> list[IfChain]:
    chains: list[IfChain] = []
    for m in re.finditer(r"\bif\b", masked):
        if preceding_word(masked, m.start()) == "else":
            continue  # consumed by an earlier chain's walk
        chains.append(walk_if_chain(masked, m.start()))
    return chains


def contains(outer: ForSpan, inner: ForSpan) -> bool:
    return outer is not inner and outer.start <= inner.start and inner.end <= outer.end


def containment_count(span: ForSpan, all_spans: Sequence[ForSpan]) -> int:
    return 1 + sum(1 for other in all_spans if contains(other, span))


def dace_comment_marker(line: str) -> str:
    stripped = line.strip()
    if not stripped.startswith("//"):
        return ""
    body = stripped[2:].strip()
    return body if body.startswith(DACE_MARKER_PREFIXES) else ""


def own_pragma_lines(lines: list[str], header_line: int) -> list[str]:
    """Pragma line(s) directly above ``header_line``, closest first walking upward."""
    out: list[str] = []
    idx = header_line - 1
    while idx >= 0 and PRAGMA_LINE_RE.match(lines[idx]):
        out.append(lines[idx])
        idx -= 1
    return out


def preceding_marker(lines: list[str], header_line: int, own_pragma_count: int) -> str:
    idx = header_line - 1 - own_pragma_count
    while idx >= 0 and lines[idx].strip() == "":
        idx -= 1
    return dace_comment_marker(lines[idx]) if idx >= 0 else ""


def base_kind(pragma_lines: Sequence[str]) -> NestKind:
    lowered = [p.lower() for p in pragma_lines]
    if any("parallel" in p for p in lowered) or any(WORKSHARING_FOR_RE.search(p) for p in lowered):
        return NestKind.PARALLEL
    if any("simd" in p for p in lowered):
        return NestKind.SIMD_ONLY
    return NestKind.SEQUENTIAL


def classify_root(
    root: ForSpan, all_spans: Sequence[ForSpan], lines: list[str], newline_offsets: list[int]
) -> LoopNest:
    subtree = [s for s in all_spans if contains(root, s) or s is root]
    depth = max(containment_count(s, all_spans) for s in subtree)
    end_line = min(line_of(newline_offsets, max(s.end for s in subtree)) + 1, len(lines))
    own = own_pragma_lines(lines, root.header_line)
    # own pragmas sit ABOVE header_line, so the aggregate scan must start there, not at the for itself.
    span_lines = lines[max(0, root.header_line - len(own)) : end_line]
    kind = base_kind(span_lines)
    own_kind = base_kind(own)
    lowered_span = [p.lower() for p in span_lines if PRAGMA_LINE_RE.match(p)]
    return LoopNest(
        kind=kind,
        header_line=root.header_line,
        depth=depth,
        has_reduction=any("reduction(" in p for p in lowered_span),
        has_scan=any("scan" in p for p in lowered_span),
        has_if_clause=any(IF_CLAUSE_RE.search(p) for p in lowered_span),
        has_atomic=any("atomic" in p for p in lowered_span),
        worksharing_only=kind == NestKind.PARALLEL and not any("parallel" in p for p in lowered_span),
        nested_parallel_region=kind == NestKind.PARALLEL and own_kind != NestKind.PARALLEL,
        dace_comment=preceding_marker(lines, root.header_line, len(own)),
    )


def nearest_chain_index(root: ForSpan, chain_branches: Sequence[tuple[int, tuple[int, int]]]) -> int | None:
    """Index into ``chain_branches`` (chain_id, (start, end)) of the tightest branch containing ``root``."""
    best: tuple[int, int] | None = None
    best_id: int | None = None
    for chain_id, (start, end) in chain_branches:
        if start <= root.start and root.end <= end:
            if best is None or (end - start) < (best[1] - best[0]):
                best = (start, end)
                best_id = chain_id
    return best_id


def apply_contract_guards(nests: list[LoopNest], roots: list[ForSpan], chains: Sequence[IfChain]) -> None:
    """Reclassify PARALLEL/SIMD_ONLY nests to CONTRACT_GUARDED where a sibling branch is sequential."""
    chain_branches = [(cid, branch) for cid, chain in enumerate(chains) for branch in chain.branches]
    groups: dict[int, list[int]] = {}
    for i, root in enumerate(roots):
        chain_id = nearest_chain_index(root, chain_branches)
        if chain_id is not None:
            groups.setdefault(chain_id, []).append(i)
    for indices in groups.values():
        kinds = {nests[i].kind for i in indices}
        parallel_flavored = {NestKind.PARALLEL, NestKind.SIMD_ONLY}
        if kinds & parallel_flavored and NestKind.SEQUENTIAL in kinds:
            for i in indices:
                if nests[i].kind in parallel_flavored:
                    nests[i] = dataclasses.replace(nests[i], kind=NestKind.CONTRACT_GUARDED)


def analyze_source(text: str) -> list[LoopNest]:
    """Every loop nest in ``text``, classified. ``text`` may be C or C++; the pragma vocabulary
    used here (parallel for, for, simd, reduction, scan, atomic, if-clause) is the same either way.
    """
    masked = mask_comments_and_strings(text)
    # A separate buffer for FOR/IF keyword scanning only: an OMP clause like the "if(cond)" in
    # "omp parallel for if(cond)" is not C control flow, and letting the if-chain walk see it
    # derails every span after it. Tag/pragma detection still reads the untouched source lines.
    structural = blank_pragma_lines(masked)
    lines = text.splitlines()
    newline_offsets = build_newline_offsets(masked)
    for_spans = find_for_spans(structural, newline_offsets)
    roots = [s for s in for_spans if containment_count(s, for_spans) == 1]
    nests = [classify_root(r, for_spans, lines, newline_offsets) for r in roots]
    chains = find_if_chains(structural)
    apply_contract_guards(nests, roots, chains)
    return nests


def language_for_path(path_suffix: str) -> str:
    return LANGUAGE_BY_SUFFIX.get(path_suffix.lower(), path_suffix.lstrip("."))


def breakdown_for_source(text: str, language: str, kernel: str) -> ParallelismBreakdown:
    """The pure entry point: source text plus language in, one kernel's typed breakdown out."""
    return ParallelismBreakdown.from_nests(kernel, language, analyze_source(text))
