"""Comment stripping across the benchmark languages (python / c / cpp /
fortran / cuda / hip).

Backed by **tree-sitter** (``tree-sitter-language-pack``, a core dependency): uniform, grammar-faithful comment
removal that never touches string or char literals.
"""

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tree_sitter

__all__ = [
    "ATTRIBUTION_RE",
    "COMMENT_STARTS",
    "TS_GRAMMAR",
    "carries_attribution",
    "leading_license_block",
    "normalize_lang",
    "strip_comments",
    "strip_dispatch",
    "strip_with_tree_sitter",
    "ts_get_parser",
]


def normalize_lang(lang: str) -> str:
    key = lang.strip().lower()
    aliases = {"py": "python", "c++": "cpp", "f90": "fortran", "f": "fortran"}
    return aliases.get(key, key)


def ts_get_parser(grammar: str) -> "tree_sitter.Parser":
    from tree_sitter_language_pack import get_parser

    return get_parser(grammar)


# Map our language keys onto tree-sitter grammar names.
TS_GRAMMAR = {
    "python": "python",
    "c": "c",
    "cpp": "cpp",
    "fortran": "fortran",
    "cuda": "cpp",  # CUDA/HIP are C++-family for comment scanning
    "hip": "cpp",
}


def strip_with_tree_sitter(src: str, lang: str) -> str:
    """Remove every node whose type contains ``comment`` by blanking its byte
    span (preserving newlines so line numbers / layout are stable)."""
    parser = ts_get_parser(TS_GRAMMAR[lang])
    data = src.encode("utf-8")
    tree = parser.parse(data)

    spans: list[tuple[int, int]] = []

    def walk(node: "tree_sitter.Node") -> None:
        if "comment" in node.type:
            spans.append((node.start_byte, node.end_byte))
            return
        for child in node.children:
            walk(child)

    walk(tree.root_node)

    if not spans:
        return src

    out = bytearray(data)
    for start, end in spans:
        for i in range(start, end):
            if out[i] != ord("\n"):
                out[i] = ord(" ")
    return out.decode("utf-8")


# License / attribution notices we must preserve verbatim: CC-BY and friends
# REQUIRE the notice to survive redistribution, so stripping it from a ported
# kernel (a microapp adapted from a real, licensed code) would violate the
# license. Synthetic microkernels carry no such header, so nothing is kept.
# The last alternative is the copyright SIGN, spelled as an escape so this source stays ASCII --
# it is the only branch that catches a bare glyph, which "copyright" and "\(c\)" both miss.
ATTRIBUTION_RE = re.compile("licen[sc]e|attribution|copyright|spdx|creative commons|\\(c\\)|\\u00a9", re.IGNORECASE)

# The leading COMMENT-line marker per language. Deliberately language-specific: '#' is a
# comment in python/shell but a PREPROCESSOR directive in C, and '*' is a pointer in C --
# so neither may count as a header comment there (the C '/* */' continuation is tracked
# separately). Anything not listed is C-family and uses '//'.
COMMENT_STARTS = {"python": ("#",), "fortran": ("!",)}


def carries_attribution(text: str) -> bool:
    """True iff ``text`` carries a license / attribution notice to preserve."""
    return bool(ATTRIBUTION_RE.search(text))


def leading_license_block(src: str, norm: str) -> str:
    """The top-of-file license / attribution comment block to keep verbatim, or ``""``.

    It is the FIRST contiguous run of comment lines (language-aware: python ``#``,
    fortran ``!``, C-family ``//`` + ``/* */``), starting at the top of the file and
    STOPPING at the first blank line or blank comment line. Stopping there is what keeps
    a DESCRIPTION block sitting below the notice out of the preserved region -- e.g.
    force_lj's closed-form formula, separated from the copyright by a bare ``#``: only the
    notice survives, the description is stripped with the body. Returns ``""`` when that
    first block carries no license marker (a synthetic kernel)."""
    starts = COMMENT_STARTS.get(norm, ("//",))
    c_family = norm not in COMMENT_STARTS
    block: list[str] = []
    in_block = False  # inside a /* ... */ comment
    for line in src.splitlines(keepends=True):
        s = line.strip()
        if in_block:
            block.append(line)
            if "*/" in s:
                in_block = False
            continue
        if not s:
            break  # a blank line ends the leading block
        if c_family and s.startswith("/*"):
            block.append(line)
            if "*/" not in s[2:]:
                in_block = True
            continue
        if s.startswith(starts):
            marker = next(m for m in starts if s.startswith(m))
            if s[len(marker) :].strip() == "":
                break  # a blank comment line separates the notice from a description below
            block.append(line)
            continue
        break  # first line of real code (or a docstring -- the stripper leaves those alone)
    header = "".join(block)
    return header if carries_attribution(header) else ""


def strip_dispatch(src: str, norm: str) -> str:
    """Language dispatch: tree-sitter's grammar for ``norm`` (a core dependency)."""
    return strip_with_tree_sitter(src, norm)


def strip_comments(src: str, lang: str) -> str:
    """Return ``src`` with all comments removed for ``lang``.

    Uses tree-sitter's grammar for the language; string / char literals are never disturbed. A leading license / attribution NOTICE (the first
    top-of-file comment block, per :func:`leading_license_block`) is preserved verbatim
    so a ported microapp keeps its CC-BY / copyright line; everything else -- including
    any description comments below the notice -- is stripped.
    """
    norm = normalize_lang(lang)
    if norm not in TS_GRAMMAR:
        raise ValueError(f"strip_comments: unsupported lang {lang!r}; supported = {sorted(TS_GRAMMAR)}")

    header = leading_license_block(src, norm)
    if header:
        return header + strip_dispatch(src[len(header) :], norm)
    return strip_dispatch(src, norm)
