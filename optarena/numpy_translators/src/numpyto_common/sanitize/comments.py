"""Comment stripping across the benchmark languages (python / c / cpp /
fortran / cuda / hip).

Backed by **tree-sitter** when its runtime + prebuilt grammars are importable
(uniform, grammar-faithful comment removal); otherwise a robust **stdlib
fallback**:

* python  -- ``tokenize`` (drops ``COMMENT`` tokens; never touches string
  literals because the tokenizer classifies them separately).
* c / cpp / cuda / hip -- a careful character scanner that respects string and
  char literals (and escapes / raw-ish forms) so ``//`` or ``/* */`` inside a
  string is left alone.
* fortran -- ``!`` line comments, honouring quoted strings.

Availability is detected with :func:`importlib.util.find_spec` (no bare
try-import-on-string dispatch).
"""
import importlib.util
import io
import re
import tokenize
from typing import List, Tuple

# Languages handled by the C-family block-and-line comment scanner.
C_FAMILY = frozenset({"c", "cpp", "c++", "cuda", "hip"})
SUPPORTED_LANGS = frozenset({"python", "py", "c", "cpp", "c++", "fortran", "f90", "cuda", "hip"})


def _normalize_lang(lang: str) -> str:
    key = lang.strip().lower()
    aliases = {"py": "python", "c++": "cpp", "f90": "fortran", "f": "fortran"}
    return aliases.get(key, key)


def tree_sitter_available() -> bool:
    """True iff the maintained ``tree-sitter-language-pack`` grammar bundle is
    importable (it vendors the tree-sitter runtime). Detected via ``find_spec``
    (no import side effects). tree-sitter is an OPTIONAL enhancement -- when
    absent, every entry point below falls back to the stdlib scanners, so it is
    deliberately not a hard dependency (the dead ``tree-sitter-languages`` had
    no wheels past Python 3.11)."""
    return importlib.util.find_spec("tree_sitter_language_pack") is not None


# --- tree-sitter API adapter -------------------------------------------------
# We support whichever tree-sitter the grammar bundle ships. The bundled binding
# in tree-sitter-language-pack 1.x differs from the official ``tree-sitter`` PyPI
# wheel: ``Node.kind`` instead of ``Node.type``, ``Tree.root_node`` is a method,
# children are reached via ``child(i)``/``child_count`` (no ``.children`` list),
# and ``Parser.parse`` takes ``str``. These helpers normalize both shapes; byte
# offsets are utf-8 byte indices in every variant, so span math is unaffected.
def _ts_attr(obj, name):
    """Read ``obj.name`` whether the binding exposes it as a property or a
    nullary method (the two tree-sitter bindings disagree on which)."""
    v = getattr(obj, name, None)
    return v() if callable(v) else v


def _ts_get_parser(grammar: str):
    from tree_sitter_language_pack import get_parser
    return get_parser(grammar)


def _ts_parse(parser, src: str):
    try:
        return parser.parse(src)  # language-pack: str
    except TypeError:
        return parser.parse(src.encode("utf-8"))  # official: bytes


def _ts_root(tree):
    return _ts_attr(tree, "root_node")


def _ts_type(node) -> str:
    t = _ts_attr(node, "type")  # official binding
    return t if isinstance(t, str) else _ts_attr(node, "kind")  # language-pack


def _ts_span(node):
    return _ts_attr(node, "start_byte"), _ts_attr(node, "end_byte")


def _ts_children(node):
    ch = _ts_attr(node, "children")  # official: list property
    if ch is not None:
        return ch
    return [node.child(i) for i in range(_ts_attr(node, "child_count"))]


# Map our language keys onto tree-sitter grammar names.
TS_GRAMMAR = {
    "python": "python",
    "c": "c",
    "cpp": "cpp",
    "fortran": "fortran",
    "cuda": "cpp",  # CUDA/HIP are C++-family for comment scanning
    "hip": "cpp",
}


def _strip_with_tree_sitter(src: str, lang: str) -> str:
    """Remove every node whose type contains ``comment`` by blanking its byte
    span (preserving newlines so line numbers / layout are stable)."""
    parser = _ts_get_parser(TS_GRAMMAR[lang])
    data = src.encode("utf-8")
    tree = _ts_parse(parser, src)

    spans: List[tuple] = []

    def walk(node):
        if "comment" in _ts_type(node):
            spans.append(_ts_span(node))
            return
        for child in _ts_children(node):
            walk(child)

    walk(_ts_root(tree))

    if not spans:
        return src

    out = bytearray(data)
    for start, end in spans:
        for i in range(start, end):
            if out[i] != ord("\n"):
                out[i] = ord(" ")
    return out.decode("utf-8")


def _strip_python_tokenize(src: str) -> str:
    """Drop ``#`` comments with the tokenizer, editing the original text in
    place so all other layout / tokens are preserved byte-for-byte (string
    literals are a different token kind, so they are never disturbed)."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError):
        # Malformed input: fall back to the line scanner.
        return _strip_python_line_scan(src)

    lines = src.splitlines(keepends=True)
    # Collect comment spans per (1-based) line; a COMMENT token never spans
    # multiple lines, so start row == end row.
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        (srow, scol), (_erow, ecol) = tok.start, tok.end
        if 1 <= srow <= len(lines):
            line = lines[srow - 1]
            lines[srow - 1] = line[:scol] + line[ecol:]

    out = "".join(lines)
    out = "\n".join(seg.rstrip() for seg in out.split("\n"))
    return out


def _strip_python_line_scan(src: str) -> str:
    """Last-resort python stripper: remove ``#`` outside of string literals,
    line by line. Used only if tokenize raises on malformed input."""
    return _strip_c_family(src, hashes=True, slashes=False, fortran_bang=False)


def _strip_c_family(src: str, *, slashes: bool = True, hashes: bool = False, fortran_bang: bool = False) -> str:
    """Character scanner that strips comments while respecting string and char
    literals.

    * ``slashes``      -> handle ``//`` line and ``/* ... */`` block comments.
    * ``hashes``       -> handle ``#`` line comments (python fallback only).
    * ``fortran_bang`` -> handle ``!`` line comments.
    """
    out: List[str] = []
    i = 0
    n = len(src)
    in_string = False
    string_quote = ""
    while i < n:
        ch = src[i]
        nxt = src[i + 1] if i + 1 < n else ""

        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                # Escaped char: copy the next char verbatim.
                out.append(nxt)
                i += 2
                continue
            if ch == string_quote:
                in_string = False
            i += 1
            continue

        # Not currently inside a string.
        if ch in ("'", '"', "`"):
            in_string = True
            string_quote = ch
            out.append(ch)
            i += 1
            continue

        if slashes and ch == "/" and nxt == "/":
            # Line comment: skip to end of line (keep the newline).
            while i < n and src[i] != "\n":
                i += 1
            continue

        if slashes and ch == "/" and nxt == "*":
            # Block comment: skip to closing */, preserving embedded newlines.
            i += 2
            while i < n and not (src[i] == "*" and i + 1 < n and src[i + 1] == "/"):
                if src[i] == "\n":
                    out.append("\n")
                i += 1
            i += 2  # consume the closing */
            continue

        if hashes and ch == "#":
            while i < n and src[i] != "\n":
                i += 1
            continue

        if fortran_bang and ch == "!":
            while i < n and src[i] != "\n":
                i += 1
            continue

        out.append(ch)
        i += 1

    text = "".join(out)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text


# License / attribution notices we must preserve verbatim: CC-BY and friends
# REQUIRE the notice to survive redistribution, so stripping it from a ported
# kernel (a microapp adapted from a real, licensed code) would violate the
# license. Synthetic microkernels carry no such header, so nothing is kept.
_ATTRIBUTION_RE = re.compile(r"licen[sc]e|attribution|copyright|spdx|creative commons|©", re.IGNORECASE)

# Line prefixes that begin a top-of-file header line in the benchmark languages.
_HEADER_LINE_STARTS = ("#", "//", "*", "!")


def _carries_attribution(header: str) -> bool:
    """True iff a header block carries a license / attribution notice to keep."""
    return bool(_ATTRIBUTION_RE.search(header))


def _split_leading_header(src: str) -> Tuple[str, str]:
    """Split ``src`` into (header, body): the header is the top-of-file run of
    blank lines, comment lines, and a leading triple-quoted docstring, up to the
    first line of real code -- the block a license / attribution notice lives in.
    Tracks multi-line ``/* */`` and docstrings so a wrapped notice stays whole."""
    lines = src.splitlines(keepends=True)
    doc_delim = None  # inside a python docstring
    in_block = False  # inside a /* ... */ comment
    k = 0
    for line in lines:
        s = line.strip()
        if in_block:
            k += 1
            if "*/" in s:
                in_block = False
            continue
        if doc_delim is not None:
            k += 1
            if doc_delim in s:
                doc_delim = None
            continue
        if not s:
            k += 1
            continue
        if s[:3] in ('"""', "'''"):
            k += 1
            if s[:3] not in s[3:]:  # multi-line docstring: track its close
                doc_delim = s[:3]
            continue
        if s.startswith("/*"):
            k += 1
            if "*/" not in s[2:]:  # multi-line block comment: track its close
                in_block = True
            continue
        if s.startswith(_HEADER_LINE_STARTS):
            k += 1
            continue
        break  # first line of real code
    return "".join(lines[:k]), "".join(lines[k:])


def _strip_dispatch(src: str, norm: str) -> str:
    """Language dispatch: tree-sitter when importable, else a stdlib fallback."""
    if tree_sitter_available():
        return _strip_with_tree_sitter(src, norm)
    if norm == "python":
        return _strip_python_tokenize(src)
    if norm == "fortran":
        return _strip_c_family(src, slashes=False, hashes=False, fortran_bang=True)
    # c / cpp / cuda / hip
    return _strip_c_family(src, slashes=True, hashes=False, fortran_bang=False)


def strip_comments(src: str, lang: str) -> str:
    """Return ``src`` with all comments removed for ``lang``.

    Uses tree-sitter when importable, else a stdlib fallback per language.
    String / char literals are never disturbed. A leading license / attribution
    header is preserved VERBATIM (see :data:`_ATTRIBUTION_RE`); only the body is
    stripped -- so a ported microapp keeps its CC-BY notice while a synthetic
    microkernel (no header) is fully stripped.
    """
    norm = _normalize_lang(lang)
    if norm not in TS_GRAMMAR:
        raise ValueError(f"strip_comments: unsupported lang {lang!r}; "
                         f"supported = {sorted(TS_GRAMMAR)}")

    header, body = _split_leading_header(src)
    if _carries_attribution(header):
        return header + _strip_dispatch(body, norm)
    return _strip_dispatch(src, norm)
