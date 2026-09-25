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

# Languages handled by the C-family block-and-line comment scanner.
C_FAMILY = frozenset({"c", "cpp", "c++", "cuda", "hip"})


def normalize_lang(lang: str) -> str:
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


# tree-sitter API adapter
# We support whichever tree-sitter the grammar bundle ships. The bundled binding
# in tree-sitter-language-pack 1.x differs from the official ``tree-sitter`` PyPI
# wheel: ``Node.kind`` instead of ``Node.type``, ``Tree.root_node`` is a method,
# children are reached via ``child(i)``/``child_count`` (no ``.children`` list),
# and ``Parser.parse`` takes ``str``. These helpers normalize both shapes; byte
# offsets are utf-8 byte indices in every variant, so span math is unaffected.
def ts_attr(obj, name):
    """Read ``obj.name`` whether the binding exposes it as a property or a
    nullary method (the two tree-sitter bindings disagree on which).

    Keep the getattr: these are C-extension objects with no ``__dict__``, so ``vars(obj)`` raises."""
    v = getattr(obj, name, None)
    return v() if callable(v) else v


def ts_get_parser(grammar: str):
    from tree_sitter_language_pack import get_parser

    return get_parser(grammar)


def ts_parse(parser, src: str):
    try:
        return parser.parse(src)  # language-pack: str
    except TypeError:
        return parser.parse(src.encode("utf-8"))  # official: bytes


def ts_root(tree):
    return ts_attr(tree, "root_node")


def ts_type(node) -> str:
    t = ts_attr(node, "type")  # official binding
    return t if isinstance(t, str) else ts_attr(node, "kind")  # language-pack


def ts_span(node):
    return ts_attr(node, "start_byte"), ts_attr(node, "end_byte")


def ts_children(node):
    ch = ts_attr(node, "children")  # official: list property
    if ch is not None:
        return ch
    return [node.child(i) for i in range(ts_attr(node, "child_count"))]


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
    tree = ts_parse(parser, src)

    spans: list[tuple] = []

    def walk(node) -> None:
        if "comment" in ts_type(node):
            spans.append(ts_span(node))
            return
        for child in ts_children(node):
            walk(child)

    walk(ts_root(tree))

    if not spans:
        return src

    out = bytearray(data)
    for start, end in spans:
        for i in range(start, end):
            if out[i] != ord("\n"):
                out[i] = ord(" ")
    return out.decode("utf-8")


def strip_python_tokenize(src: str) -> str:
    """Drop ``#`` comments with the tokenizer, editing the original text in
    place so all other layout / tokens are preserved byte-for-byte (string
    literals are a different token kind, so they are never disturbed)."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError):
        # Malformed input: fall back to the line scanner.
        return strip_python_line_scan(src)

    lines = src.splitlines(keepends=True)
    # Collect comment spans per (1-based) line; a COMMENT token never spans
    # multiple lines, so start row == end row.
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        (srow, scol), ecol = tok.start, tok.end[1]
        if 1 <= srow <= len(lines):
            line = lines[srow - 1]
            lines[srow - 1] = line[:scol] + line[ecol:]

    out = "".join(lines)
    out = "\n".join(seg.rstrip() for seg in out.split("\n"))
    return out


def strip_python_line_scan(src: str) -> str:
    """Last-resort python stripper: remove ``#`` outside of string literals,
    line by line. Used only if tokenize raises on malformed input."""
    return strip_c_family(src, hashes=True, slashes=False, fortran_bang=False)


def line_end(src: str, i: int) -> int:
    """Index of the newline ending the line ``i`` is on (``len(src)`` on the last line)."""
    n = len(src)
    while i < n and src[i] != "\n":
        i += 1
    return i


def block_comment_end(src: str, i: int, newlines: list[str] | None = None) -> int:
    """Index just past the ``*/`` closing the ``/*`` comment at ``i``; each newline inside the comment
    is appended to ``newlines`` when given."""
    n = len(src)
    i += 2
    while i < n and not (src[i] == "*" and i + 1 < n and src[i + 1] == "/"):
        if newlines is not None and src[i] == "\n":
            newlines.append("\n")
        i += 1
    return i + 2


def strip_c_family(src: str, *, slashes: bool = True, hashes: bool = False, fortran_bang: bool = False) -> str:
    """Character scanner that strips comments while respecting string and char
    literals.

    * ``slashes``      -> handle ``//`` line and ``/* ... */`` block comments.
    * ``hashes``       -> handle ``#`` line comments (python fallback only).
    * ``fortran_bang`` -> handle ``!`` line comments.
    """
    out: list[str] = []
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

        if (slashes and ch == "/" and nxt == "/") or (hashes and ch == "#") or (fortran_bang and ch == "!"):
            # Line comment: skip to end of line (keep the newline).
            i = line_end(src, i)
            continue

        if slashes and ch == "/" and nxt == "*":
            # Block comment: skip past the closing */, preserving embedded newlines.
            i = block_comment_end(src, i, out)
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
    """Language dispatch: tree-sitter when importable, else a stdlib fallback."""
    if tree_sitter_available():
        return strip_with_tree_sitter(src, norm)
    if norm == "python":
        return strip_python_tokenize(src)
    if norm == "fortran":
        return strip_c_family(src, slashes=False, hashes=False, fortran_bang=True)
    # c / cpp / cuda / hip
    return strip_c_family(src, slashes=True, hashes=False, fortran_bang=False)


def strip_comments(src: str, lang: str) -> str:
    """Return ``src`` with all comments removed for ``lang``.

    Uses tree-sitter when importable, else a stdlib fallback per language. String /
    char literals are never disturbed. A leading license / attribution NOTICE (the first
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
