# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rewrites of a translator-emitted ``*_pluto_input.c`` that polycc sees and nothing else does.

Two pet/Pluto defects cost affine kernels the Pluto column, and both are avoided by respelling the
same computation before polycc parses it (``pluto_affine.KNOWN_POLYCC_ISSUES`` POLYCC-014/015):

* POLYCC-014: a FUNCTION-LOCAL scalar written inside a scop (a reduction accumulator, a carried
  value, an if-conversion flag) aborts Pluto (``pluto_auto_transform`` assertion) or is dropped /
  shared across threads in the transformed output. The identical code on a POINTER PARAMETER cell
  transforms and validates. :func:`externalize_scop_scalars` moves the scop into a static function
  that receives each such scalar as a one-element ``restrict`` pointer; the exported symbol keeps
  its signature, declares the scalars and calls it.
* POLYCC-015: a non-unit literal stride (``i += 2``) is DROPPED by pet, so the loop visits every
  element. :func:`normalize_strided_loops` runs the loop over a unit-stride counter and substitutes
  ``lo + s * n`` for the index.

Applied only on the Pluto column's own path (:func:`hpcagent_bench.pluto_transform.run_polycc`), to a
scratch copy: the file on disk is also PPCG's input and the freshness key of the build.
"""

import re

#: A scalar declaration the emitter writes at function top (``_emit_body``'s int/implicit locals).
_SCALAR_DECL_RE = re.compile(
    r"^(?P<indent>[ \t]+)(?P<ctype>(?:double|float) _Complex|double|float|_Float16|bool|u?int(?:8|16|32|64)_t)"
    r" (?P<name>[A-Za-z_]\w*);$",
    re.MULTILINE,
)

#: The emitted entry point: ``void <symbol>(<params>) {`` on one line, closed by ``}`` in column 0.
_FUNC_RE = re.compile(r"^void (?P<name>\w+)\((?P<params>[^\n]*)\) \{\n(?P<body>.*?)^\}\n", re.MULTILINE | re.DOTALL)

#: A for header the emitter writes for a literal step other than +1 (``_CBodyEmitter._emit_for``):
#: ``i += s`` or the reverse ``--i``.
_STRIDED_FOR_RE = re.compile(
    r"^(?P<indent>[ \t]*)for \(int64_t (?P<var>\w+) = (?P<lo>[^;\n]+); (?P=var) (?P<op>[<>]) (?P<hi>[^;\n]+); "
    r"(?:(?P=var) \+= \(?(?P<step>-?\d+)\)?|(?P<down>--)(?P=var))\) \{$",
    re.MULTILINE,
)

_SCOP_REGION_RE = re.compile(r"#pragma scop.*?#pragma endscop", re.DOTALL)


def _ident_re(name: str) -> "re.Pattern[str]":
    """``name`` as a whole C identifier that is not a member access."""
    return re.compile(rf"(?<![\w.>]){re.escape(name)}(?!\w)")


def _closing(text: str, open_idx: int, pair: tuple[str, str]) -> int:
    """Index of the bracket closing the one at ``open_idx``; ``len(text)`` when unbalanced."""
    depth = 0
    for j in range(open_idx, len(text)):
        if text[j] == pair[0]:
            depth += 1
        elif text[j] == pair[1]:
            depth -= 1
            if not depth:
                return j
    return len(text)


def _index_spans(body: str) -> list[tuple[int, int]]:
    """Spans whose names act as integers polycc must model: subscripts, for headers, if conditions."""
    spans: list[tuple[int, int]] = []
    for m in re.finditer(r"\[", body):
        spans.append((m.start(), _closing(body, m.start(), ("[", "]"))))
    for m in re.finditer(r"\b(?:for|if|while) \(", body):
        spans.append((m.end() - 1, _closing(body, m.end() - 1, ("(", ")"))))
    return spans


def _param_names(params: str) -> list[str]:
    """Parameter names of a one-line C parameter list (VLA extents carry no commas)."""
    names: list[str] = []
    for part in params.split(","):
        m = re.search(r"(\w+)\s*(?:\[.*)?$", part.strip())
        if m is None:
            raise ValueError(f"unparseable parameter {part!r}")
        names.append(m.group(1))
    return names


def externalize_scop_scalars(text: str) -> str:
    """POLYCC-014: every local scalar used only as a VALUE inside scop regions becomes ``name[0]`` of a
    pointer parameter of a static ``<symbol>_pluto_scop``; unchanged when there is none.

    A scalar that appears outside a region, or inside a subscript, a loop header or an ``if``
    condition, is left alone: as ``name[0]`` it would turn an affine index into a data-dependent one.
    """
    out: list[str] = []
    pos = 0
    for fm in _FUNC_RE.finditer(text):
        body = fm.group("body")
        regions = [(m.start(), m.end()) for m in _SCOP_REGION_RE.finditer(body)]
        if not regions:
            continue
        index_spans = _index_spans(body)
        moved: list[tuple[str, str, str]] = []  # (decl line, ctype, name)
        for dm in _SCALAR_DECL_RE.finditer(body):
            uses = [
                m.start() for m in _ident_re(dm.group("name")).finditer(body) if not dm.start() <= m.start() < dm.end()
            ]
            if not uses:
                continue
            if not all(any(lo <= u < hi for lo, hi in regions) for u in uses):
                continue
            if any(lo <= u <= hi for u in uses for lo, hi in index_spans):
                continue
            moved.append((dm.group(0), dm.group("ctype"), dm.group("name")))
        if not moved:
            continue
        name, params = fm.group("name"), fm.group("params")
        inner_body = body
        for line, _ctype, var in moved:
            inner_body = inner_body.replace(line + "\n", "", 1)
            inner_body = _ident_re(var).sub(f"{var}[0]", inner_body)
        inner = f"{name}_pluto_scop"
        extra = ", ".join(f"{ctype} *restrict {var}" for _line, ctype, var in moved)
        args = ", ".join([*_param_names(params), *(f"&{var}" for _l, _c, var in moved)])
        decls = "".join(f"{line}\n" for line, _c, _v in moved)
        indent = moved[0][0][: len(moved[0][0]) - len(moved[0][0].lstrip())]
        out.append(text[pos : fm.start()])
        out.append(f"static void {inner}({params}, {extra}) {{\n{inner_body}}}\n")
        out.append(f"void {name}({params}) {{\n{decls}{indent}{inner}({args});\n}}\n")
        pos = fm.end()
    out.append(text[pos:])
    return "".join(out)


def normalize_strided_loops(text: str) -> str:
    """POLYCC-015: ``for (i = lo; i < hi; i += s)`` with a literal ``|s| > 1``, and the reverse
    ``for (i = lo; i > hi; --i)``, become a forward unit-stride loop over ``i_pn`` with
    ``(lo + s * i_pn)`` substituted for ``i`` in its condition and body."""
    while True:
        m = _STRIDED_FOR_RE.search(text)
        if m is None:
            return text
        step = -1 if m.group("down") else int(m.group("step"))
        indent, var = m.group("indent"), m.group("var")
        if step in (0, 1):
            raise ValueError(f"step {step} spelled as += in {m.group(0)!r}")
        counter = f"{var}_pn"
        if re.search(rf"\b{counter}\b", text):
            raise ValueError(f"loop counter {counter} already in use")
        end = text.find(f"\n{indent}}}", m.end())
        if end < 0:
            raise ValueError(f"no closing brace for {m.group(0)!r}")
        value = f"({m.group('lo')} + ({step}) * {counter})"
        header = f"{indent}for (int64_t {counter} = 0; {value} {m.group('op')} {m.group('hi')}; ++{counter}) {{"
        body = _ident_re(var).sub(value, text[m.end() : end])
        text = text[: m.start()] + header + body + text[end:]


def normalize_scop_input(text: str) -> str:
    """Both rewrites, stride first (it leaves no new scalar behind)."""
    return externalize_scop_scalars(normalize_strided_loops(text))
