# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rewrites of a translator-emitted ``*_pluto_input.c`` that polycc sees and nothing else does.

pet/Pluto defects cost affine kernels the Pluto column; each is avoided by respelling the same
computation before polycc parses it (``pluto_affine.KNOWN_POLYCC_ISSUES``):

* POLYCC-014: a FUNCTION-LOCAL scalar written inside a scop aborts Pluto or is dropped / shared in
  the output; :func:`externalize_scop_scalars` hands it to a static ``<symbol>_pluto_scop`` as a
  one-element ``restrict`` pointer cell, the exported symbol keeping its signature.
* POLYCC-015: a literal non-unit or reverse step is dropped; :func:`normalize_strided_loops`.
* POLYCC-016: C23 ``constexpr`` knobs stop the parse; :func:`inline_pinned_constants`.
* POLYCC-017: integer locals that carry an index are data to pet;
  :func:`forward_substitute_scalars` (single assignment), :func:`fold_constant_sign_ternaries`
  (the runtime-sign condition of a now-literal step) and :func:`substitute_induction_scalars`
  (literal-step induction scalars as closed forms of the counter).
* POLYCC-010: pet outlines helper calls into an undeclared ``__pet_ret``; :func:`floord_subscripts`
  and :func:`opaque_helper_calls` (undone on the output by :func:`restore_output`).

Applied to scratch copies only: :func:`hpcagent_bench.pluto_transform.run_polycc` gets
:func:`normalize_scop_input`, :func:`hpcagent_bench.ppcg_transform.run_ppcg` the signature-preserving
subset :func:`normalize_ppcg_input`. The file on disk is the build's freshness key and never changes.
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
        serial = 0
        while re.search(rf"\b{counter}\b", text):
            serial += 1
            counter = f"{var}_pn{serial}"
        end = text.find(f"\n{indent}}}", m.end())
        if end < 0:
            raise ValueError(f"no closing brace for {m.group(0)!r}")
        value = f"({m.group('lo')} + ({step}) * {counter})"
        header = f"{indent}for (int64_t {counter} = 0; {value} {m.group('op')} {m.group('hi')}; ++{counter}) {{"
        body = _ident_re(var).sub(value, text[m.end() : end])
        text = text[: m.start()] + header + body + text[end:]


#: A file-scope pinned knob (``numpyto_c.emit.pinned_const_block``). C23 ``constexpr`` is not C that
#: pet's libclang parses, and the whole translation unit is refused over it.
_CONSTEXPR_RE = re.compile(r"^constexpr (?P<ctype>[\w ]+?) (?P<name>\w+) = (?P<value>[^;\n]+);$", re.MULTILINE)

#: The emitter's integer scalar locals, the ones that can carry an index.
_INT_DECL_RE = re.compile(r"^(?P<indent>[ \t]+)(?P<ctype>u?int(?:8|16|32|64)_t) (?P<name>[A-Za-z_]\w*);$", re.MULTILINE)

#: Calls in a scop that name a SCALAR helper pet may see the body of; everything else is left alone.
_KEEP_CALLS = frozenset({"floord", "ceild", "min", "max", "for", "if", "while", "return", "sizeof"})

#: Prefix of the body-less stand-in :func:`opaque_helper_calls` gives pet, stripped by :func:`restore_output`.
OPAQUE_PREFIX = "__pluto_opaque_"


def inline_pinned_constants(text: str) -> str:
    """POLYCC-016: each ``constexpr`` knob becomes ``static const`` and its literal value is substituted
    into every use, so pet parses the unit and models the value as the constant it is."""
    for m in list(_CONSTEXPR_RE.finditer(text)):
        name, value = m.group("name"), m.group("value")
        head, tail = text[: m.start()], text[m.end() :]
        text = f"{head}static const {m.group('ctype')} {name} = {value};{_ident_re(name).sub(f'({value})', tail)}"
    return text


def _block_end(text: str, start: int, indent: str) -> int:
    """End of the block holding the statement that starts at ``start`` with ``indent``: the first later
    line indented less, or a scop pragma at that indent."""
    pos = text.find("\n", start)
    while pos >= 0:
        line_end = text.find("\n", pos + 1)
        line = text[pos + 1 : line_end if line_end >= 0 else len(text)]
        stripped = line.lstrip()
        if stripped and (len(line) - len(stripped) < len(indent) or stripped.startswith("#pragma")):
            return pos
        pos = line_end
    return len(text)


def forward_substitute_scalars(text: str) -> str:
    """POLYCC-017: an integer local assigned ONCE, from an expression of loop counters, parameters and
    literals, is replaced by that expression at every use (which must all follow it in its own block).

    pet otherwise keeps such a scalar as data: ``j = floord(i, 2); c[j]`` is an indirect read, and
    ``W = 7; ii += W`` a loop whose step it cannot see."""
    changed = True
    while changed:
        changed = False
        for dm in _INT_DECL_RE.finditer(text):
            name = dm.group("name")
            assigns = list(
                re.finditer(rf"^(?P<indent>[ \t]+){re.escape(name)} = (?P<rhs>[^;\n]+);$", text, re.MULTILINE)
            )
            if len(assigns) != 1:
                continue
            am = assigns[0]
            rhs = am.group("rhs")
            if "[" in rhs or "?" in rhs:
                continue
            locals_ = {d.group("name") for d in _SCALAR_DECL_RE.finditer(text)}
            if any(tok in locals_ for tok in re.findall(r"[A-Za-z_]\w*", rhs)):
                continue
            region = next((r for r in _SCOP_REGION_RE.finditer(text) if r.start() < am.start() < r.end()), None)
            if region is None:
                continue
            end = min(_block_end(text, am.start(), am.group("indent")), region.end())
            uses = [
                u.start()
                for u in _ident_re(name).finditer(text)
                if u.start() not in (dm.start("name"), am.start("indent") + len(am.group("indent")))
            ]
            if not uses or not all(am.end() <= u < end for u in uses):
                continue
            body = _ident_re(name).sub(f"({rhs})", text[am.end() : end])
            text = text[: dm.start()] + text[dm.end() + 1 : am.start()] + body[1:] + text[end:]
            changed = True
            break
    return text


def fold_constant_sign_ternaries(text: str) -> str:
    """``((<literal>) > 0 ? a : b)``, the emitter's runtime-sign loop condition, folded to ``a`` or ``b``
    once the step is a literal (after :func:`forward_substitute_scalars`)."""
    while True:
        m = re.search(r"\(\(\((?P<lit>-?\d+)\)\) > 0 \? |\(\((?P<lit2>-?\d+)\) > 0 \? ", text)
        if m is None:
            return text
        lit = int(m.group("lit") or m.group("lit2"))
        close = _closing(text, m.start(), ("(", ")"))
        inner = text[m.end() : close]
        depth, split = 0, -1
        for j, ch in enumerate(inner):
            depth += ch in "([" and 1 or 0
            depth -= ch in ")]" and 1 or 0
            if ch == ":" and depth == 0:
                split = j
                break
        if split < 0:
            return text
        pick = inner[:split].strip() if lit > 0 else inner[split + 1 :].strip()
        text = text[: m.start()] + pick + text[close + 1 :]


_UNIT_FOR_RE = re.compile(
    r"^(?P<indent>[ \t]*)for \(int64_t (?P<var>\w+) = (?P<lo>[^;\n]+); [^;\n]+; \+\+(?P=var)\) \{$", re.MULTILINE
)
_STEP_ASSIGN_RE = re.compile(r"^(?P<var>\w+) = \(?(?P<src>\w+)(?: (?P<op>[-+]) (?P<c>\d+))?\)?;$")


def substitute_induction_scalars(text: str) -> str:
    """POLYCC-017: integer locals a unit-step loop only ever advances by literals (``k = (j + 1);
    j = (k + 1);``) become closed forms of the loop counter (``-1 + 2 * (i - lo) + 1``).

    Each must be initialised by a literal before the loop, assigned nowhere else and read nowhere
    after it; one that is READ in the body before it is written must come back to itself plus a
    constant, which is its per-iteration step."""
    for lm in _UNIT_FOR_RE.finditer(text):
        indent, var, lo = lm.group("indent"), lm.group("var"), lm.group("lo")
        end = text.find(f"\n{indent}}}", lm.end())
        if end < 0:
            continue
        body = text[lm.end() : end]
        lines = body.split("\n")
        inner = indent + "  "
        ints = {d.group("name") for d in _INT_DECL_RE.finditer(text)}
        state: dict[str, tuple[str, int]] = {}
        order: list[str] = []
        ok = True
        for line in lines:
            if not line.startswith(inner) or line.startswith(inner + " "):
                continue
            sm = _STEP_ASSIGN_RE.match(line.strip())
            if sm is None or sm.group("var") not in ints:
                continue
            src, c = sm.group("src"), int(sm.group("c") or 0) * (-1 if sm.group("op") == "-" else 1)
            if src not in ints:
                ok = False
                break
            base, off = state.get(src, (src, 0))
            state[sm.group("var")] = (base, off + c)
            order.append(sm.group("var"))
        if not ok or not state:
            continue
        cands = set(state)
        # every assignment of a candidate inside the loop must be one of the step lines above
        step_lines = [ln for ln in lines if (m2 := _STEP_ASSIGN_RE.match(ln.strip())) and m2.group("var") in cands]
        if any(re.match(rf"\s*{re.escape(v)} = ", ln) for v in cands for ln in lines if ln not in step_lines):
            continue
        bases = {b for b, _ in state.values()}
        if not bases <= cands or any(state[b][0] != b for b in bases):
            continue
        head = text[: lm.start()]
        inits: dict[str, re.Match[str]] = {}
        for b in bases:
            ims = list(re.finditer(rf"^{re.escape(indent)}{re.escape(b)} = (?P<lit>-?\d+);$", head, re.MULTILINE))
            if ims:
                inits[b] = ims[-1]
        if set(inits) != bases:
            continue
        tail = text[end:]
        if any(
            _ident_re(v).search(tail[: tail.find("#pragma endscop") if "#pragma endscop" in tail else len(tail)])
            for v in cands
        ):
            continue
        if any(_ident_re(v).search(head[inits[b].end() :]) for v in cands for b in inits):
            continue
        # symbolic walk: each use takes the value its variable has at that point of the body
        cur: dict[str, tuple[str, int]] = {b: (b, 0) for b in bases}
        new_lines: list[str] = []
        for line in lines:
            sm = _STEP_ASSIGN_RE.match(line.strip())
            if sm is not None and sm.group("var") in cands and line in step_lines:
                src, c = sm.group("src"), int(sm.group("c") or 0) * (-1 if sm.group("op") == "-" else 1)
                b, off = cur[src]
                cur[sm.group("var")] = (b, off + c)
                continue
            for v in cands:
                if _ident_re(v).search(line):
                    if v not in cur:
                        ok = False
                        break
                    b, off = cur[v]
                    expr = f"({inits[b].group('lit')} + ({state[b][1]}) * ({var} - ({lo})) + ({off}))"
                    line = _ident_re(v).sub(expr, line)
            new_lines.append(line)
        if not ok:
            continue
        new_head = head
        for b in sorted(bases, key=lambda n: inits[n].start(), reverse=True):
            im = inits[b]
            new_head = new_head[: im.start()] + new_head[im.end() + 1 :]
        for v in cands:
            new_head = re.sub(
                rf"^[ \t]+u?int(?:8|16|32|64)_t {re.escape(v)};\n", "", new_head, count=1, flags=re.MULTILINE
            )
        header = lm.group(0)
        return substitute_induction_scalars(new_head + header + "\n".join(new_lines) + tail)
    return text


def floord_subscripts(text: str) -> str:
    """POLYCC-010, subscript half: ``floord(a, c)`` with a literal ``c`` inside a scop SUBSCRIPT is
    spelled as the quasi-affine ternary pet models, instead of a call it outlines into an undeclared
    ``__pet_ret``."""
    out: list[str] = []
    pos = 0
    for region in _SCOP_REGION_RE.finditer(text):
        chunk = region.group(0)
        spans = [(m.start(), _closing(chunk, m.start(), ("[", "]"))) for m in re.finditer(r"\[", chunk)]

        def repl(m: "re.Match[str]", spans: list[tuple[int, int]] = spans) -> str:
            if not any(lo < m.start() < hi for lo, hi in spans):
                return m.group(0)
            a, c = m.group("a"), m.group("c")
            return f"((({a}) < 0) ? -((-({a}) + {c} - 1) / {c}) : ({a}) / {c})"

        chunk = re.sub(r"floord\((?P<a>[^,()]+(?:\([^()]*\)[^,()]*)*), (?P<c>[1-9]\d*)\)", repl, chunk)
        out.append(text[pos : region.start()])
        out.append(chunk)
        pos = region.end()
    out.append(text[pos:])
    return "".join(out)


def opaque_helper_calls(text: str) -> str:
    """POLYCC-010, value half: a call to a macro or ``static inline`` helper of the prelude inside a scop
    (``python_mod``) goes to a body-less stand-in, so pet keeps it as a call rather than outlining it
    into an undeclared ``__pet_ret``; :func:`restore_output` renames it back."""
    arity = {
        m.group(1): len(m.group(2).split(",")) for m in re.finditer(r"^#define (\w+)\(([^)]*)\)", text, re.MULTILINE)
    }
    for m in re.finditer(r"^static inline [\w ]+?\b(\w+)\(([^)]*)\)", text, re.MULTILINE):
        arity.setdefault(m.group(1), len(m.group(2).split(",")))
    helpers = set(arity) - _KEEP_CALLS
    used: set[str] = set()
    out: list[str] = []
    pos = 0
    for region in _SCOP_REGION_RE.finditer(text):
        chunk = region.group(0)
        for h in helpers:
            if re.search(rf"(?<![\w.]){h}\(", chunk):
                used.add(h)
                chunk = re.sub(rf"(?<![\w.]){h}\(", f"{OPAQUE_PREFIX}{h}(", chunk)
        out.append(text[pos : region.start()])
        out.append(chunk)
        pos = region.end()
    out.append(text[pos:])
    text = "".join(out)
    if not used:
        return text
    protos = "".join(f"double {OPAQUE_PREFIX}{h}({', '.join(['double'] * arity[h])});\n" for h in sorted(used))
    fm = _FUNC_RE.search(text)
    at = fm.start() if fm else 0
    return text[:at] + protos + text[at:]


def restore_output(transformed: str) -> str:
    """polycc's output with :func:`opaque_helper_calls`' stand-ins renamed back to the real helper."""
    transformed = re.sub(rf"^double {OPAQUE_PREFIX}\w+\([^)\n]*\);\n", "", transformed, flags=re.MULTILINE)
    return transformed.replace(OPAQUE_PREFIX, "")


def normalize_ppcg_input(text: str) -> str:
    """The rewrites PPCG's pet needs too: they keep the entry point's signature and add no host
    local, so ppcg_hip's device-resident host rewrite still sees only its own parameters. Local scalars
    stay where they are (a pointer cell would be a host stack address on the device), and helper calls
    keep their names (``ppcg_transform.device_helpers`` copies them by name)."""
    text = inline_pinned_constants(text)
    text = forward_substitute_scalars(text)
    text = fold_constant_sign_ternaries(text)
    text = normalize_strided_loops(text)
    text = substitute_induction_scalars(text)
    return floord_subscripts(text)


def normalize_scop_input(text: str) -> str:
    """Every rewrite, in dependency order: constants and single-assignment scalars first (they expose
    literal steps and affine subscripts), then strides, induction scalars, subscript divisions, opaque
    helper calls, and last the scalars still local, which move to pointer cells."""
    return externalize_scop_scalars(opaque_helper_calls(normalize_ppcg_input(text)))
