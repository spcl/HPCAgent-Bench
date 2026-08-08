# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Affine-index detection for a ``#pragma scop`` Pluto input.

Pluto is a polyhedral (affine) optimizer: an array access whose subscript index is non-affine is
outside its model, and ``polycc`` may silently MISCOMPILE such a scop rather than reject it. The
detector here scans the scop's array subscripts and returns the first non-affine index pattern, so a
caller can deem Pluto inapplicable (a clean skip) instead of trusting a transform it cannot soundly
make.

This lives in the package (not under ``tests/``) so both the numerical oracle AND external consumers
(e.g. the nest-forge arena's Pluto lane) import the SAME detector rather than reimplementing it.
``tests.numerical_oracle`` re-exports it under its historical private name.

Beside the detector sits :data:`KNOWN_POLYCC_ISSUES`, the registry of what pet / Pluto / polycc were
MEASURED to do wrong, and of the standing caveats that make a green polycc run mean less than it
looks. Both are consumed the same way: import from here, never restate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional


def scop_nonaffine_reason(scop_c: str) -> Optional[str]:
    """Return the first non-affine array-subscript pattern in a ``#pragma scop`` body, or ``None`` when
    every subscript index is affine.

    The patterns, outside Pluto's polyhedral model, are ``indirection`` (``b[ip[i]]``), ``modulo``
    (``a[i % k]``) and ``integer-division`` (``a[i / k]``). Only the index INSIDE ``[...]`` matters --
    value-side ``/`` and ``%`` are ignored, so an affine program Pluto merely miscompiles stays a tracked
    FAIL, not a skip. When no ``#pragma scop``/``#pragma endscop`` pair is present the whole string is
    scanned (an already-extracted scop body)."""
    m = re.search(r"#pragma scop(.*?)#pragma endscop", scop_c, re.S)
    body = m.group(1) if m else scop_c
    i, n = 0, len(body)
    while i < n:
        if body[i] != "[":
            i += 1
            continue
        depth, j, inner = 1, i + 1, []
        while j < n and depth:
            c = body[j]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    break
            inner.append(c)
            j += 1
        sub = "".join(inner)
        if "[" in sub:
            return "indirection"
        if "%" in sub:
            return "modulo"
        if "/" in sub:
            return "integer-division"
        i = j + 1
    return None


@dataclass(slots=True, frozen=True)
class PolyccIssue:
    """One entry of :data:`KNOWN_POLYCC_ISSUES`.

    ``kind`` is ``bug`` (something is WRONG and was observed) or ``caveat`` (nothing is broken, but
    the polycc result means less than it appears to). ``component`` names who owns it: ``pet``,
    ``pluto``, ``polycc`` or ``translator`` (ours). ``severity`` is ``miscompile`` (wrong answer, no
    diagnostic), ``refusal`` (polycc gives up or aborts), ``latent`` (wrong only under a condition
    that does not hold today), ``detector-gap`` (we fail to SPOT an unsupported construct) or
    ``caveat``. ``symptom`` is one observational line -- what was seen, not what it implies.
    ``repro`` points at the pytest node or the file that reproduces it.

    ``avoided_by`` is the dotted path of the desugar / guard that keeps the bug out of the emitted
    scop, empty when nothing avoids it yet. It is a TRIPWIRE: a well-formedness test resolves every
    non-empty path, so deleting a desugar reds the gate BY NAME instead of silently re-opening the
    bug. ``upstream`` is ``not filed``, a tracker URL, or ``n/a`` when the owner is us.
    """

    id: str
    kind: str
    component: str
    severity: str
    symptom: str
    repro: str
    avoided_by: str
    upstream: str


_BENCH = "hpcagent_bench/benchmarks/scientific_computing"
_TRANS_TESTS = "hpcagent_bench/numpy_translators/tests"

#: Insertion-ordered registry of measured polycc/pet/Pluto defects (``POLYCC-nnn``) followed by the
#: standing caveats (``C-nnn``). Keyed by id. Every entry states what was OBSERVED; an entry with an
#: empty ``avoided_by`` is a bug we currently ship into polycc's input.
KNOWN_POLYCC_ISSUES: Dict[str, PolyccIssue] = {
    i.id: i
    for i in (
        PolyccIssue(
            id="POLYCC-001",
            kind="bug",
            component="pet",
            severity="miscompile",
            symptom=("A loop-invariant scalar assign at depth d whose consumers sit deeper is dropped "
                     "from the transformed output (conv_2d's w = w_box[..]), so the consumers read a "
                     "stale value; reproduced standalone. The assign no longer reaches pet -- the "
                     "lowering replays its RHS at the deeper use sites and deletes it."),
            repro=f"{_BENCH}/structured_grids/conv_2d -- polycc --pet --tile --parallel on its pluto input",
            avoided_by="numpyto_common.lowering._ForwardSubstituteInvariantScalars",
            upstream="not filed",
        ),
        PolyccIssue(
            id="POLYCC-002",
            kind="bug",
            component="pet",
            severity="miscompile",
            symptom=("A scop-external scalar used as an accumulator carries false WAW/WAR dependences "
                     "and stays SHARED across threads under --parallel (symm, trmm)."),
            repro=f"{_TRANS_TESTS}/test_scalar_accumulator_retarget.py",
            avoided_by="numpyto_common.lib_nodes._retarget_scalar_accumulator",
            upstream="not filed",
        ),
        PolyccIssue(
            id="POLYCC-003",
            kind="bug",
            component="pet",
            severity="refusal",
            symptom=("Writing a signature parameter inside the scop (X = X) is read as a data-dependent "
                     "condition (pet_to_pluto.cpp:565) and aborts on an isl assert "
                     "(constraints_isl.c:429) -- a core dump, not a refusal."),
            repro=f"{_TRANS_TESTS}/test_no_self_assign_in_scop.py",
            avoided_by="numpyto_common.lowering._SelfAssignDropper",
            upstream="not filed",
        ),
        PolyccIssue(
            id="POLYCC-004",
            kind="bug",
            component="polycc",
            severity="latent",
            symptom=("polycc prepends its own #define min/max/floord/ceild above the translator's "
                     "prelude. For min/max it is a non-NaN-propagating shadow, harmless only while pet "
                     "macro-expands the body first. For floord/ceild it is not harmless: unguarded, the "
                     "prepended macro expands the prelude's own static inline DECLARATORS and the "
                     "transformed output stops compiling, which is why those two are #ifndef-guarded."),
            repro="the polycc-transformed *.pluto.c preamble vs numpyto_c/emit.py's min/max + floord guards",
            avoided_by="",
            upstream="not filed",
        ),
        PolyccIssue(
            id="POLYCC-005",
            kind="bug",
            component="pluto",
            severity="refusal",
            symptom=("trmm crashed Pluto internally before the POLYCC-002 retarget existed; with the "
                     "retarget in place the same scop transforms cleanly (rc 0, 2 parallel bands), so "
                     "it is closed rather than filed."),
            repro=f"{_BENCH}/dense_linear_algebra/trmm -- polycc --pet --tile --parallel on its pluto input",
            avoided_by="numpyto_common.lib_nodes._retarget_scalar_accumulator",
            upstream="n/a",
        ),
        PolyccIssue(
            id="POLYCC-006",
            kind="bug",
            component="translator",
            severity="detector-gap",
            symptom=("scop_nonaffine_reason inspects literal subscript text only, so scalar-laundered "
                     "indirection (first_i = box_offsets[l]; rv[first_i + i]) read as affine and "
                     "reached polycc (lavamd). The laundering scalar is now substituted away, so the "
                     "subscript carries the gather literally and the detector declines the kernel."),
            repro=f"{_BENCH}/n_body_methods/lavamd -- its pluto input fails scop_nonaffine_reason",
            avoided_by="numpyto_common.lowering._ForwardSubstituteInvariantScalars",
            upstream="n/a",
        ),
        PolyccIssue(
            id="POLYCC-007",
            kind="bug",
            component="translator",
            severity="refusal",
            symptom=("A local malloc is emitted INSIDE #pragma scop when its size depends on a scop-local "
                     "scalar (dbcsr, gmres, minife, needleman_wunsch, histogram_equalization, vexx), "
                     "against the emitter's own stated invariant that allocations sit outside. It costs "
                     "the whole kernel: pet reports 'unsupported' on the sizeof and extracts no scop at "
                     "all. NOT hoistable. dbcsr's sizes (m, k, n) are read per iteration out of m_sizes / "
                     "k_sizes / n_sizes, so there is no scop-invariant expression to allocate from -- the "
                     "deferred-malloc path exists for exactly that. Of the six only needleman_wunsch is "
                     "affine enough to reach polycc, and its size scalar (M = N) IS invariant, but "
                     "hoisting the assign above the scop makes M a SECOND pluto parameter and the "
                     "schedule comes back with 2**64-scale coefficients that divide by zero (SIGFPE); "
                     "hoisting only the malloc leaves the assign to be dropped by POLYCC-009. Removing M "
                     "by forward substitution transforms and validates, but _ForwardSubstituteInvariantScalars "
                     "excludes a function-level assign by design (it would replay deriche's exp() "
                     "coefficients down a nest), so there is no fix on this entry's own ground."),
            repro=f"{_BENCH}/dynamic_programming/needleman_wunsch -- polycc --pet extracts no scop from "
            "its pluto input; invariant stated in numpyto_c.emit.emit_pluto",
            avoided_by="",
            upstream="n/a",
        ),
        PolyccIssue(
            id="POLYCC-008",
            kind="bug",
            component="pet",
            severity="refusal",
            symptom=("An int_floor call in a scop LOOP BOUND is read as a data-dependent condition "
                     "(pet_to_pluto.cpp:565) and aborts pet; first measured 08-07 on pagerank, whose "
                     "blocked-reduction lowering put int_floor(N, 128) there. pet name-matches a call "
                     "spelled floord and models it as quasi-affine, so the pluto emit spells integer "
                     "floor division that way and the same scop transforms. The blocked lowering is "
                     "gone (reassociation is sanctioned), but the spelling stays load-bearing for the "
                     "divisions the SOURCE writes -- tsvc_2_s128's floord(LEN_1D, 2) bound transforms "
                     "clean 08-07, and int_floor there would abort pet the same way."),
            repro=f"{_TRANS_TESTS}/test_pluto_named_div_builtins.py",
            avoided_by="numpyto_c.emit.pluto_floordiv",
            upstream="n/a",
        ),
        PolyccIssue(
            id="POLYCC-009",
            kind="bug",
            component="pet",
            severity="miscompile",
            symptom=("Every statement whose only write is a scop-EXTERNAL scalar is dropped from the "
                     "transformed output, so its consumers read an uninitialized value; rc 0, no "
                     "diagnostic. Reduced to a 12-line scop (s = 0.0; s += a[i]; out[i] = a[i] / s "
                     "loses both writes to s), so no floor division is involved. Remeasured 08-07 with "
                     "the blocked reduction removed: NOT collapsed. pagerank still keeps 4 of its 7 "
                     "statements (teleport, __cb2 = 0.0 and the __cb2 accumulation loop go) and still "
                     "computes inf; tsvc_2_s128 drops all three writes to its induction scalars j / k "
                     "and reads k uninitialized. What did collapse is the sub-class the POLYCC-002 "
                     "retarget reaches: a full np.sum whose result IS an array cell now accumulates "
                     "into that cell and mints no scalar -- one corpus kernel, fft_3d (__cb7 gone). "
                     "pagerank is not in it: its sum feeds an elementwise divide, so the scalar stays."),
            repro=f"{_BENCH}/graph_traversal/pagerank -- its pluto input transforms clean and computes inf",
            avoided_by="",
            upstream="not filed",
        ),
        PolyccIssue(
            id="POLYCC-010",
            kind="bug",
            component="pet",
            severity="refusal",
            symptom=("pet mints a return temporary __pet_ret_0 and never declares it, so polycc exits 0 "
                     "and the transformed output does not compile ('__pet_ret_0' undeclared). Measured "
                     "08-07 on 4 kernels once POLYCC-008 let them reach the transform: "
                     "disjoint_halves_gather, ext_floordiv_offset, triplet_margin_loss, tsvc_2_s173. "
                     "The trigger is a static inline call pet can see the BODY of -- it outlines that; "
                     "libm calls (pow, sqrt) have no body in the TU and survive verbatim. Reduced to two "
                     "spellings: floord in a SUBSCRIPT (pet name-matches it only in a loop BOUND, so "
                     "POLYCC-008's spelling stands) and __npb_fmax/__npb_fmin. The pluto emit hoists an "
                     "invariant subscript floord to a scop-external temp and spells min/max with the "
                     "prelude's own NaN-propagating macro, which pet expands away; all 4 now compile and "
                     "3 agree with the oracle. triplet_margin_loss agrees only up to POLYCC-009, which "
                     "compiling unmasks: its __cb8 accumulator is dropped from the transformed output."),
            repro=f"{_TRANS_TESTS}/test_pluto_no_helper_calls_in_scop.py",
            avoided_by="numpyto_c.emit.pluto_call_free",
            upstream="not filed",
        ),
        PolyccIssue(
            id="C-001",
            kind="caveat",
            component="pluto",
            severity="caveat",
            symptom=("--parallel privatizes only Pluto's own tile counters -- any scalar live across the "
                     "scop stays shared between threads."),
            repro=f"{_TRANS_TESTS}/test_scalar_accumulator_retarget.py -- POLYCC-002 is this caveat's instance",
            avoided_by="",
            upstream="n/a",
        ),
        PolyccIssue(
            id="C-002",
            kind="caveat",
            component="pluto",
            severity="caveat",
            symptom=("A non-affine construct is an EXPECTED refusal, not a defect: indirection (icon_*), "
                     "integer division (doitgen), sparse formats."),
            repro="tests/test_pluto_affine_detect.py",
            avoided_by="",
            upstream="n/a",
        ),
        PolyccIssue(
            id="C-003",
            kind="caveat",
            component="pet",
            severity="caveat",
            symptom=("pet macro-expands the body before Pluto emits, so the translator's macro semantics "
                     "(the NaN-propagating min/max guards) are not present in polycc's output."),
            repro="numpyto_c/emit.py -- the _C_HEADER min/max guard pair",
            avoided_by="",
            upstream="n/a",
        ),
        PolyccIssue(
            id="C-004",
            kind="caveat",
            component="translator",
            severity="caveat",
            symptom=("The numerical oracle invokes polycc --pet WITHOUT --tile --parallel, so an "
                     "oracle-green kernel does not certify the production schedule."),
            repro="tests/numerical_oracle.py:1154",
            avoided_by="",
            upstream="n/a",
        ),
        PolyccIssue(
            id="C-005",
            kind="caveat",
            component="translator",
            severity="caveat",
            symptom=("Every declaration sits above #pragma scop because mallocs are non-affine, so each "
                     "scalar declared there is scop-external by construction (see C-001)."),
            repro="numpyto_c/emit.py:2631",
            avoided_by="",
            upstream="n/a",
        ),
    )
}
