# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Iteration bounds shared by the translators' fixpoint passes.

Each pass loops until nothing changes; the bound only stops a pass that would otherwise never converge,
so it is far above the depth any real kernel needs."""

__all__ = [
    "CALL_GRAPH_ROUNDS",
    "FIXPOINT_ROUNDS",
    "HELPER_INLINE_ROUNDS",
    "HELPER_NESTING_ROUNDS",
    "SHALLOW_ROUNDS",
]

#: Rounds of a straight-line propagation (dtype kinds, ranks, shape tables) before it is declared converged.
FIXPOINT_ROUNDS: int = 8

#: Rounds for a table whose entries refer to each other at most one level deep (a temp naming another temp).
SHALLOW_ROUNDS: int = 4

#: Rounds of a pass over the call graph (helper return kinds and ranks, transposed-name chains).
CALL_GRAPH_ROUNDS: int = 6

#: Levels of nested helper definitions a helper is flattened through.
HELPER_NESTING_ROUNDS: int = 16

#: Passes of helper inlining: one pass inlines one nesting level of calls.
HELPER_INLINE_ROUNDS: int = 64
