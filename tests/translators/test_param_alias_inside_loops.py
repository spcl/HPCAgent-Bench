# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""``local = <param>`` aliases are folded onto the parameter at any nesting depth.

numpy's ``f = field`` is the same buffer, so a helper that fills halo corners through ``f`` writes the
caller's array. Inlining that helper into a time-step loop leaves the alias INSIDE the loop body, where
the fold used to look only at the function's top-level statements: the native backends then copied
``q`` into a fresh ``f``, the corner fills never reached ``q`` and every later read of ``q`` saw
unfilled corners (fv3_dycore's C result disagreed with numpy from the first step on).
"""

import ast
import textwrap

from hpcagent_bench.translators.numpyto_common.frontend.body_rewrites import SubstituteParamAliases


def folded(src: str, params: list[str]) -> str:
    fn = ast.parse(textwrap.dedent(src)).body[0]
    assert isinstance(fn, ast.FunctionDef)
    sub = SubstituteParamAliases(params)
    sub.collect(fn)
    sub.visit(fn)
    return ast.unparse(fn)


def test_an_alias_bound_inside_a_loop_is_folded_onto_the_parameter() -> None:
    got = folded(
        """
        def kernel(q, nsteps):
            for _step in range(nsteps):
                f = q
                f[0, 0] = f[5, 0]
                q[1, 1] -= f[1, 1]
        """,
        ["q", "nsteps"],
    )
    assert "f = q" not in got and "f[" not in got, got
    assert "q[0, 0] = q[5, 0]" in got, got


def test_an_alias_rebound_in_the_loop_is_left_alone() -> None:
    src = """
        def kernel(q, nsteps):
            f = q
            for _step in range(nsteps):
                f = f * 2.0
                f[0, 0] = 1.0
        """
    got = folded(src, ["q", "nsteps"])
    assert "f = q" in got and "f = f * 2.0" in got, got
