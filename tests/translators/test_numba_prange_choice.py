# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Which ``range`` loop the numba emitter turns into ``nb.prange``."""

import textwrap

from hpcagent_bench.translators.numpyto_numba.parfor import parallelize_one_range_loop

#: rb_sor's shape: a time loop whose body updates ``u`` through a helper, so step t+1 reads what
#: step t wrote; a prange over the time loop raced and graded 6e-5 .. 1e-4 off numpy, run to run.
#: The helper's own column loop is independent, so that loop is the one prange may take.
TIME_LOOP_THROUGH_A_HELPER = textwrap.dedent(
    """
    def half_sweep(u, f, n, parity):
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                if (i + j) % 2 == parity:
                    u[i, j] = (u[i - 1, j] + u[i + 1, j] + f[i, j]) / 3.0


    def kernel(f, u, n, steps):
        for t in range(steps):
            half_sweep(u, f, n, 0)
            half_sweep(u, f, n, 1)
    """
)


def test_a_loop_that_mutates_an_array_through_a_helper_stays_serial() -> None:
    out = parallelize_one_range_loop(TIME_LOOP_THROUGH_A_HELPER)
    assert "for t in range(steps):" in out
    assert "for j in nb.prange(1, n - 1):" in out


def test_a_helper_that_only_reads_its_arrays_does_not_block_the_loop() -> None:
    """A call that writes none of the arrays it is handed carries no dependence, so the loop
    around it may still run in parallel."""
    src = textwrap.dedent(
        """
        def weight(a, i):
            return a[i] * 2.0


        def kernel(a, out, n):
            for i in range(n):
                out[i] = weight(a, i)
        """
    )
    assert "for i in nb.prange(n):" in parallelize_one_range_loop(src)


#: The writes the helper-write check has to see past a direct ``p[...] = ...`` in the callee: a
#: wrapper around the writing helper, a view of the parameter, an ``out=`` argument, a mutating
#: method, and the array handed over by keyword. In each, the time loop's step t+1 reads what step
#: t wrote, so a prange over it races.
INDIRECT_HELPER_WRITES = {
    "through_a_wrapper": """
        def relax(u, n):
            for j in range(1, n - 1):
                u[j] = 0.5 * (u[j - 1] + u[j + 1])


        def step(u, n):
            relax(u, n)
        """,
    "through_a_view": """
        def step(u, n):
            row = u[1:n]
            row[0] = row[1] * 0.5
        """,
    "through_out": """
        def step(u, n):
            np.multiply(u, 0.5, out=u)
        """,
    "through_a_method": """
        def step(u, n):
            u.sort()
        """,
}


def test_a_loop_that_mutates_an_array_through_any_helper_write_stays_serial() -> None:
    for name, helper in INDIRECT_HELPER_WRITES.items():
        src = textwrap.dedent(helper) + textwrap.dedent(
            """

            def kernel(u, n, steps):
                for t in range(steps):
                    step(u, n)
            """
        )
        out = parallelize_one_range_loop(src)
        assert "for t in range(steps):" in out, f"{name}: the time loop became a prange\n{out}"


def test_an_array_handed_by_keyword_to_a_writing_helper_keeps_the_loop_serial() -> None:
    src = textwrap.dedent(
        """
        def step(u, n):
            u[0] = u[n - 1]


        def kernel(u, n, steps):
            for t in range(steps):
                step(n=n, u=u)
        """
    )
    assert "for t in range(steps):" in parallelize_one_range_loop(src)
