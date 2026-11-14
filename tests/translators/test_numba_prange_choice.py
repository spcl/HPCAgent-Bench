# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Which ``range`` loops the numba emitter turns into ``nb.prange`` (:mod:`numpyto_numba.prange`): the outermost
proven loop of every nest, a reduction numba carries, never a loop with a carried dependence, an aliasing or
indirect write, or a loop inside another prange -- and the pranged forms compute numpy's answer on 1 and 24
threads."""

import importlib.util
import pathlib
import sys
import textwrap

import numpy as np
import pytest

from hpcagent_bench import modules
from hpcagent_bench.translators.numpyto_numba.emit import emit_numba
from hpcagent_bench.translators.numpyto_numba.prange import parallelize_range_loops
from tests.own_process import fresh_interpreter


def pranged(src: str) -> str:
    """``src`` with its proven loops rewritten."""
    return parallelize_range_loops(textwrap.dedent(src))[0]


#: rb_sor's shape: a time loop whose body updates ``u`` through a helper, so step t+1 reads what
#: step t wrote. The helper's own row loop reads rows i-1 and i+1 of the array it writes, so it is not
#: proven either; the column loop inside it is.
TIME_LOOP_THROUGH_A_HELPER = """
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


def test_a_loop_that_mutates_an_array_through_a_helper_stays_serial() -> None:
    out = pranged(TIME_LOOP_THROUGH_A_HELPER)
    assert "for t in range(steps):" in out
    assert "for i in range(1, n - 1):" in out


def test_a_helper_that_only_reads_its_arrays_does_not_block_the_loop() -> None:
    src = """
        def weight(a, i):
            return a[i] * 2.0


        def kernel(a, out, n):
            for i in range(n):
                out[i] = weight(a, i)
        """
    out, nested = parallelize_range_loops(textwrap.dedent(src))
    assert "for i in nb.prange(n):" in out
    assert nested == {"weight"}  # it runs inside the prange, so it must be a serial build


#: The writes the helper-write check has to see past a direct ``p[...] = ...`` in the callee.
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
    "in_place_without_a_subscript": """
        def step(u, n):
            u += 1.0
            u *= 0.5
        """,
}


@pytest.mark.parametrize("helper", list(INDIRECT_HELPER_WRITES.values()), ids=list(INDIRECT_HELPER_WRITES))
def test_a_loop_that_mutates_an_array_through_any_helper_write_stays_serial(helper: str) -> None:
    src = textwrap.dedent(helper) + textwrap.dedent(
        """

        def kernel(u, n, steps):
            for t in range(steps):
                step(u, n)
        """
    )
    assert "for t in range(steps):" in parallelize_range_loops(src)[0]


#: Loops that must stay serial: a carried dependence, a written array reached through another name, an indirect
#: write, a reduction into an array cell, an exit from the body, a body scalar or the index read after the loop.
SERIAL = {
    "scan": """
        def f(a, x, n):
            for i in range(1, n):
                a[i] = a[i - 1] + x[i]
        """,
    "carried_scalar": """
        def f(a, b, n):
            s = 0.0
            for i in range(n):
                b[i] = s
                s = a[i]
        """,
    "alias_view": """
        def f(a, n):
            b = a[1:]
            for i in range(n - 1):
                b[i] = a[i] * 2.0
        """,
    "alias_name": """
        def f(a, n):
            b = a
            for i in range(n):
                b[i] = a[n - 1 - i]
        """,
    "indirect_write": """
        def f(a, idx, x, n):
            for i in range(n):
                a[idx[i]] = x[i]
        """,
    "cell_reduction": """
        def f(a, out, n):
            for i in range(n):
                out[0] = out[0] + a[i]
        """,
    "break": """
        def f(a, out, n):
            for i in range(n):
                if a[i] < 0.0:
                    break
                out[i] = a[i]
        """,
    "live_out_scalar": """
        def f(a, out, n):
            last = 0.0
            for i in range(n):
                last = a[i] * 2.0
                out[i] = last
            out[0] = last
        """,
    "live_out_index": """
        def f(a, out, n):
            i = 0
            for i in range(n):
                out[i] = a[i]
            out[0] = i
        """,
    "reduction_seeded_from_an_array": """
        def f(a, out, n):
            s = a[0]
            for i in range(n):
                s += a[i]
            out[0] = s
        """,
    "helper_handed_the_same_array_twice": """
        def g(dst, src, n):
            for i in range(1, n):
                dst[i] = src[i - 1]


        def f(a, n):
            g(a, a, n)
        """,
}


@pytest.mark.parametrize("src", list(SERIAL.values()), ids=list(SERIAL))
def test_an_unproven_loop_stays_a_serial_range(src: str) -> None:
    assert "nb.prange" not in pranged(src)


def test_only_the_outermost_proven_loop_of_a_nest_is_a_prange() -> None:
    out = pranged(
        """
        def f(a, b, n, m):
            for i in range(n):
                for j in range(m):
                    b[i, j] = a[i, j] * 2.0
            for t in range(1, n):
                for j in range(m):
                    a[t, j] = a[t - 1, j] + 1.0
        """
    )
    assert "for i in nb.prange(n):" in out
    assert "for j in range(m):\n            b[i, j]" in out
    assert "for t in range(1, n):" in out  # carried along t
    assert "for j in nb.prange(m):\n            a[t, j]" in out  # independent along j, inside t


def test_a_function_running_inside_a_prange_keeps_no_prange_of_its_own() -> None:
    out, nested = parallelize_range_loops(
        textwrap.dedent(
            """
            def row(a, out, i, m):
                for j in range(m):
                    out[i, j] = a[i, j] + 1.0


            def f(a, out, n, m):
                for i in range(n):
                    row(a, out, i, m)
            """
        )
    )
    # ``row`` writes the array it is handed, so the caller's loop stays serial and ``row``'s own loop is the prange.
    assert "for i in range(n):" in out
    assert "for j in nb.prange(m):" in out
    assert nested == set()


#: Forms that become a prange and must still compute numpy's answer: an element map, a row accumulator private
#: to each iteration, a sum, a conditional sum, a product and a max reduction, and a nest whose inner loop runs
#: serial inside each outer iteration.
PARALLEL = textwrap.dedent(
    """
    import numpy as np


    def f(a, m, out, rows):
        n = a.shape[0]
        for i in range(n):
            out[i] = a[i] * 2.0 + 1.0
        for i in range(n):
            acc = 0.0
            for j in range(m.shape[1]):
                acc += m[i, j] * a[j % n]
            rows[i] = acc
        total = 0.0
        for i in range(n):
            total += a[i]
        positive = 0.0
        for i in range(n):
            if a[i] > 0.0:
                positive = positive + a[i]
        prod = 1.0
        for i in range(n):
            prod *= 1.0 + a[i] * 1e-6
        top = -np.inf
        for i in range(n):
            top = max(top, a[i])
        out[0] = total
        out[1] = positive
        out[2] = prod
        out[3] = top
    """
)


def test_the_proven_forms_become_pranges() -> None:
    emitted = emit_numba(PARALLEL)
    assert emitted.count("nb.prange(") == 6, emitted
    assert "for j in range(m.shape[1]):" in emitted


def run_threads(source: str, threads: int, workdir: str) -> list[np.ndarray]:
    """``f`` of ``source`` on ``threads`` numba threads, in this (fresh) interpreter."""
    import numba

    numba.set_num_threads(threads)
    path = pathlib.Path(workdir) / f"pranged_{threads}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    rng = np.random.default_rng(11)
    a, m = rng.standard_normal(200_000), rng.standard_normal((200_000, 8))
    out, rows = np.zeros(200_000), np.zeros(200_000)
    modules.function(module, "f")(a, m, out, rows)
    return [out, rows]


@pytest.mark.parametrize("threads", [1, 24])
def test_the_pranged_forms_compute_numpys_answer(threads: int, tmp_path: pathlib.Path) -> None:
    namespace: dict[str, object] = {}
    exec(compile(PARALLEL, "parallel", "exec"), namespace)  # noqa: S102 -- the test's own numpy source
    rng = np.random.default_rng(11)
    a, m = rng.standard_normal(200_000), rng.standard_normal((200_000, 8))
    want = [np.zeros(200_000), np.zeros(200_000)]
    namespace["f"](a, m, *want)  # type: ignore[operator]
    got = fresh_interpreter(
        run_threads,
        emit_numba(PARALLEL),
        threads,
        str(tmp_path),
        env={"NUMBA_NUM_THREADS": "24", "NUMBA_THREADING_LAYER": "workqueue"},
    )
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(g, w, rtol=1e-12, atol=1e-9)  # sums reassociate across threads
