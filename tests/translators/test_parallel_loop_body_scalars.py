# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scalars a parallel loop body binds: private to each thread, and never read after the loop.

Two defects of the same parallel-loop decision:

* the OpenMP emits declare every body scalar once at function top and tagged the loop with a bare
  ``parallel for`` / ``parallel do``, so all threads shared the scalar -- a data race (14 + 157441
  wrong elements at -O0 with 16 threads on a two-temp body);
* a body scalar that is also read AFTER the loop passed the parallel-safety check whenever the body
  read it too, so numba's ``prange`` handed back an arbitrary iteration's value instead of the last.
"""

import ast
import ctypes
import importlib.util
import pathlib
import subprocess

import numpy as np
import pytest

from hpcagent_bench import numerical_oracle
from hpcagent_bench.translators.numpyto_c.emit import emit_c_omp
from hpcagent_bench.translators.numpyto_common.ir import KernelIR
from hpcagent_bench.translators.numpyto_common.lowering import lower
from hpcagent_bench.translators.numpyto_common.parallelism import loop_is_parallel_safe
from hpcagent_bench.translators.numpyto_fortran.emit import emit_fortran_omp
from hpcagent_bench.translators.numpyto_numba.emit import emit_numba
from tests.fresh_module import module_at
from tests.translators.op_oracle import native, parse_source

TWO_TEMPS = (
    "def f(x, r, q):\n"
    "    for i in range(N):\n"
    "        t = x[i] * 2.0\n"
    "        u = t + 1.0\n"
    "        r[i] = t * u\n"
    "        q[i] = u - t * t\n"
)
SHAPES = {"x": "(N,)", "r": "(N,)", "q": "(N,)"}
#: Large enough that a shared temp is overwritten by another thread between its store and its load.
RACE_N = 2_000_000
#: Thread count the OpenMP build runs with: a shared temp races at any count above one.
RACE_THREADS = 8


def lowered(src: str, n: int) -> KernelIR:
    return lower(parse_source(src, "f", ["x"], ["r", "q"], SHAPES, {"N": n}))


def test_c_omp_privatises_the_body_temps() -> None:
    c = emit_c_omp(lowered(TWO_TEMPS, 16), fn_name="f")
    pragma = next(line for line in c.splitlines() if "#pragma omp parallel for" in line)
    assert "private(t, u)" in pragma, pragma


def test_fortran_omp_privatises_the_body_temps() -> None:
    f = emit_fortran_omp(lowered(TWO_TEMPS, 16), fn_name="f")
    pragma = next(line for line in f.splitlines() if "!$omp parallel do" in line)
    assert "private(t, u)" in pragma, pragma


def test_c_omp_with_body_temps_matches_numpy_on_many_threads(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native()
    monkeypatch.setenv("OMP_NUM_THREADS", str(RACE_THREADS))
    kir = lowered(TWO_TEMPS, RACE_N)
    src = tmp_path / "f.c"
    src.write_text(emit_c_omp(kir, fn_name="f"))
    so = tmp_path / "libf.so"
    # -O0 keeps every temp in memory, where a shared one is visibly overwritten.
    cmd = numerical_oracle.native_build_command("c", src, so, extra_compile=("-fopenmp", "-O0"))
    built = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert built.returncode == 0, built.stderr[-2000:]
    x = np.random.default_rng(0).standard_normal(RACE_N)
    r, q = np.zeros(RACE_N), np.zeros(RACE_N)
    buffers = {"x": x, "r": r, "q": q}
    args = [
        ctypes.c_void_p(buffers[n].ctypes.data) if n in buffers else ctypes.c_int64(RACE_N) for n in kir.param_order()
    ]
    ctypes.CDLL(str(so)).f(*args)
    t = x * 2.0
    u = t + 1.0
    np.testing.assert_array_equal(r, t * u)
    np.testing.assert_array_equal(q, u - t * t)


LIVE_AFTER = (
    "import numpy as np\n"
    "def f(a, b, c):\n"
    "    t = 0.0\n"
    "    for i in range(a.shape[0]):\n"
    "        t = a[i] * 2.0\n"
    "        b[i] = t\n"
    "    c[0] = t\n"
)
REBOUND_AFTER = (
    "import numpy as np\n"
    "def f(a, b, c):\n"
    "    for i in range(a.shape[0]):\n"
    "        t = a[i] * 2.0\n"
    "        b[i] = t\n"
    "    t = 5.0\n"
    "    c[0] = t\n"
)


def test_a_loop_whose_body_scalar_is_read_after_it_stays_serial() -> None:
    tree = ast.parse(LIVE_AFTER)
    assert not loop_is_parallel_safe(first_loop(tree), tree)
    assert "nb.prange" not in emit_numba(LIVE_AFTER)


def test_a_body_scalar_rebound_before_any_later_read_keeps_the_loop_parallel() -> None:
    """The check is liveness, not "the name appears again": a later rebinding kills the value."""
    tree = ast.parse(REBOUND_AFTER)
    assert loop_is_parallel_safe(first_loop(tree), tree)
    assert "nb.prange" not in emit_numba(REBOUND_AFTER)


def first_loop(tree: ast.Module) -> ast.For:
    """The first ``for`` loop of ``tree``."""
    return next(n for n in ast.walk(tree) if isinstance(n, ast.For))


def test_numba_returns_the_last_iterations_scalar(tmp_path: pathlib.Path) -> None:
    if importlib.util.find_spec("numba") is None:
        pytest.skip("numba not installed")
    mod = tmp_path / "live_after_nb.py"
    mod.write_text(emit_numba(LIVE_AFTER))
    m = module_at(mod, "live_after_nb")
    a = np.arange(1000.0)
    b, c = np.zeros(1000), np.zeros(1)
    m.f(a, b, c)
    assert c[0] == a[-1] * 2.0, c[0]


RESPELLED = (
    "def f(x, r, q):\n"
    "    for i in range(N):\n"
    "        default = x[i] * 2.0\n"
    "        _t = default + 1.0\n"
    "        r[i] = default * _t\n"
    "        q[i] = _t\n"
)


def test_the_private_clause_names_scalars_as_the_emit_declares_them() -> None:
    """A C keyword and a leading underscore are respelled in the declaration; the clause must
    name the declared variable, or it does not compile."""
    kir = lowered(RESPELLED, 16)
    c = emit_c_omp(kir, fn_name="f")
    pragma = next(line for line in c.splitlines() if "#pragma omp parallel for" in line)
    for name in pragma.split("private(", 1)[1].rstrip(")").split(", "):
        assert f" {name};" in c or f" {name} =" in c, (name, c)
    f = emit_fortran_omp(kir, fn_name="f")
    pragma = next(line for line in f.splitlines() if "!$omp parallel do" in line)
    for name in pragma.split("private(", 1)[1].rstrip(")").split(", "):
        assert f":: {name}" in f, (name, f)
