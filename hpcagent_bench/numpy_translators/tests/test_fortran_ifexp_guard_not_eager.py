# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A Fortran ``IfExp`` must run only the TAKEN branch.

Fortran has no ternary operator, so the emitter used to lower ``b if t else c`` to
``merge(b, c, t)``. ``merge`` is an ORDINARY FUNCTION CALL: all three arguments are evaluated
before it selects one. That is exactly backwards for the guard an ``IfExp`` is usually written
for -- ``y = a / x if x != 0.0 else 0.0`` divides by zero on the excluded value anyway, and
``a[idx[i]] if idx[i] < n else 0.0`` reads out of bounds anyway. C's ``?:`` short-circuits, so
this was Fortran-only. ``numpyto_fortran.emit._hoist_ifexp`` now lowers every ``IfExp`` to an
explicit ``if/else`` over a fresh temp on the Fortran-only tree copy.

Both probes run the compiled kernel from a generated Fortran PROGRAM, in a subprocess:

* the FPE trap ``-ffpe-trap=zero`` arms is installed by ``libgfortran``'s PROGRAM-level startup
  code, which never runs for a bare ``dlopen``ed subroutine -- a ctypes call into a shared object
  does not trap however the object was compiled;
* an aborting kernel is the SIGNAL these tests read, so it must not be able to take the pytest
  worker down with it.

Compiled at ``-O0``. Measured on this box at ``-O1``/``-O2``/``-O3``: GCC will not speculate a
trapping operation, so it turns the eager ``merge`` back into a real branch and neither probe
fires for the unfixed emitter either -- the test would pass vacuously. ``-O0`` tests the emitted
SOURCE's semantics rather than an optimizer's mercy, which is why these do not reuse the shared
oracle's ``-O2`` flags (``tests/numerical_oracle.py::COMPILE``).
"""
import json
import pathlib
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pytest

import _op_oracle as oo
from numpyto_common import dtypes

#: Deliberately NOT the shared oracle's flags -- see the module docstring for why -O0.
_FORTRAN_O0 = ["gfortran", "-O0", "-ffree-form", "-ffree-line-length-none"]

_N = 6

_DIV_SRC = ("import numpy as np\n"
            "def f(a, x, out):\n"
            " for i in range(x.shape[0]):\n"
            "  out[i] = a[i] / x[i] if x[i] != 0.0 else 0.0\n")

_OOB_SRC = ("import numpy as np\n"
            "def g(a, idx, out):\n"
            " for i in range(idx.shape[0]):\n"
            "  out[i] = a[idx[i]] if idx[i] < a.shape[0] else 0.0\n")

#: ``x[1]``, ``x[3]``, ``x[5]`` are the excluded (guarded-out) elements: the eager ``merge`` form
#: divides ``a[i]`` by that zero before selecting the 0.0 the guard asks for.
_DIV_A = np.array([10.0, 999.0, 20.0, -999.0, 30.0, 5.0])
_DIV_X = np.array([1.0, 0.0, 2.0, 0.0, -3.0, 0.0])

#: ``idx[1]``, ``idx[3]``, ``idx[5]`` are far past the end of ``a``: the eager form subscripts
#: ``a`` with them anyway. 4096 elements past a 6-element array is well outside the allocation,
#: so this is a genuine out-of-bounds read, not a neighbouring-object one.
_OOB_A = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
_OOB_IDX = np.array([0, 4096, 2, 4096, 4, 4096], dtype=np.int64)


def _python_reference(src: str, func: str, args: Sequence[np.ndarray], out: np.ndarray) -> np.ndarray:
    """Run the kernel's OWN Python body. Python's conditional expression short-circuits, so the
    excluded branch is never evaluated -- no divide-by-zero warning and no out-of-range index, and
    no hand-derived vectorized stand-in that could silently drift from the source under test."""
    ns: Dict[str, Any] = {}
    exec(compile(src, "<ref>", "exec"), ns)  # noqa: S102 -- the kernel source is a module constant
    ns[func](*args, out)
    return out


def _emit(tdp: pathlib.Path, src: str, func: str, inputs: List[str], outputs: List[str],
          elem_dtypes: Dict[str, str]) -> Tuple[str, Dict[str, Any]]:
    """Emit ``src`` through the real translator front end; return the Fortran text and its binding."""
    npy = tdp / f"{func}.py"
    npy.write_text(src)
    shapes = {name: f"({_N}, )" for name in inputs + outputs}
    bi = tdp / "bench_info.json"
    bi.write_text(json.dumps(oo._bench_info(func, inputs, outputs, shapes, {"N": _N}, elem_dtypes)))
    oo._emit_native(npy, bi, tdp, func)
    return (tdp / f"{func}.f90").read_text(), json.loads((tdp / f"{func}_binding.json").read_text())


def _driver_source(binding: Dict[str, Any], out_names: Sequence[str]) -> str:
    """A Fortran PROGRAM calling the emitted ``bind(C)`` subroutine through an interface built from
    its own binding JSON: every array arg is read from stdin in binding order and every output is
    written back to stdout, so the driver needs no knowledge of the kernel beyond that file."""
    iface: List[str] = []
    decls: List[str] = []
    setup: List[str] = []
    actual: List[str] = []
    for arg in binding["args"]:
        name, kind = arg["name"], arg["kind"]
        ftype = dtypes.info_for_kind(kind).fortran
        actual.append(name)
        if kind in dtypes.SCALAR_KINDS:  # a by-value size symbol
            iface.append(f"{ftype}, value, intent(in) :: {name}")
            decls.append(f"{ftype} :: {name}")
            setup.append(f"{name} = {_N}")
        else:
            iface.append(f"{ftype}, intent(inout) :: {name}(*)")
            decls.append(f"{ftype} :: {name}({_N})")
            setup.append(f"read (*, *) {name}")
    body = "\n".join(f"    {line}" for line in setup)
    writes = "\n".join(f"    write (*, '(ES24.16)') {name}" for name in out_names)
    return ("program drv\n"
            "    use, intrinsic :: iso_c_binding\n"
            "    implicit none\n"
            "    interface\n"
            f"        subroutine hpcagent_kernel({', '.join(actual)}) bind(C, name=\"{binding['kernel']}\")\n"
            "            use, intrinsic :: iso_c_binding\n" + "".join(f"            {d}\n" for d in iface) +
            "        end subroutine hpcagent_kernel\n"
            "    end interface\n" + "".join(f"    {d}\n" for d in decls) + f"{body}\n"
            f"    call hpcagent_kernel({', '.join(actual)})\n"
            f"{writes}\n"
            "end program drv\n")


def _run_driver(tdp: pathlib.Path, f90: pathlib.Path, binding: Dict[str, Any], buffers: Dict[str, np.ndarray],
                out_names: Sequence[str], extra_flags: Sequence[str]) -> subprocess.CompletedProcess:
    """Compile kernel + generated driver at -O0 with ``extra_flags`` and run it, feeding every array
    arg on stdin in binding order. Returns the completed subprocess (exit status IS the probe)."""
    drv = tdp / "drv.f90"
    drv.write_text(_driver_source(binding, out_names))
    exe = tdp / "drv"
    cc = subprocess.run(_FORTRAN_O0 + list(extra_flags) + [str(f90), str(drv), "-o", str(exe)],
                        capture_output=True,
                        text=True)
    assert cc.returncode == 0, cc.stderr
    ptr_args = [a["name"] for a in binding["args"] if a["kind"] not in dtypes.SCALAR_KINDS]
    stdin = "".join(" ".join(repr(v.item()) for v in buffers[name]) + "\n" for name in ptr_args)
    return subprocess.run([str(exe)], input=stdin, capture_output=True, text=True, timeout=120)


def _outputs(proc: subprocess.CompletedProcess, count: int) -> np.ndarray:
    values = [float(line) for line in proc.stdout.split()]
    assert len(values) == count, f"driver printed {len(values)} values, expected {count}: {proc.stdout!r}"
    return np.array(values)


@pytest.mark.integration
def test_guarded_division_does_not_divide_by_zero():
    if not shutil.which("gfortran"):
        pytest.skip("gfortran needed to compile the emitted Fortran")
    expected = _python_reference(_DIV_SRC, "f", (_DIV_A, _DIV_X), np.zeros(_N))
    assert np.isfinite(expected).all()  # the reference itself never divides by zero

    with tempfile.TemporaryDirectory() as td:
        tdp = pathlib.Path(td)
        text, binding = _emit(tdp, _DIV_SRC, "f", ["a", "x"], ["out"], {})
        assert "merge(" not in text, f"IfExp still lowers to an eager merge():\n{text}"
        buffers = {"a": _DIV_A, "x": _DIV_X, "out": np.zeros(_N)}
        # -ffpe-trap=zero: the excluded a[i] / 0.0 raises SIGFPE the instant it executes.
        proc = _run_driver(tdp, tdp / "f.f90", binding, buffers, ["out"], ["-ffpe-trap=zero,invalid,overflow"])
        assert proc.returncode == 0, ("the guarded-out branch's division by zero ran -- the IfExp is "
                                      f"evaluating both sides (rc={proc.returncode}):\n{proc.stderr}")
        np.testing.assert_allclose(_outputs(proc, _N), expected, rtol=1e-12, atol=0.0)


@pytest.mark.integration
def test_guarded_subscript_does_not_read_out_of_bounds():
    if not shutil.which("gfortran"):
        pytest.skip("gfortran needed to compile the emitted Fortran")
    expected = _python_reference(_OOB_SRC, "g", (_OOB_A, _OOB_IDX), np.zeros(_N))

    with tempfile.TemporaryDirectory() as td:
        tdp = pathlib.Path(td)
        text, binding = _emit(tdp, _OOB_SRC, "g", ["a", "idx"], ["out"], {"idx": "int64"})
        assert "merge(" not in text, f"IfExp still lowers to an eager merge():\n{text}"
        buffers = {"a": _OOB_A, "idx": _OOB_IDX, "out": np.zeros(_N)}
        # -fcheck=bounds: the excluded a(4097) subscript aborts the instant it executes.
        proc = _run_driver(tdp, tdp / "g.f90", binding, buffers, ["out"], ["-fcheck=bounds"])
        assert proc.returncode == 0, ("the guarded-out branch's out-of-bounds read ran -- the IfExp is "
                                      f"evaluating both sides (rc={proc.returncode}):\n{proc.stderr}")
        np.testing.assert_allclose(_outputs(proc, _N), expected, rtol=1e-12, atol=0.0)


def test_while_test_ifexp_is_reevaluated_across_continue():
    """A ``while`` whose TEST is an ``IfExp`` re-runs that test every iteration, so the hoisted temp
    has to be recomputed at the loop tail AND before every ``continue`` -- ``continue`` emits as
    Fortran ``cycle``, which jumps straight past the tail. Reading a stale temp is a wrong VALUE,
    not a trap, so the ordinary oracle catches it: on this data the stale form runs two extra
    iterations and reports ``t = 3.0, i = 4`` instead of ``t = 2.0, i = 2``."""
    src = ("import numpy as np\n"
           "def w(x, out):\n"
           " i = 0\n"
           " t = 0.0\n"
           " while (1.0 / x[i] if x[i] != 0.0 else 0.0) >= 0.25:\n"
           "  i = i + 1\n"
           "  if x[i] < 0.0:\n"
           "   continue\n"
           "  t = t + x[i]\n"
           " out[0] = t\n"
           " out[1] = float(i)\n")
    x = np.array([4.0, 2.0, -1.0, 1.0, 0.0, 8.0])
    status = oo.run_op(src,
                       "w", {"x": x}, {"out": (_N, )}, {"N": _N},
                       shapes={
                           "x": "(N,)",
                           "out": "(N,)"
                       },
                       backends=("c", "cpp", "fortran"))
    assert status == {"c": "ok", "cpp": "ok", "fortran": "ok"}, status


def test_guarded_ifexp_c_and_cpp_unaffected():
    # C's ?: already short-circuits; the Fortran-only hoist must not have disturbed those emitters.
    shapes = {"a": "(N,)", "x": "(N,)", "out": "(N,)"}
    status = oo.run_op(_DIV_SRC,
                       "f", {
                           "a": _DIV_A,
                           "x": _DIV_X
                       }, {"out": (_N, )}, {"N": _N},
                       shapes=shapes,
                       backends=("c", "cpp"))
    assert status == {"c": "ok", "cpp": "ok"}, status
