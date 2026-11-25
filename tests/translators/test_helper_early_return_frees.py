# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generated C/C++ frees its heap locals on EVERY exit, not just the last one.

A helper's frees are emitted after its body, so a data-dependent early ``return`` jumped straight
over them. Only helpers can return at all -- the kernel is void and its returns are dropped -- so
the leak was invisible in every kernel-level test, and it is the worst possible shape: the caller
is a benchmark loop, so the helper leaks its workspace once per element per rep, and a run long
enough to measure is a run long enough to exhaust the box.

Under AddressSanitizer, which fails the run on a leak rather than asking a human to read a number.
"""

import pytest

from hpcagent_bench.translators.numpyto_c.emit import emit_c, emit_cpp
from hpcagent_bench.translators.numpyto_common.lowering import lower
from tests.translators.native_tu import build_run_c, have_gcc, have_gpp
from tests.translators.op_oracle import parse_source

#: ``scratch`` allocates a workspace AFTER a guard that returns early, so one call in three exits
#: with the buffer live and one exits with it allocated. The early return is also what stops the
#: inliner absorbing the helper, which is why it survives as a real function with a real free.
SOURCE = (
    "import numpy as np\n\n\n"
    "def scratch(v, N):\n"
    "    if v < 0.0:\n"
    "        return 0.0\n"
    "    t = np.zeros((N,))\n"
    "    for k in range(N):\n"
    "        t[k] = v * (k + 1)\n"
    "    if v > 1.0:\n"
    "        return t[0]\n"
    "    return t[1]\n\n\n"
    "def k(a, out, N):\n"
    "    for i in range(N):\n"
    "        out[i] = scratch(a[i], N)\n"
)

DRIVER = (
    "int main(void) {\n"
    "    enum { N = 8 };\n"
    "    double a[N], out[N];\n"
    "    for (int i = 0; i < N; ++i) a[i] = (double)i - 3.5;\n"
    "    for (int rep = 0; rep < 32; ++rep) k(a, out, N);\n"
    "    return out[N - 1] == out[N - 1] ? 0 : 1;\n"
    "}\n"
)


def emitted(cpp: bool, source: str = SOURCE) -> str:
    kir = lower(parse_source(source, "k", ["a"], ["out"], {"a": "(N,)", "out": "(N,)"}, {"N": 8}))
    return emit_cpp(kir, fn_name="k") if cpp else emit_c(kir, fn_name="k")


def helper_of(text: str) -> str:
    """Just the helper, so a free in the KERNEL cannot stand in for one the helper owes."""
    start = text.index("scratch(")
    return text[start : text.index("\n}\n", start)]


@pytest.mark.parametrize("cpp", [False, True], ids=["c", "cpp"])
def test_the_helper_frees_its_workspace_on_every_return(cpp: bool) -> None:
    helper = helper_of(emitted(cpp=cpp))
    assert "malloc(" in helper, f"the helper stopped allocating, so this proves nothing:\n{helper}"
    # The allocation is hoisted to function top, so ALL THREE exits owe a free -- including the
    # guard that reads as "before" it in the numpy source. No fourth: the body ends in a return, so
    # a closing free would be unreachable.
    assert helper.count("return ") == 3, helper
    assert helper.count("free(") == 3, helper


#: A helper returning a heap local BY VALUE. The array return is supposed to become an out-param;
#: reaching a scalar return with a live buffer means the classification went wrong upstream.
RETURNS_BUFFER = (
    "import numpy as np\n\n\n"
    "def build(v, N):\n"
    "    t = np.zeros((N,))\n"
    "    s = np.zeros((N,))\n"
    "    for k in range(N):\n"
    "        t[k] = v * (k + 1)\n"
    "    if v < 0.0:\n"
    "        return s\n"
    "    return t\n\n\n"
    "def k(a, out, N):\n"
    "    for i in range(N):\n"
    "        r = build(a[i], N)\n"
    "        out[i] = r[0]\n"
)


def test_returning_a_heap_local_by_value_is_refused_by_name() -> None:
    """Freeing on the way out must never hand back a dangling pointer. Emitting the free would; NOT
    emitting it leaks; the emitted C does not even typecheck. So it is refused, naming the buffer.

    The refusal is raised while the helper is CLASSIFIED, not while its return is emitted: the C
    emitter's own guard fires per backend, and Fortran had none -- it emitted a subroutine gfortran
    rejects outright ("VALUE attribute conflicts with FUNCTION attribute"). Refusing once, in the
    frontend, covers every backend from one place; the name of the buffer is still in the message.
    """
    with pytest.raises(NotImplementedError, match=r"returns its own array 't' by value"):
        emitted(cpp=False, source=RETURNS_BUFFER)
    # Same refusal for C++ and Fortran, which is what moving it upstream buys.
    with pytest.raises(NotImplementedError, match=r"returns its own array 't' by value"):
        emitted(cpp=True, source=RETURNS_BUFFER)


#: A local whose EXTENT is computed in the loop body, so its allocation is deferred to a marker
#: inside that loop -- emitted once, executed once per iteration.
IN_LOOP_ALLOC = (
    "import numpy as np\n\n\n"
    "def k(a, out, N):\n"
    "    for i in range(N):\n"
    "        m = i + 1\n"
    "        t = np.zeros((m,))\n"
    "        for j in range(m):\n"
    "            t[j] = a[j] * 2.0\n"
    "        out[i] = t[0]\n"
)

IN_LOOP_DRIVER = (
    "int main(void) {\n"
    "    enum { N = 8 };\n"
    "    double a[N], out[N];\n"
    "    for (int i = 0; i < N; ++i) a[i] = (double)i + 1.0;\n"
    "    for (int rep = 0; rep < 16; ++rep) k(a, out, N);\n"
    "    return out[0] == out[0] ? 0 : 1;\n"
    "}\n"
)


@pytest.mark.parametrize(
    ("source", "driver", "cpp"),
    [
        pytest.param(IN_LOOP_ALLOC, IN_LOOP_DRIVER, False, marks=have_gcc, id="c_deferred_loop_allocation"),
        pytest.param(SOURCE, DRIVER, False, marks=have_gcc, id="c_generated_helper"),
        pytest.param(SOURCE, DRIVER, True, marks=have_gpp, id="cpp_generated_helper"),
    ],
)
def test_generated_code_runs_leak_free_under_address_sanitizer(source: str, driver: str, cpp: bool) -> None:
    run = build_run_c(emitted(cpp=cpp, source=source), driver, cpp=cpp, sanitize=True)
    assert run.returncode == 0, f"{run.stdout}\n{run.stderr}"
    assert "detected memory leaks" not in run.stderr, run.stderr
