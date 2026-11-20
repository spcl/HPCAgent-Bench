# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A buffer the deferred-malloc table records inside a branch or a loop body is allocated on every path that
uses it, not only on the first one emitted.

The table recorded sizes in emission order, so the ``else`` branch "reused" the buffer the ``if`` branch
allocated -- on a run taking only the ``else``, a NULL pointer (warpx_esirkepov_deposition's ``cum_z`` in every
non-3D geometry, a judge-side SIGSEGV). Each case runs under AddressSanitizer at N=8, so the branch or loop that
records first never executes.
"""

import pytest

from hpcagent_bench.translators.numpyto_c.emit import emit_c
from hpcagent_bench.translators.numpyto_common.lowering import lower
from tests.translators.native_tu import build_run_c, have_gcc
from tests.translators.op_oracle import parse_source

HEADER = "import numpy as np\n\n\ndef k(a, out, N):\n"

#: Case -> (body, the value ``out[1]`` holds for a = ones(8)). ``lo`` is body-computed, so each ``c``'s
#: allocation is deferred to where it is bound.
CASES = {
    # the original: two sibling branches, inside a loop; only the else runs
    "else-branch": (
        "    for i in range(N):\n"
        "        lo = i % 2\n"
        "        if N > 100:\n"
        "            c = np.cumsum(a[lo:N])\n"
        "        else:\n"
        "            c = np.cumsum(a[lo:N] * 2.0)\n"
        "        out[1] = c[N - lo - 1]\n",
        14.0,
    ),
    # an if without else that never runs, then the same size bound after it
    "if-without-else": (
        "    lo = N % 2\n"
        "    if N > 100:\n"
        "        c = np.cumsum(a[lo:N])\n"
        "        out[0] = c[0]\n"
        "    c = np.cumsum(a[lo:N] * 2.0)\n"
        "    out[1] = c[N - lo - 1]\n",
        16.0,
    ),
    # a while loop that runs zero times, then the same size bound after it
    "while-zero-trips": (
        "    lo = N % 2\n"
        "    j = N\n"
        "    while j < N:\n"
        "        c = np.cumsum(a[lo:N])\n"
        "        out[0] = c[0]\n"
        "        j += 1\n"
        "    c = np.cumsum(a[lo:N] * 2.0)\n"
        "    out[1] = c[N - lo - 1]\n",
        16.0,
    ),
}

DRIVER = (
    "int main(void) {{\n"
    "    enum {{ N = 8 }};\n"
    "    double a[N], out[N] = {{0}};\n"
    "    for (int i = 0; i < N; ++i) a[i] = 1.0;\n"
    "    k(a, out, N);\n"
    "    return out[1] == {expected} ? 0 : 1;\n"
    "}}\n"
)


@have_gcc
@pytest.mark.parametrize("case", sorted(CASES))
def test_a_deferred_buffer_is_allocated_on_the_path_that_runs(case: str) -> None:
    body, expected = CASES[case]
    kir = lower(parse_source(HEADER + body, "k", ["a"], ["out"], {"a": "(N,)", "out": "(N,)"}, {"N": 8}))
    run = build_run_c(emit_c(kir, fn_name="k"), DRIVER.format(expected=expected), sanitize=True)
    assert run.returncode == 0, run.stderr[-2000:]


if __name__ == "__main__":
    for name in sorted(CASES):
        test_a_deferred_buffer_is_allocated_on_the_path_that_runs(name)
