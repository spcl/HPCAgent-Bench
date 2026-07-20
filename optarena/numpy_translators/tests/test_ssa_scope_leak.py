"""A shape-changing rebinding inside nested control flow must not leak its rename outward.

The shape-versioning pass renames a name that is re-bound to a DIFFERENT extent (``x`` ->
``x__v1``) so each extent gets its own buffer. Nested scopes were walked with the SAME rename map
rather than a copy -- the code did not do what its own comment said -- so a rename minted inside an
``if`` branch stayed active after the branch closed. Every later read of ``x`` then resolved to a
buffer that was only written on the path not taken:

    c, cpp   -> SIGSEGV (read past the end of the original allocation)
    fortran  -> silently wrong values, no diagnostic at all

The silent Fortran result is why this is pinned per backend. A rebinding that is only reachable
conditionally cannot be resolved statically at all, so the pass refuses it rather than guessing;
the unconditional top-level case keeps working exactly as before.
"""
import numpy as np
from _op_oracle import run_op

_NATIVE = ("c", "cpp", "fortran")


def _assert_ok(res):
    for backend, status in res.items():
        assert status == "ok" or status.startswith("skip"), f"{backend}: {status}"
    assert any(status == "ok" for status in res.values()), f"all skipped (vacuous): {res}"


def _run(src, n=4):
    return run_op(src,
                  "f", {"a": np.arange(1, n + 1, dtype=np.float64)}, {"out": (n, )}, {"N": n},
                  shapes={
                      "a": "(N,)",
                      "out": "(N,)"
                  },
                  dtypes={
                      "a": "float64",
                      "out": "float64"
                  },
                  backends=_NATIVE)


_REBIND_IN_UNTAKEN_BRANCH = ("import numpy as np\n"
                             "def f(a, out):\n"
                             "    n = a.shape[0]\n"
                             "    x = np.zeros(n)\n"
                             "    for i in range(n):\n"
                             "        x[i] = a[i]\n"
                             "    if n > 1000:\n"
                             "        x = np.zeros(2 * n)\n"
                             "    for i in range(n):\n"
                             "        out[i] = x[i]\n")


def test_conditional_shape_rebinding_is_refused_not_miscompiled():
    # Whichever branch runs is a RUNTIME fact, so no static buffer choice is correct. Refusing is
    # the only sound answer -- silently binding to either one is the bug this pins. The oracle
    # reports an emit-time refusal as a status rather than propagating it, so assert on that.
    for backend, status in _run(_REBIND_IN_UNTAKEN_BRANCH).items():
        assert status.startswith("FAIL:emit:NotImplementedError"), f"{backend}: {status}"
        assert "conditional control flow" in status, f"{backend}: {status}"


def test_unconditional_rebinding_at_top_level_still_versions():
    # The supported case: both extents are live on every path, so each gets its own buffer.
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           "    n = a.shape[0]\n"
           "    x = np.zeros(n)\n"
           "    for i in range(n):\n"
           "        x[i] = a[i]\n"
           "    x = np.zeros(2 * n)\n"
           "    for i in range(2 * n):\n"
           "        x[i] = 1.0\n"
           "    for i in range(n):\n"
           "        out[i] = x[i] + a[i]\n")
    _assert_ok(_run(src))


def test_same_shape_rebinding_reuses_one_buffer():
    # A rebinding to the SAME extent is not a new buffer and must not be renamed at all, inside
    # control flow or out of it.
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           "    n = a.shape[0]\n"
           "    x = np.zeros(n)\n"
           "    for i in range(n):\n"
           "        x[i] = a[i]\n"
           "    x = np.zeros(n)\n"
           "    for i in range(n):\n"
           "        x[i] = a[i] * 2.0\n"
           "    for i in range(n):\n"
           "        out[i] = x[i]\n")
    _assert_ok(_run(src))


def test_sibling_loop_nests_may_reuse_a_name_at_different_shapes():
    # The shape ICON's velocity_tendencies actually has: two INDEPENDENT temporaries that happen to
    # share the name `t`, each written and fully consumed inside its own nest, at different extents.
    # Nothing reads `t` after either nest, so there is no ambiguity -- refusing this would reject a
    # working kernel, which an earlier, liveness-blind version of the guard did.
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           "    n = a.shape[0]\n"
           "    for i in range(n):\n"
           "        t = np.zeros(n)\n"
           "        for j in range(n):\n"
           "            t[j] = a[j] * 2.0\n"
           "        out[i] = t[i]\n"
           "    for i in range(n):\n"
           "        t = np.zeros(2 * n)\n"
           "        for j in range(2 * n):\n"
           "            t[j] = a[i]\n"
           "        out[i] = out[i] + t[0]\n")
    _assert_ok(_run(src))


def test_two_shapes_per_loop_iteration_are_not_refused():
    """daubechies_dwt2d's shape: a name re-bound to a second extent inside a loop body, where the
    re-entry read is preceded by a re-binding at the TOP of the body.

    The rows pass binds ``e`` to one extent, the columns pass re-binds it to another, and the next
    iteration re-assigns ``e`` before reading it -- so the read never sees the columns binding and
    there is nothing ambiguous. A liveness check that scanned the whole re-entry prefix without
    stopping at that kill refused this, breaking daubechies_dwt2d and ls3df_scf on all three native
    backends. Both had always emitted correct code.
    """
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           "    n = a.shape[0]\n"
           "    for lvl in range(2):\n"
           "        e = np.zeros(n)\n"
           "        for j in range(n):\n"
           "            e[j] = a[j] + lvl\n"
           "        s = 0.0\n"
           "        for j in range(n):\n"
           "            s = s + e[j]\n"
           "        e = np.zeros(2 * n)\n"
           "        for j in range(2 * n):\n"
           "            e[j] = s\n"
           "        out[lvl] = e[0]\n")
    _assert_ok(_run(src))


def test_reentry_read_before_any_rebinding_is_still_refused():
    """The kill is what makes the loop case safe, so a body with NO kill before the read must still
    be refused -- otherwise this change would have traded a false positive for a false negative.

    Here ``x`` is bound before the loop and re-bound at a second extent inside it, with the re-entry
    read at the top of the body reaching that second binding. Which extent that read sees depends on
    the iteration, so no static buffer choice is right.
    """
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           "    n = a.shape[0]\n"
           "    x = np.zeros(n)\n"
           "    for i in range(n):\n"
           "        out[i] = x[0]\n"
           "        x = np.zeros(2 * n)\n"
           "        for j in range(2 * n):\n"
           "            x[j] = a[i]\n")
    for backend, status in _run(src).items():
        assert status.startswith("FAIL:emit:NotImplementedError"), f"{backend}: {status}"
        assert "conditional control flow" in status, f"{backend}: {status}"


def test_rebinding_confined_to_a_loop_body_does_not_escape():
    # x is re-bound and fully consumed inside the loop body; nothing after the loop reads it, so
    # there is no ambiguity to refuse and the kernel must still translate.
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           "    n = a.shape[0]\n"
           "    for i in range(n):\n"
           "        t = np.zeros(2 * n)\n"
           "        for j in range(2 * n):\n"
           "            t[j] = a[i]\n"
           "        out[i] = t[0]\n")
    _assert_ok(_run(src))


# --- holes found by review of the first fix; each was CONFIRMED before being closed -------------
# These drive the PASS directly instead of emitting and running. The property under test belongs to
# the lowering pass, and going end-to-end is actively unsafe here: the while-loop case below is a
# non-terminating kernel whenever the guard fails, so a regression would HANG the suite for minutes
# per backend instead of failing in milliseconds.
import ast  # noqa: E402

import pytest  # noqa: E402

from numpyto_common.lowering import _ssa_rename_reassigned  # noqa: E402

_SHAPES = {"a": ["n"], "out": ["n"]}


def _lower(body):
    _ssa_rename_reassigned(ast.parse("import numpy as np\n" + body), dict(_SHAPES))


def _assert_refused(body):
    with pytest.raises(NotImplementedError, match="conditional control flow"):
        _lower(body)


def test_rebinding_nested_below_the_loop_body_is_refused():
    """The guard consulted only the IMMEDIATELY enclosing block, so a rebinding one level deeper
    escaped it -- the same miscompile the guard exists for, just nested. Every enclosing loop's
    re-entry POINT is carried down now and truncated per name at the mint site (the truncation
    depends on the name, so a pre-truncated prefix cannot be passed down)."""
    _assert_refused("""
def k(a, out, n, m, iters):
    e = np.zeros((n,))
    for it in range(iters):
        out[it] = e[0]
        for j in range(m):
            e = np.zeros((m,))
            e[j] = 1.0
""")


def test_the_killing_statement_own_rhs_read_still_counts():
    # `X = np.zeros(n) + X[0]` kills X, but its RHS reads the previous binding FIRST. Truncating
    # the prefix before the whole statement skipped that read.
    _assert_refused("""
def k(a, out, n, m, T):
    X = np.zeros((n,))
    for t in range(T):
        X = np.zeros((n,)) + X[0]
        out[t] = X[0]
        X = np.zeros((m,))
        X[0] = 1.0
""")


def test_augmented_assignment_counts_as_a_read():
    # `e += 1.0` reads and writes the SAME buffer, but its bare-Name target carries ctx=Store, so
    # it was neither a kill (correctly) nor a read (wrongly) and liveness answered "dead".
    _assert_refused("""
def k(a, out, n, m):
    e = np.zeros((n,))
    if n > 0:
        e = np.zeros((m,))
        e[0] = 2.0
    e += 1.0
""")


def test_a_zero_trip_for_target_is_not_a_kill():
    """`for e in range(k)` with a runtime k == 0 never binds e, so the previous binding survives --
    may-define, not must-define. Treating it as a kill dropped every read before it."""
    _assert_refused("""
def k(a, out, n, m, iters):
    e = np.zeros((n,))
    for it in range(iters):
        for e in range(0):
            pass
        out[it] = e[0]
        e = np.zeros((m,))
        e[0] = 1.0
""")


def test_while_test_runs_after_the_body():
    # A While re-tests its condition after the body, so the test is live-after code too.
    _assert_refused("""
def k(a, out, n, m):
    x = np.zeros((n,))
    c = 0
    while x[0] < 1.0:
        x = np.zeros((m,))
        c = c + 1
""")


def test_starred_and_nested_unpacking_are_recognised_as_kills():
    """A target form the kill scan does not recognise is not symmetric: the prefix is not truncated,
    extra reads are counted, and a WORKING kernel is refused -- the regression class this whole
    guard has already caused once."""
    for target in ("first, *e = a", "(first, (e, second)) = a", "[e, second] = a"):
        _lower(f"""
def k(a, out, n, m, T):
    e = np.zeros((n,))
    for t in range(T):
        {target}
        out[t] = e[0]
        e = np.zeros((m,))
        e[0] = 1.0
""")
