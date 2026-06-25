"""Loop / op parallelism classification (the IR annotation JAX consumes).
Pure-logic unit test; imports resolve via PYTHONPATH."""
import ast

from numpyto_common.parallelism import (PARALLEL, SEQ, classify,
                                         is_timestep_loop)


def _stmt(src):
    return ast.parse(src).body[0]


def test_independent_elementwise_range_for_is_parallel():
    # a[i] = f(b[i]) with no carry -> a pure map -> PARALLEL (devectorisable).
    assert classify(_stmt("for i in range(N):\n    a[i] = b[i]\n")) == PARALLEL
    assert classify(_stmt("for i in range(N):\n    a[i] = b[i] + c[i] * 2\n")) == PARALLEL


def test_carried_range_for_is_seq():
    # reads the array it writes (a[i-1]) -> loop-carried -> SEQ.
    assert classify(_stmt("for i in range(1, N):\n    a[i] = a[i - 1] + b[i]\n")) == SEQ


def test_non_i_subscript_store_is_seq():
    # target subscript is not exactly i (a[i - 1]) -> not a clean map -> SEQ.
    assert classify(_stmt("for i in range(N):\n    a[i - 1] = b[i]\n")) == SEQ


def test_offset_read_stencil_is_seq():
    # an offset read (b[i - 1]) leaves i behind when [i] subscripts are dropped,
    # so the emitter cannot devectorise it -> SEQ (matches JAX _classify_for).
    assert classify(_stmt("for i in range(1, N):\n    a[i] = b[i] + b[i - 1]\n")) == SEQ


def test_bare_index_use_is_seq():
    # i used as a bare scalar in the RHS (not just inside a subscript) -> SEQ,
    # since dropping the loop index would lose it.
    assert classify(_stmt("for i in range(N):\n    a[i] = i * 2.0\n")) == SEQ


def test_range_for_with_break_is_seq():
    assert classify(_stmt("for i in range(N):\n    a[i] = b[i]\n    if a[i] > 0:\n        break\n")) == SEQ


def test_timestep_for_is_seq_and_flagged():
    node = _stmt("for t in range(TSTEPS):\n    step(t)\n")
    assert classify(node) == SEQ          # still sequential...
    assert is_timestep_loop(node)         # ...and flagged no-unroll


def test_plain_range_is_not_timestep():
    assert not is_timestep_loop(_stmt("for i in range(N):\n    a[i] = 0\n"))


def test_niter_counts_as_timestep():
    assert is_timestep_loop(_stmt("for k in range(NITER):\n    pass\n"))


def test_slice_op_is_parallel():
    assert classify(_stmt("a[1:-1] = b[:-2] + b[2:]\n")) == PARALLEL


def test_2d_slice_op_is_parallel():
    assert classify(_stmt("p[1:-1, 1:-1] = q[2:, 1:-1] + q[:-2, 1:-1]\n")) == PARALLEL


def test_aug_slice_op_is_parallel():
    assert classify(_stmt("a[1:-1] += b[2:]\n")) == PARALLEL


def test_scalar_assign_is_neither():
    assert classify(_stmt("x = y + 1\n")) is None


def test_point_subscript_assign_is_neither():
    # a single-element store is not a whole-array slice op
    assert classify(_stmt("a[i] = b[i] + 1\n")) is None
