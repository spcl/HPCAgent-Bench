"""numpy's two indexing rules produce different RANKS, and a length-1 slice is where they differ.

    a[0:N, 0]    integer index  -> DROPS the axis  -> (N,)
    a[0:N, 0:1]  slice          -> KEEPS the axis  -> (N, 1)

A kept length-1 axis BROADCASTS: every position along it reads the same source element. The
scalarizer mapped it with the loop variable instead of its slice start, so ``a[:, 0:1] + b`` was
emitted as ``a[i][j] + b[i][j]`` -- a whole row where one column belongs. Wrong numbers in C, C++
and Fortran alike, and nothing in the pipeline could notice: all three backends agreed with each
other, and the oracle only compares each backend against numpy.

numpy is the oracle here rather than the emitted text, because the text was plausible.
"""

import numpy as np

from _op_oracle import run_op

BACKENDS = ("c", "cpp", "fortran", "numba", "pythran")
M, N = 4, 6
SHAPES_2D = {"a": "(M, N)", "b": "(M, N)", "out": "(M, N)"}


def check(body: str, out_shape: tuple, shapes: dict) -> None:
    rng = np.random.default_rng(0)
    src = f"import numpy as np\n\n\ndef f(a, b, out):\n{body}"
    result = run_op(
        src,
        "f",
        {"a": rng.random((M, N)), "b": rng.random((M, N))},
        {"out": out_shape},
        {"M": M, "N": N},
        shapes=shapes,
        backends=BACKENDS,
    )
    bad = {k: v for k, v in result.items() if v != "ok" and not v.startswith("skip")}
    assert not bad, bad


def test_a_length_1_slice_broadcasts_instead_of_advancing():
    """``a[:, 0:1]`` is column 0 read for EVERY output column -- the case that was miscompiled."""
    check("    out[:, :] = a[:, 0:1] + b[:, :]\n", (M, N), SHAPES_2D)


def test_a_length_1_slice_broadcasts_under_multiplication_too():
    """Not addition-specific: the defect is in the index mapping, so every operator inherits it."""
    check("    out[:, :] = a[:, 0:1] * b[:, :]\n", (M, N), SHAPES_2D)


def test_an_integer_index_drops_the_axis():
    """The other rule, and the reason the first cannot simply be 'squeeze size-1 dims': ``a[i, 0]``
    is a scalar spread along the row, which is a DIFFERENT computation from ``a[:, 0:1]``."""
    check("    for i in range(a.shape[0]):\n        out[i, :] = a[i, 0] + b[i, :]\n", (M, N), SHAPES_2D)


def test_a_length_1_slice_as_the_assignment_TARGET_keeps_its_axis():
    check("    out[:, 0:1] = a[:, 0:1]\n", (M, N), SHAPES_2D)


def test_a_length_1_row_slice_keeps_the_leading_axis():
    check("    out[0:1, :] = a[0:1, :] + b[0:1, :]\n", (M, N), SHAPES_2D)


def test_reducing_over_a_length_1_axis_yields_the_lower_rank():
    """``np.sum(a[:, 0:1], axis=1)`` reduces a kept axis of extent 1 -- the rank drops because the
    REDUCTION removed it, not because the slice was size 1."""
    check("    out[:] = np.sum(a[:, 0:1], axis=1)\n", (M,), {"a": "(M, N)", "b": "(M, N)", "out": "(M,)"})


K = 5
#: A DECLARED size-1 axis, not a length-1 slice. ``p`` is the shape cfd's ``pressure`` has.
SHAPES_BCAST = {"p": "(M, 1)", "n": "(M, K, 3)", "out": "(M, K, 3)"}


def check_broadcast(body: str) -> None:
    rng = np.random.default_rng(0)
    src = f"import numpy as np\n\n\ndef f(p, n, out):\n{body}"
    result = run_op(
        src,
        "f",
        {"p": rng.random((M, 1)), "n": rng.random((M, K, 3))},
        {"out": (M, K, 3)},
        {"M": M, "K": K},
        shapes=SHAPES_BCAST,
        backends=BACKENDS,
    )
    bad = {k: v for k, v in result.items() if v != "ok" and not v.startswith("skip")}
    assert not bad, bad


def test_a_declared_size_1_axis_broadcasts_under_a_newaxis():
    """cfd's ``pressure[..., np.newaxis] * normal``.

    ``...`` expands to full slices, so the operand reaches the scalarizer as ``p[:, :, None]`` --
    a SLICE over an axis the array declares as 1, which the length-1-slice rule above never sees.
    The slice consumed the result's extent-``K`` iter, so ``p[i, k]`` read along a single column:
    the next row's element for every cell but the last, and past the allocation for that one.
    """
    check_broadcast("    out[:] = p[..., np.newaxis] * n\n")


def test_the_same_operand_spelled_without_the_ellipsis():
    """``p[:, :, None]`` is what the ellipsis expands to; both spellings must agree."""
    check_broadcast("    out[:] = p[:, :, None] * n\n")


def test_a_size_1_axis_still_broadcasts_when_it_is_the_leading_one():
    """The rule is per-axis, not "the last axis": pinning must follow the size-1 axis wherever the
    array declares it."""
    rng = np.random.default_rng(0)
    src = "import numpy as np\n\n\ndef f(p, n, out):\n    out[:] = p[:, :, np.newaxis] * n\n"
    result = run_op(
        src,
        "f",
        {"p": rng.random((1, K)), "n": rng.random((M, K, 3))},
        {"out": (M, K, 3)},
        {"M": M, "K": K},
        shapes={"p": "(1, K)", "n": "(M, K, 3)", "out": "(M, K, 3)"},
        backends=BACKENDS,
    )
    bad = {k: v for k, v in result.items() if v != "ok" and not v.startswith("skip")}
    assert not bad, bad


def test_a_newaxis_BEFORE_the_size_1_axis_still_pins_the_right_one():
    """A leading newaxis shifts every following element one result axis to the right without
    consuming a source axis. Reading the operand's extent at the subscript POSITION rather than at
    the source axis would ask ``(1, 3)`` about axis 1 and pin the wrong one.

    pythran is skipped on its own numbers, not on ours: the emitted module is this body verbatim,
    and pythran compiles the identical hand-written source to a wrong result too (checked
    directly -- max abs error 2.47 against numpy on the same inputs). Emitting correct numpy is
    the whole of this backend's job, so the gap is the tool's.
    """
    rng = np.random.default_rng(0)
    src = "import numpy as np\n\n\ndef f(p, n, out):\n    out[:] = p[np.newaxis, :, :] * n\n"
    result = run_op(
        src,
        "f",
        {"p": rng.random((1, 3)), "n": rng.random((M, K, 3))},
        {"out": (M, K, 3)},
        {"M": M, "K": K},
        shapes={"p": "(1, 3)", "n": "(M, K, 3)", "out": "(M, K, 3)"},
        backends=BACKENDS,
        skip_backends={"pythran": "pythran miscompiles a LEADING-newaxis broadcast"},
    )
    bad = {k: v for k, v in result.items() if v != "ok" and not v.startswith("skip")}
    assert not bad, bad
