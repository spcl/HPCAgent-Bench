# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A view whose base is written later in the block is MATERIALIZED before the write.

dace's simplify fuses a straight-line block into one dataflow state. Inside one state there is no
ordering edge between a read through a View and a write to the array that view reads, so codegen may
serialize the write first. daubechies_dwt2d bound ``block = out[:s, :s]``, read it for both column
bands, then wrote four quadrants of ``out``; the generated C emitted one term of the high band AFTER
the first quadrant write and read back what that write had just replaced -- ``lo`` and both quadrants
derived from it stayed bit-exact while ``hi`` was wrong in exactly the columns the write had touched.

The copy is only taken where every read of the view precedes the first store: a kernel that reads the
view AFTER writing the base is relying on the aliasing, and a copy would answer the wrong array.
"""

import ast

import pytest

from hpcagent_bench.translators.numpyto_c.dace_emit import views_of_written_bases, written_through

# The daubechies shape, minimised: bind a view of `out`, read it, then write disjoint slices of
# `out` through a TUPLE target -- which is how the four quadrant stores are spelled.
READ_THEN_WRITE = """def f(out, n, half):
    block = out[:n, :n]
    lo = block[:, 0:2 * half:2]
    hi = block[:, 1:2 * half:2]
    out[:half, :half], out[half:2 * half, :half] = lo, hi
"""

# The same block with the read moved AFTER the store: the kernel means the aliasing.
WRITE_THEN_READ = """def f(out, n, half):
    block = out[:n, :n]
    out[:half, :half] = out[half:2 * half, :half]
    lo = block[:, 0:2 * half:2]
    out[half:2 * half, :half] = lo
"""

# A view whose base is never stored into needs no copy.
NEVER_WRITTEN = """def f(src, out, n, half):
    block = src[:n, :n]
    out[:half, :half] = block[:, 0:2 * half:2]
"""


def names_(src: str) -> set[str]:
    return views_of_written_bases(ast.parse(src).body[0])


@pytest.mark.parametrize(
    "src", [WRITE_THEN_READ, NEVER_WRITTEN], ids=["read_after_the_base_is_written", "base_never_written"]
)
def test_a_view_that_needs_no_copy_is_declined(src: str) -> None:
    assert names_(src) == set()


def test_a_tuple_target_counts_as_a_store_on_every_element() -> None:
    # The predicate above is only reachable because written_through descends into tuple targets;
    # before it did, the four quadrant writes read as no store at all and nothing was copied.
    assert written_through(ast.parse("def f(a, b):\n    a[0], b[1] = 1, 2\n").body[0]) == {"a", "b"}
