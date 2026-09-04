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

import numpy as np
from _op_oracle import run_op

from numpyto_c.dace_emit import views_of_written_bases, written_through

# The daubechies shape, minimised: bind a view of `out`, read it, then write disjoint slices of
# `out` through a TUPLE target -- which is how the four quadrant stores are spelled.
_READ_THEN_WRITE = """def f(out, n, half):
    block = out[:n, :n]
    lo = block[:, 0:2 * half:2]
    hi = block[:, 1:2 * half:2]
    out[:half, :half], out[half:2 * half, :half] = lo, hi
"""

# The same block with the read moved AFTER the store: the kernel means the aliasing.
_WRITE_THEN_READ = """def f(out, n, half):
    block = out[:n, :n]
    out[:half, :half] = out[half:2 * half, :half]
    lo = block[:, 0:2 * half:2]
    out[half:2 * half, :half] = lo
"""

# A view whose base is never stored into needs no copy.
_NEVER_WRITTEN = """def f(src, out, n, half):
    block = src[:n, :n]
    out[:half, :half] = block[:, 0:2 * half:2]
"""


def _names(src):
    return views_of_written_bases(ast.parse(src).body[0])


def test_a_view_read_before_the_base_is_written_is_named():
    assert _names(_READ_THEN_WRITE) == {"block"}


def test_a_view_read_after_the_base_is_written_is_declined():
    assert _names(_WRITE_THEN_READ) == set()


def test_a_view_of_an_unwritten_base_is_declined():
    assert _names(_NEVER_WRITTEN) == set()


def test_a_tuple_target_counts_as_a_store_on_every_element():
    # The predicate above is only reachable because written_through descends into tuple targets;
    # before it did, the four quadrant writes read as no store at all and nothing was copied.
    assert written_through(ast.parse("def f(a, b):\n    a[0], b[1] = 1, 2\n").body[0]) == {"a", "b"}


def test_the_emitted_dace_program_materializes_the_view():
    """End to end on the kernel that found this: the emitted port copies `block`, not views it.

    The port is regenerated rather than read off the tree -- ``*_dace.py`` is a gitignored artifact,
    so whatever a previous run left behind says nothing about the emitter as it stands now.
    """
    from hpcagent_bench import autogen, paths
    from hpcagent_bench.spec import BenchSpec

    spec = BenchSpec.load("daubechies_dwt2d")
    emitted = paths.BENCHMARKS / spec.relative_path / f"{spec.module_name}_dace.py"
    emitted.unlink(missing_ok=True)
    autogen.ensure("daubechies_dwt2d", ["dace"])
    text = emitted.read_text()
    assert "block = np.copy(out[:s, :s])" in text, text


def test_reading_a_view_before_writing_its_base_matches_numpy():
    # The C/C++/Fortran emitters answer this correctly already -- the case is here so the numpy
    # semantics the desugar preserves are pinned by a run, not only by the AST predicate.
    src = (
        "import numpy as np\n"
        "def band(out, n):\n"
        "    half = n // 2\n"
        "    block = out[:n, :n]\n"
        "    lo = block[:, 0:2 * half:2] + block[:, 1:2 * half:2]\n"
        "    out[:half, :half] = lo[:half, :]\n"
        "    out[half:2 * half, :half] = lo[half:2 * half, :]\n"
    )
    N = 8
    out = np.random.default_rng(0).standard_normal((N, N))
    res = run_op(
        src.replace("def band(out, n):\n    half = n // 2\n", "def band(out):\n    n = 8\n    half = 4\n"),
        "band",
        {"out": out},
        {},
        {"N": N},
        shapes={"out": "(N, N)"},
        backends=("c", "cpp", "fortran"),
    )
    assert all(v == "ok" or v.startswith("skip") for v in res.values()), res
