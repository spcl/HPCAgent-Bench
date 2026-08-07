"""The ISO standard-algorithm C++ backend (``numpyto --target cpp_isopar``).

``emit_cpp_isopar`` emits the same ABI as ``emit_cpp``, but every loop with a faithful
``<algorithm>``/``<numeric>`` spelling is emitted as that call instead of as a hand-written loop --
a map as ``std::transform``, a reduction as ``std::reduce``/``std::transform_reduce``, a prefix
recurrence as ``std::inclusive_scan``, a constant store as ``std::fill``, a plain move as
``std::copy``. The source then states the kernel's STRUCTURE and leaves the schedule to the
toolchain, the way Fortran array intrinsics and ``do concurrent`` do.

Every converted call carries an execution policy -- ``par_unseq`` everywhere except
``inclusive_scan``, which carries ``unseq`` because libstdc++'s PARALLEL scan miscomputes any
combine whose identity is not zero (see ``_ISOPAR_SCAN_POLICY``). Without a policy at all, ISO
specifies the algorithm as sequential and the emitted source would license nothing the loop did not.

Two halves, both on real output:

* the conversions fire and produce exactly the call they claim to (including the explicit
  ``static_cast`` at every width change, which is what keeps the generated C++ warning-clean);
* every shape that has NO faithful algorithm stays a loop -- a stencil, a strided or reversed
  sweep, a scatter, a scaled recurrence, a multi-statement body. A wrong algorithm here is a silent
  miscompile, so these are the load-bearing tests.

Numerics run the emitted C++ against numpy through the shared oracles: ``run_op`` for the shape
probes, ``run_kernel`` for corpus kernels across the four shapes (elementwise, reduction, scan,
convolution nest).
"""
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _op_oracle import _bench_info, run_op  # noqa: E402
from numpyto_c.emit import emit_cpp, emit_cpp_isopar  # noqa: E402
from numpyto_common.frontend import parse_kernel  # noqa: E402
from numpyto_common.lowering import lower  # noqa: E402

#: Every algorithm this backend may emit. A conversion outside this set is a bug, not a feature.
_ALGORITHMS = ("std::transform", "std::reduce", "std::transform_reduce", "std::inclusive_scan", "std::fill",
               "std::copy")

_SYMS = {"N": 8}
_SHAPE_1D = {"a": "(N,)", "b": "(N,)", "out": "(N,)"}
_A = np.array([-3.5, -1.0, 0.0, 2.5, 5.0, -7.25, 1.5, 4.0], dtype=np.float64)
_B = np.array([2.0, 3.0, 1.5, 2.0, 4.0, 3.0, 0.5, 1.25], dtype=np.float64)


def _emit(body: str, args="a, b, out", shapes=None, syms=None, dtypes=None, isopar=True) -> str:
    """Emit ``def k(<args>, N): <body>`` through the C++ backend under test (or plain ``emit_cpp``).

    ``args`` is the numpy signature; the last name is declared the graded OUTPUT of the synthesized
    bench_info. That says which buffer is compared, not where it lands in the emitted signature --
    the emitted parameter order is the ABI's canonical one (pointers by name, then scalars), which
    both backends get from the same :func:`_emit_signature`.
    """
    src = f"import numpy as np\n\n\ndef k({args}, N):\n{body}"
    names = [a.strip() for a in args.split(",")]
    shapes = shapes or {n: "(N,)" for n in names}
    with tempfile.TemporaryDirectory() as td:
        d = pathlib.Path(td)
        (d / "k_numpy.py").write_text(src)
        info = _bench_info("k", names[:-1], names[-1:], shapes, syms or _SYMS, dtypes)
        (d / "bi.json").write_text(json.dumps(info))
        kir = lower(parse_kernel(d / "k_numpy.py", d / "bi.json"))
        return (emit_cpp_isopar if isopar else emit_cpp)(kir, fn_name="k")


def _signature(text: str) -> str:
    """The emitted kernel's signature line."""
    return next(ln.strip() for ln in text.splitlines() if ln.startswith("void k("))


def _calls(text: str) -> list:
    """The algorithm calls in emitted output, one entry per occurrence, in source order."""
    return [m.group(0) for m in re.finditer(r"std::[a-z_]+\(", text)]


def _body(text: str) -> str:
    """The emitted kernel function body, without the shared prelude (which names no algorithm)."""
    return text[text.index('extern "C"'):]


def _stayed_a_loop(text: str) -> bool:
    return "for (int64_t" in _body(text) and not _calls(_body(text))


# --- the ABI and the prelude are unchanged --------------------------------------------------------


@pytest.mark.parametrize(
    "body,args,shapes",
    [
        # Output sorts LAST among the pointers ...
        ("    for i in range(N):\n        out[i] = a[i] + b[i]\n", "a, b, out", None),
        # ... and FIRST: the ABI orders pointers by name, so the output has no reserved position.
        ("    for i in range(N):\n        out[i] = x[i] * 2.0\n", "x, out", None),
        # A converted loop next to one that stays a loop, with a by-value scalar in the signature.
        ("    s = 0.0\n    for i in range(N):\n        s = s + z[i]\n    out[0] = s * alpha\n", "z, alpha, out", {
            "z": "(N,)",
            "out": "(N,)"
        }),
    ],
    ids=["output-last", "output-first", "scalar-param"],
)
def test_signature_is_byte_identical_to_the_plain_cpp_backend(body, args, shapes):
    """isopar changes the BODY, never the interface: same symbol, same canonical parameter order
    (pointers by name, then scalars/symbols by name), same types, same ``__restrict__``. Both
    backends read it from one ``_emit_signature``, and this pins that they still do."""
    assert _signature(_emit(body, args=args,
                            shapes=shapes)) == _signature(_emit(body, args=args, shapes=shapes, isopar=False))


def test_c_linkage_block_is_opened_once():
    text = _emit("    for i in range(N):\n        out[i] = a[i] + b[i]\n")
    assert text.count('extern "C" {') == 1 and text.count('} // extern "C"') == 1


def test_library_headers_precede_the_arithmetic_prelude():
    """<algorithm> must be parsed BEFORE the prelude's ``max``/``min`` templates are declared: a
    same-named global visible while libstdc++ is being parsed is what detonates inside it."""
    text = _emit("    for i in range(N):\n        out[i] = a[i] + b[i]\n")
    for header in ("<algorithm>", "<execution>", "<numeric>", "<functional>"):
        assert text.index(f"#include {header}") < text.index("constexpr auto max("), header


#: Map / reduce shapes: the strongest policy, on every call.
_PAR_UNSEQ_BODIES = [
    "    for i in range(N):\n        out[i] = a[i] + b[i]\n",
    "    for i in range(N):\n        out[i] = a[i]\n",
    "    for i in range(N):\n        out[i] = 1.0\n",
    "    s = 0.0\n    for i in range(N):\n        s = s + a[i]\n    out[0] = s\n",
    "    s = 0.0\n    for i in range(N):\n        s = s + a[i] * b[i]\n    out[0] = s\n",
    "    s = 0.0\n    for i in range(N):\n        s = s + np.abs(a[i])\n    out[0] = s\n",
]


@pytest.mark.parametrize("body", _PAR_UNSEQ_BODIES, ids=range(len(_PAR_UNSEQ_BODIES)))
def test_map_and_reduce_calls_carry_par_unseq(body):
    """The policy is the point. Without one, ISO specifies the algorithm as sequential and the
    emitted source licenses nothing the loop did not already license; ``par_unseq`` is what permits
    both threading and vectorization, which is what makes this the analogue of an array intrinsic."""
    text = _body(_emit(body))
    calls = _calls(text)
    assert calls, body
    assert text.count("std::execution::par_unseq, ") == len(calls), (calls, text)
    for weaker in ("std::execution::par,", "std::execution::seq", "std::execution::unseq"):
        assert weaker not in text, (weaker, body)


def test_scan_carries_unseq_because_the_parallel_scan_is_wrong_here():
    """libstdc++'s parallel scan seeds a block with a value-initialized element instead of the init,
    so a prefix PRODUCT under ``par``/``par_unseq`` comes back all zeros -- measured on g++ 15.2 at
    every size, both float and double. ``unseq`` reaches the same serial recurrence the loop does
    (still vectorizable), so it is correct by construction rather than by the accident that zero is
    ``plus``'s identity. Both scan combines take it, so the emitter never depends on that accident."""
    for body in ("    for i in range(1, N):\n        out[i] = out[i - 1] + a[i]\n",
                 "    for i in range(1, N):\n        out[i] = out[i - 1] * a[i]\n"):
        text = _body(_emit(body))
        assert _calls(text) == ["std::inclusive_scan("], text
        assert "std::inclusive_scan(std::execution::unseq, " in text, text
        assert "par_unseq" not in text, text


def test_execution_header_precedes_the_arithmetic_prelude():
    text = _emit("    for i in range(N):\n        out[i] = a[i] + b[i]\n")
    assert text.index("#include <execution>") < text.index("constexpr auto max(")


def test_never_std_accumulate():
    """``std::accumulate`` is specified strictly left-to-right, which forecloses exactly the
    reassociation this backend exists to allow. Reductions must use std::reduce."""
    text = _emit("    s = 0.0\n    for i in range(N):\n        s = s + a[i]\n    out[0] = s\n")
    assert "std::accumulate" not in text
    assert "std::reduce(" in text


# --- map: transform / copy / fill -----------------------------------------------------------------


def test_binary_elementwise_map_is_one_transform():
    text = _body(_emit("    for i in range(N):\n        out[i] = a[i] + b[i]\n"))
    assert ("std::transform(std::execution::par_unseq, a, a + __n0, b, out, "
            "[](double __v0, double __v1) { return static_cast<double>((__v0 + __v1)); });") in text
    assert _calls(text) == ["std::transform("]


def test_in_place_map_reuses_the_destination_range():
    """``out[i] = out[i] + b[i]``: std::transform explicitly allows result == first1, and the ranges
    here are exactly equal -- not merely overlapping."""
    text = _body(_emit("    for i in range(N):\n        out[i] = out[i] + b[i]\n", args="b, out"))
    assert "std::transform(std::execution::par_unseq, out, out + __n0, b, out, [](double __v0, double __v1)" in text


def test_unary_map_carries_the_call_into_the_lambda():
    text = _body(_emit("    for i in range(N):\n        out[i] = np.sqrt(a[i]) * 2.0\n"))
    assert "std::transform(std::execution::par_unseq, a, a + __n0, out, [](double __v0) { return static_cast<double>((sqrt(__v0) * 2.0)); });" \
        in text


def test_plain_move_is_a_copy_not_a_transform():
    text = _body(_emit("    for i in range(N):\n        out[i] = a[i]\n"))
    assert "std::copy(std::execution::par_unseq, a, a + __n0, out);" in text


def test_constant_store_is_a_fill_with_an_explicit_cast():
    text = _body(_emit("    for i in range(N):\n        out[i] = 0.0\n"))
    assert "std::fill(std::execution::par_unseq, out, out + __n0, static_cast<double>(0.0));" in text


def test_shifted_read_shifts_the_input_range():
    """``out[i] = a[i-1]`` over ``range(1, N)`` is the same map on a shifted source range; the
    destination is a DIFFERENT array, so the ranges cannot overlap."""
    text = _body(_emit("    for i in range(1, N):\n        out[i] = a[i - 1] + b[i]\n"))
    assert "const int64_t __n0 = (N) > (1) ? (N) - (1) : 0;" in text
    assert "std::transform(std::execution::par_unseq, a + ((1) - 1), a + ((1) - 1) + __n0, b + ((1)), out + ((1))," in text


def test_invariant_read_of_the_destination_stays_a_loop():
    """``out[i] = a[i] + out[0]`` reads a cell this same call is writing. The loop reads it in its
    own order; std::transform specifies NO order, so the two are not the same computation."""
    assert _stayed_a_loop(_emit("    for i in range(N):\n        out[i] = a[i] + out[0]\n", args="a, out"))


def test_invariant_operand_is_captured_not_parameterised():
    """A loop-invariant read stays inline in the lambda body, which then captures; only the element
    reads become parameters."""
    text = _body(_emit("    for i in range(N):\n        out[i] = a[i] * b[0]\n"))
    assert "std::transform(std::execution::par_unseq, a, a + __n0, out, [&](double __v0) { return static_cast<double>((__v0 * b[0])); });" in text


def test_trip_count_is_clamped_so_an_empty_range_is_never_inverted():
    """A loop whose end runs before its start executes zero times; the same pointer pair handed to
    an algorithm is undefined, so the count is clamped at 0."""
    text = _body(_emit("    for i in range(2, N):\n        out[i] = a[i]\n"))
    assert "const int64_t __n0 = (N) > (2) ? (N) - (2) : 0;" in text


def test_row_of_a_2d_array_converts_on_the_contiguous_axis():
    """The inner loop walks the FASTEST axis, so one iteration is one element: that row is a range.
    The outer loop stays a loop -- no algorithm expresses a nest."""
    text = _body(
        _emit("    for i in range(N):\n        for j in range(N):\n            out[i, j] = a[i, j] * 2.0\n",
              args="a, out",
              shapes={
                  "a": "(N, N)",
                  "out": "(N, N)"
              }))
    assert "for (int64_t i = 0; i < N; ++i) {" in text
    assert "std::transform(std::execution::par_unseq, a + ((i)*(N) + (0)), a + ((i)*(N) + (0)) + __n0, out + ((i)*(N) + (0))," in text


def test_two_reads_on_different_outer_rows_are_two_ranges():
    """``out[i, j] = a[2*i, j] + a[2*i+1, j]`` sweeps the same LAST axis twice but two different
    rows. Keying a range on the fastest axis alone collapsed them into one parameter and emitted
    ``__v0 + __v0`` -- a silent miscompile (found on dwt2d's Haar column pass)."""
    text = _body(
        _emit(
            "    for i in range(N):\n"
            "        for j in range(N):\n"
            "            out[i, j] = (a[2 * i, j] + a[2 * i + 1, j]) * 0.5\n",
            args="a, out",
            shapes={
                "a": "(N, N)",
                "out": "(N, N)"
            }))
    assert "std::transform(std::execution::par_unseq, a + (((2 * i))*(N) + (0)), a + (((2 * i))*(N) + (0)) + __n0, " \
           "a + ((((2 * i) + 1))*(N) + (0)), out + ((i)*(N) + (0)), " \
           "[](double __v0, double __v1) { return static_cast<double>(((__v0 + __v1) * 0.5)); });" in text


def test_column_sweep_stays_a_loop():
    """``out[j, i]`` walks the SLOW axis: stride N, not 1. No standard algorithm takes a strided
    range, and pretending it does would read the wrong elements."""
    text = _emit("    for i in range(N):\n        for j in range(N):\n            out[j, i] = a[j, i] * 2.0\n",
                 args="a, out",
                 shapes={
                     "a": "(N, N)",
                     "out": "(N, N)"
                 })
    assert _stayed_a_loop(text)


# --- reduce -------------------------------------------------------------------------------------


def test_sum_reduction_is_std_reduce_seeded_with_the_live_accumulator():
    """The accumulator's current value is the init, so no pattern-match of the preceding
    ``s = 0.0`` is needed and a pre-seeded accumulator stays correct."""
    text = _body(_emit("    s = 0.0\n    for i in range(N):\n        s = s + a[i]\n    out[0] = s\n"))
    assert "s = std::reduce(std::execution::par_unseq, a, a + __n0, s);" in text


def test_product_reduction_names_its_combine():
    text = _body(_emit("    s = 1.0\n    for i in range(N):\n        s = s * a[i]\n    out[0] = s\n"))
    assert "s = std::reduce(std::execution::par_unseq, a, a + __n0, s, std::multiplies<double>{});" in text


def test_max_reduction_uses_the_nan_propagating_combine():
    """numpy's maximum propagates NaN, and so does the prelude's ``max`` -- which makes it
    commutative and associative, hence a legal std::reduce combine."""
    text = _body(_emit("    s = a[0]\n    for i in range(N):\n        s = max(s, a[i])\n    out[0] = s\n"))
    assert "s = std::reduce(std::execution::par_unseq, a, a + __n0, s, [](double __a, double __b) { return max(__a, __b); });" in text


def test_dot_product_is_the_default_transform_reduce():
    text = _body(_emit("    s = 0.0\n    for i in range(N):\n        s = s + a[i] * b[i]\n    out[0] = s\n"))
    assert "s = std::transform_reduce(std::execution::par_unseq, a, a + __n0, b, s);" in text


def test_transformed_reduction_keeps_the_combine_in_the_accumulator_type():
    text = _body(_emit("    s = 0.0\n    for i in range(N):\n        s = s + a[i] * a[i]\n    out[0] = s\n"))
    assert ("s = std::transform_reduce(std::execution::par_unseq, a, a + __n0, s, std::plus<double>{}, "
            "[](double __v0) { return static_cast<double>((__v0 * __v0)); });") in text


def test_reduction_into_an_output_cell_converts_too():
    """``out[0] = out[0] + ...`` is the same reduction with the accumulator living in a buffer."""
    text = _body(_emit("    for i in range(N):\n        out[0] = out[0] + a[i] * b[i]\n"))
    assert "out[0] = std::transform_reduce(std::execution::par_unseq, a, a + __n0, b, out[0]);" in text


def test_reduction_over_its_own_array_stays_a_loop():
    """``out[0] = out[0] + out[i]`` sweeps a range that CONTAINS the accumulator cell: each
    iteration reads what the previous wrote, which std::reduce does not do."""
    assert _stayed_a_loop(_emit("    for i in range(N):\n        out[0] = out[0] + out[i]\n", args="a, out"))


def test_index_valued_body_stays_a_loop():
    """``out[i] = a[i] * i`` needs the INDEX inside the callable, and an algorithm hands its
    callable elements, not indices."""
    assert _stayed_a_loop(_emit("    for i in range(N):\n        out[i] = a[i] * i\n"))


# --- scan ---------------------------------------------------------------------------------------


def test_prefix_sum_is_an_inclusive_scan_seeded_from_the_preceding_element():
    text = _body(_emit("    for i in range(1, N):\n        out[i] = out[i - 1] + a[i]\n"))
    assert ("std::inclusive_scan(std::execution::unseq, a + ((1)), a + ((1)) + __n0, out + ((1)), "
            "std::plus<double>{}, out[(1) - 1]);") in text
    # The init READS the element before the range, which an empty range does not have.
    assert "if (__n0 > 0) {" in text


def test_prefix_product_scans_under_multiplies():
    text = _body(_emit("    for i in range(1, N):\n        out[i] = out[i - 1] * a[i]\n"))
    assert ("std::inclusive_scan(std::execution::unseq, a + ((1)), a + ((1)) + __n0, out + ((1)), "
            "std::multiplies<double>{}, out[(1) - 1]);") in text


def test_per_row_scan_of_a_2d_array_converts():
    text = _body(
        _emit(
            "    for i in range(N):\n        for j in range(1, N):\n            out[i, j] = out[i, j - 1] + a[i, j]\n",
            args="a, out",
            shapes={
                "a": "(N, N)",
                "out": "(N, N)"
            }))
    assert ("std::inclusive_scan(std::execution::unseq, a + ((i)*(N) + ((1))), "
            "a + ((i)*(N) + ((1))) + __n0, out + ((i)*(N) + ((1))),") in text
    assert "std::plus<double>{}, out[(i)*(N) + ((1) - 1)]);" in text


def test_scaled_recurrence_stays_a_loop():
    """``out[i] = out[i-1]*0.9 + a[i]`` is a first-order recurrence. Its scan form is over affine
    MAPS, not over doubles under plus -- writing it as an inclusive_scan of the elements would
    compute a different function, not a reassociated one."""
    assert _stayed_a_loop(_emit("    for i in range(1, N):\n        out[i] = out[i - 1] * 0.9 + a[i]\n"))


def test_stride_two_recurrence_stays_a_loop():
    """``out[i] = out[i-2] + b[i]`` carries over TWO elements: two interleaved scans, not one."""
    assert _stayed_a_loop(_emit("    for i in range(2, N):\n        out[i] = out[i - 2] + b[i]\n", args="b, out"))


def test_recurrence_with_a_third_operand_stays_a_loop():
    """``out[i] = out[i] + out[i-1]*b[i]`` reads the destination at two different offsets: neither a
    map (overlapping ranges) nor a scan (the combine is not the bare associative one)."""
    assert _stayed_a_loop(
        _emit("    for i in range(1, N):\n        out[i] = out[i] + out[i - 1] * b[i]\n", args="b, out"))


# --- shapes with no faithful spelling stay loops ---------------------------------------------------


def test_stencil_stays_a_loop():
    """``out[i] = out[i+1] + b[i]`` is a SHIFTED self-read: as std::transform the input and output
    ranges would overlap without being equal, which is undefined."""
    assert _stayed_a_loop(_emit("    for i in range(N - 1):\n        out[i] = out[i + 1] + b[i]\n", args="b, out"))


def test_strided_loop_stays_a_loop():
    assert _stayed_a_loop(_emit("    for i in range(0, N, 2):\n        out[i] = a[i] + b[i]\n"))


def test_reversed_loop_stays_a_loop():
    assert _stayed_a_loop(_emit("    for i in range(N - 1, 0, -1):\n        out[i] = a[i] + b[i]\n"))


def test_scaled_index_stays_a_loop():
    assert _stayed_a_loop(
        _emit("    for i in range(N):\n        out[i] = a[2 * i]\n", shapes={
            "a": "(N,)",
            "b": "(N,)",
            "out": "(N,)"
        }))


def test_indirect_gather_stays_a_loop():
    """``out[i] = a[ip[i]]`` is a gather: the range it touches is data-dependent."""
    assert _stayed_a_loop(
        _emit("    for i in range(N):\n        out[i] = a[ip[i]]\n",
              args="a, ip, out",
              shapes={
                  "a": "(N,)",
                  "ip": "(N,)",
                  "out": "(N,)"
              },
              dtypes={"ip": "int64"}))


def test_indirect_scatter_stays_a_loop():
    assert _stayed_a_loop(
        _emit("    for i in range(N):\n        out[ip[i]] = a[i]\n",
              args="a, ip, out",
              shapes={
                  "a": "(N,)",
                  "ip": "(N,)",
                  "out": "(N,)"
              },
              dtypes={"ip": "int64"}))


def test_multi_statement_body_stays_a_loop():
    """Two stores per iteration is a schedule of two maps; converting only one would reorder them
    against each other."""
    assert _stayed_a_loop(
        _emit("    for i in range(N):\n        out[i] = a[i] + b[i]\n        out[i] = out[i] * 2.0\n"))


def test_conditional_body_stays_a_loop():
    assert _stayed_a_loop(_emit("    for i in range(N):\n        if a[i] > 0.0:\n            out[i] = a[i]\n"))


def test_body_calling_a_kernel_helper_stays_a_loop():
    """``par_unseq`` forbids allocation inside an element access function. A kernel HELPER is
    emitted from the same IR as the kernel, so its body may ``malloc`` a local array -- which a
    lambda calling it would then do once per element, on an unspecified thread. Refused."""
    # The early return is what stops the frontend inlining it, so it survives as a real function.
    src = ("import numpy as np\n\n\n"
           "def scratch(v, N):\n"
           "    if v < 0.0:\n"
           "        return 0.0\n"
           "    t = np.zeros((N,))\n"
           "    for k in range(N):\n"
           "        t[k] = v\n"
           "    return t[N - 1]\n\n\n"
           "def k(a, b, out, N):\n"
           "    for i in range(N):\n"
           "        out[i] = scratch(a[i], N)\n")
    with tempfile.TemporaryDirectory() as td:
        d = pathlib.Path(td)
        (d / "k_numpy.py").write_text(src)
        (d / "bi.json").write_text(json.dumps(_bench_info("k", ["a", "b"], ["out"], dict(_SHAPE_1D), _SYMS, None)))
        kir = lower(parse_kernel(d / "k_numpy.py", d / "bi.json"))
        text = emit_cpp_isopar(kir, fn_name="k")
    # It really did survive as its own function, and it really does allocate.
    assert "static double scratch(" in text and "malloc(" in text, text[-900:]
    # So the loop that CALLS it stays a loop. (The helper's own body still converts -- a call
    # there is an ordinary call site, not an element access function.)
    kernel = text[text.index("void k("):]
    assert not _calls(kernel), kernel


def test_narrow_int_elements_stay_a_loop():
    """An int32 element PROMOTES to int64 on read (numpy's arithmetic width). A lambda taking it by
    value would compute in int32 and wrap where the loop does not."""
    assert _stayed_a_loop(
        _emit("    for i in range(N):\n        out[i] = a[i] + b[i]\n",
              dtypes={
                  "a": "int32",
                  "b": "int32",
                  "out": "int32"
              }))


def test_every_algorithm_emitted_is_one_we_claim():
    """A conversion outside the documented set means an unreviewed algorithm reached the output."""
    bodies = [
        "    for i in range(N):\n        out[i] = a[i] + b[i]\n",
        "    s = 0.0\n    for i in range(N):\n        s = s + a[i]\n    out[0] = s\n",
        "    for i in range(1, N):\n        out[i] = out[i - 1] + a[i]\n",
        "    for i in range(N):\n        out[i] = 1.0\n",
        "    for i in range(N):\n        out[i] = a[i]\n",
    ]
    for body in bodies:
        for call in _calls(_body(_emit(body))):
            assert call[:-1] in _ALGORITHMS, (call, body)


# --- numerics: the emitted C++ against numpy --------------------------------------------------------

_NUMERIC = ("cpp", "cpp_isopar")


def _run(body: str):
    """Run one 1-D probe on both C++ backends. The trip count is the literal 8 because the oracle
    calls the numpy reference with the arrays alone; the emitted signature still carries ``N``, as
    the shapes declare it."""
    src = "import numpy as np\n\n\ndef k(a, b, out):\n" + body
    return run_op(src,
                  "k", {
                      "a": _A.copy(),
                      "b": _B.copy()
                  }, {"out": (8, )},
                  _SYMS,
                  shapes=_SHAPE_1D,
                  backends=_NUMERIC)


def _ok(res):
    return all(v == "ok" for v in res.values()), res


@pytest.mark.integration
@pytest.mark.skipif(not shutil.which("g++"), reason="g++ needed to build the emitted C++")
@pytest.mark.parametrize(
    "name,body",
    [
        ("transform", "    for i in range(8):\n        out[i] = a[i] * b[i] + 1.0\n"),
        ("copy", "    for i in range(8):\n        out[i] = a[i]\n"),
        ("fill", "    for i in range(8):\n        out[i] = 2.5\n"),
        ("reduce", "    s = 0.0\n    for i in range(8):\n        s = s + a[i]\n    out[0] = s\n"),
        ("reduce_max", "    s = a[0]\n    for i in range(8):\n        s = max(s, a[i])\n    out[0] = s\n"),
        ("dot", "    s = 0.0\n    for i in range(8):\n        s = s + a[i] * b[i]\n    out[0] = s\n"),
        ("transform_reduce", "    s = 0.0\n    for i in range(8):\n        s = s + np.abs(a[i])\n    out[0] = s\n"),
        ("scan", "    out[0] = a[0]\n    for i in range(1, 8):\n        out[i] = out[i - 1] + a[i]\n"),
        ("shifted_map", "    for i in range(1, 8):\n        out[i] = a[i - 1] + b[i]\n"),
        # Unconvertible shapes must still be CORRECT: they fall back to the loop form.
        ("stencil_loop", "    for i in range(1, 7):\n        out[i] = a[i - 1] + a[i + 1]\n"),
        ("strided_loop", "    for i in range(0, 8, 2):\n        out[i] = a[i] + b[i]\n"),
    ],
)
def test_shapes_match_numpy(name, body):
    ok, res = _ok(_run(body))
    assert ok, (name, res)


@pytest.mark.integration
@pytest.mark.skipif(not shutil.which("g++"), reason="g++ needed to build the emitted C++")
def test_two_dimensional_row_map_and_scan_match_numpy():
    a2 = np.arange(16, dtype=np.float64).reshape(4, 4) - 7.0
    shapes = {"a": "(M, M)", "out": "(M, M)"}
    src = ("import numpy as np\n\n\ndef k(a, out):\n"
           "    for i in range(4):\n"
           "        out[i, 0] = a[i, 0]\n"
           "        for j in range(1, 4):\n"
           "            out[i, j] = out[i, j - 1] + a[i, j] * 2.0\n")
    res = run_op(src, "k", {"a": a2}, {"out": (4, 4)}, {"M": 4}, shapes=shapes, backends=_NUMERIC)
    assert all(v == "ok" for v in res.values()), res


# --- numerics: corpus kernels end to end -------------------------------------------------------------

#: One registered kernel per shape the backend converts, plus one it deliberately does not.
_CORPUS = [
    ("tsvc_2_vpv", "elementwise map -> std::transform"),
    ("tsvc_2_vsumr", "sum reduction -> std::reduce"),
    ("tsvc_2_vdotr", "dot product -> std::transform_reduce"),
    ("safety_map_of_scans", "per-row prefix sum -> std::inclusive_scan"),
    ("conv_standard_1d", "convolution nest: inner contraction -> std::transform_reduce"),
    ("vertical_flux_prefix_scan", "scaled recurrence: stays a loop"),
    # Two ranges on different outer rows of ONE array; keying on the last axis alone miscompiled it.
    ("dwt2d", "Haar column pass: a[2*i, :] and a[2*i+1, :] are two distinct ranges"),
]


def _oracle():
    repo = pathlib.Path(__file__).resolve().parents[3]
    path = str(repo / "tests")
    if path not in sys.path:
        sys.path.insert(0, path)
    import numerical_oracle as no
    if not shutil.which("g++"):
        pytest.skip("g++ needed to build the emitted C++")
    return no


@pytest.mark.integration
@pytest.mark.parametrize("kernel,shape", _CORPUS, ids=[k for k, _ in _CORPUS])
def test_corpus_kernel_matches_numpy(kernel, shape):
    no = _oracle()
    status = no.run_kernel(kernel, preset="S", precision="fp64", only_backends={"cpp", no.ISOPAR})
    assert status.get(no.ISOPAR) == "ok", f"{kernel} ({shape}): {status}"
    assert status.get("cpp") == "ok", f"{kernel} plain cpp regressed: {status}"


#: The no-implicit-conversion gate emitted C/C++ is held to. ``-Wunused-parameter`` is deliberately
#: absent: the ABI fixes the parameter list, so an unread parameter is required, not a defect.
_NO_IMPLICIT_CONVERSION = ("-Werror=conversion", "-Werror=sign-conversion", "-Werror=float-conversion",
                           "-Werror=double-promotion")

#: One converted case per algorithm, all mixed into one kernel per parametrization below.
_CONVERSION_CASES = [
    ("transform+reduce", "    s = 0.0\n"
     "    for i in range(N):\n"
     "        out[i] = np.sqrt(np.abs(a[i])) * b[i]\n"
     "    for i in range(N):\n"
     "        s = s + out[i] * b[i]\n"
     "    out[0] = s\n", None),
    ("scan+fill+copy", "    for i in range(N):\n"
     "        out[i] = 0.0\n"
     "    for i in range(1, N):\n"
     "        out[i] = out[i - 1] + a[i]\n"
     "    for i in range(N):\n"
     "        b[i] = out[i]\n", None),
    ("integer elements", "    for i in range(N):\n"
     "        out[i] = a[i] * b[i] + 3\n", {
         "a": "int64",
         "b": "int64",
         "out": "int64"
     }),
]


@pytest.mark.integration
@pytest.mark.skipif(not shutil.which("g++"), reason="g++ needed to build the emitted C++")
@pytest.mark.parametrize("name,body,dtypes", _CONVERSION_CASES, ids=[c[0] for c in _CONVERSION_CASES])
def test_emitted_source_has_no_implicit_conversion(name, body, dtypes):
    """Every width or signedness change in the emitted C++ is written as an explicit
    ``static_cast``. Inside a lambda that is load-bearing: the callable's result is converted on the
    way into the output range, where the loop form's assignment used to hide it."""
    from hpcagent_bench import languages
    text = _emit(body, dtypes=dtypes)
    with tempfile.TemporaryDirectory() as td:
        src = pathlib.Path(td) / "k.cpp"
        src.write_text(text)
        cc = subprocess.run([
            "g++", "-O1",
            languages.std_flag("cpp"), "-Wall", "-Wextra", "-Wno-unused-parameter", *_NO_IMPLICIT_CONVERSION,
            "-fsyntax-only",
            str(src)
        ],
                            capture_output=True,
                            text=True)
    assert cc.returncode == 0, cc.stderr
