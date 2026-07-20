"""Parallel-loop analysis + OpenMP parallel-scope emission.

Two layers: pure-AST predicate unit tests (loop classification / reduction /
scatter detection) and end-to-end ``emit_c_omp`` text assertions (the pragma a
parallel loop nest gets). Imports resolve via PYTHONPATH.
"""
import ast
import json
import pathlib
import tempfile

import pytest

from numpyto_common.parallelism import (any_parallelizable_loop, has_indirect_scatter, is_timestep_loop,
                                        loop_is_parallel_safe, loop_reduction, subscript_idx_safe,
                                        UnsupportedParallelError)


def _stmt(src):
    return ast.parse(src).body[0]


def _sub(src):
    return ast.parse(src).body[0].value


# --- timestep detection (JAX consumer) ------------------------------------------------------------
def test_timestep_for_is_flagged():
    assert is_timestep_loop(_stmt("for t in range(TSTEPS):\n    step(t)\n"))


def test_plain_range_is_not_timestep():
    assert not is_timestep_loop(_stmt("for i in range(N):\n    a[i] = 0\n"))


def test_niter_counts_as_timestep():
    assert is_timestep_loop(_stmt("for k in range(NITER):\n    pass\n"))


def test_non_for_node_is_not_timestep():
    assert not is_timestep_loop(_stmt("a[1:-1] = b[:-2] + b[2:]\n"))


# --- subscript_idx_safe ---------------------------------------------------------------------------
def test_bare_index_is_safe():
    assert subscript_idx_safe(_sub("a[i]"), "i")


def test_shifted_index_is_unsafe():
    assert not subscript_idx_safe(_sub("a[i - 1]"), "i")


def test_scaled_index_is_unsafe():
    assert not subscript_idx_safe(_sub("a[2 * i]"), "i")


def test_indirect_index_is_unsafe():
    assert not subscript_idx_safe(_sub("a[p[i]]"), "i")


def test_absent_index_is_unsafe():
    assert not subscript_idx_safe(_sub("a[0]"), "i")


def test_multidim_bare_in_one_axis_is_safe():
    assert subscript_idx_safe(_sub("a[i, j]"), "i")


# --- loop_is_parallel_safe ------------------------------------------------------------------------
def test_elementwise_map_is_parallel_safe():
    assert loop_is_parallel_safe(_stmt("for i in range(N):\n    c[i] = a[i] + b[i]\n"))


def test_jacobi_read_only_neighbours_is_parallel_safe():
    # b written idx-safe; a is READ-ONLY (never written), so its shifted read does not race.
    assert loop_is_parallel_safe(_stmt("for i in range(N):\n    b[i] = a[i - 1] + a[i + 1]\n"))


def test_scalar_reduction_is_not_parallel_safe():
    assert not loop_is_parallel_safe(_stmt("for i in range(N):\n    s = s + a[i]\n"))


def test_inplace_stencil_is_not_parallel_safe():
    # a is written AND read shifted -> loop-carried dependence.
    assert not loop_is_parallel_safe(_stmt("for i in range(N):\n    a[i] = a[i - 1] + a[i + 1]\n"))


def test_scatter_is_not_parallel_safe():
    assert not loop_is_parallel_safe(_stmt("for i in range(N):\n    a[p[i]] = a[p[i]] + b[i]\n"))


# --- loop_reduction -------------------------------------------------------------------------------
def test_sum_reduction_assign_form():
    assert loop_reduction(_stmt("for i in range(N):\n    s = s + a[i]\n")) == ("+", "s")


def test_sum_reduction_augassign_form():
    assert loop_reduction(_stmt("for i in range(N):\n    s += a[i]\n")) == ("+", "s")


def test_product_reduction():
    assert loop_reduction(_stmt("for i in range(N):\n    p = p * a[i]\n")) == ("*", "p")


def test_max_reduction_via_np_maximum():
    assert loop_reduction(_stmt("for i in range(N):\n    m = np.maximum(m, a[i])\n")) == ("max", "m")


def test_min_reduction_via_builtin():
    assert loop_reduction(_stmt("for i in range(N):\n    m = min(m, a[i])\n")) == ("min", "m")


def test_pure_map_has_no_reduction():
    assert loop_reduction(_stmt("for i in range(N):\n    c[i] = a[i] + b[i]\n")) is None


def test_two_accumulators_is_not_a_single_reduction():
    assert loop_reduction(_stmt("for i in range(N):\n    s = s + a[i]\n    t = t + b[i]\n")) is None


def test_private_temp_is_not_a_reduction():
    # ``x`` is recomputed each iteration (not self-referential) -> not an accumulator.
    assert loop_reduction(_stmt("for i in range(N):\n    x = a[i] * 2.0\n    c[i] = x\n")) is None


# --- has_indirect_scatter / any_parallelizable_loop -----------------------------------------------
def test_scatter_write_is_indirect():
    assert has_indirect_scatter(ast.parse("for i in range(N):\n    out[idx[i]] += x[i]\n"))


def test_affine_write_is_not_indirect():
    assert not has_indirect_scatter(ast.parse("for i in range(N):\n    out[i] += x[i]\n"))


def test_any_parallelizable_true_for_map():
    assert any_parallelizable_loop(ast.parse("for i in range(N):\n    c[i] = a[i]\n"))


def test_any_parallelizable_true_for_reduction():
    assert any_parallelizable_loop(ast.parse("for i in range(N):\n    s = s + a[i]\n"))


def test_any_parallelizable_false_for_scatter_only():
    assert not any_parallelizable_loop(ast.parse("for i in range(N):\n    out[idx[i]] += x[i]\n"))


# --- emit_c_omp (end to end: parse -> lower -> emit) -----------------------------------------------
def _kir(src, args, shapes, dtypes=None, params=None):
    from numpyto_common.frontend import parse_kernel
    from numpyto_common.lowering import lower
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "k_numpy.py").write_text(src)
    bi = {
        "benchmark": {
            "name": "k",
            "short_name": "k",
            "relative_path": "",
            "module_name": "k",
            "func_name": "f",
            "parameters": {
                "S": params or {
                    "N": 16
                }
            },
            "input_args": args,
            "array_args": args,
            "output_args": [args[-1]],
            "init": {
                "shapes": shapes,
                "dtypes": dtypes or {}
            },
        }
    }
    (d / "bi.json").write_text(json.dumps(bi))
    return lower(parse_kernel(d / "k_numpy.py", d / "bi.json"))


def test_emit_c_omp_elementwise_parallel_for():
    from numpyto_c.emit import emit_c, emit_c_omp
    kir = _kir("def f(x, y, out):\n    for i in range(N):\n        out[i] = y[i] + 2.0 * x[i]\n", ["x", "y", "out"], {
        "x": "(N,)",
        "y": "(N,)",
        "out": "(N,)"
    })
    c = emit_c_omp(kir, fn_name="f")
    assert "#pragma omp parallel for\n" in c
    assert "reduction(" not in c
    assert "#pragma omp" not in emit_c(kir, fn_name="f")  # sequential emit is unchanged


def test_emit_c_omp_sum_reduction_clause():
    from numpyto_c.emit import emit_c_omp
    kir = _kir("def f(x, out):\n    s = 0.0\n    for i in range(N):\n        s = s + x[i]\n    out[0] = s\n",
               ["x", "out"], {
                   "x": "(N,)",
                   "out": "(N,)"
               })
    assert "#pragma omp parallel for reduction(+:s)" in emit_c_omp(kir, fn_name="f")


def test_emit_c_omp_nested_tags_outer_only():
    from numpyto_c.emit import emit_c_omp
    kir = _kir(
        "def f(a, out):\n    for i in range(N):\n        for j in range(N):\n            out[i, j] = a[i, j] * 2.0\n",
        ["a", "out"], {
            "a": "(N, N)",
            "out": "(N, N)"
        })
    c = emit_c_omp(kir, fn_name="f")
    assert c.count("#pragma omp parallel for") == 1  # only the outermost loop, no nested regions


def test_emit_c_omp_scatter_is_refused():
    from numpyto_c.emit import emit_c_omp
    kir = _kir("def f(idx, x, out):\n    for i in range(N):\n        out[idx[i]] = out[idx[i]] + x[i]\n",
               ["idx", "x", "out"], {
                   "idx": "(N,)",
                   "x": "(N,)",
                   "out": "(N,)"
               }, {"idx": "int64"})
    with pytest.raises(UnsupportedParallelError):
        emit_c_omp(kir, fn_name="f")


def test_emit_fortran_omp_elementwise_parallel_do():
    from numpyto_fortran.emit import emit_fortran, emit_fortran_omp
    kir = _kir("def f(x, y, out):\n    for i in range(N):\n        out[i] = y[i] + 2.0 * x[i]\n", ["x", "y", "out"], {
        "x": "(N,)",
        "y": "(N,)",
        "out": "(N,)"
    })
    f = emit_fortran_omp(kir, fn_name="f")
    assert "!$omp parallel do" in f
    assert "reduction(" not in f
    assert "!$omp" not in emit_fortran(kir, fn_name="f")  # sequential emit is unchanged


def test_emit_fortran_omp_sum_reduction_clause():
    from numpyto_fortran.emit import emit_fortran_omp
    kir = _kir("def f(x, out):\n    s = 0.0\n    for i in range(N):\n        s = s + x[i]\n    out[0] = s\n",
               ["x", "out"], {
                   "x": "(N,)",
                   "out": "(N,)"
               })
    assert "!$omp parallel do reduction(+:s)" in emit_fortran_omp(kir, fn_name="f")


def test_emit_fortran_omp_scatter_is_refused():
    from numpyto_fortran.emit import emit_fortran_omp
    kir = _kir("def f(idx, x, out):\n    for i in range(N):\n        out[idx[i]] = out[idx[i]] + x[i]\n",
               ["idx", "x", "out"], {
                   "idx": "(N,)",
                   "x": "(N,)",
                   "out": "(N,)"
               }, {"idx": "int64"})
    with pytest.raises(UnsupportedParallelError):
        emit_fortran_omp(kir, fn_name="f")


def test_reduction_rejects_self_referential_recurrence():
    # Genuine reductions: the non-accumulator operand is loop data, not the accumulator.
    assert loop_reduction(_stmt("for i in range(N):\n s = s + a[i]\n")) == ("+", "s")
    assert loop_reduction(_stmt("for i in range(N):\n p = p * a[i]\n")) == ("*", "p")
    assert loop_reduction(_stmt("for i in range(N):\n m = max(m, a[i])\n")) == ("max", "m")
    # A self-referential recurrence reads the accumulator on BOTH sides -- reduction(+:ema)
    # would start each thread at the identity and lose the compounding, so it must NOT match.
    assert loop_reduction(_stmt("for i in range(N):\n ema = ema + ema * alpha\n")) is None
    assert loop_reduction(_stmt("for i in range(N):\n m = max(m, m * 2.0)\n")) is None
