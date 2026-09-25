# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""What a call to a KEPT helper hands dace, and what dace can solve from it.

A nested ``@dc.program`` call is bound by the frontend: the callee's symbols are SOLVED from the
shapes the call site passes, and both halves of that can fail. Too little and there is no solution
("Cannot infer values for symbols in inference"); too much, or the wrong kind, and there is more
than one ("Ambiguous values for symbols in inference"). Both halves were live at once --
jfnk_bratu's ``bratu_jvp(u, Q[:, :, k], ...)`` and sgs_pcg's ``sgs_apply``, the two kernels these
cases come from.
"""

import ast
import json
import pathlib
import tempfile
from typing import TYPE_CHECKING

from hpcagent_bench.translators.numpyto_c.dace_emit import contiguous_subscript, emit_dace, with_solvable_extents

if TYPE_CHECKING:
    from hpcagent_bench.translators.numpyto_common.ir import KernelIR

#: A helper whose CSR parameters are declared over symbols its body never reads, and whose one body
#: symbol sizes four vectors. sgs_pcg in miniature.
CSR_KERNEL = """import numpy as np


def sweep(A_data, A_indptr, diag, r, y, N):
    for i in range(N):
        s = r[i]
        for k in range(A_indptr[i], A_indptr[i + 1]):
            s = s - A_data[k] * y[i]
        y[i] = s / diag[i]


def csr_demo(A_data, A_indptr, b, x, NX, NY):
    N = NX * NY
    diag = np.zeros((N,), dtype=np.float64)
    y = np.zeros((N,), dtype=np.float64)
    for i in range(N):
        diag[i] = 1.0 + b[i]
    sweep(A_data, A_indptr, diag, b, y, N)
    for i in range(N):
        x[i] = y[i]
"""

CSR_BENCH = {
    "benchmark": {
        "func_name": "csr_demo",
        "array_args": ["A_data", "A_indptr", "b", "x"],
        "input_args": ["A_data", "A_indptr", "b", "x", "NX", "NY"],
        "output_args": ["x"],
        "init": {
            "shapes": {
                "A_data": "((3 * NX - 2) * (3 * NY - 2),)",
                "A_indptr": "(NX * NY + 1,)",
                "b": "(NX * NY,)",
                "x": "(NX * NY,)",
            },
            "dtypes": {"A_data": "float64", "A_indptr": "int64", "b": "float64", "x": "float64"},
        },
        "parameters": {"S": {"NX": 4, "NY": 4}},
        "short_name": "csr_demo",
    },
    "track": "loop_level_reasoning",
    "precisions": ["fp64"],
}

#: A helper called with a TRAILING-index plane of a rank-3 array -- the non-contiguous view -- and
#: one called with a leading-index plane, which is contiguous. jfnk_bratu's Arnoldi loop in
#: miniature.
PLANE_KERNEL = """import numpy as np


def scale_plane(v, out, N):
    for i in range(N):
        for j in range(N):
            out[i, j] = out[i, j] + v[i, j] * 2.0


def plane_demo(Q, R, out, N, K):
    for k in range(K):
        scale_plane(Q[:, :, k], out, N)
        scale_plane(R[k, :, :], out, N)
"""

PLANE_BENCH = {
    "benchmark": {
        "func_name": "plane_demo",
        "array_args": ["Q", "R", "out"],
        "input_args": ["Q", "R", "out", "N", "K"],
        "output_args": ["out"],
        "init": {
            "shapes": {"Q": "(N,N,K)", "R": "(K,N,N)", "out": "(N,N)"},
            "dtypes": {"Q": "float64", "R": "float64", "out": "float64"},
        },
        "parameters": {"S": {"N": 4, "K": 3}},
        "short_name": "plane_demo",
    },
    "track": "loop_level_reasoning",
    "precisions": ["fp64"],
}

#: A helper that WRITES the plane it is handed, so the copy has to come back.
WRITE_KERNEL = """import numpy as np


def fill_plane(v, src, N):
    for i in range(N):
        for j in range(N):
            v[i, j] = src[i, j] * 3.0


def write_demo(Q, src, N, K):
    for k in range(K):
        fill_plane(Q[:, :, k], src, N)
"""

WRITE_BENCH = {
    "benchmark": {
        "func_name": "write_demo",
        "array_args": ["Q", "src"],
        "input_args": ["Q", "src", "N", "K"],
        "output_args": ["Q"],
        "init": {
            "shapes": {"Q": "(N,N,K)", "src": "(N,N)"},
            "dtypes": {"Q": "float64", "src": "float64"},
        },
        "parameters": {"S": {"N": 4, "K": 3}},
        "short_name": "write_demo",
    },
    "track": "loop_level_reasoning",
    "precisions": ["fp64"],
}

#: A helper that accumulates one number over row dot products and returns it. Its call site binds a
#: fresh local, so the target says nothing -- and the call's own arguments are (N, N).
DOT_KERNEL = """import numpy as np


def row_dot(A, B, N):
    s = 0.0
    for i in range(N):
        s = s + A[i, :] @ B[i, :]
    return s


def dot_demo(A, B, H, N):
    for k in range(N):
        h = row_dot(A, B, N)
        H[k] = h
"""

DOT_BENCH = {
    "benchmark": {
        "func_name": "dot_demo",
        "array_args": ["A", "B", "H"],
        "input_args": ["A", "B", "H", "N"],
        "output_args": ["H"],
        "init": {
            "shapes": {"A": "(N,N)", "B": "(N,N)", "H": "(N,)"},
            "dtypes": {"A": "float64", "B": "float64", "H": "float64"},
        },
        "parameters": {"S": {"N": 4}},
        "short_name": "dot_demo",
    },
    "track": "loop_level_reasoning",
    "precisions": ["fp64"],
}


#: resnet101 in miniature: one stem conv+bn+relu+pool, then ONE bottleneck block (conv1x1 ->
#: conv3x3 -> conv1x1, each batch-normed), the exact shape ``_conv2d``'s third (1x1) call takes
#: at resnet101's ``layer1_0``. ``_bottleneck`` recomputes its own ``oh2``/``ow2`` from
#: ``h_in``/``w_in`` rather than reading the second conv's output extent directly (readability in
#: the numpy reference), and once ``_bottleneck`` is inlined into the kernel, that recomputed pair
#: collapses through the caller's OWN symbol for the same quantity (``sh1``/``sw1``, already minted
#: for the second conv) -- a two-hop rename the emitter must resolve to the end, or the
#: intermediate name is left in the third conv's body with no ``dc.symbol`` declaring it. The
#: batch-norm wrapping is load-bearing: a bare chain of ``_conv2d`` calls falls back to a "kept
#: helper calling a kept helper" form this emitter does not support at all, never reaching the bug.
CONV_CHAIN_KERNEL = """import numpy as np


def _conv2d(x, weight, stride, padding, n, c_in, h, w, c_out, kh, kw):
    if kh == 1 and kw == 1:
        sub = x[:, :, ::stride, ::stride] if stride > 1 else x
        oh = (h - 1) // stride + 1
        ow = (w - 1) // stride + 1
        nhwc = np.transpose(sub, (0, 2, 3, 1)).reshape(n * oh * ow, c_in)
        out = nhwc @ weight[:, :, 0, 0].T
        return np.transpose(out.reshape(n, oh, ow, c_out), (0, 3, 1, 2))
    oh = (h + 2 * padding - kh) // stride + 1
    ow = (w + 2 * padding - kw) // stride + 1
    if padding > 0:
        padded = np.zeros((n, c_in, h + 2 * padding, w + 2 * padding), x.dtype)
        padded[:, :, padding : padding + h, padding : padding + w] = x
    else:
        padded = x
    nhwc = np.transpose(padded, (0, 2, 3, 1))
    acc = np.zeros((n * oh * ow, c_out), x.dtype)
    for ky in range(kh):
        for kx in range(kw):
            patch = nhwc[:, ky : ky + (oh - 1) * stride + 1 : stride, kx : kx + (ow - 1) * stride + 1 : stride, :]
            acc += np.reshape(patch, (n * oh * ow, c_in)) @ np.transpose(weight[:, :, ky, kx])
    return np.transpose(np.reshape(acc, (n, oh, ow, c_out)), (0, 3, 1, 2))


def _batch_norm(x, weight, bias, running_mean, running_var, eps, c):
    shape = (1, c, 1, 1)
    return (x - np.reshape(running_mean, shape)) / np.sqrt(np.reshape(running_var, shape) + eps) * np.reshape(
        weight, shape
    ) + np.reshape(bias, shape)


def _bottleneck(x, w1, g1, b1, m1, v1, w2, g2, b2, m2, v2, w3, g3, b3, m3, v3, stride, eps, n, c_in, h_in, w_in, c_mid):
    a1 = np.maximum(
        _batch_norm(_conv2d(x, w1, 1, 0, n, c_in, h_in, w_in, c_mid, 1, 1), g1, b1, m1, v1, eps, c_mid), 0.0
    )
    oh2 = (h_in - 1) // stride + 1
    ow2 = (w_in - 1) // stride + 1
    a2 = np.maximum(
        _batch_norm(_conv2d(a1, w2, stride, 1, n, c_mid, h_in, w_in, c_mid, 3, 3), g2, b2, m2, v2, eps, c_mid), 0.0
    )
    a3 = _batch_norm(_conv2d(a2, w3, 1, 0, n, c_mid, oh2, ow2, c_in, 1, 1), g3, b3, m3, v3, eps, c_in)
    return np.maximum(a3 + x, 0.0)


def conv_chain_demo(
    x, conv1_weight, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var,
    b_w1, b_g1, b_b1, b_m1, b_v1, b_w2, b_g2, b_b2, b_m2, b_v2, b_w3, b_g3, b_b3, b_m3, b_v3,
    out, n, c_in, height, width, c_mid, eps,
):
    x1 = _conv2d(x, conv1_weight, 2, 3, n, c_in, height, width, c_mid, 7, 7)
    h1 = np.maximum(_batch_norm(x1, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, eps, c_mid), 0.0)
    sh1 = ((height + 2 * 3 - 7) // 2 + 1 + 2 * 1 - 3) // 2 + 1
    sw1 = ((width + 2 * 3 - 7) // 2 + 1 + 2 * 1 - 3) // 2 + 1
    out[:] = _bottleneck(
        h1, b_w1, b_g1, b_b1, b_m1, b_v1, b_w2, b_g2, b_b2, b_m2, b_v2, b_w3, b_g3, b_b3, b_m3, b_v3,
        1, eps, n, c_mid, sh1, sw1, c_mid,
    )
"""

CONV_CHAIN_ARRAYS = (
    "x",
    "conv1_weight",
    "bn1_weight",
    "bn1_bias",
    "bn1_running_mean",
    "bn1_running_var",
    "b_w1",
    "b_g1",
    "b_b1",
    "b_m1",
    "b_v1",
    "b_w2",
    "b_g2",
    "b_b2",
    "b_m2",
    "b_v2",
    "b_w3",
    "b_g3",
    "b_b3",
    "b_m3",
    "b_v3",
    "out",
)

CONV_CHAIN_BENCH = {
    "benchmark": {
        "func_name": "conv_chain_demo",
        "array_args": list(CONV_CHAIN_ARRAYS),
        "input_args": [*CONV_CHAIN_ARRAYS, "n", "c_in", "height", "width", "c_mid", "eps"],
        "output_args": ["out"],
        "init": {
            "shapes": {
                "x": "(n, c_in, height, width)",
                "conv1_weight": "(c_mid, c_in, 7, 7)",
                "bn1_weight": "(c_mid,)",
                "bn1_bias": "(c_mid,)",
                "bn1_running_mean": "(c_mid,)",
                "bn1_running_var": "(c_mid,)",
                "b_w1": "(c_mid, c_mid, 1, 1)",
                "b_g1": "(c_mid,)",
                "b_b1": "(c_mid,)",
                "b_m1": "(c_mid,)",
                "b_v1": "(c_mid,)",
                "b_w2": "(c_mid, c_mid, 3, 3)",
                "b_g2": "(c_mid,)",
                "b_b2": "(c_mid,)",
                "b_m2": "(c_mid,)",
                "b_v2": "(c_mid,)",
                "b_w3": "(c_mid, c_mid, 1, 1)",
                "b_g3": "(c_mid,)",
                "b_b3": "(c_mid,)",
                "b_m3": "(c_mid,)",
                "b_v3": "(c_mid,)",
                "out": "(n, c_mid, ((height + 2 * 3 - 7) // 2 + 1 + 2 * 1 - 3) // 2 + 1, "
                "((width + 2 * 3 - 7) // 2 + 1 + 2 * 1 - 3) // 2 + 1)",
            },
            "dtypes": {k: "float64" for k in CONV_CHAIN_ARRAYS},
        },
        "parameters": {"S": {"n": 2, "c_in": 3, "height": 32, "width": 32, "c_mid": 8, "eps": 1e-5}},
        "short_name": "conv_chain_demo",
    },
    "track": "loop_level_reasoning",
    "precisions": ["fp64"],
}


def free_names(source: str, program: str) -> set:
    """Names ``program``'s body READS but never binds -- as a parameter, an assignment or loop
    target, or a module-level ``dc.symbol``/constant. What dace's frontend calls "undefined
    variable" (``DaceSyntaxError: Use of undefined variable ...``), read off the AST alone."""
    module = ast.parse(source)
    bound = {"np", "dc", "math", "sin", "cos", "log", "exp", "pow", "sqrt", "dc_float", "dc_complex_float"}
    for node in module.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            bound.add(node.targets[0].id)
    fn = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == program)
    bound |= {a.arg for a in fn.args.args}
    bound |= {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    loads = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return (loads - bound) - set(dir(__builtins__))


def test_a_helpers_own_recipe_collapses_through_to_the_caller_symbol() -> None:
    # ``_bottleneck``'s ``oh2``/``ow2`` are a SECOND spelling of the same extent the second conv's
    # own OUT-param already carries as ``sh1``/``sw1`` (both mean "this bottleneck's spatial
    # size"). Once inlined, the specialised third ``_conv2d`` reads ``oh2``/``ow2`` for its own
    # ``h``/``w``, and the fix must chase that all the way to ``sh1``/``sw1`` -- stopping one hop
    # short leaves the intermediate ``__inl<k>_oh2`` name in the body with no declaration.
    source = emit_dace(kir_of(CONV_CHAIN_KERNEL, CONV_CHAIN_BENCH, "conv_chain_demo"))
    programs = [n.name for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
    third_conv = programs[-2]  # the kept helpers precede the kernel; the third conv is the last one
    assert free_names(source, third_conv) == set(), (third_conv, source)
    assert "sh1" in source and "sw1" in source, source


def kir_of(source: str, bench: dict, stem: str) -> "KernelIR":
    from hpcagent_bench.translators.numpyto_common.frontend import parse_kernel

    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        numpy_py = root / f"{stem}_numpy.py"
        numpy_py.write_text(source)
        info = root / "bench_info.json"
        info.write_text(json.dumps(bench))
        return parse_kernel(numpy_py, info)


def annotation_of(source: str, program: str, param: str) -> str:
    """The declared annotation of one parameter of one emitted ``@dc.program``."""
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == program)
    arg = next(a for a in fn.args.args if a.arg == param)
    return ast.unparse(arg.annotation)


def call_to(source: str, program: str, callee: str) -> ast.Call:
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == program)
    return next(
        n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == callee
    )


def test_a_body_unread_compound_extent_becomes_one_symbol_dace_can_solve() -> None:
    # Three symbols in one product is one equation for three unknowns, and sympy answers with a
    # family. The body reads none of them, so the whole extent is one symbol instead.
    kir = kir_of(CSR_KERNEL, CSR_BENCH, "csr_demo")
    helper = next(h for h in kir.helpers if h.kernel_name == "sweep")
    settled = with_solvable_extents(helper)
    shapes = {a.name: tuple(str(d) for d in a.shape) for a in settled.arrays}
    assert shapes["A_data"] == ("sweep_extent0",), shapes
    assert shapes["A_indptr"] == ("sweep_extent1",), shapes
    # ``N`` is a bare extent and the body reads it: untouched, and still what dace solves from diag.
    assert shapes["diag"] == ("N",) and shapes["y"] == ("N",), shapes
    assert [s.name for s in settled.symbols] == [], [s.name for s in settled.symbols]


def test_the_retired_symbols_are_neither_declared_nor_passed() -> None:
    # A symbol the callee no longer names is one dace can neither solve nor accept: passing it as a
    # keyword is "Invalid keyword argument".
    source = emit_dace(kir_of(CSR_KERNEL, CSR_BENCH, "csr_demo"))
    assert "sweep_extent0 = dc.symbol(" in source, source
    assert annotation_of(source, "sweep", "A_data") == "dc_float[sweep_extent0]", source
    call = call_to(source, "csr_demo", "sweep")
    assert [kw.arg for kw in call.keywords] == [], ast.unparse(call)
    assert all(not isinstance(a, ast.Name) or a.id not in {"NX", "NY"} for a in call.args), ast.unparse(call)


def test_a_trailing_index_plane_is_copied_before_it_is_passed() -> None:
    # ``Q[:, :, k]`` strides over the third axis; a parameter declared [N, N] declares strides
    # (N, 1), and the two say __SOLVE_N = N and __SOLVE_N = K * N at once.
    source = emit_dace(kir_of(PLANE_KERNEL, PLANE_BENCH, "plane_demo"))
    assert "__hslice_0 = np.empty((N, N), dtype=np.float64)" in source, source
    assert "__hslice_0[:] = Q[:, :, k]" in source, source
    calls = [
        ast.unparse(n)
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "scale_plane"
    ]
    assert any(c.startswith("scale_plane(__hslice_0,") for c in calls), calls
    # The leading-index plane is already contiguous; copying it would be pure cost.
    assert any(c.startswith("scale_plane(R[k, :, :],") for c in calls), calls


def test_the_allocation_sits_at_function_scope_not_in_the_loop() -> None:
    # One name, one shape, one descriptor: these calls are inside a loop, and an allocation per
    # trip is a fresh descriptor per trip.
    source = emit_dace(kir_of(PLANE_KERNEL, PLANE_BENCH, "plane_demo"))
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "plane_demo")
    assert "__hslice_0 = np.empty" in ast.unparse(fn.body[0]), ast.unparse(fn)


def test_contiguity_reads_the_subscript_form() -> None:
    def form(text: str) -> bool:
        return contiguous_subscript(ast.parse(text, mode="eval").body)

    assert form("Q[k, :, :]") and form("Q[k]") and form("Q[:, :]") and form("Q[k, 2:5, :]")
    assert not form("Q[:, :, k]") and not form("Q[:, k, :]")
    assert not form("Q[0:2, 0:2]") and not form("Q[::2, :]") and not form("Q[None, :]")


def test_a_scalar_accumulator_helper_returns_by_value() -> None:
    # The target is a fresh local bound by the call itself, so the only other reading of the return
    # shape is a broadcast join over the call's own (N, N) arguments -- an out-param for one number.
    kir = kir_of(DOT_KERNEL, DOT_BENCH, "dot_demo")
    helper = next(h for h in kir.helpers if h.kernel_name == "row_dot")
    assert helper.return_kind == "scalar", (helper.return_kind, [(a.name, a.shape) for a in helper.arrays])
    assert [a.name for a in helper.arrays] == ["A", "B"], [a.name for a in helper.arrays]
    source = emit_dace(kir)
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "row_dot")
    assert [a.arg for a in fn.args.args] == ["A", "B"], ast.unparse(fn)
    assert any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(fn)), ast.unparse(fn)


def test_a_written_plane_is_copied_back_after_the_call() -> None:
    # A copy-in alone would drop the helper's whole result: the callee writes the temp and nothing
    # ever puts it back into the strided plane it stood for.
    source = emit_dace(kir_of(WRITE_KERNEL, WRITE_BENCH, "write_demo"))
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "write_demo")
    loop = next(n for n in ast.walk(fn) if isinstance(n, ast.For))
    lines = [ast.unparse(s) for s in loop.body]
    assert lines == [
        "__hslice_0[:] = Q[:, :, k]",
        "fill_plane(__hslice_0, src)",
        "Q[:, :, k] = __hslice_0",
    ], lines
