"""Returning a transposed VIEW must materialize into a fresh output buffer.

Two halves: (1) every transpose spelling desugars to the single ``np.transpose``
function form (``_TransposeRewriter``); (2) a returned transpose promotes to an
output buffer with the reversed / permuted shape (``_shape_from_transpose`` +
the frontend return-promotion), and reproduces numpy bit-exact on c/cpp/fortran.
"""
import ast
import json
import pathlib
import subprocess
import tempfile

import numpy as np

from numpyto_common.lowering import _TransposeRewriter


def _rewrite(src, sparse=()):
    tree = ast.parse(src)
    _TransposeRewriter(set(sparse)).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


# --------------------------------------------------------------------------- #
# desugar: every transpose spelling -> np.transpose(...)                       #
# --------------------------------------------------------------------------- #


def test_dot_T_desugars_to_np_transpose():
    assert _rewrite("y = x.T") == "y = np.transpose(x)"


def test_method_no_args_desugars():
    assert _rewrite("y = x.transpose()") == "y = np.transpose(x)"


def test_method_varargs_packs_into_tuple():
    assert _rewrite("y = x.transpose(1, 0)") == "y = np.transpose(x, (1, 0))"


def test_method_tuple_arg_preserved():
    assert _rewrite("y = x.transpose((0, 2, 1))") == "y = np.transpose(x, (0, 2, 1))"


def test_np_transpose_left_alone():
    assert _rewrite("y = np.transpose(x, (1, 0))") == "y = np.transpose(x, (1, 0))"


def test_sparse_transpose_not_densified():
    # a sparse matrix's ``A.T`` / ``A.transpose()`` must stay a method so the
    # SpMV hoister can flip CSR<->CSC on its own buffers.
    assert _rewrite("y = A.T @ x", sparse=["A"]) == "y = A.T @ x"
    assert _rewrite("y = A.transpose()", sparse=["A"]) == "y = A.transpose()"


# --------------------------------------------------------------------------- #
# frontend: a returned transpose promotes to an output buffer                  #
# --------------------------------------------------------------------------- #


def _parse(src, input_args, shapes, syms):
    from numpyto_common.frontend import parse_kernel
    from numpyto_common.lowering import lower
    d = pathlib.Path(tempfile.mkdtemp())
    npy = d / "k_numpy.py"
    npy.write_text(src)
    bi = {
        "benchmark": {
            "name": "k",
            "short_name": "k",
            "relative_path": "",
            "module_name": "k",
            "func_name": "f",
            "parameters": {
                "S": dict(syms)
            },
            "input_args": input_args,
            "array_args": [a for a in input_args if a in shapes],
            "output_args": [],
            "init": {
                "shapes": shapes
            }
        }
    }
    p = d / "bi.json"
    p.write_text(json.dumps(bi))
    return lower(parse_kernel(npy, p))


def test_return_transpose_promotes_reversed_shape():
    kir = _parse("import numpy as np\ndef f(x):\n return x.T\n", ["x"], {"x": "(M, N)"}, {"M": 3, "N": 4})
    outs = [a for a in kir.arrays if a.is_output]
    assert len(outs) == 1 and tuple(outs[0].shape) == ("N", "M")


def test_return_transpose_axes_promotes_permuted_shape():
    kir = _parse("import numpy as np\ndef f(x):\n return np.transpose(x, (0, 2, 1))\n", ["x"], {"x": "(A, B, C)"}, {
        "A": 2,
        "B": 3,
        "C": 4
    })
    outs = [a for a in kir.arrays if a.is_output]
    assert len(outs) == 1 and tuple(outs[0].shape) == ("A", "C", "B")


def test_tuple_return_with_transpose_promotes_both_into_outputs():
    kir = _parse("import numpy as np\ndef f(x, y):\n return x.T, y * 2\n", ["x", "y"], {
        "x": "(M, N)",
        "y": "(M, N)"
    }, {
        "M": 3,
        "N": 4
    })
    outs = {a.name: tuple(a.shape) for a in kir.arrays if a.is_output}
    assert len(outs) == 2 and ("N", "M") in outs.values() and ("M", "N") in outs.values()


# --------------------------------------------------------------------------- #
# numerical: bit-exact vs numpy on c / cpp / fortran                          #
# --------------------------------------------------------------------------- #


def _validate_native(src, x, expected, out_shape, shapes, syms):
    import _op_oracle as oo
    import numerical_oracle as no
    d = pathlib.Path(tempfile.mkdtemp())
    npy = d / "k_numpy.py"
    npy.write_text(src)
    bi = {
        "benchmark": {
            "name": "k",
            "short_name": "k",
            "relative_path": "",
            "module_name": "k",
            "func_name": "f",
            "parameters": {
                "S": dict(syms)
            },
            "input_args": ["x"],
            "array_args": ["x"],
            "output_args": [],
            "init": {
                "shapes": shapes
            }
        }
    }
    (d / "bi.json").write_text(json.dumps(bi))
    oo._emit_native(npy, d / "bi.json", d, "f")
    binding = json.loads((d / "f_binding.json").read_text())
    by = {"x": x, "ret_arr0": np.zeros(out_shape, dtype=np.float64)}
    exp = {"ret_arr0": no._norm(expected)}
    for b, ext in (("c", ".c"), ("cpp", ".cpp"), ("fortran", ".f90")):
        so = d / f"l_{b}.so"
        cc = subprocess.run(no.COMPILE[b] + [str(d / f"f{ext}"), "-o", str(so)], capture_output=True, text=True)
        assert cc.returncode == 0, f"{b} compile: {cc.stderr[-200:]}"
        st = no._invoke_isolated(b, binding, so, by, syms, exp, ["ret_arr0"], 1e-9, 1e-9)
        assert st == "ok", f"{b}: {st}"


def test_return_dot_T_matches_numpy_native():
    x = np.arange(12, dtype=np.float64).reshape(3, 4)
    _validate_native("import numpy as np\ndef f(x):\n return x.T\n", x, x.T, (4, 3), {"x": "(M, N)"}, {"M": 3, "N": 4})


def test_return_method_transpose_matches_numpy_native():
    x = np.arange(12, dtype=np.float64).reshape(3, 4)
    _validate_native("import numpy as np\ndef f(x):\n return x.transpose(1, 0)\n", x, x.transpose(1, 0), (4, 3),
                     {"x": "(M, N)"}, {
                         "M": 3,
                         "N": 4
                     })
