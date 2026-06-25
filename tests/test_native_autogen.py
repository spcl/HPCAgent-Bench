"""Native (C / C++ / Fortran) on-demand generation + the canonical ABI.

Locks the post-collapse native scheme:

* the emitter writes ``<short>_<fptype>.<ext>`` with the bare base as the
  exported symbol (no ``_auto`` / per-compiler suffix), carrying the
  ``optarena-autogen`` marker on line 1;
* ``cpp_runtime.wrap_kernel`` builds ``lib<short>_<framework>.so`` per framework
  (cc=gcc, llvm=clang, fortran=gfortran) and the result matches the numpy
  reference for both precisions.

Skips cleanly where the translators or a compiler are absent.
"""
import importlib.util
import pathlib
import shutil
import sys
import tempfile

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
TRANSLATORS = REPO / "optarena" / "NumpyTranslators" / "src"
for _p in (str(REPO), str(TRANSLATORS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from optarena import paths  # noqa: E402
from optarena.spec import BenchSpec  # noqa: E402

KERNEL = "tsvc_2_s212"  # 1-D: a,b outputs; c,d inputs; LEN_1D symbol
#: framework -> the compiler binary that must be present to build it. Polly/Pluto
#: are flag presets on the C++ source, so they need clang(++) like llvm.
_COMPILER = {"cc": "gcc", "llvm": "clang", "fortran": "gfortran", "polly": "clang", "pluto": "clang"}


def _emitter_present() -> bool:
    return (TRANSLATORS / "numpyto_c" / "cli.py").exists()


@pytest.mark.skipif(not _emitter_present(), reason="translators absent")
def test_emit_names_and_marker():
    """numpyto_c writes <short>_fp64/<short>_fp32 sources whose symbol == stem."""
    from optarena.emit_bridge import emit_kernel
    spec = BenchSpec.load(KERNEL)
    numpy_py = paths.BENCHMARKS / spec.relative_path / f"{spec.module_name}_numpy.py"
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d)
        assert emit_kernel(KERNEL, numpy_py, out, target="c") == 0
        assert emit_kernel(KERNEL, numpy_py, out, target="c", precision="float32") == 0
        for fptype in ("fp64", "fp32"):
            for ext in ("c", "cpp"):
                src = out / f"{KERNEL}_{fptype}.{ext}"
                assert src.exists(), src.name
                text = src.read_text()
                assert text.splitlines()[0].lstrip("/ ").startswith("optarena-autogen")
                assert f"{KERNEL}_{fptype}(" in text  # symbol == file stem
                assert "_auto" not in text  # no legacy suffix


@pytest.mark.parametrize("framework", ["cc", "llvm", "fortran", "polly", "pluto"])
@pytest.mark.parametrize("dtype,fptype", [(np.float64, "fp64"), (np.float32, "fp32")])
def test_wrap_kernel_matches_numpy(framework, dtype, fptype):
    if not _emitter_present() or not shutil.which(_COMPILER[framework]):
        pytest.skip(f"translators or {_COMPILER[framework]} absent")
    from optarena.emit_bridge import emit_kernel
    from optarena.benchmarks import cpp_runtime

    spec = BenchSpec.load(KERNEL)
    numpy_py = paths.BENCHMARKS / spec.relative_path / f"{spec.module_name}_numpy.py"
    sm = importlib.util.spec_from_file_location(KERNEL, numpy_py)
    mod = importlib.util.module_from_spec(sm)
    sm.loader.exec_module(mod)
    ref = getattr(mod, spec.func_name)

    with tempfile.TemporaryDirectory() as d:
        cpp = pathlib.Path(d) / "cpp_backend"
        cpp.mkdir()
        for tgt in ("c", "fortran"):
            for prec in ("", "float32"):
                assert emit_kernel(KERNEL, numpy_py, cpp, target=tgt, precision=prec) == 0
        wrapper = pathlib.Path(d) / f"{KERNEL}_cpp.py"
        wrapper.write_text("# test wrapper\n")

        rng = np.random.default_rng(0)
        a, b, c, dd = (rng.random(1024).astype(dtype) for _ in range(4))
        LEN = 1024
        ea, eb = a.copy(), b.copy()
        ref(ea, eb, c.copy(), dd.copy(), LEN)  # numpy expected

        call = cpp_runtime.wrap_kernel(str(wrapper), KERNEL, framework)
        na, nb = a.copy(), b.copy()
        call(na, nb, c.copy(), dd.copy(), LEN)  # native, in place

        rt = 1e-6 if dtype == np.float32 else 1e-9
        assert np.allclose(ea, na, rtol=rt, atol=rt)
        assert np.allclose(eb, nb, rtol=rt, atol=rt)
        assert cpp_runtime.LAST_NATIVE_NS > 0  # self-timed


# A sparse kernel is emitted ONE source per configuration; the layout IS the
# sub-benchmark. The buffer-style numpy ref takes the unpacked CSR buffers, so
# the bench_info is flattened (emit_bridge) and the emitted source carries the
# <short>_<config>_<fptype> name (no duplicate parameters). C/C++ match numpy
# (Fortran also compiles now -- see the kind-unification tests below).
@pytest.mark.parametrize("framework", ["cc", "llvm"])
@pytest.mark.parametrize("dtype,fptype", [(np.float64, "fp64"), (np.float32, "fp32")])
def test_sparse_layout_is_a_subbenchmark(framework, dtype, fptype):
    if not _emitter_present() or not shutil.which(_COMPILER[framework]):
        pytest.skip(f"translators or {_COMPILER[framework]} absent")
    pytest.importorskip("scipy")
    import scipy.sparse as sp
    from optarena.autogen import ensure_native, _native_targets
    from optarena.benchmarks import cpp_runtime

    spec = BenchSpec.load("spmv")
    assert _native_targets(spec) == [("csr", "spmv_csr")]  # the layout = the sub-bench
    # Each layout is a registered sub-benchmark with its own native stem.
    assert spec.native_base("csr") == "spmv_csr"
    assert BenchSpec.load("gemm").native_base() == "gemm"  # dense -> bare short

    numpy_py = paths.BENCHMARKS / spec.relative_path / f"{spec.module_name}_numpy.py"
    sm = importlib.util.spec_from_file_location("spmv", numpy_py)
    mod = importlib.util.module_from_spec(sm)
    sm.loader.exec_module(mod)
    ref = getattr(mod, spec.func_name)

    ensure_native("spmv")
    cpp = paths.BENCHMARKS / spec.relative_path / "cpp_backend"
    sig = next(l for l in (cpp / f"spmv_csr_{fptype}.c").read_text().splitlines() if "void spmv_csr" in l)
    assert sig.count("A_data") == 1  # no duplicate params

    rng = np.random.default_rng(0)
    M = N = 64
    A = sp.random(M, N, density=0.1, format="csr", random_state=0, dtype=dtype)
    data, ind, ptr = A.data.copy(), A.indices.astype(np.int64), A.indptr.astype(np.int64)
    x = rng.random(N).astype(dtype)
    y_ref = np.zeros(M, dtype=dtype)
    ref(data.copy(), ind.copy(), ptr.copy(), x.copy(), y_ref)

    wf = paths.BENCHMARKS / spec.relative_path / f"{spec.module_name}_cpp.py"
    call = cpp_runtime.wrap_kernel(str(wf), "spmv_csr", framework)
    y = np.zeros(M, dtype=dtype)
    call(data.copy(), ind.copy(), ptr.copy(), x.copy(), y, M, N, A.nnz)
    rt = 1e-5 if dtype == np.float32 else 1e-9
    assert np.allclose(y, y_ref, rtol=rt, atol=rt)
    assert cpp_runtime.LAST_NATIVE_NS > 0


# --- Canonical integer width: int64 symbols/iterators + int32-array promotion ---


def test_symbols_and_iterators_are_int64():
    """Every backend declares size symbols AND loop iterators at the int64 ABI
    width (abi_contract.md): a 32-bit symbol/iterator would mix widths with the
    int64 index arithmetic and (in Fortran) fail to compile."""
    if not _emitter_present():
        pytest.skip("translators absent")
    from optarena.emit_bridge import emit_kernel
    spec = BenchSpec.load("gemm")
    numpy_py = paths.BENCHMARKS / spec.relative_path / f"{spec.module_name}_numpy.py"
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d)
        assert emit_kernel("gemm", numpy_py, out, target="c") == 0
        assert emit_kernel("gemm", numpy_py, out, target="fortran") == 0
        c = (out / "gemm_fp64.c").read_text()
        f = (out / "gemm_fp64.f90").read_text()
        # C: symbols are int64_t scalars; loop iterators are int64_t (not `int`).
        assert "int64_t NI" in c and "int64_t NJ" in c
        assert "for (int64_t " in c and "for (int " not in c
        # Fortran: symbols + iterators are integer(c_int64_t); no bare `integer ::`.
        assert "integer(c_int64_t), value" in f
        assert "integer(c_int64_t) ::" in f


_INT32_SRC = ("import numpy as np\n\n\n"
              "def gather_scale(idx, out, scale, N):\n"
              "    for i in range(N):\n"
              "        out[i] = idx[i] * scale + idx[i]\n")
_INT32_BENCH = {
    "benchmark": {
        "func_name": "gather_scale",
        "short_name": "gather_scale",
        "name": "gather_scale",
        "relative_path": "x",
        "module_name": "gather_scale",
        "array_args": ["idx", "out"],
        "input_args": ["idx", "out", "scale", "N"],
        "output_args": ["out"],
        "parameters": {
            "S": {
                "N": 16
            }
        },
        "init": {
            "func_name": "initialize",
            "input_args": ["N"],
            "output_args": ["idx", "out"],
            "shapes": {
                "idx": "(N,)",
                "out": "(N,)"
            },
            "dtypes": {
                "idx": "int32",
                "out": "int64"
            },
            "scalars": {
                "scale": 3
            }
        },
    },
    "track": "foundation",
    "precisions": ["fp64"],
}


@pytest.mark.parametrize("framework,target,compiler,ext", [
    ("cc", "c", "gcc", "c"),
    ("llvm", "c", "clang++", "cpp"),
    ("fortran", "fortran", "gfortran", "f90"),
])
def test_int32_array_promoted_on_read(framework, target, compiler, ext):
    """A user-supplied int32 index/value array stays CORRECT: its elements are
    promoted to int64 on read, so ``idx[i] * scale`` (int32 elem * int64 scalar)
    is a single-width op. Without promotion the Fortran build fails outright
    (mixed integer kinds) and any backend risks a narrow-width miscompile."""
    import ctypes
    import json
    import subprocess
    if not _emitter_present() or not shutil.which(compiler):
        pytest.skip(f"translators or {compiler} absent")

    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d)
        numpy_py = out / "gather_scale_numpy.py"
        numpy_py.write_text(_INT32_SRC)
        bi = out / "bi.json"
        bi.write_text(json.dumps(_INT32_BENCH))
        env = {**__import__("os").environ, "PYTHONPATH": str(TRANSLATORS)}
        # Always emit C (it writes the .c/.cpp + the canonical binding JSON, the
        # single source of the ABI arg order); also emit Fortran when needed.
        mods = ["numpyto_c.cli"] + (["numpyto_fortran.cli"] if target == "fortran" else [])
        for mod in mods:
            r = subprocess.run([
                sys.executable, "-m", mod, "emit", "--kernel",
                str(numpy_py), "--bench-info",
                str(bi), "--out",
                str(out)
            ],
                               env=env,
                               capture_output=True,
                               text=True)
            assert r.returncode == 0, r.stderr

        base = "gather_scale_fp64"
        src = out / f"{base}.{ext}"
        # The narrow int32 element is explicitly promoted on read (not a bare
        # mixed-width op): cast in C, INT(..) in Fortran.
        text = src.read_text()
        assert ("(int64_t)" in text) if target != "fortran" else ("INT(idx" in text)

        so = out / "libgs.so"
        if target == "fortran":
            cmd = [
                compiler, "-O2", "-ffree-form", "-ffree-line-length-none", "-std=f2018", "-fPIC", "-shared",
                str(src), "-o",
                str(so)
            ]
        else:
            std = "-std=c++23" if ext == "cpp" else "-std=c17"
            cmd = [compiler, "-O2", std, "-D_POSIX_C_SOURCE=199309L", "-fPIC", "-shared", str(src), "-o", str(so)]
        rc = subprocess.run(cmd, capture_output=True, text=True)
        assert rc.returncode == 0, rc.stderr

        binding = json.loads((out / f"{base}_binding.json").read_text())
        order = [a["name"] for a in binding["args"]]

        N = 16
        idx = (np.arange(N) * 7 % N).astype(np.int32)
        scale = 3
        out_buf = np.zeros(N, dtype=np.int64)
        expected = idx.astype(np.int64) * scale + idx  # numpy reference
        vals = {"idx": idx, "out": out_buf, "N": N, "scale": scale}
        cargs = []
        keep = []
        for nm in order:
            v = vals[nm]
            if isinstance(v, np.ndarray):
                v = np.ascontiguousarray(v)
                keep.append(v)
                cargs.append(v.ctypes.data_as(ctypes.c_void_p))
            else:
                cargs.append(ctypes.c_int64(int(v)))
        t = np.zeros(1, dtype=np.int64)
        cargs.append(t.ctypes.data_as(ctypes.c_void_p))
        getattr(ctypes.CDLL(str(so)), base)(*cargs)
        assert np.array_equal(out_buf, expected), (framework, out_buf, expected)
