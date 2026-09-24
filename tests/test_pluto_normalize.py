# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The respelling polycc gets of a translator scop (``hpcagent_bench.pluto_normalize``).

POLYCC-014: a function-local scalar written inside a scop aborts Pluto's ``pluto_auto_transform``
(tsvc_2_s316, s3110, s3111, argmax_with_index) or is dropped / shared in the output (s252, s255,
s3112, s319, s2710). POLYCC-015: pet drops a literal non-unit stride (quasi_affine_reduce_odd's
``i += 2`` ran over every element). The text tests pin the rewrites and what they refuse to touch;
the toolchain tests are the numerical consumers, graded by the same oracle the column gates on.
"""

import pathlib
import shutil

import pytest

from hpcagent_bench import pluto_normalize, pluto_transform

NO_POLYCC = "polycc absent: the Pluto toolchain is built from source, see containers/pluto.Dockerfile"

S316 = """#include <stdint.h>
void s316_fp64(int64_t LEN_1D, const double *restrict a, double *restrict result) {
        int64_t pluto_pred0;
        double x;
        #pragma scop
        x = a[0];
        for (int64_t i = 1; i < LEN_1D; ++i) {
          pluto_pred0 = ((a[i] < x) ? 1 : 0);
          x = (pluto_pred0 ? a[i] : x);
        }
        result[0] = x;
        #pragma endscop
}
"""


def test_a_scop_local_scalar_becomes_a_pointer_parameter_cell() -> None:
    """Every use of each moved scalar is ``name[0]`` inside the static scop function, and the exported
    symbol keeps its exact signature and hands the scalars' addresses in, in declaration order."""
    out = pluto_normalize.externalize_scop_scalars(S316)

    inner = (
        "static void s316_fp64_pluto_scop(int64_t LEN_1D, const double *restrict a, double *restrict result, "
        "int64_t *restrict pluto_pred0, double *restrict x) {"
    )
    assert inner in out
    assert "void s316_fp64(int64_t LEN_1D, const double *restrict a, double *restrict result) {" in out
    assert "s316_fp64_pluto_scop(LEN_1D, a, result, &pluto_pred0, &x);" in out
    scop = out[out.index("#pragma scop") : out.index("#pragma endscop")]
    assert "x[0] = (pluto_pred0[0] ? a[i] : x[0]);" in scop
    assert "result[0] = x[0];" in scop
    assert " x " not in scop and "(x)" not in scop


def test_a_scalar_used_as_an_index_or_outside_the_scop_stays_local() -> None:
    """``k`` indexes an array and ``n`` bounds a loop: as ``k[0]`` / ``n[0]`` both would read as
    data-dependent. ``out`` is also read after the region. None of them move, so nothing changes."""
    src = """void f_fp64(int64_t N, const double *restrict a, double *restrict r) {
        int64_t k;
        int64_t n;
        double out;
        #pragma scop
        k = 0;
        n = N;
        out = 0.0;
        for (int64_t i = 0; i < n; ++i) {
          out = out + a[k];
        }
        #pragma endscop
        r[0] = out;
}
"""
    assert pluto_normalize.externalize_scop_scalars(src) == src


def test_a_file_with_no_scop_scalar_is_returned_byte_identical() -> None:
    """The rewrite only exists for scops that need it; every other kernel's transform is untouched."""
    src = """void g_fp64(int64_t N, const double *restrict a, double *restrict b) {
        #pragma scop
        for (int64_t i = 0; i < N; ++i) {
          b[i] = a[i];
        }
        #pragma endscop
}
"""
    assert pluto_normalize.normalize_scop_input(src) == src


def test_a_literal_stride_loop_runs_over_a_unit_counter() -> None:
    """``i += 2`` becomes ``++i_pn`` with ``(1 + (2) * i_pn)`` in the condition and every use."""
    src = """void q_fp64(int64_t LEN_1D, const double *restrict a, double *restrict out) {
        #pragma scop
        out[0] = 0.0;
        for (int64_t i = 1; i < LEN_1D; i += 2) {
          out[0] = (out[0] + a[i]);
        }
        #pragma endscop
}
"""
    out = pluto_normalize.normalize_strided_loops(src)

    assert "for (int64_t i_pn = 0; (1 + (2) * i_pn) < LEN_1D; ++i_pn) {" in out
    assert "out[0] = (out[0] + a[(1 + (2) * i_pn)]);" in out
    assert "i +=" not in out


def test_a_negative_stride_keeps_its_direction() -> None:
    src = """void r_fp64(int64_t N, double *restrict a) {
        #pragma scop
        for (int64_t i = N - 1; i > 0; i += -3) {
          a[i] = 0.0;
        }
        #pragma endscop
}
"""
    out = pluto_normalize.normalize_strided_loops(src)

    assert "for (int64_t i_pn = 0; (N - 1 + (-3) * i_pn) > 0; ++i_pn) {" in out
    assert "a[(N - 1 + (-3) * i_pn)] = 0.0;" in out


def test_run_polycc_hands_polycc_the_normalized_copy_and_leaves_the_input_alone(tmp_path, monkeypatch) -> None:
    """The file on disk is PPCG's input and the build's freshness key: polycc must see the respelled
    COPY, and the original bytes must survive the call."""
    scop = tmp_path / "s316_fp64_pluto_input.c"
    scop.write_text(S316)
    seen: dict[str, str] = {}

    def fake_run_bounded(cmd, cwd=None, timeout=None, env=None):
        src = pathlib.Path(cmd[-3])
        seen["text"] = src.read_text()
        seen["path"] = str(src)
        pathlib.Path(cmd[-1]).write_text("/* transformed */\n")
        return pluto_transform.subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(pluto_transform, "polycc_exe", lambda: "/usr/bin/polycc")
    monkeypatch.setattr(pluto_transform, "run_bounded", fake_run_bounded)

    pluto_transform.run_polycc(scop, tmp_path / "s316_fp64_pluto.c")

    assert seen["path"] != str(scop)
    assert seen["text"] == pluto_normalize.normalize_scop_input(S316)
    assert scop.read_text() == S316


def test_a_reverse_unit_loop_runs_forward_over_a_counter() -> None:
    """``--i`` is dropped the same way (tsvc_2_s1112 ran forward over the wrong range)."""
    src = """void s_fp64(int64_t LEN_1D, double *restrict a, const double *restrict b) {
        #pragma scop
        for (int64_t i = (LEN_1D - 1); i > -1; --i) {
          a[i] = (b[i] + 1.0);
        }
        #pragma endscop
}
"""
    out = pluto_normalize.normalize_strided_loops(src)

    assert "for (int64_t i_pn = 0; ((LEN_1D - 1) + (-1) * i_pn) > -1; ++i_pn) {" in out
    assert "a[((LEN_1D - 1) + (-1) * i_pn)] = (b[((LEN_1D - 1) + (-1) * i_pn)] + 1.0);" in out


@pytest.mark.skipif(pluto_transform.polycc_exe() is None, reason=NO_POLYCC)
@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc absent: the oracle compiles polycc's C with it")
@pytest.mark.parametrize(
    "kernel",
    [
        "tsvc_2_s316",  # POLYCC-014, Pluto assertion
        "argmax_with_index",  # POLYCC-014, Pluto assertion, value + index carried
        "tsvc_2_s252",  # POLYCC-014, carried scalars dropped
        "tsvc_2_s319",  # POLYCC-014, accumulator dropped
        "quasi_affine_reduce_odd",  # POLYCC-015, stride dropped
        "tsvc_2_s1112",  # POLYCC-015, reverse loop
        "tsvc_2_s128",  # POLYCC-017, literal induction scalars
        "tsvc_2_s4117",  # POLYCC-017 + POLYCC-010, floord index scalar
        "tsvc_2_s315",  # POLYCC-010, python_mod value call
        "cond_reduce_sym",  # POLYCC-016, constexpr knob
    ],
)
def test_the_respelled_kernel_transforms_and_agrees_with_numpy(kernel: str) -> None:
    """Each was ``skip:unsupported:polycc`` or ``pluto-miscompile`` before the respelling."""
    assert pluto_transform.oracle_pluto_status(kernel) == "ok"


@pytest.mark.skipif(pluto_transform.polycc_exe() is None, reason=NO_POLYCC)
@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc absent: the oracle compiles polycc's C with it")
def test_a_data_dependent_subscript_is_still_declined() -> None:
    """The control: compact_threshold_pack writes ``packed[n]`` with a data-dependent ``n``. ``n`` is
    an index, so it stays local, and the kernel stays outside what the column may time."""
    assert pluto_transform.oracle_pluto_status("compact_threshold_pack") != "ok"


def test_a_pinned_constant_is_inlined_and_no_longer_constexpr() -> None:
    """POLYCC-016: pet's libclang refuses the C23 ``constexpr`` line and with it the whole unit."""
    src = """constexpr int64_t K = 1;

void c_fp64(int64_t N, const double *restrict a, double *restrict out) {
        #pragma scop
        for (int64_t i = 0; i < N; ++i) {
          out[0] = ((a[i] > K) ? (out[0] + a[i]) : out[0]);
        }
        #pragma endscop
}
"""
    out = pluto_normalize.inline_pinned_constants(src)

    assert "constexpr" not in out
    assert "static const int64_t K = 1;" in out
    assert "(a[i] > (1))" in out


def test_a_single_assignment_index_scalar_is_forward_substituted() -> None:
    """POLYCC-017: ``j = floord(i, 2); c[j]`` reads as indirection until ``j`` is its expression."""
    src = """void s4117_fp64(int64_t LEN_1D, double *restrict a, const double *restrict c) {
        int64_t j;
        #pragma scop
        for (int64_t i = 0; i < LEN_1D; ++i) {
          j = floord(i, 2);
          a[i] = c[j];
        }
        #pragma endscop
}
"""
    out = pluto_normalize.forward_substitute_scalars(src)

    assert "int64_t j;" not in out
    assert "j = " not in out
    assert "a[i] = c[(floord(i, 2))];" in out


def test_a_scalar_assigned_twice_or_used_after_its_block_is_not_substituted() -> None:
    src = """void t_fp64(int64_t N, double *restrict a, double *restrict r) {
        int64_t k;
        #pragma scop
        k = 0;
        for (int64_t i = 0; i < N; ++i) {
          a[k] = 1.0;
          k = (k + 1);
        }
        #pragma endscop
}
"""
    assert pluto_normalize.forward_substitute_scalars(src) == src


def test_a_literal_tile_width_becomes_a_literal_stride() -> None:
    """jacobi_2d_tile_w7: ``W = 7`` inlined, the runtime-sign condition folded, the stride normalized."""
    src = """void w_fp64(int64_t N, double *restrict a) {
        int64_t W;
        #pragma scop
        W = 7;
        for (int64_t ii = 1; ((W) > 0 ? ii < (N - 1) : ii > (N - 1)); ii += W) {
          a[ii] = 0.0;
        }
        #pragma endscop
}
"""
    text = pluto_normalize.forward_substitute_scalars(src)
    text = pluto_normalize.fold_constant_sign_ternaries(text)
    out = pluto_normalize.normalize_strided_loops(text)

    assert "for (int64_t ii_pn = 0; (1 + (7) * ii_pn) < (N - 1); ++ii_pn) {" in out
    assert "a[(1 + (7) * ii_pn)] = 0.0;" in out
    assert "W" not in out.split("{", 1)[1]


def test_literal_induction_scalars_become_closed_forms_of_the_counter() -> None:
    """tsvc_2_s128: ``k = j + 1; ... j = k + 1`` from ``j = -1`` is ``k = 2i`` at every read."""
    src = """void s128_fp64(int64_t LEN_1D, double *restrict a, double *restrict b, const double *restrict d) {
        int64_t j;
        int64_t k;
        #pragma scop
        j = -1;
        for (int64_t i = 0; i < LEN_1D; ++i) {
          k = (j + 1);
          a[i] = (b[k] - d[i]);
          j = (k + 1);
          b[k] = a[i];
        }
        #pragma endscop
}
"""
    out = pluto_normalize.substitute_induction_scalars(src)

    k = "(-1 + (2) * (i - (0)) + (1))"
    assert f"a[i] = (b[{k}] - d[i]);" in out
    assert f"b[{k}] = a[i];" in out
    assert "j = " not in out and "k = " not in out
    assert "int64_t j;" not in out and "int64_t k;" not in out


def test_a_conditional_advance_is_not_an_induction_scalar() -> None:
    """tsvc_2_s341's packing index moves only where the data says so: it stays data-dependent."""
    src = """void s341_fp64(int64_t LEN_1D, double *restrict a, const double *restrict b) {
        int64_t j;
        #pragma scop
        j = -1;
        for (int64_t i = 0; i < LEN_1D; ++i) {
          j = ((b[i] > 0.0) ? (j + 1) : j);
          a[j] = b[i];
        }
        #pragma endscop
}
"""
    assert pluto_normalize.substitute_induction_scalars(src) == src


def test_floord_in_a_subscript_is_spelled_as_the_quasi_affine_ternary() -> None:
    """POLYCC-010's subscript half; a floord in a loop BOUND keeps the spelling pet name-matches."""
    src = """void q_fp64(int64_t N, const double *restrict a, double *restrict b) {
        #pragma scop
        for (int64_t i = 0; i < floord(N, 2); ++i) {
          b[floord(i, 2)] = a[i];
        }
        #pragma endscop
}
"""
    out = pluto_normalize.floord_subscripts(src)

    assert "i < floord(N, 2)" in out
    assert "b[(((i) < 0) ? -((-(i) + 2 - 1) / 2) : (i) / 2)] = a[i];" in out


def test_a_prelude_helper_call_goes_to_a_stand_in_and_comes_back() -> None:
    """POLYCC-010's value half: pet outlines ``python_mod`` into an undeclared ``__pet_ret``."""
    src = """#define python_mod(a, b) ((a) % (b))
void m_fp64(int64_t N, double *restrict a) {
        #pragma scop
        for (int64_t i = 0; i < N; ++i) {
          a[i] = python_mod((i * 7), N);
        }
        #pragma endscop
}
"""
    out = pluto_normalize.opaque_helper_calls(src)

    assert "double __pluto_opaque_python_mod(double, double);\nvoid m_fp64(" in out
    assert "a[i] = __pluto_opaque_python_mod((i * 7), N);" in out
    assert "#define python_mod(a, b)" in out
    assert pluto_normalize.restore_output(out) == src


def test_ppcg_gets_no_pointer_cells_and_keeps_helper_names() -> None:
    """ppcg_hip strips device mirrors from the host half: a pointer cell there would be a host
    stack address handed to the device, so PPCG's copy keeps its locals and its signature."""
    out = pluto_normalize.normalize_ppcg_input(S316)

    assert out == S316
