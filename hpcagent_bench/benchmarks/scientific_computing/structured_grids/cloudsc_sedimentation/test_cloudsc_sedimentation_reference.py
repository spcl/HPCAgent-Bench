# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Proves the NumPy port of CLOUDSC's sedimentation is the upstream recurrence
(``cloudsc_sedimentation_reference.f90``, cloudsc.F90:694, 849, 1720-1789, 2603, 2631, 2680-2687, 2705-2718).

The NumPy arrays are C-contiguous ``(NSPEC, KLEV, KLON)`` and ``(NSPEC, KLEV + 1, KLON)``, the same
memory Fortran reads as ``(KLON, KLEV, NSPEC)`` and ``(KLON, KLEV + 1, NSPEC)``. Agreement with the
Fortran is bit-exact: every operation is an IEEE add, multiply, divide, min or max, and the reference is
built with ``-ffp-contract=off``. The comparisons call ``sedimentation_step``, one pass; the last tests say
the kernel's ``nsteps`` loop is that pass repeated with the amounts relaxed toward their initial field.

A scalar per-column loop, written independently of both, checks the same numbers one column at a time and
counts the cells where the clip to vapour, the cover overlap and the cover reset act: a guard that never
fires would make the comparisons tautologies. The remaining tests pin what the recurrence means: the
column water budget closes with the vapour the clip receives, the flux at the model top is zero whatever
the buffer held, and a dry column stays dry.
"""

import ctypes
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from numpy.ctypeslib import ndpointer

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "cloudsc_sedimentation_reference.f90"
NAMES = ("za", "zdtgdp", "zrdtgdp", "zrho", "vqx", "zqx", "zqv", "pfplsx", "zqxn", "zcovptot")


def load_module(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: dataclasses resolves a string annotation through
    # sys.modules[cls.__module__], which is None for a module loaded by path alone.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


NUMPY = load_module("cloudsc_sedimentation_numpy")
INITIALIZE = load_module("cloudsc_sedimentation").initialize
ZEPSEC, RCOVPMIN = NUMPY.ZEPSEC, NUMPY.RCOVPMIN


def build_reference(tmp_path: Path):
    library = tmp_path / "libcloudsc_sedimentation_reference.so"
    subprocess.run(
        ["gfortran", "-O2", "-shared", "-fPIC", "-fno-fast-math", "-ffp-contract=off", str(SOURCE), "-o", str(library)],
        check=True,
    )
    f64 = ndpointer(np.float64, flags="C_CONTIGUOUS")
    function = ctypes.CDLL(str(library)).cloudsc_sedimentation_reference
    function.argtypes = [f64] * 10 + [ctypes.c_int] * 2
    function.restype = None
    return function


def scalar_columns(buffers, KLEV, KLON):
    """One column at a time, plain floats. Returns zqxn, pfplsx, zqv, zcovptot and the counts of cells where
    the clip, the cover overlap and the cover reset acted."""
    za, zdtgdp, zrdtgdp, zrho, vqx, zqx, zqv = buffers[:7]
    species = zqx.shape[0]
    zqxn = np.zeros((species, KLEV, KLON))
    pfplsx = np.zeros((species, KLEV + 1, KLON))
    vapour = zqv.copy()
    zcovptot = np.zeros((KLEV, KLON))
    clipped = overlapped = reset = 0
    for jl in range(KLON):
        cover = 0.0
        for jk in range(KLEV):
            pretot = 0.0
            if jk > 0:
                for jm in range(species):
                    pretot += float(zqx[jm, jk, jl]) + float(pfplsx[jm, jk, jl]) * float(zdtgdp[jk, jl])
                if pretot > ZEPSEC:
                    above = float(za[jk - 1, jl])
                    cover = 1.0 - (
                        (1.0 - cover) * (1.0 - max(float(za[jk, jl]), above)) / (1.0 - min(above, 1.0 - 1.0e-6))
                    )
                    cover = max(cover, RCOVPMIN)
                    overlapped += 1
                else:
                    cover = 0.0
            else:
                cover = 0.0
            for jm in range(species):
                sink = float(zdtgdp[jk, jl]) * (float(vqx[jm]) * float(zrho[jk, jl]))
                source = float(pfplsx[jm, jk, jl]) * float(zdtgdp[jk, jl])
                amount = (float(zqx[jm, jk, jl]) + source) / (1.0 + sink)
                if amount < ZEPSEC:
                    clipped += amount > 0.0
                    vapour[jk, jl] += amount
                    amount = 0.0
                zqxn[jm, jk, jl] = amount
                pfplsx[jm, jk + 1, jl] = sink * amount * float(zrdtgdp[jk, jl])
            if float(pfplsx[2, jk + 1, jl]) + float(pfplsx[1, jk + 1, jl]) < ZEPSEC:
                reset += cover > 0.0
                cover = 0.0
            zcovptot[jk, jl] = cover
    return zqxn, pfplsx, vapour, zcovptot, (clipped, overlapped, reset)


def run_step(KLEV, KLON):
    buffers = INITIALIZE(KLEV, KLON)
    NUMPY.sedimentation_step(*buffers, KLEV, KLON)
    return buffers


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")
@pytest.mark.parametrize("KLEV,KLON", [(137, 512), (137, 37), (1, 3), (2, 5)])
def test_numpy_matches_upstream_reference(tmp_path, KLEV, KLON) -> None:
    """The manifest's S preset, a column count no vector width divides, a single level and two levels."""
    reference = build_reference(tmp_path)

    buffers = INITIALIZE(KLEV, KLON)
    ref_buffers = [b.copy() for b in buffers]

    NUMPY.sedimentation_step(*buffers, KLEV, KLON)
    reference(*ref_buffers, KLEV, KLON)

    for name, got, want in zip(NAMES, buffers, ref_buffers, strict=True):
        assert np.array_equal(got, want), name


def test_numpy_matches_an_independent_scalar_loop_and_every_guard_fires() -> None:
    """Bit-exact against plain per-column floats; at S the clip to vapour, the cover overlap and the cover
    reset each act on some cells, and every buffer is finite."""
    KLEV, KLON = 137, 512
    before = INITIALIZE(KLEV, KLON)
    buffers = INITIALIZE(KLEV, KLON)
    NUMPY.sedimentation_step(*buffers, KLEV, KLON)
    want_amount, want_flux, want_vapour, want_cover, (clipped, overlapped, reset) = scalar_columns(before, KLEV, KLON)
    assert np.array_equal(buffers[8], want_amount) and np.array_equal(buffers[7], want_flux)
    assert np.array_equal(buffers[6], want_vapour) and np.array_equal(buffers[9], want_cover)
    assert clipped > 0 and overlapped > 0 and reset > 0, (clipped, overlapped, reset)
    assert all(np.all(np.isfinite(b)) for b in buffers)


def test_the_column_water_budget_closes_with_the_vapour_and_the_flux_reaches_the_ground() -> None:
    """Each layer's change in amount, plus the vapour the clip takes, is the flux in minus the flux out, in
    units of zdtgdp, so the column loses exactly what leaves through the surface interface."""
    KLEV, KLON = 137, 512
    vapour_before = INITIALIZE(KLEV, KLON)[6]
    buffers = run_step(KLEV, KLON)
    zdtgdp, zqx, zqv, pfplsx, zqxn = buffers[1], buffers[5], buffers[6], buffers[7], buffers[8]
    lost = -np.sum((zqxn - zqx) / zdtgdp[None, :, :], axis=1) - np.sum((zqv - vapour_before) / zdtgdp, axis=0)
    assert pfplsx[:, KLEV, :].max() > 0.0, "no precipitation reaches the ground: the budget is vacuous"
    # The clip passes an amount under ZEPSEC to the vapour and drops the flux that amount would have carried,
    # at most ZEPSEC per level, so the budget closes to KLEV * ZEPSEC.
    assert np.allclose(lost, pfplsx[:, KLEV, :], rtol=1e-12, atol=KLEV * ZEPSEC)


def test_the_top_interface_and_the_cover_are_written_whatever_the_buffers_held() -> None:
    """Output buffers poisoned with NaN come out finite: interface 0 is set to zero by the step, the flux at
    each interface follows from the layer above, and the cover starts at zero at the top."""
    KLEV, KLON = 8, 32
    buffers = INITIALIZE(KLEV, KLON)
    for index in (7, 8, 9):
        buffers[index][:] = np.nan
    NUMPY.sedimentation_step(*buffers, KLEV, KLON)
    assert np.all(buffers[7][:, 0, :] == 0.0) and np.all(buffers[9][0] == 0.0)
    assert all(np.all(np.isfinite(buffers[index])) for index in (7, 8, 9))


def lone_amount(species: int, level: int, amount: float):
    """Two columns of 12 levels; column 1 holds ``amount`` of one species at one level and nothing else."""
    KLEV, KLON = 12, 2
    buffers = INITIALIZE(KLEV, KLON)
    buffers[5][:] = 0.0
    buffers[5][species, level, 1] = amount
    NUMPY.sedimentation_step(*buffers, KLEV, KLON)
    return buffers


def test_a_dry_column_stays_dry_and_a_wet_one_wets_the_levels_below_it() -> None:
    """Snow at level 3 of column 1: the flux, the amounts and the cover appear below it and nowhere in the
    dry column 0."""
    buffers = lone_amount(2, 3, 1.0e-5)
    pfplsx, zqxn, zcovptot = buffers[7], buffers[8], buffers[9]
    assert np.all(pfplsx[:, :, 0] == 0.0) and np.all(zqxn[:, :, 0] == 0.0) and np.all(zcovptot[:, 0] == 0.0)
    assert np.all(pfplsx[2, 4:, 1] > 0.0) and np.all(pfplsx[:, :4, 1] == 0.0)
    assert np.all(zqxn[2, 4:, 1] > 0.0) and np.all(zqxn[:, :3, 1] == 0.0)
    assert np.all(zcovptot[3:, 1] >= RCOVPMIN) and np.all(zcovptot[:3, 1] == 0.0)


def test_the_clip_takes_a_tiny_amount_into_the_vapour() -> None:
    """An amount far under ZEPSEC leaves nothing behind and arrives, untouched, as vapour; an ordinary one
    does not."""
    clipped = lone_amount(0, 3, 5.0e-16)
    kept = lone_amount(0, 3, 1.0e-3)
    assert clipped[8][0, 3, 1] == 0.0 and np.all(clipped[7][0, 4:, 1] == 0.0)
    assert clipped[6][3, 1] > INITIALIZE(12, 2)[6][3, 1]
    assert kept[8][0, 3, 1] > 0.0 and kept[6][3, 1] == INITIALIZE(12, 2)[6][3, 1]


def test_the_cover_is_reset_where_the_flux_leaving_the_layer_is_under_epsec() -> None:
    """At level 1 the air is so thin that a small amount of snow falls out as a flux under ZEPSEC while the
    amount itself is over it: the overlap would set the cover, the reset clears it. A larger amount keeps
    the cover at its minimum or more. Ice alone never keeps a cover: the reset reads snow plus rain."""
    thin, thick, ice = lone_amount(2, 1, 1.0e-13), lone_amount(2, 1, 1.0e-4), lone_amount(0, 1, 1.0e-4)
    assert thin[8][2, 1, 1] > ZEPSEC and thin[7][2, 2, 1] < ZEPSEC and thin[9][1, 1] == 0.0
    assert thick[7][2, 2, 1] > ZEPSEC and thick[9][1, 1] >= RCOVPMIN
    assert ice[7][0, 2, 1] > ZEPSEC and ice[9][1, 1] == 0.0


def test_the_kernel_repeats_the_step_and_relaxes_the_amounts_toward_their_initial_field() -> None:
    """``nsteps`` passes equal that many hand-written steps, each followed by zqx = (zqx0 + zqxn) / 2; the
    amounts stay at least half their initial field, and the repeats change the answer (a hoisted pass would
    not)."""
    KLEV, KLON = 30, 128
    looped = list(INITIALIZE(KLEV, KLON))
    by_hand = [b.copy() for b in looped]
    initial = looped[5].copy()
    NUMPY.cloudsc_sedimentation(*looped, KLEV, KLON, 4)
    for _ in range(4):
        NUMPY.sedimentation_step(*by_hand, KLEV, KLON)
        by_hand[5][:] = 0.5 * (initial + by_hand[8])
    for name, got, want in zip(NAMES, looped, by_hand, strict=True):
        assert np.array_equal(got, want), name
    once = list(INITIALIZE(KLEV, KLON))
    NUMPY.cloudsc_sedimentation(*once, KLEV, KLON, 1)
    assert not np.array_equal(once[8], looped[8])
    assert np.all(looped[5] >= 0.5 * initial) and all(np.all(np.isfinite(b)) for b in looped)
