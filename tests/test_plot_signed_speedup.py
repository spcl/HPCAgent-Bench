# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``scripts/plot_speedup.py`` -- the signed-change speed-up chart.

The load-bearing assertions are about the AXIS, not the drawing. A 2x speed-up and a 2x
slow-down must be the same distance from 0 (the whole reason the figure replaces a ratio axis),
and a cell that cannot be turned into a speed-up must not be able to land on 0, which is the exact
value of "measured, and nothing changed". Both are pure functions, so both are tested without
rendering anything.
"""
import importlib.util
import math
import pathlib
from typing import List, Tuple

import pandas as pd
import pytest

from hpcagent_bench import plotting

REPO = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    """Import ``scripts/plot_speedup.py`` as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("plot_speedup", REPO / "scripts" / "plot_speedup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


speedup = load_script()


def summary_for(cells) -> pd.DataFrame:
    """A :func:`plotting.cell_summary` frame built the way the figure gets one -- from per-sample
    rows through the shipped summariser -- so the column shape can never drift from the real one.

    ``cells`` is ``(kernel, framework, milliseconds)``; each cell is given identical samples, which
    keeps the cleaned median exact and the bootstrap CI degenerate (nothing to warn about).
    """
    rows = [dict(benchmark=k, domain="Physics", framework=f, time=t) for k, f, ms in cells for t in [ms] * 5]
    return plotting.cell_summary(pd.DataFrame(rows))


# --- the signed transform ---------------------------------------------------------------------


@pytest.mark.parametrize("ratio,expected", [(1.0, 0.0), (2.0, 1.0), (3.0, 2.0), (0.5, -1.0), (0.25, -3.0)])
def test_the_landmarks_the_spec_names(ratio: float, expected: float) -> None:
    assert speedup.signed_change(ratio) == pytest.approx(expected)


@pytest.mark.parametrize("magnitude", [1.0, 1.25, 1.5, 2.0, 3.0, 10.0, 100.0])
def test_a_win_and_a_loss_of_the_same_size_are_the_same_distance_from_zero(magnitude: float) -> None:
    """⛔ THE point of the figure. On a raw ratio axis a 0.5x regression sits 0.5 below 1.0 while
    the 2.0x win sits 1.0 above it, so the eye reads the regression as the smaller event."""
    win = speedup.signed_change(magnitude)
    loss = speedup.signed_change(1.0 / magnitude)
    assert win == pytest.approx(-loss)
    assert win == pytest.approx(magnitude - 1.0)


@pytest.mark.parametrize("ratio", [0.0, -1.0, -0.5, math.inf, -math.inf, math.nan])
def test_an_unusable_ratio_is_nan_never_zero(ratio: float) -> None:
    """0 means "measured, nothing changed". A cell that was never measured must not claim it."""
    value = speedup.signed_change(ratio)
    assert math.isnan(value), f"{ratio} became {value}, which will be plotted"


# --- band assignment --------------------------------------------------------------------------


@pytest.mark.parametrize("ratio,band", [
    (1.0, speedup.BAND_LOW),
    (1.999, speedup.BAND_LOW),
    (0.51, speedup.BAND_LOW),
    (2.0, speedup.BAND_MID),
    (10.0, speedup.BAND_MID),
    (0.5, speedup.BAND_MID),
    (0.1, speedup.BAND_MID),
    (10.5, speedup.BAND_HIGH),
    (100.0, speedup.BAND_HIGH),
    (1.0 / 10.5, speedup.BAND_HIGH),
])
def test_the_band_edges(ratio: float, band: str) -> None:
    """The band named for an edge owns it: 2x and 10x are both ``2x .. 10x``."""
    assert speedup.band_of(speedup.signed_change(ratio)) == band


@pytest.mark.parametrize("magnitude", [1.0, 1.5, 2.0, 9.9, 10.0, 50.0])
def test_a_band_holds_a_win_and_its_mirrored_loss(magnitude: float) -> None:
    """Bands are keyed on magnitude, never on sign -- a 3x regression is read on the same axis as
    a 3x win, which is what makes the panels comparable."""
    assert (speedup.band_of(speedup.signed_change(magnitude)) == speedup.band_of(speedup.signed_change(1.0 /
                                                                                                       magnitude)))


def test_an_unplottable_change_has_no_band() -> None:
    assert speedup.band_of(math.nan) is None


def test_band_limits_are_anchored_at_the_band_edge_and_open_only_at_the_top() -> None:
    """Every panel shows its band's inner edge, so a point's height means the same thing each time
    and a lone ``> 10x`` point is read against the 10x boundary rather than an arbitrary window.
    Only the top band's OUTER end follows the data -- which is why one 100x outlier there cannot
    flatten the panels below it."""
    assert speedup.band_limits(speedup.BAND_LOW, [0.2, -0.3]) == (-1.0, 1.0)
    assert speedup.band_limits(speedup.BAND_MID, [1.5, 4.0]) == (1.0, 9.0)  # wins only
    assert speedup.band_limits(speedup.BAND_MID, [-1.5, -4.0]) == (-9.0, -1.0)  # losses only
    assert speedup.band_limits(speedup.BAND_MID, [-1.5, 4.0]) == (-9.0, 9.0)
    assert speedup.band_limits(speedup.BAND_HIGH, [40.0]) == (9.0, pytest.approx(42.0))
    assert speedup.band_limits(speedup.BAND_HIGH, [-40.0]) == (pytest.approx(-42.0), -9.0)


# --- points from the results summary ------------------------------------------------------------


def test_points_carry_the_median_speedup_over_the_baseline() -> None:
    frame = summary_for([("heat3d", plotting.BASELINE, 10.0), ("heat3d", "dace_cpu", 5.0)])
    points: List[speedup.Point] = speedup.speedup_points(frame)
    assert len(points) == 1, "the baseline is the divisor, not a series"
    assert points[0].framework == "dace_cpu"
    assert points[0].ratio == pytest.approx(2.0)
    assert points[0].change == pytest.approx(1.0)
    assert points[0].band == speedup.BAND_MID


def test_a_kernel_with_no_baseline_is_dropped_and_named() -> None:
    frame = summary_for([("heat3d", "dace_cpu", 5.0), ("jacobi2d", plotting.BASELINE, 10.0),
                         ("jacobi2d", "dace_cpu", 20.0)])
    with pytest.warns(UserWarning, match="heat3d@dace_cpu"):
        points = speedup.speedup_points(frame)
    assert [point.kernel for point in points] == ["jacobi2d"]
    assert points[0].change == pytest.approx(-1.0), "a 2x slow-down is -1, the mirror of a 2x win"


def test_a_non_positive_median_is_dropped_not_plotted_at_zero() -> None:
    frame = summary_for([("heat3d", plotting.BASELINE, 10.0), ("heat3d", "dace_cpu", 0.0)])
    with pytest.warns(UserWarning, match="heat3d@dace_cpu"):
        assert speedup.speedup_points(frame) == []


# --- the figures ---------------------------------------------------------------------------------


def rendered_panels(monkeypatch: pytest.MonkeyPatch, points, kernels, output: str) -> int:
    """Render the banded figure and count the panels ON THE FIGURE, not in the code path."""
    seen: List[int] = []
    original = plotting.save_figure

    def spy(path: str, fig) -> str:
        seen.append(len(fig.axes))
        return original(path, fig)

    monkeypatch.setattr(plotting, "save_figure", spy)
    speedup.banded_figure(points, kernels, output)
    assert len(seen) == 1
    return seen[0]


def test_an_empty_band_is_dropped_rather_than_drawn_empty(monkeypatch: pytest.MonkeyPatch,
                                                          tmp_path: pathlib.Path) -> None:
    """Two kernels, one band -> ONE panel. An empty panel carries no information and its y scale
    would be invented rather than measured, so the band is dropped from the layout."""
    frame = summary_for([("heat3d", plotting.BASELINE, 10.0), ("heat3d", "dace_cpu", 5.0),
                         ("jacobi2d", plotting.BASELINE, 10.0), ("jacobi2d", "dace_cpu", 2.5)])
    points = speedup.speedup_points(frame)
    assert {point.band for point in points} == {speedup.BAND_MID}
    assert rendered_panels(monkeypatch, points, ["heat3d", "jacobi2d"], str(tmp_path / "speedup.pdf")) == 1


def test_every_non_empty_band_gets_its_own_panel(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Three magnitudes -> three panels, each with its own y scale."""
    frame = summary_for([("heat3d", plotting.BASELINE, 10.0), ("heat3d", "dace_cpu", 9.5),
                         ("jacobi2d", plotting.BASELINE, 10.0), ("jacobi2d", "dace_cpu", 2.0),
                         ("gemm", plotting.BASELINE, 10.0), ("gemm", "dace_cpu", 0.05)])
    points = speedup.speedup_points(frame)
    assert {point.band for point in points} == {speedup.BAND_LOW, speedup.BAND_MID, speedup.BAND_HIGH}
    assert rendered_panels(monkeypatch, points, ["gemm", "heat3d", "jacobi2d"], str(tmp_path / "speedup.pdf")) == 3


def test_the_simplified_figure_shows_the_band_with_the_most_points(tmp_path: pathlib.Path) -> None:
    frame = summary_for([("heat3d", plotting.BASELINE, 10.0), ("heat3d", "dace_cpu", 5.0),
                         ("jacobi2d", plotting.BASELINE, 10.0), ("jacobi2d", "dace_cpu", 2.5),
                         ("gemm", plotting.BASELINE, 10.0), ("gemm", "dace_cpu", 9.5)])
    points = speedup.speedup_points(frame)
    assert speedup.dominant_band(points) == speedup.BAND_MID
    out = speedup.simple_figure(points, ["gemm", "heat3d", "jacobi2d"], str(tmp_path / "speedup-simple.svg"))
    blob = pathlib.Path(out).read_bytes()
    assert blob.lstrip().startswith(b"<?xml"), "the simplified variant must be a real SVG"
    assert b"<svg" in blob


def test_the_mini_variant_prunes_the_ticks_that_do_not_survive_embed_size(monkeypatch: pytest.MonkeyPatch,
                                                                          tmp_path: pathlib.Path) -> None:
    """At 3.4in wide a real kernel name and a y-tick number are both an unreadable smear, so the x
    ticks are ``K1..Kn`` and the y numbers are gone -- the band title carries the order of magnitude
    instead. What is left still has to say which axis it is."""
    seen: List[Tuple[List[str], List[str], List[str]]] = []
    original = plotting.save_figure

    def spy(path: str, fig) -> str:
        xticks = [text.get_text() for text in fig.axes[-1].get_xticklabels()]
        yticks = [text.get_text() for ax in fig.axes for text in ax.get_yticklabels()]
        seen.append((xticks, yticks, [text.get_text() for text in fig.texts]))
        return original(path, fig)

    monkeypatch.setattr(plotting, "save_figure", spy)
    points = speedup.demo_points()
    kernels = speedup.plotted_kernels(points)
    speedup.mini_figure(points, kernels, str(tmp_path / "speedup-mini.svg"))
    assert len(seen) == 1
    xticks, yticks, texts = seen[0]
    assert xticks == [f"K{i + 1}" for i in range(len(kernels))]
    assert yticks == [], "a number this small is clutter, not a reading"
    assert "Speedup" in texts


def test_every_output_is_written_per_machine(tmp_path: pathlib.Path) -> None:
    """End to end over a real results DB, through the shipped reader: the banded PDF plus the two
    SVG variants, each carrying the machine label (two nodes may never share a figure)."""
    from tests.test_inference_plots import build_results_db

    db = tmp_path / "results.db"
    build_results_db(db, shift=0.5)  # dace_cpu at half the numpy runtime -> a clean 2x
    written = speedup.plot_signed_speedup(db=str(db), preset="S", output=str(tmp_path / "speedup.pdf"), usetex=False)
    pdfs = [p for p in written if p.endswith(".pdf")]
    svgs = sorted(p for p in written if p.endswith(".svg"))
    assert len(pdfs) == 1 and len(svgs) == 2, written
    assert pathlib.Path(pdfs[0]).name.startswith("speedup.")
    assert [pathlib.Path(p).name.split(".")[0] for p in svgs] == ["speedup-mini", "speedup-simple"]
    assert pathlib.Path(pdfs[0]).read_bytes().startswith(b"%PDF-")
    assert all(b"<svg" in pathlib.Path(p).read_bytes() for p in svgs)


def baseline_only_db(path: pathlib.Path) -> None:
    """A results DB carrying the BASELINE and nothing else -- the shipped Result model, so the
    fixture cannot drift from the table the reader queries."""
    from sqlmodel import Session

    from hpcagent_bench.frameworks.schema import Result, results_engine
    with Session(results_engine(str(path))) as session:
        for value in (10.0, 10.5, 9.5):
            session.add(
                Result(timestamp=0,
                       benchmark="heat3d",
                       domain="Physics",
                       preset="S",
                       framework=plotting.BASELINE,
                       agent=None,
                       validated=True,
                       cpu="test-cpu",
                       time=value,
                       native_time=None,
                       datatype="float64",
                       variant=None,
                       prompt_hash=None,
                       execution="native"))
        session.commit()


def test_the_demo_populates_every_band_with_both_signs() -> None:
    """``--demo`` exists so the three panels can be LOOKED at. A change to the synthetic layout
    that emptied a band would quietly turn it back into a one-panel figure, and a demo that only
    ever shows wins would not demonstrate the mirroring it is there to demonstrate."""
    points = speedup.demo_points()
    per_band = {band: {point.kernel for point in points if point.band == band} for band in speedup.BANDS}
    assert all(2 <= len(kernels) <= 3 for kernels in per_band.values()), per_band
    for band in speedup.BANDS:
        assert any(point.change < 0.0 for point in points if point.band == band), f"{band}: no slow-down"
    assert speedup.demo_points() == points, "the demo seed must make the figure reproducible"


def test_a_db_with_only_the_baseline_fails_loudly(tmp_path: pathlib.Path) -> None:
    """No candidate framework means no speed-up exists. Writing no file while exiting 0 is the
    failure that reads as a clean run."""
    db = tmp_path / "baseline_only.db"
    baseline_only_db(db)
    with pytest.warns(UserWarning, match="no kernel has a plottable speed-up"):
        with pytest.raises(RuntimeError, match="no speed-up to plot"):
            speedup.plot_signed_speedup(db=str(db), preset="S", output=str(tmp_path / "speedup.pdf"), usetex=False)


# --- the boxes -----------------------------------------------------------------------------------


def test_a_boxs_samples_are_divided_by_a_fixed_baseline_not_paired_elementwise() -> None:
    """The load-bearing claim of the box: it is the CANDIDATE's spread, not a ratio distribution.

    Repetition i of a candidate and repetition i of the baseline are independent timings of
    different code, so dividing them elementwise would manufacture a spread out of two unrelated
    ones. Holding the divisor at the baseline's cleaned median keeps the box's width a property of
    the candidate alone -- which is what pins it here: identical candidate samples must produce a
    box of ZERO width no matter what the baseline's own samples did.
    """
    flat = speedup.cell_changes([5.0, 5.0, 5.0, 5.0, 5.0], base_time=10.0)
    assert len(flat) == 5
    assert min(flat) == pytest.approx(max(flat)), "identical candidate times cannot produce a spread"
    assert flat[0] == pytest.approx(1.0), "10 ms baseline over a 5 ms candidate is a 2x win, i.e. +1"
    spread = speedup.cell_changes([4.0, 5.0, 6.0], base_time=10.0)
    assert min(spread) < max(spread), "differing candidate times must produce one"


def test_a_cell_whose_baseline_is_unusable_yields_no_samples() -> None:
    """No divisor means no speed-up, and a box drawn at 0 would claim 'measured, nothing changed'."""
    assert speedup.cell_changes([1.0, 2.0, 3.0], base_time=0.0) == ()
    assert speedup.cell_changes([1.0, 2.0, 3.0], base_time=math.nan) == ()


def test_points_carry_their_repetitions_only_when_asked() -> None:
    """``speedup_points`` must not change the POSITIONS it computes by being asked for spread --
    the median and the band come from the summary either way, and only ``samples`` is added."""
    cells = [("heat3d", plotting.BASELINE, 10.0), ("heat3d", "dace_cpu", 5.0)]
    rows = pd.DataFrame(
        [dict(benchmark=k, domain="Physics", framework=f, time=t) for k, f, ms in cells for t in [ms] * 5])
    frame = plotting.cell_summary(rows)
    without = speedup.speedup_points(frame)
    with_samples = speedup.speedup_points(frame, data=rows)
    assert without[0].samples == ()
    assert len(with_samples[0].samples) == 5
    assert without[0].change == pytest.approx(with_samples[0].change), "asking for boxes moved the median"
    assert without[0].band == with_samples[0].band


def test_a_thinly_sampled_cell_keeps_its_marker_instead_of_faking_quartiles(tmp_path: pathlib.Path) -> None:
    """A box over two points draws a quartile range nobody measured. Such a cell must come BACK
    from ``draw_boxes`` so the caller still plots it -- dropping it would silently shrink the
    figure's population, which is the worse of the two errors."""
    import matplotlib.pyplot as plt

    thin = speedup.Point("heat3d", "dace_cpu", 2.0, 1.0, speedup.BAND_MID, (0.9, 1.1))
    fat = speedup.Point("jacobi2d", "dace_cpu", 2.0, 1.0, speedup.BAND_MID, tuple([1.0] * speedup.MIN_BOX_SAMPLES))
    fig, ax = plt.subplots()
    left = speedup.draw_boxes(ax, [thin, fat], {"heat3d": 0, "jacobi2d": 1}, {"dace_cpu": "#1f77b4"})
    plt.close(fig)
    assert [point.kernel for point in left] == ["heat3d"], "the thin cell must fall back to a marker"


def test_every_cell_is_drawn_exactly_once_whichever_way_it_is_drawn(tmp_path: pathlib.Path) -> None:
    """Boxes and markers coexist in one panel, so the risk is a cell drawn twice or not at all."""
    import matplotlib.pyplot as plt

    points = [
        speedup.Point("heat3d", "dace_cpu", 2.0, 1.0, speedup.BAND_MID, (0.9, 1.1)),
        speedup.Point("jacobi2d", "dace_cpu", 2.0, 1.0, speedup.BAND_MID, (0.9, 1.0, 1.1, 1.2, 1.05)),
    ]
    fig, ax = plt.subplots()
    speedup.draw_band(ax, speedup.BAND_MID, points, {"heat3d": 0, "jacobi2d": 1}, {"dace_cpu": "#1f77b4"}, boxes=True)
    markers = [line for line in ax.get_lines() if line.get_marker() == "o"]
    drawn = sum(len(line.get_xdata()) for line in markers)
    plt.close(fig)
    assert drawn == 1, f"expected the one thin cell as a marker, got {drawn}"


def test_dodged_frameworks_do_not_share_an_x_position() -> None:
    """Two frameworks' boxes at one kernel must not sit on top of each other, and a single
    framework must stay ON its kernel's tick -- where the marker figure puts it."""
    assert speedup.dodge_offsets(1) == [0.0]
    two = speedup.dodge_offsets(2)
    assert two[0] < 0.0 < two[1]
    assert sum(two) == pytest.approx(0.0), "the group must stay centred on the tick"
    assert max(speedup.dodge_offsets(4)) - min(speedup.dodge_offsets(4)) <= 0.8, "the group must stay in its slot"


def test_compact_weights_panel_heights_by_population_and_is_otherwise_the_equal_split() -> None:
    """``--compact`` is a LAYOUT knob. Without it the panels keep matplotlib's equal split, and with
    it a band holding one outlier stops costing the same height as one holding forty kernels."""
    points = ([speedup.Point(f"k{i}", "dace_cpu", 2.0, 1.0, speedup.BAND_MID)
               for i in range(9)] + [speedup.Point("solo", "dace_cpu", 20.0, 19.0, speedup.BAND_HIGH)])
    present = [speedup.BAND_HIGH, speedup.BAND_MID]
    assert speedup.panel_heights(points, present, compact=False) is None
    heights = speedup.panel_heights(points, present, compact=True)
    assert heights[1] > heights[0], "the crowded band must get the taller panel"
    assert min(heights) >= 0.18, "a sparse band must stay readable, not collapse to a rule"


def test_the_demo_draws_enough_repetitions_for_a_box() -> None:
    """The demo exists to be looked at, and with --boxplot that means it must actually produce
    boxes. A demo whose cells fell under the threshold would render as the marker figure while
    claiming to show spread."""
    points = speedup.demo_points()
    assert all(len(point.samples) >= speedup.MIN_BOX_SAMPLES for point in points)
    assert any(len(set(point.samples)) > 1 for point in points), "jitter must actually vary"
    assert speedup.demo_points() == points, "the demo seed must keep the boxes reproducible too"


# --- the square embed figure ---------------------------------------------------------------------


def two_framework_points(kernels, band: str, magnitude: float = 3.0):
    """Both demo frameworks on every kernel in ``kernels`` -- the complete groups the figure wants."""
    return [
        speedup.Point(kernel, framework, magnitude, magnitude - 1.0, band, tuple([magnitude - 1.0] * 12))
        for kernel in kernels for framework in speedup.DEMO_FRAMEWORKS
    ]


def test_the_square_figure_keeps_its_kernels_on_one_order_of_magnitude() -> None:
    """A single ``> 10x`` cell would set a y range that collapses every other box to a line -- the
    same problem the banded layout exists to solve, except a square panel has no second band to
    move the outlier to. So the kernels must come from ONE band, never a mix."""
    points = two_framework_points(["k0", "k1"], speedup.BAND_MID)
    points += two_framework_points(["huge"], speedup.BAND_HIGH, magnitude=90.0)
    kernels, frameworks = speedup.square_kernels(points)
    assert frameworks == sorted(speedup.DEMO_FRAMEWORKS)
    assert "huge" not in kernels, "the outlier would flatten the others"
    assert kernels == ["k0", "k1"]


def test_a_kernel_missing_one_framework_is_not_drawn_as_a_half_group() -> None:
    """The figure's claim is a comparison. A kernel where only one agent has a box invites reading
    the gap as a result rather than as data that was never collected."""
    points = two_framework_points(["paired"], speedup.BAND_MID)
    points.append(speedup.Point("lonely", speedup.DEMO_FRAMEWORKS[0], 3.0, 2.0, speedup.BAND_MID, tuple([2.0] * 12)))
    kernels, _frameworks = speedup.square_kernels(points)
    assert kernels == ["paired"], "a group with a missing agent must be left out entirely"


def test_the_square_figure_groups_both_agents_on_one_axis(tmp_path: pathlib.Path) -> None:
    """It mimics the banded figure: kernel names on x, a speedup label, a legend naming the agents,
    and one hue per agent -- so the two are read against each other rather than in two panels."""
    out = tmp_path / "square.svg"
    written = speedup.square_figure(speedup.demo_points(), str(out))
    assert pathlib.Path(written).exists()
    body = out.read_text()
    for band in speedup.BANDS:
        assert band not in body, f"band title {band!r} leaked into a single-band panel"
    for framework in speedup.DEMO_FRAMEWORKS:
        assert framework in body, f"the legend must name {framework!r}"
    assert "speedup" in body, "the y axis must say what it measures"


def test_the_square_figure_is_square() -> None:
    """It is specified as a square panel; a drifting aspect would quietly become a wide strip."""
    assert speedup.SQUARE_SIDE > 0
    assert speedup.SQUARE_CELLS == 4


def test_the_bare_simple_figure_clears_the_LEFT_title_not_just_the_centre(tmp_path: pathlib.Path) -> None:
    """Regression. ``draw_band`` sets the band label with ``loc="left"``, which matplotlib keeps as
    a DIFFERENT artist from the centre title -- so ``set_title("")`` cleared nothing visible and the
    band label shipped on a figure asked to have no titles."""
    points = speedup.demo_points()
    kernels = speedup.plotted_kernels(points)
    out = tmp_path / "bare.svg"
    speedup.simple_figure(points, kernels, str(out), boxes=True, bare=True)
    body = out.read_text()
    for band in speedup.BANDS:
        assert band not in body, f"band title {band!r} survived --bare"
    for framework in speedup.DEMO_FRAMEWORKS:
        assert framework not in body, f"legend text {framework!r} survived --bare"
    assert "signed relative change" not in body, "the y label survived --bare"
    for kernel, _low, _high, _sign in speedup.DEMO_CELLS:
        assert f">{kernel}<" not in body, f"kernel name {kernel!r} survived --bare"


def test_the_unbare_simple_figure_still_states_what_it_hides(tmp_path: pathlib.Path) -> None:
    """The default figure must keep saying it shows one band of several. --bare drops that
    statement on purpose, which is only safe because the default does not."""
    points = speedup.demo_points()
    kernels = speedup.plotted_kernels(points)
    out = tmp_path / "titled.svg"
    speedup.simple_figure(points, kernels, str(out), boxes=True)
    assert "not shown" in out.read_text(), "the simplification must be stated on the default figure"


def test_the_square_ticks_always_show_the_axis_landmarks() -> None:
    """-1, 0 and +1 are not round numbers on this axis, they are 2x slower / unchanged / 2x faster.
    A generic locator omits them routinely -- on a -1.6..4.3 panel it chose 0.0 and 2.5, leaving the
    figure unable to say whether a box below zero was a small regression or a catastrophic one."""
    assert speedup.square_ticks(-1.8, 4.4) == [-1.0, 0.0, 1.0, 4.0]
    # A bottom the landmarks do not reach gets one whole number; -1.8 above did not need one,
    # because ceil(-1.8) is -1 and the landmark already sits there.
    assert speedup.square_ticks(-2.4, 4.4) == [-2.0, -1.0, 0.0, 1.0, 4.0]
    assert -1.0 in speedup.square_ticks(-1.2, 6.0)
    assert 0.0 in speedup.square_ticks(-3.0, 3.0)
    # A range that misses every landmark still gets ticks rather than an empty axis.
    assert speedup.square_ticks(20.0, 40.0) == [20, 40]


def test_the_square_figure_shows_a_slow_down_when_its_band_has_one() -> None:
    """At this size the figure IS the summary somebody reads. One that shows only wins while the
    band it came from also holds losses is the wrong summary."""
    kernels, _frameworks = speedup.square_kernels(speedup.demo_points())
    points = {(p.kernel, p.framework): p for p in speedup.demo_points()}
    changes = [points[(k, f)].change for k in kernels for f in speedup.DEMO_FRAMEWORKS]
    assert any(change < 0.0 for change in changes), "no regression is shown"
    assert any(change > 0.0 for change in changes), "no speed-up is shown"
