# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``hpcagent_bench.stats.stub_data`` -- the stub-random data generator: seeded determinism, and
that its rows are the real extractor's schema (:func:`hpcagent_bench.experiments.read_observations`
reads them without warning or error, and the population helpers every figure calls accept them)."""

import pathlib
import warnings

import pytest

from hpcagent_bench import experiments
from hpcagent_bench.stats import population, stub_data
from hpcagent_bench.stats.figures import per_kernel, scaling

SMALL_KERNELS = ("gemm", "heat_3d", "lavamd")
SMALL_SCALING_KERNELS = ("dist_softmax", "dist_layer_norm")
SMALL_MODELS = ("qwen38", "oss120b")


def small_dataset(seed: int = 1) -> stub_data.StubDataset:
    return stub_data.generate(
        seed=seed, kernels=SMALL_KERNELS, scaling_kernels=SMALL_SCALING_KERNELS, models=SMALL_MODELS
    )


def test_generate_is_deterministic_for_one_seed() -> None:
    """Two calls at the same seed produce byte-identical CSVs -- required for a reproducible
    sample-plots run and for a stable test fixture."""
    first = small_dataset(seed=7).combined().to_csv(index=False)
    second = small_dataset(seed=7).combined().to_csv(index=False)
    assert first == second


def test_generate_differs_across_seeds() -> None:
    """A different seed must draw different measurements, or 'seeded' would be a decoration."""
    first = small_dataset(seed=1).per_kernel
    second = small_dataset(seed=2).per_kernel
    assert not first["tokens"].equals(second["tokens"])


def test_per_kernel_rows_carry_the_columns_graded_episode_rows_requires() -> None:
    """:func:`hpcagent_bench.stats.population.graded_episode_rows` refuses a frame missing
    ``speedup``, ``suspect`` or a timing-reduction stamp -- the stub rows must carry all three."""
    frame = small_dataset().per_kernel
    required = {
        "arm",
        "benchmark",
        "record",
        "speedup",
        "tokens",
        "suspect",
        "timing_reduction",
        *population.EPISODE_KEY,
    }
    assert required <= set(frame.columns)
    submissions = frame[frame["record"] == "submission"]
    assert (submissions["speedup"] > 0).all()


def test_per_kernel_frame_is_readable_by_the_population_layer(tmp_path: pathlib.Path) -> None:
    """The stub CSV, round-tripped through :func:`hpcagent_bench.experiments.read_observations`
    exactly as a real campaign's CSV is, produces cells :func:`hpcagent_bench.stats.figures.
    per_kernel.speedup_cells` can draw -- with no warning, which would mean a dropped row."""
    csv_path = tmp_path / "observations.csv"
    small_dataset().combined().to_csv(csv_path, index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frame = experiments.read_observations(csv_path)
    one_arm = frame[frame["arm"] == "sample-plots-qwen38-hip-cpf"]
    cells = per_kernel.speedup_cells(one_arm)
    assert {cell.kernel for cell in cells} == set(SMALL_KERNELS)
    for cell in cells:
        assert cell.episodes and all(v > 0 for v in cell.episodes)


def test_scaling_rows_carry_the_columns_scaling_curves_requires() -> None:
    frame = small_dataset().scaling
    assert set(scaling.REQUIRED_COLUMNS) <= set(frame.columns)
    assert set(frame["ranks"].unique()) == set(stub_data.RANKS)


def test_scaling_frame_is_readable_and_draws_full_curves(tmp_path: pathlib.Path) -> None:
    """Every (arm, kernel, mode) curve reaches :func:`hpcagent_bench.stats.figures.scaling.curves`
    with :data:`hpcagent_bench.stats.figures.scaling.MIN_CURVE_POINTS` or more points -- the stub
    sweep must not silently produce a one-point curve nothing can draw."""
    csv_path = tmp_path / "observations.csv"
    small_dataset().combined().to_csv(csv_path, index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frame = experiments.read_observations(csv_path)
    curves = scaling.curves(frame)
    assert curves
    for curve in curves:
        assert len(curve.points) >= scaling.MIN_CURVE_POINTS
    modes = {curve.mode for curve in curves}
    assert modes == set(scaling.MODES)


def test_a_control_arm_centres_near_one_x_and_a_treated_arm_shows_a_real_win() -> None:
    """The per-kernel dataset's whole point is a summary column with something to show: the
    control packet's geomean is near 1x, the treated packet's is well above it."""
    frame = small_dataset().per_kernel
    control = per_kernel.speedup_cells(frame[frame["arm"] == "sample-plots-qwen38-hip"], served=False)
    treated = per_kernel.speedup_cells(frame[frame["arm"] == "sample-plots-qwen38-hip-cpf"], served=False)
    control_geomean, _, _ = per_kernel.summary_point_speedup(control)
    treated_geomean, _, _ = per_kernel.summary_point_speedup(treated)
    assert 0.5 < control_geomean < 2.0
    assert treated_geomean > control_geomean


def test_generate_has_no_undelivered_kernels_dropped_silently() -> None:
    """An undelivered episode (:mod:`hpcagent_bench.stats.stub_data`'s ~6% chance) must still leave
    a ``task`` row: no episode disappears from the frame entirely."""
    frame = small_dataset().per_kernel
    episodes = frame[["run_root", "job", "run_id", "benchmark"]].drop_duplicates()
    for _, episode in episodes.iterrows():
        rows = frame[
            (frame["run_root"] == episode["run_root"])
            & (frame["job"] == episode["job"])
            & (frame["run_id"] == episode["run_id"])
        ]
        assert (rows["record"] == "task").any()


@pytest.mark.parametrize("seed", [0, 1, 20260923])
def test_generate_never_raises_at_any_seed(seed: int) -> None:
    stub_data.generate(seed=seed, kernels=SMALL_KERNELS, scaling_kernels=SMALL_SCALING_KERNELS, models=SMALL_MODELS)
