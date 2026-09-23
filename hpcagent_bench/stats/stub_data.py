# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stub-random observations: seeded, deterministic rows in the SAME schema the real extractor
writes (:func:`hpcagent_bench.experiments.read_observations`), for rendering sample figures with no
cluster data.

Every plotting entry point in this repo (``statistics/plot_per_kernel.py``,
``statistics/plot_scaling.py``, the library under :mod:`hpcagent_bench.stats.figures`) reads one
long-format observations table and does not care whether its rows came from a judge database or
were made up -- so this module builds that exact table, not a shortcut past it. Two row shapes:

* an EPISODE (:func:`_episode_rows`): a ``submission`` row (``speedup``) and a ``task`` row
  (``tokens``), the shape :func:`hpcagent_bench.stats.population.graded_episode_rows` /
  ``episode_tokens`` read, one per (arm, kernel) -- see ``tests/test_plot_per_kernel.py::episode``,
  which this mirrors so a stub row can never drift from what the real tests already pin.
* a SCALING POINT (:func:`_scaling_rows`): one row per (arm, kernel, mode, P), the shape
  :mod:`hpcagent_bench.stats.figures.scaling` reads under ``record == "scaling"`` -- mirrors
  ``tests/test_plot_scaling.py::row``.

Nothing here recomputes a statistic (no geomean, no efficiency): the numbers are synthetic INPUTS,
and the figures compute every summary from them exactly as they would from real data. Seeded with
:mod:`random`'s Mersenne Twister so the same seed always writes the same CSV bytes.
"""

import dataclasses
import math
import random
from collections.abc import Sequence

import pandas as pd

#: Real benchmark names (``hpcagent_bench/benchmarks/*/*``), so a sample figure reads like a real
#: one instead of "kernel_07". Order is fixed so a caller asking for N kernels always gets the same
#: N regardless of ``seed`` (the seed only randomizes the MEASUREMENTS, not which kernels appear).
KERNEL_NAMES: tuple[str, ...] = (
    "gemm",
    "heat_3d",
    "lavamd",
    "conv2d_relu_bias_add",
    "conv2d_bias",
    "conv2d_gelu_global_avg_pool",
    "conv2d_tanh_scaling_bias_add_max",
    "conv2d_activation_batch_norm",
    "conv2d_divide_leaky_relu",
    "conv2d_relu_hardswish",
    "conv2d_scaling_min",
    "conv2d_subtract_hardswish_max_pool_mish",
    "conv2d_multiply_leaky_relu_gelu",
    "conv2d_group_norm_scale_max_pool_clamp",
    "conv2d_add_scale_sigmoid_group_norm",
    "conv3d_hardswish_group_norm_mean",
    "conv3d_mish_tanh",
    "conv3d_multiply_instance_norm_clamp_multiply_max",
    "conv3d_divide_max_global_avg_pool_bias_add_sum",
    "average_pooling_3d",
    "max_pooling_3d",
    "bmm_instance_norm_sum_residual_add_multiply",
    "matmul_sigmoid_sum",
    "vanilla_rnn_hidden",
    "swin_mlp",
    "vision_attention",
    "backtrack_branch_bound",
    "wavefront2d",
    "two_stream_reftrans",
    "scatter_accum_dup",
    "compact_threshold_pack",
    "argmax_value",
    "argmin_value",
    "argmin_over_a_dimension",
    "unroll_prime_17_uniform",
    "unroll_partial_5_then_12",
    "tsvc_2_s111",
    "tsvc_2_s273",
    "tsvc_2_s279",
    "tsvc_2_s441",
    "tsvc_2_s482",
    "tsvc_2_vpv",
)

#: The models a sample figure overlays, in registry order (``hpcagent_bench/envs/registry.yaml``
#: ``models``) so ``palette.model_color``/``marker`` resolve them to real hues and shapes rather
#: than the "unknown tag" fallback.
MODELS: tuple[str, ...] = ("qwen38", "oss120b", "kimi27sglang")

#: The packets a sample figure's arms carry (``registry.yaml`` ``packets``): no packet (control)
#: and one treatment.
PACKETS: tuple[str, ...] = ("", "cpf")

#: Rank counts the stub scaling sweep measures -- the same grid the real mlscale track runs
#: (:mod:`hpcagent_bench.stats.figures.scaling` docstring).
RANKS: tuple[int, ...] = (1, 2, 4, 8, 16)

#: Distributed kernel names for the scaling figures (mlscale's own roster shape: ``dist_*``).
SCALING_KERNELS: tuple[str, ...] = (
    "dist_softmax",
    "dist_layer_norm",
    "dist_cross_entropy",
    "dist_matmul_large_k",
    "dist_sdpa",
    "dist_matmul_gelu_softmax",
    "dist_gemm_add_relu",
    "dist_gemm_group_norm_swish",
    "dist_mlp_tp",
    "dist_moe_dispatch",
)


def _arm_name(model: str, packet: str, mode: str = "") -> str:
    """One arm label, in the ``<experiment>-<model>-<language>[-<packet>][-<mode>]`` shape
    :func:`hpcagent_bench.experiment_tags.packet_of`/``language_of`` parse back out of an arm name."""
    tail = f"-{packet}" if packet else ""
    modepart = f"-{mode}" if mode else ""
    return f"sample-plots{modepart}-{model}-hip{tail}"


def _lognormal(rng: random.Random, mu: float, sigma: float) -> float:
    return math.exp(rng.gauss(mu, sigma))


def _episode_rows(rng: random.Random, arm: str, kernel: str, rep: int, mu_log_speedup: float) -> list[dict]:
    """One episode: a ``submission`` row (log-normal speed-up around ``exp(mu_log_speedup)``) and a
    ``task`` row (log-normal token count), the shape ``tests/test_plot_per_kernel.py::episode``
    pins. A small chance of a non-delivery (``suspect`` speedup <= 0 is what
    :func:`hpcagent_bench.stats.population.graded_episode_rows` drops, so a "never verified" episode
    here just omits the ``submission`` row and keeps only ``task`` -- read as unserved-not-answered
    by :func:`hpcagent_bench.stats.figures.per_kernel.speedup_cells`'s ``served`` fallback)."""
    run = f"{arm}-{kernel}-w{rep}"
    common = {
        "arm": arm,
        "benchmark": kernel,
        "run_root": "sample-plots-0923",
        "job": run,
        "run_id": run,
        "attempt_index": 1,
        "ts_ms": 1_000 + rep,
        "suspect": 0,
        "timing_reduction": "mwd-v2",
    }
    rows = []
    delivered = rng.random() > 0.06  # ~6% undelivered, so the "No Verified Answer" cross renders
    if delivered:
        speedup = max(_lognormal(rng, mu_log_speedup, 0.35), 1e-3)
        rows.append({**common, "record": "submission", "speedup": speedup, "tokens": None})
    tokens = _lognormal(rng, math.log(120_000.0), 0.4)
    rows.append({**common, "record": "task", "speedup": None, "tokens": tokens})
    return rows


def per_kernel_frame(
    seed: int = 20260923,
    kernels: Sequence[str] = KERNEL_NAMES,
    models: Sequence[str] = MODELS,
    packets: Sequence[str] = PACKETS,
    episodes_per_cell: int = 1,
) -> pd.DataFrame:
    """One observations table for the per-kernel + geomean figure: every (model, packet) arm times
    every kernel, ``episodes_per_cell`` episodes each. The control packet centres near 1x
    (log-mean 0); the treatment centres near a real win with per-kernel spread, so the geomean
    column has something to summarize."""
    rng = random.Random(seed)
    rows: list[dict] = []
    for model in models:
        for packet in packets:
            arm = _arm_name(model, packet)
            base_mu = math.log(1.0) if not packet else math.log(2.2)
            for kernel in kernels:
                kernel_mu = base_mu + rng.gauss(0.0, 0.5)
                for rep in range(episodes_per_cell):
                    rows += _episode_rows(rng, arm, kernel, rep, kernel_mu)
    return pd.DataFrame(rows)


def _scaling_rows(
    rng: random.Random, arm: str, kernel: str, mode: str, base_efficiency: float, jitter: float
) -> list[dict]:
    """One (arm, kernel, mode) curve over :data:`RANKS`, the shape
    ``tests/test_plot_scaling.py::row`` pins, plus the task-identity columns
    (:data:`hpcagent_bench.experiments.TASK_KEY`) every extracted row carries -- a real curve's P
    points are one graded submission replayed at several rank counts, so they share one
    ``run_root``/``job``/``run_id``/``attempt_index``, the same episode identity an episode row's
    ``submission``/``task`` pair carries. Efficiency decays gently with P around
    ``base_efficiency`` (communication overhead growing with rank count), with per-P jitter."""
    t1 = 4096.0
    run = f"{arm}-{kernel}-{mode}"
    rows = []
    for p in RANKS:
        eta = max(0.05, min(1.05, base_efficiency - 0.015 * math.log2(p) + rng.gauss(0.0, jitter)))
        if mode == "strong":
            ranked_ns = t1 / (p * eta)
            work_ratio = float("nan")
        else:
            ranked_ns = t1 / eta
            work_ratio = float(p)
        rows.append(
            {
                "record": "scaling",
                "arm": arm,
                "benchmark": kernel,
                "run_root": "sample-plots-0923",
                "job": run,
                "run_id": run,
                "attempt_index": 1,
                "scaling_mode": mode,
                "ranks": p,
                "nodes": -(-p // 4),
                "ranked_ns": ranked_ns,
                "single_rank_ns": t1,
                "work_ratio": work_ratio,
                "scaling_note": "",
                "ts_ms": 10,
            }
        )
    return rows


def scaling_frame(
    seed: int = 20260923,
    kernels: Sequence[str] = SCALING_KERNELS,
    models: Sequence[str] = MODELS,
    packets: Sequence[str] = ("", "dist-rccl-amd"),
) -> pd.DataFrame:
    """One observations table for the weak/strong scaling figures: every (model, packet) arm's
    weak AND strong curve over every kernel in :data:`RANKS`. The RCCL packet scales a little
    better (higher base efficiency, less jitter) than the control, so the per-arm geomean summary
    has a real gap to show."""
    rng = random.Random(seed + 1)
    rows: list[dict] = []
    for model in models:
        for packet in packets:
            base_eta = 0.62 if not packet else 0.82
            jitter = 0.05 if not packet else 0.03
            for mode in ("weak", "strong"):
                arm = _arm_name(model, packet, mode=mode)
                for kernel in kernels:
                    kernel_eta = max(0.1, min(1.0, base_eta + rng.gauss(0.0, 0.06)))
                    rows += _scaling_rows(rng, arm, kernel, mode, kernel_eta, jitter)
    return pd.DataFrame(rows)


@dataclasses.dataclass(frozen=True, slots=True)
class StubDataset:
    """The two tables a sample-plots run needs, plus the combined observations CSV
    :func:`hpcagent_bench.experiments.read_observations` reads (each row shape ignores columns it
    does not use, so one CSV serves both figure families)."""

    per_kernel: pd.DataFrame
    scaling: pd.DataFrame

    def combined(self) -> pd.DataFrame:
        return pd.concat([self.per_kernel, self.scaling], ignore_index=True, sort=False)


def generate(
    seed: int = 20260923,
    kernels: Sequence[str] = KERNEL_NAMES,
    scaling_kernels: Sequence[str] = SCALING_KERNELS,
    models: Sequence[str] = MODELS,
) -> StubDataset:
    """The full stub dataset at one seed. Deterministic: two calls with the same seed produce
    byte-identical CSVs (:func:`hpcagent_bench.stats.stub_data.generate` has no wall-clock or
    hash-order dependence -- every draw is on the seeded :class:`random.Random`, and DataFrame
    construction preserves row order). ``kernels``/``scaling_kernels``/``models`` narrow the
    default rosters -- a test wanting a handful of rows does not have to pay for the full 40
    kernels x 6 arms."""
    return StubDataset(
        per_kernel=per_kernel_frame(seed=seed, kernels=kernels, models=models),
        scaling=scaling_frame(seed=seed, kernels=scaling_kernels, models=models),
    )
