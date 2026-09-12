# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""The POPULATION an aggregate is taken over, made explicit so a wrong one cannot be expressed.

Three defects in this repo's published tables were the same shape: a correct statistic applied to a
population the claim was not about.

* Speed-ups over DIFFERENT DENOMINATORS were pooled. The judge stamps ``baseline`` on every graded
  row and it is a per-JOB property: the llr40v9/v10 jobs graded against the single-core C lowering
  or against parallel numba, and the same agent work reads 95.3x under one and 1.82x under the
  other while ``native_ns`` moves 7%. A mean over both is a ratio with no denominator, so
  :class:`ArmAggregate` carries its ``baseline`` and :func:`ratio` REFUSES two that disagree.
* Each arm's geomean was taken over a DIFFERENT KERNEL SET -- whatever that arm happened to solve.
  Ranking those numbers ranks coverage as much as quality, and it penalises an arm for reaching the
  hard kernels at all. So an aggregate carries the exact ``kernels`` behind it and :func:`ratio`
  refuses two whose kernel tuples differ; :func:`align` is how a caller gets two that do not.
* The EPISODE key was wrong. ``runs.run_id`` is a PRIMARY KEY inside ONE results database, and a
  launcher derives it from the rank layout (``<arm>.n<node>.p<problem>.w<worker>``), so two jobs of
  one arm reuse it: 154 of 226 llr40 run_ids appear under more than one job. Deduplicating on
  ``run_id`` alone therefore discards whole agent runs. :data:`EPISODE_KEY` is the key that is
  actually one agent on one kernel.

TWO POLICIES, AND A TABLE MUST NAME ITS OWN. ``solved`` is "how good when it works" -- the geomean
over the kernels the arm verified. ``served`` is "how good overall" -- every kernel the arm was
GIVEN, with a non-delivery entered at 1.0, because an agent that died or never verified anything
left the baseline standing and that is a real outcome of the arm. Both are legitimate and they
answer different questions, so :class:`ArmAggregate` stores which one it is and :func:`ratio`
refuses to divide one by the other.

The ``served`` roster is the kernels the arm has a RECORDED observation for, never the full roster:
an unserved kernel is a scheduling fact, not a failure, and entering one at 1.0 would score an arm
on how long its job ran. A snapshot of an unfinished campaign therefore reports both columns.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from hpcagent_bench.stats import summary

if TYPE_CHECKING:
    import pandas as pd

#: One agent on one kernel. ``run_root`` and ``job`` scope the ``run_id``, which is only unique
#: inside one results database; ``benchmark`` is carried because a caller may reduce a frame that
#: spans kernels, and one episode is one kernel by construction.
EPISODE_KEY: tuple[str, str, str, str] = ("run_root", "job", "run_id", "benchmark")

#: Which population a number is over. Never a default: a table that does not state one is the
#: defect this module exists to prevent.
KernelPolicy = Literal["solved", "served"]

POLICIES: tuple[KernelPolicy, ...] = ("solved", "served")

#: What a kernel the arm was served but never verified scores under ``served``. A speed-up of 1.0
#: is exactly "the baseline stands", which is what a non-delivery leaves behind.
NOT_DELIVERED: float = 1.0


class MixedPopulationError(ValueError):
    """Raised when an aggregate would be formed over two populations the claim is not about."""


def is_named(value: object) -> bool:
    """Whether a cell records a denominator at all: not ``None``, not NaN, not blank."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() != "nan"


def one_denominator(values: Iterable[object], label: str = "") -> str:
    """The single ``baseline`` a slice was graded against, or raise.

    ``figures/results.baseline_of`` takes the MODE and warns, which is right for drawing one
    campaign that carries a few stale rows. An AGGREGATE cannot do that: the majority denominator
    is still not the denominator of the minority's rows, so this refuses instead of picking.

    A blank, ``None`` or NaN entry is a row whose writer recorded no denominator; those are skipped
    so a recoverable gap does not read as a second reference, and a slice of nothing but gaps raises.
    """
    named = sorted({str(value).strip() for value in values if is_named(value)})
    prefix = f"{label}: " if label else ""
    if not named:
        raise MixedPopulationError(f"{prefix}no baseline recorded; a speed-up with no denominator is not a ratio")
    if len(named) > 1:
        raise MixedPopulationError(
            f"{prefix}this slice mixes baseline denominators {named}; split it by baseline rather than pooling it"
        )
    return named[0]


def last_per_episode(frame: pd.DataFrame, order: Sequence[str]) -> pd.DataFrame:
    """One row per episode: the LAST graded row that episode produced, by ``order``.

    Evaluation is single-shot, so within an episode the answer the agent stopped at is the answer.
    Keyed on :data:`EPISODE_KEY` rather than on ``run_id``, which collides across jobs.
    """
    missing = [column for column in (*EPISODE_KEY, *order) if column not in frame.columns]
    if missing:
        raise MixedPopulationError(f"cannot identify an episode without {missing}")
    return frame.sort_values(list(order)).drop_duplicates(list(EPISODE_KEY), keep="last")


def per_episode_max(frame: pd.DataFrame, column: str, keep: Sequence[str] = ()) -> pd.DataFrame:
    """One row per episode carrying that episode's MAXIMUM of ``column``, plus ``keep``.

    For a cumulative counter this is the episode's own total. ``calls.tokens`` is cumulative through
    a call, so summing its rows counts every earlier call once per later one and inflates a long
    repair loop quadratically; taking the maximum reads the total the episode actually reached.
    Callers aggregate the episodes themselves, because a kernel's total spend (a SUM over its
    episodes) and an arm's typical task cost (a MEDIAN over them) are different quantities and this
    module will not pick one for them.

    ``keep`` names columns that are constant within an episode -- the arm, the model, the condition
    -- so a caller can group on them afterwards without a second join.
    """
    missing = [name for name in (column, *EPISODE_KEY, *keep) if name not in frame.columns]
    if missing:
        raise MixedPopulationError(f"cannot reduce episodes without {missing}")
    return frame.groupby([*EPISODE_KEY, *keep], as_index=False)[column].max()


def final_answers(frame: pd.DataFrame, order: Sequence[str], by: Sequence[str]) -> pd.DataFrame:
    """The rows that are each ``by`` group's best FINAL answer, as whole rows.

    The scoring policy in two steps, in one place. WITHIN an episode the LAST verified submission
    counts, because evaluation is single-shot and a max over an episode's submissions scores
    best-of-N attempts rather than the answer the agent stopped at; ACROSS episodes the maximum is
    kept, because how many agents an arm runs is a property of the arm. Whole rows come back so a
    caller can take the timings, the source path or the denominator of the row that won.

    ``frame`` must be the GRADED rows. A ``call`` row carries a speed-up for a round the judge did
    not persist, and a reduction over those is over a population no claim is about.
    """
    if "speedup" not in frame.columns:
        raise MixedPopulationError("a final answer is decided by speedup; the frame carries none")
    episodes = last_per_episode(frame[frame.speedup > 0], order)
    return episodes.sort_values("speedup", ascending=False).drop_duplicates(list(by), keep="first")


@dataclass(frozen=True, slots=True)
class ArmAggregate:
    """One arm's speed-up aggregate, carrying the population it is over.

    ``kernels`` and ``values`` are parallel and are the exact set behind the number, so two of these
    can be checked for comparability rather than assumed to be comparable.
    """

    arm: str
    baseline: str
    policy: KernelPolicy
    kernels: tuple[str, ...]
    values: tuple[float, ...]
    n_solved: int

    def __post_init__(self) -> None:
        if not self.baseline:
            raise MixedPopulationError(f"{self.arm}: an aggregate must name the denominator it is over")
        if self.policy not in POLICIES:
            raise MixedPopulationError(f"{self.arm}: policy must be one of {POLICIES}, got {self.policy!r}")
        if len(self.kernels) != len(self.values):
            raise MixedPopulationError(f"{self.arm}: {len(self.kernels)} kernels against {len(self.values)} values")
        if len(set(self.kernels)) != len(self.kernels):
            raise MixedPopulationError(f"{self.arm}: a kernel may enter an aggregate only once")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.values):
            raise MixedPopulationError(f"{self.arm}: every value must be a finite positive ratio")

    @property
    def n(self) -> int:
        """Kernels behind the number."""
        return len(self.kernels)

    def geomean(self) -> float:
        """Geometric mean over :attr:`kernels`; NaN when the population is empty."""
        return summary.geomean(self.values) if self.values else math.nan

    def median(self) -> float:
        """Median over :attr:`kernels` -- a spread cue beside the geomean, never the headline."""
        return statistics.median(self.values) if self.values else math.nan

    def label(self) -> str:
        """One-line population statement a table or a caption must carry beside the number."""
        return f"geomean over {self.n} kernels vs {self.baseline} ({self.policy}; {self.n_solved} solved)"

    def restricted_to(self, kernels: Sequence[str]) -> ArmAggregate:
        """The same arm over exactly ``kernels``, which must all be present."""
        index = {kernel: value for kernel, value in zip(self.kernels, self.values, strict=True)}
        absent = [kernel for kernel in kernels if kernel not in index]
        if absent:
            raise MixedPopulationError(f"{self.arm}: cannot restrict to kernels it has no value for: {absent[:4]}")
        keep = tuple(kernels)
        solved = min(self.n_solved, len(keep))
        return ArmAggregate(self.arm, self.baseline, self.policy, keep, tuple(index[k] for k in keep), solved)


@dataclass(frozen=True, slots=True)
class Coverage:
    """What an intersection KEPT and what it DROPPED, so nothing vanishes unremarked."""

    n_both: int
    n_only_left: int
    n_only_right: int
    n_neither: int
    only_left: tuple[str, ...]
    only_right: tuple[str, ...]


def aggregate_arm(
    arm: str,
    baseline: str,
    solved: Mapping[str, float],
    served: Collection[str],
    policy: KernelPolicy,
) -> ArmAggregate:
    """Build one arm's aggregate under ``policy``.

    ``solved`` is the arm's one verified value per kernel; ``served`` is every kernel it has a
    recorded observation for. Under ``served`` a kernel in ``served`` and not in ``solved`` enters
    at :data:`NOT_DELIVERED`, which is what a non-delivery left behind.
    """
    if policy not in POLICIES:
        raise MixedPopulationError(f"{arm}: policy must be one of {POLICIES}, got {policy!r}")
    unserved = sorted(set(solved) - set(served))
    if unserved:
        raise MixedPopulationError(f"{arm}: verified kernels that were never served: {unserved[:4]}")
    kernels = tuple(sorted(solved)) if policy == "solved" else tuple(sorted(served))
    values = tuple(float(solved.get(kernel, NOT_DELIVERED)) for kernel in kernels)
    return ArmAggregate(arm, baseline, policy, kernels, values, len(solved))


def common_kernels(aggregates: Sequence[ArmAggregate]) -> tuple[str, ...]:
    """The kernels every aggregate in ``aggregates`` carries a value for."""
    if not aggregates:
        return ()
    shared = set(aggregates[0].kernels)
    for item in aggregates[1:]:
        shared &= set(item.kernels)
    return tuple(sorted(shared))


def align(aggregates: Sequence[ArmAggregate]) -> list[ArmAggregate]:
    """Every aggregate restricted to the one kernel set they share, or raise on mixed denominators.

    This is the only supported way to get aggregates that :func:`ratio` will accept, so a
    cross-arm number cannot be formed over two different kernel sets by accident.
    """
    if not aggregates:
        return []
    one_denominator([item.baseline for item in aggregates], label="align")
    policies = {item.policy for item in aggregates}
    if len(policies) > 1:
        raise MixedPopulationError(f"cannot align aggregates under different policies {sorted(policies)}")
    shared = common_kernels(aggregates)
    return [item.restricted_to(shared) for item in aggregates]


def coverage(left: ArmAggregate, right: ArmAggregate, roster: Collection[str] = ()) -> Coverage:
    """What restricting ``left`` and ``right`` to their shared kernels keeps and drops.

    ``roster`` is the set both arms were asked for, which is what makes ``n_neither`` -- the kernels
    neither reached -- a number rather than an assumption. Without it that count is 0.
    """
    lhs, rhs = set(left.kernels), set(right.kernels)
    return Coverage(
        n_both=len(lhs & rhs),
        n_only_left=len(lhs - rhs),
        n_only_right=len(rhs - lhs),
        n_neither=len(set(roster) - lhs - rhs),
        only_left=tuple(sorted(lhs - rhs)),
        only_right=tuple(sorted(rhs - lhs)),
    )


def mcnemar_exact(only_left: int, only_right: int) -> float:
    """Two-sided exact McNemar p on the DISCORDANT counts of a paired success outcome.

    The concordant pairs carry no information about a difference, so the null is that each of the
    ``only_left + only_right`` disagreements was equally likely to go either way: a binomial(n, 1/2)
    tail on the smaller count, doubled. This is what turns the kernels an intersection dropped into
    a tested claim rather than a footnote. ``experiments/ablation_stats.py`` keeps a stdlib copy for
    the login node, exactly as it does for the signed-rank rule, and the two are proved to agree.
    """
    n = only_left + only_right
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_left, only_right) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def ratio(left: ArmAggregate, right: ArmAggregate) -> float:
    """``left / right`` as a comparison of two arms, or raise when they are not comparable.

    Refuses a different denominator, a different policy and a different kernel set. Those three
    refusals are the whole point of the module: each one was a published comparison that read as a
    statement about the arms and was partly a statement about what they were divided by, what was
    counted as a failure, and which kernels each happened to reach.
    """
    one_denominator([left.baseline, right.baseline], label=f"{left.arm} / {right.arm}")
    if left.policy != right.policy:
        raise MixedPopulationError(f"{left.arm} / {right.arm}: {left.policy} is not comparable with {right.policy}")
    if left.kernels != right.kernels:
        gap = coverage(left, right)
        raise MixedPopulationError(
            f"{left.arm} / {right.arm}: different kernel sets ({left.n} against {right.n}, "
            f"{gap.n_both} shared); call align() first"
        )
    if not left.kernels:
        return math.nan
    return left.geomean() / right.geomean()


def log_differences(left: ArmAggregate, right: ArmAggregate) -> list[float]:
    """``log(left / right)`` per kernel, the paired quantity a signed-rank test is taken over."""
    if left.kernels != right.kernels:
        raise MixedPopulationError(f"{left.arm} / {right.arm}: pairing needs one kernel set; call align() first")
    one_denominator([left.baseline, right.baseline], label=f"{left.arm} / {right.arm}")
    return [math.log(a / b) for a, b in zip(left.values, right.values, strict=True)]
