"""Unified auto-tuner interface + search budget (Workstream G).

The agentic optimizer IS an auto-tuner: it takes a kernel + a BUDGET (search
trials / wall-clock / $) and returns an optimized artifact, scored by the SAME
correctness + perf machinery as any framework. TVM MetaSchedule, the Triton
config sweep, the Polly/Pluto flag presets, and an AI ``Agent`` are all
AutoTuners under one contract -- so the leaderboard treats "agent" as just
another (auto-tuning) framework and the search budget is set in ONE place
instead of a per-backend env var.

* :class:`TuningBudget` -- the single source of "how much search", resolved from
  ``OPTARENA_TUNE_BUDGET`` (a scale name or an integer) and exposing the
  per-backend caps (TVM trials, Triton config cap). The legacy per-framework env
  vars remain honoured as overrides for back-compat.
* :class:`AutoTuner` -- ``tune(program, budget) -> optimized program``. A
  framework that does not search inherits the identity tuner
  (:meth:`optarena.infrastructure.framework.Framework.autotune` default).
"""
from __future__ import annotations

import abc
import os
from dataclasses import dataclass
from typing import Any, Optional

#: named scale -> (TVM MetaSchedule trials, Triton config-sweep cap). ONE knob
#: drives every backend's search width; ``full`` effectively uncaps Triton.
SCALES = {"small": (64, 4), "full": (1024, 1_000_000)}
DEFAULT_SCALE = "small"


@dataclass(frozen=True)
class TuningBudget:
    """How much search a tuner may spend.

    :ivar scale: the named scale (``small`` / ``full`` / ``custom``).
    :ivar trials: candidate schedules to evaluate (TVM MetaSchedule).
    :ivar configs: autotune-config cap (Triton).
    :ivar time_s: optional wall-clock ceiling (an Agent / time-boxed tuner).
    :ivar cost: optional dollar/token ceiling (an Agent).
    """
    scale: str = DEFAULT_SCALE
    trials: int = 64
    configs: int = 4
    time_s: Optional[float] = None
    cost: Optional[float] = None

    @classmethod
    def from_env(cls, scale: Optional[str] = None) -> "TuningBudget":
        """Resolve the budget from ``scale`` or ``$OPTARENA_TUNE_BUDGET`` -- a
        named scale, or a bare integer that caps both backends explicitly."""
        raw = scale or os.environ.get("OPTARENA_TUNE_BUDGET", DEFAULT_SCALE)
        if raw in SCALES:
            trials, configs = SCALES[raw]
            return cls(scale=raw, trials=trials, configs=configs)
        try:
            n = int(raw)
            return cls(scale="custom", trials=n, configs=n)
        except (TypeError, ValueError):
            trials, configs = SCALES[DEFAULT_SCALE]
            return cls(scale=DEFAULT_SCALE, trials=trials, configs=configs)

    def tvm_trials(self) -> int:
        """MetaSchedule trial count. The legacy
        ``OPTARENA_TVM_METASCHEDULE_TRIALS`` wins when set (back-compat)."""
        raw = os.environ.get("OPTARENA_TVM_METASCHEDULE_TRIALS")
        if raw in SCALES:
            return SCALES[raw][0]
        if raw and raw.lstrip("-").isdigit():
            return int(raw)
        return self.trials

    def triton_config_cap(self) -> int:
        """Triton autotune-config cap. The legacy
        ``OPTARENA_TRITON_AUTOTUNE_SIZE`` / ``OPTARENA_TRITON_AUTOTUNE_N`` win when
        set (back-compat); ``full`` removes the cap."""
        size = os.environ.get("OPTARENA_TRITON_AUTOTUNE_SIZE")
        if size in SCALES:
            return SCALES[size][1]
        cap = os.environ.get("OPTARENA_TRITON_AUTOTUNE_N")
        if cap and cap.isdigit():
            return int(cap)
        return self.configs


class AutoTuner(abc.ABC):
    """One interface for every optimizer that SEARCHES for a faster artifact.

    :meth:`tune` takes a kernel handle plus a :class:`TuningBudget` and returns
    the optimized artifact (the same type the framework would otherwise run), so
    the harness scores a tuner exactly like a plain framework. Implementations:
    TVM MetaSchedule, Triton autotune, Polly/Pluto (a one-point "search" = a flag
    preset), and the AI ``Agent`` (budget = tokens/$/time). A framework that does
    not search inherits :class:`IdentityTuner`.
    """

    name: str = "autotuner"

    @abc.abstractmethod
    def tune(self, program: Any, budget: TuningBudget) -> Any:
        """Search within ``budget`` and return the optimized ``program``."""


class IdentityTuner(AutoTuner):
    """No search -- returns the program unchanged (the default for a framework
    that is not an auto-tuner)."""

    name = "identity"

    def tune(self, program: Any, budget: TuningBudget) -> Any:
        return program
