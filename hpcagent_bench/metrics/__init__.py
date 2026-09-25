# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-kernel metrics a framework sweep records beside its timings, one module per metric.

A module in this package is a SWEEP METRIC when it defines the three functions of
:class:`SweepMetric`; :func:`sweep_metrics` finds every such module by scanning the package, so a
new metric is one file here plus its ``metrics.<module name>`` switch in ``config.yaml``. A module
without that interface is a library the scan skips.
"""

import importlib
import pkgutil
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from hpcagent_bench.frameworks.benchmark import Benchmark
    from hpcagent_bench.frameworks.framework import Framework, KernelImpl
    from hpcagent_bench.frameworks.schema import KernelMetric

#: The functions a sweep metric module defines.
SWEEP_METRIC_API = ("enabled", "measure_sweep", "rows")


class SweepMetric(Protocol):
    """What :class:`hpcagent_bench.frameworks.test.Test` calls on a metric module, per implementation."""

    def enabled(self) -> bool:
        """Whether ``metrics.<name>`` is switched on."""
        ...

    def measure_sweep(
        self, frmwrk: "Framework", impl: "KernelImpl", bench: "Benchmark", reports: dict[str, str | None], datatype: str
    ) -> Any:
        """The measured artifact's value, or ``None`` when this framework has nothing to measure."""
        ...

    def rows(
        self,
        measured: Any,
        *,
        timestamp: int,
        benchmark: str,
        framework: str,
        flavor: str | None,
        impl: str,
        datatype: str,
    ) -> "list[KernelMetric]":
        """The ``kernel_metrics`` rows of one measured value."""
        ...


def sweep_metrics() -> list[tuple[str, SweepMetric]]:
    """``(name, module)`` for every sweep metric module in this package, in name order. The name is the
    module's own and doubles as the metric's ``metrics.<name>`` config key."""
    found: list[tuple[str, SweepMetric]] = []
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
        module = importlib.import_module(f"{__name__}.{info.name}")
        if all(callable(vars(module).get(fn)) for fn in SWEEP_METRIC_API):
            found.append((info.name, cast(SweepMetric, module)))
    return found
