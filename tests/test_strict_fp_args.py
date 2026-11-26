# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The strict-FP rebuild of a dace variant rounds as written: scalar libm, no contraction, no reciprocal rewrite."""

import copy
import shlex
from typing import Any

import dace
import pytest

from hpcagent_bench import flags
from hpcagent_bench.frameworks import dace_framework
from hpcagent_bench.frameworks.dace_framework import STRICT_FP_FLAG, strict_fp_args

GCC_VECMATH = f"-march=native -fopenmp -ffp-contract=fast -include {shlex.quote(str(flags.VECMATH_H))}"


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        pytest.param(
            GCC_VECMATH, f"-march=native -fopenmp -ffp-contract=fast {STRICT_FP_FLAG}", id="gcc-vecmath-header"
        ),
        pytest.param(
            "-march=native -fopenmp=libomp -Xarch_host -fveclib=libmvec",
            f"-march=native -fopenmp=libomp {STRICT_FP_FLAG}",
            id="clang-veclib",
        ),
        pytest.param("-include other.h", f"-include other.h {STRICT_FP_FLAG}", id="another-included-header-stays"),
    ],
)
def test_the_strict_rebuild_drops_the_vector_libm_and_keeps_every_other_flag(given: str, expected: str) -> None:
    """cloudsc_monolith: libmvec's exp/log/pow differ from libm by a few ulp, which no contraction flag undoes."""
    assert strict_fp_args(given) == expected


def test_the_rebuild_compiles_under_the_strict_flags_and_restores_the_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The only call site of :func:`strict_fp_args`: ``strict_fp_or`` compiles the variant while DaCe's configuration
    holds the strict flags, and leaves the configuration as it found it."""
    seen: list[str] = []

    def compiled(sdfg: Any) -> object:  # noqa: ANN401
        seen.append(str(dace.Config.get("compiler", "cpu", "args")))
        return object()

    framework = dace_framework.DaceFramework("dace_cpu_canon")
    monkeypatch.setattr(dace_framework, "timed_compile", compiled)
    monkeypatch.setattr(dace_framework.copy, "deepcopy", copy.copy)
    monkeypatch.setattr(dace_framework, "TimedCompiledSDFG", lambda *args: args)
    monkeypatch.setattr(dace_framework.DaceFramework, "verify", lambda *args: True)
    before = dace.Config.get("compiler", "cpu", "args")
    rebuilt = framework.strict_fp_or("canon_cpu", "fallback", dace.SDFG("k"), None, None, None)
    assert rebuilt != "fallback"
    assert STRICT_FP_FLAG in seen[0]
    assert dace.Config.get("compiler", "cpu", "args") == before
