# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A gpu CPF form renders through the experimental DaCe GPU code generator, whatever the process-wide value."""

import pytest

pytest.importorskip("dace")

from dace.config import Config, set_temporary

from hpcagent_bench.cpf import bridge as cpf_bridge
from hpcagent_bench.cpf.cache import Target


@pytest.mark.parametrize(
    ("target", "pinned"), [(Target.GPU, cpf_bridge.GPU_CODEGEN), (Target.CPU, "legacy")], ids=["gpu", "cpu"]
)
def test_a_gpu_render_uses_the_experimental_generator_even_after_a_pipeline_selected_legacy(
    target: Target, pinned: str
) -> None:
    """A gpu render pins the experimental generator; a cpu render leaves the gpu generator alone."""
    with set_temporary("compiler", "cuda", "implementation", value="legacy"):
        with cpf_bridge.gpu_codegen_pinned(target):
            assert Config.get("compiler", "cuda", "implementation") == pinned
        assert Config.get("compiler", "cuda", "implementation") == "legacy"


def test_an_environment_pin_to_another_generator_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DACE_compiler_cuda_implementation", "legacy")
    if Config.get("compiler", "cuda", "implementation") != "legacy":
        pytest.skip("this dace does not let the environment outrank the in-process setting")
    with (
        pytest.raises(RuntimeError, match="DACE_compiler_cuda_implementation"),
        cpf_bridge.gpu_codegen_pinned(Target.GPU),
    ):
        pass
