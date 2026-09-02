# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The opt-in reroute of async scheduling's sampled-token broadcast.

vLLM's V1 runner broadcasts sampled tokens on ``pp.device_group`` -- the group that otherwise
carries only the inter-stage P2P, which torch serves from per-pair 2-rank communicators. Under
lazy init the first decode bootstraps both at once and rccl reports "Message truncated". The
shipped default is ``--no-async-scheduling``; this patch is the alternative that KEEPS async
scheduling by moving the broadcast onto the sibling communicator, the way upstream's V2 runner
already does (``v1/worker/gpu/pp_utils.PPHandler``).

Run in a subprocess: importing sitecustomize rebinds ``torch.distributed.init_process_group``
process-wide, which has no business leaking into the rest of the suite.
"""

import pathlib
import subprocess
import sys

PATCH_DIR = (
    pathlib.Path(__file__).resolve().parents[1] / "containers/cluster/ce-images/inference/external-eager-pg-patch"
)

DRIVER = """
import sys, types
sys.path.insert(0, {patch_dir!r})

pp = types.SimpleNamespace(device_group="MAIN", collective_group={sibling!r})
seen = []

class GPUModelRunner:
    def _pp_broadcast_prev_sampled_token_ids(self, tokens):
        seen.append(pp.device_group)

    def _pp_receive_prev_sampled_token_ids_to_input_batch(self):
        seen.append(pp.device_group)
        raise ValueError("boom")

runner_mod = types.ModuleType("vllm.v1.worker.gpu_model_runner")
runner_mod.GPUModelRunner = GPUModelRunner
state_mod = types.ModuleType("vllm.distributed.parallel_state")
state_mod.get_pp_group = lambda: pp
for name, mod in [
    ("vllm", types.ModuleType("vllm")),
    ("vllm.v1", types.ModuleType("vllm.v1")),
    ("vllm.v1.worker", types.ModuleType("vllm.v1.worker")),
    ("vllm.v1.worker.gpu_model_runner", runner_mod),
    ("vllm.distributed", types.ModuleType("vllm.distributed")),
    ("vllm.distributed.parallel_state", state_mod),
]:
    sys.modules[name] = mod

import sitecustomize
sitecustomize.patch_pp_token_broadcast()

runner = GPUModelRunner()
runner._pp_broadcast_prev_sampled_token_ids(None)
try:
    runner._pp_receive_prev_sampled_token_ids_to_input_batch()
except ValueError:
    pass
print("SAW", seen, "AFTER", pp.device_group)
"""


def run(sibling):
    proc = subprocess.run(
        [sys.executable, "-c", DRIVER.format(patch_dir=str(PATCH_DIR), sibling=sibling)], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip().splitlines()[-1]


def test_both_directions_run_on_the_sibling():
    """Sender and receiver must move together -- one of them left behind is the collision itself."""
    assert run("SIBLING") == "SAW ['SIBLING', 'SIBLING'] AFTER MAIN"


def test_the_main_group_is_restored_even_when_the_call_raises():
    """The receiver raises here; a leaked swap would put the P2P on the sibling for good."""
    assert run("SIBLING").endswith("AFTER MAIN")


def test_a_group_without_a_sibling_is_left_alone():
    """pp world_size 1 gets no sibling, and has no broadcast worth moving."""
    assert run(None) == "SAW ['MAIN', 'MAIN'] AFTER MAIN"


def test_it_refuses_to_install_without_the_collective_split():
    """Without the split there is no sibling, so the reroute would silently do nothing."""
    text = (PATCH_DIR / "sitecustomize.py").read_text()
    assert "VLLM_PP_TOKEN_BROADCAST_SIBLING=1 needs VLLM_PP_COLLECTIVE_SPLIT=1" in text


def test_it_is_off_by_default():
    text = (PATCH_DIR / "sitecustomize.py").read_text()
    assert 'os.environ.get("VLLM_PP_TOKEN_BROADCAST_SIBLING", "0")' in text
