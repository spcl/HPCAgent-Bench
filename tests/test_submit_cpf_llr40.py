# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""experiments/submit-cpf-llr40.sh builds each CPF ablation arm with exactly its own treatment.

A control arm that carries a view key is not a control, a cpf arm that carries the drop-in key measures
two treatments, and a device arm that reads the CPU prompt is told the wrong build contract. None of
these fails at launch; each shows up as a skewed number weeks later.

Runs a temp copy of the launcher's inputs against real (fake-content) CPF cache views built with
hpcagent_bench.cpf_cache, SUBMIT=0: nothing reaches sbatch. One invocation per target, since
CPF_FORMS_DIR and CPF_DROPIN_DIR name one view for every arm of an invocation.
"""

import dataclasses
import os
import pathlib
import shutil
import subprocess
import sys
from collections.abc import Callable

import pytest

from hpcagent_bench import cpf_cache
from tests.test_submit_scicomp_dc_cpfsrc import env_dict, stub

REPO = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"

SUBMIT_INPUTS = (
    "submit-cpf-llr40.sh",
    "arm_nodes.sh",
    "roster.sh",
    "record_identity.sh",
    "submit_common.sh",
    "make_problems.py",
    "packet_env.py",
    ".env.base-qwen38",
)

#: Two llr-focus40 kernels, so the roster and coverage gates resolve them without a fabricated manifest.
ROSTER_KERNELS = ("fuse_diamond", "tsvc_2_s115")

FORM_KEY = "HPCAGENT_BENCH_SERVICE_CANONICAL_PARALLEL_FORM_DIR"
DROPIN_KEY = "CPF_DROPIN_DIR"

#: What the launcher reads from the calling shell; dropped so the host cannot steer a case.
KNOBS = frozenset(
    {
        "SUBMIT",
        "ARMS",
        "MODELS",
        "KERNELS",
        "KERNELS_FILE",
        "TAG",
        "EXPERIMENT",
        "RECORD_EXPERIMENT",
        "STAMP",
        "BEGIN",
        "CPF_CE_ENV",
        "DEVICE_LANGS",
        "EXTRA_ENV_KV",
        "DEPEND_ON",
        "CPF_FORMS_DIR",
        "CPF_VIEW",
        DROPIN_KEY,
        FORM_KEY,
        "OPT",
        "PY",
        "SCRATCH",
        "PYTHONPATH",
    }
)

#: The dialects a prerender of each target writes into its view.
TARGET_DIALECTS = {"cpu": ("c", "c++"), "gpu": ("hip",)}

#: Every arm of the wave, split by the target whose view it is gated against.
TARGET_ARMS = {"cpu": "c:plain c:cpf c:cpfsrc", "gpu": "hip:plain hip:cpf"}


@dataclasses.dataclass(slots=True, frozen=True)
class Launch:
    experiments: pathlib.Path
    view: pathlib.Path
    result: subprocess.CompletedProcess[str]


def build_view(root: pathlib.Path, kernels: tuple[str, ...], target: str) -> pathlib.Path:
    """A view pinned to ``target`` serving a form and a drop-in for ``kernels`` in its target's dialects."""
    cache, view = root / "cache", root / "view"
    cpf_cache.open_view(view, cache, target, "dace")
    for kernel in kernels:
        stem = f"{kernel}_fp64_cpf"
        for dialect in TARGET_DIALECTS[target]:
            modes: dict[str, dict[str, object]] = {}
            for mode in cpf_cache.MODES:
                options = {"kernel": kernel, "language": dialect, "target": target, "mode": mode}
                key = cpf_cache.cache_key("sdfg", "dace", options)
                source = (f"{stem}.{cpf_cache.LANGUAGE_EXT[dialect]}", f"// {kernel} {mode}\n")
                cpf_cache.publish(cache, key, {"kernel": kernel}, source, (f"{stem}_binding.json", "{}\n"))
                modes[mode] = {"key": key, "verdict": "ok", "cached": False}
            cpf_cache.record(view, kernel, dialect, "fp64", modes)
    return view


def launch(root: pathlib.Path, arms: str, kernels: tuple[str, ...], target: str) -> Launch:
    """Build ``arms`` against a ``target`` view serving ``kernels``, with SUBMIT=0."""
    experiments = root / "experiments"
    experiments.mkdir(parents=True)
    for name in SUBMIT_INPUTS:
        shutil.copy2(EXPERIMENTS / name, experiments / name)
    (experiments / "kernels.txt").write_text("\n".join(ROSTER_KERNELS) + "\n")
    stub(root / "bin", "sbatch", 'touch "${STUB_MARKERS}/sbatch-called"; exit 1')
    # the launcher runs ${SCRATCH}/venv-optarena-314/bin/python, so that path must be this interpreter
    stub(root / "scratch" / "venv-optarena-314" / "bin", "python", f'exec "{sys.executable}" "$@"')
    view = build_view(root, kernels, target)
    env = {key: value for key, value in os.environ.items() if key not in KNOBS and not key.startswith("SLURM_")}
    env.update(
        PATH=f"{root / 'bin'}:{env['PATH']}",
        SCRATCH=str(root / "scratch"),
        OPT=str(REPO),
        SUBMIT="0",
        MODELS="qwen38",
        ARMS=arms,
        KERNELS_FILE="kernels.txt",
        STAMP="20260914",
        STUB_MARKERS=str(root),
        CPF_FORMS_DIR=str(view),
        CPF_DROPIN_DIR=str(view),
    )
    result = subprocess.run(
        ["bash", str(experiments / "submit-cpf-llr40.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    return Launch(experiments, view, result)


@pytest.fixture(name="wave", scope="module")
def wave_fixture(tmp_path_factory: pytest.TempPathFactory) -> Callable[[str], Launch]:
    """Each target's full wave, built once per module and only when a test asks for that target."""
    built: dict[str, Launch] = {}

    def for_target(target: str) -> Launch:
        if target not in built:
            built[target] = launch(
                tmp_path_factory.mktemp(f"wave-{target}"), TARGET_ARMS[target], ROSTER_KERNELS, target
            )
        return built[target]

    return for_target


def arm_env(experiments: pathlib.Path, arm: str) -> pathlib.Path:
    return experiments / f".env.cpf-llr-focus40-qwen38-{arm}"


@pytest.mark.parametrize(("target", "arms"), [("cpu", ("c", "c-cpf", "c-cpfsrc")), ("gpu", ("hip", "hip-cpf"))])
def test_every_arm_of_a_target_builds_without_reaching_the_queue(
    wave: Callable[[str], Launch], target: str, arms: tuple[str, ...]
) -> None:
    built = wave(target)
    assert built.result.returncode == 0, built.result.stderr
    assert [arm for arm in arms if not arm_env(built.experiments, arm).is_file()] == []
    assert list(built.experiments.glob(".env.*.staging")) == []
    assert not (built.experiments.parent / "sbatch-called").exists()


@pytest.mark.parametrize(
    ("target", "arm", "keys"),
    [
        ("cpu", "c", ()),
        ("cpu", "c-cpf", (FORM_KEY,)),
        ("cpu", "c-cpfsrc", (DROPIN_KEY,)),
        ("gpu", "hip", ()),
        ("gpu", "hip-cpf", (FORM_KEY,)),
    ],
)
def test_an_arm_carries_only_the_cpf_key_of_its_kind(
    wave: Callable[[str], Launch], target: str, arm: str, keys: tuple[str, ...]
) -> None:
    """The caller exports both views for every arm; only the treated arm may pin one, and only its own."""
    built = wave(target)
    assert built.result.returncode == 0, built.result.stderr
    env = env_dict(arm_env(built.experiments, arm))
    carried = {key: env[key] for key in (FORM_KEY, DROPIN_KEY) if key in env}
    assert carried == dict.fromkeys(keys, str(built.view))


@pytest.mark.parametrize(
    ("target", "arm", "prompt"),
    [
        ("cpu", "c", "prompt.md"),
        ("cpu", "c-cpf", "prompt.md"),
        ("cpu", "c-cpfsrc", "prompt.md"),
        ("gpu", "hip", "prompt-gpu.md"),
        ("gpu", "hip-cpf", "prompt-gpu.md"),
    ],
)
def test_an_arm_reads_the_prompt_of_its_target(
    wave: Callable[[str], Launch], target: str, arm: str, prompt: str
) -> None:
    """The base env is a CPU arm's, and its prompt states the CPU build contract as fact."""
    built = wave(target)
    assert built.result.returncode == 0, built.result.stderr
    assert env_dict(arm_env(built.experiments, arm))["AGENT_PROMPT_FILE"] == prompt


@pytest.mark.parametrize(
    ("target", "spec", "arm"),
    [("cpu", "c:cpf", "c-cpf"), ("cpu", "c:cpfsrc", "c-cpfsrc"), ("gpu", "hip:cpf", "hip-cpf")],
)
def test_a_view_missing_a_kernel_refuses_the_arm_by_name_and_leaves_no_env(
    tmp_path: pathlib.Path, target: str, spec: str, arm: str
) -> None:
    """A missing form reads as a 200 unavailable at run time, so the arm would silently run untreated
    for that kernel; a half-built env left behind would look complete to the next launch."""
    served, absent = ROSTER_KERNELS
    built = launch(tmp_path, spec, (served,), target)
    assert built.result.returncode == 2, built.result.stderr
    assert f"  {absent}:" in built.result.stderr
    assert f"  {served}:" not in built.result.stderr
    assert not arm_env(built.experiments, arm).exists()
    assert not arm_env(built.experiments, f"{arm}.staging").exists()
    assert not (tmp_path / "sbatch-called").exists()
