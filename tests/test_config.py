# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The configuration model (hpcagent_bench/config.py): env parsing, precedence, validation, and the readers
that take their defaults from it."""

import contextlib
import pathlib
import re
from collections.abc import Callable
from typing import get_args

import pydantic
import pytest
from hpcagent_agent.coded import Coded
from hpcagent_agent.network_policy import AgentNetwork, Unsealed

from hpcagent_bench import config, containers, flags, languages, paths, seal, spec
from hpcagent_bench.api import InputMode, Oracle
from hpcagent_bench.categories import Baseline, Denominator, RecordDevice, Residency
from hpcagent_bench.config import AgentRootfs, BaselineRace, ContainerRuntime, Execution, Hardware, ScalingLaw
from hpcagent_bench.harness import (
    agent,
    grading,
    judge_scheduler,
    mpi_gang,
    native_call,
    pipeline,
    recording,
    scoring,
    task,
    timing,
)
from hpcagent_bench.harness.requests_api import Location
from tests.tracked_files import tracked_files

REPO = paths.ROOT


def field_values(dotted: str) -> set[object]:
    """The values the Literal of leaf ``dotted`` (or of its sequence items) allows; an enum's members."""
    annotation, _ = config.strip_optional(config.FIELDS[dotted].annotation)
    args = get_args(annotation)
    if args and args[-1] is Ellipsis:
        annotation = args[0]
        args = get_args(annotation)
    if isinstance(annotation, type) and issubclass(annotation, Coded):
        return {member for member in annotation if accepts(dotted, member.text)}
    return set(args)


def accepts(dotted: str, text: str) -> bool:
    """Whether leaf ``dotted`` validates the spelling ``text``."""
    try:
        config.parse_env(dotted, text)
    except ValueError:
        return False
    return True


# env text -> value


@pytest.mark.parametrize("sha", ["0123456", "4468e35", "1e5", "nan"])
def test_a_text_key_gets_the_env_text_unchanged(monkeypatch: pytest.MonkeyPatch, sha: str) -> None:
    """A short git sha that looks like a number must not lose its leading zero or become a float."""
    monkeypatch.setenv("HPCAGENT_BENCH_RECORD_COMMIT", sha)
    assert config.current().record.commit == sha


@pytest.mark.parametrize(
    ("text", "want"), [("off", False), ("no", False), ("False ", False), ("on", True), ("1", True)]
)
def test_a_flag_reads_every_boolean_spelling(monkeypatch: pytest.MonkeyPatch, text: str, want: bool) -> None:
    """``off`` and ``no`` are false; read by truthiness they were true."""
    monkeypatch.setenv("HPCAGENT_BENCH_RECORD_ENABLED", text)
    assert config.current().record.enabled is want


@pytest.mark.parametrize(
    ("name", "text"),
    [
        pytest.param("HPCAGENT_BENCH_RECORD_LOG_CALLS", "maybe", id="no-boolean-spelling"),
        pytest.param("HPCAGENT_BENCH_MEASUREMENT_REPEAT", "0", id="out-of-range"),
    ],
)
def test_an_invalid_env_value_raises_naming_its_variable(monkeypatch: pytest.MonkeyPatch, name: str, text: str) -> None:
    monkeypatch.setenv(name, text)
    with pytest.raises(ValueError, match=name):
        config.current()


@pytest.mark.parametrize("text", ["", "null", "None", " NULL "])
def test_a_nullable_key_reads_every_null_spelling(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    """A later env layer can restore a nullable key to null; ``''`` and ``null`` used to crash int()."""
    monkeypatch.setenv("HPCAGENT_BENCH_ATTEMPTS_TOKEN_BUDGET", text)
    assert config.current().attempts.token_budget is None


@pytest.mark.parametrize(
    ("name", "text", "read", "want"),
    [
        # ``/a,/b`` is two paths to hide, never the one path ``/a,/b``.
        pytest.param(
            "HPCAGENT_BENCH_GRADING_SEAL_HIDE",
            "/a, /b",
            lambda: config.current().grading.seal_hide,
            ("/a", "/b"),
            id="comma",
        ),
        pytest.param(
            "HPCAGENT_BENCH_GRADING_SEAL_HIDE",
            '["/a,b"]',
            lambda: config.current().grading.seal_hide,
            ("/a,b",),
            id="json",
        ),
        pytest.param(
            "HPCAGENT_BENCH_GRADING_SEAL_HIDE", "", lambda: config.current().grading.seal_hide, (), id="empty"
        ),
        pytest.param(
            "HPCAGENT_BENCH_TIMEOUTS_KERNEL_S_BY_LEVEL",
            '{"1": 5}',
            lambda: config.current().timeouts.kernel_s_by_level,
            {1: 5.0},
            id="map-json-object",
        ),
    ],
)
def test_a_list_key_reads_a_comma_list_or_a_json_array(
    monkeypatch: pytest.MonkeyPatch, name: str, text: str, read: Callable[[], object], want: object
) -> None:
    """A list key reads a comma list or a JSON array; a map key reads a JSON object."""
    monkeypatch.setenv(name, text)
    assert read() == want


# unknown keys


@pytest.mark.parametrize("name", ["HPCAGENT_BENCH_MEASUREMENT_REPEATS", "HPCAGENT_BENCH_TIMEOUTS_KERNEL_S_BY_LEVEL_1"])
def test_an_env_name_under_a_section_that_is_no_key_is_an_error(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """A typo, or a per-entry name of a map read whole, used to have no effect and no error."""
    monkeypatch.setenv(name, "5")
    with pytest.raises(config.UnknownConfigEnvError, match=name):
        config.current()


@pytest.mark.parametrize(
    ("name", "meant"),
    [
        ("HPCAGENT_BENCH_GRADNG_SEAL", "HPCAGENT_BENCH_GRADING_SEAL"),
        ("HPCAGENT_BENCH_RECRD_MODEL", "HPCAGENT_BENCH_RECORD_MODEL"),
        ("HPCAGENT_BENCH_MEASUREMENT_REPEATS", "HPCAGENT_BENCH_MEASUREMENT_REPEAT"),
    ],
)
def test_a_misspelled_name_is_an_error_naming_the_key_it_means(
    monkeypatch: pytest.MonkeyPatch, name: str, meant: str
) -> None:
    """A misspelled section (GRADNG for GRADING) sat outside every section prefix and was ignored: the seal stayed on
    while the operator believed it off."""
    monkeypatch.setenv(name, "false")
    with pytest.raises(config.UnknownConfigEnvError, match=rf"{name} \(did you mean {meant}\?\)"):
        config.current()


@pytest.mark.parametrize("name", sorted(config.RENAMED_ENV))
def test_a_renamed_name_is_an_error_naming_its_new_name(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """A renamed seconds knob still set under its old name would be ignored and its default used in silence."""
    monkeypatch.setenv(name, "60")
    with pytest.raises(config.UnknownConfigEnvError, match=rf"{name} \(renamed {config.RENAMED_ENV[name]}\)"):
        config.current()


@pytest.mark.parametrize(
    ("call", "exception", "match"),
    [
        pytest.param(
            lambda: config.set_override("measurement.repeats", 5),
            KeyError,
            r"measurement\.repeats",
            id="override-unknown",
        ),
        pytest.param(
            lambda: config.set_override("measurement.repeat", 0), pydantic.ValidationError, "", id="out-of-range"
        ),
        pytest.param(lambda: config.get("measurement.repeats"), KeyError, "", id="get-unknown"),
        pytest.param(
            lambda: config.set_override("measurement.timing_backend", "mannwhitney"),
            pydantic.ValidationError,
            "mannwhitney",
            id="unknown-timing-backend-not-reduced-with-min-of-k",
        ),
    ],
)
def test_overriding_an_unknown_key_raises(call: Callable[[], object], exception: type[Exception], match: str) -> None:
    with pytest.raises(exception, match=match):
        call()


#: The files that may name a renamed env name: its refusal table and the changelog.
RENAMED_HOMES = frozenset({"hpcagent_bench/config.py", "CHANGELOG.md"})


def test_every_env_name_in_the_repository_is_a_key_or_listed_as_non_config() -> None:
    """A launcher or layer that exports a name the model does not know would stop every config read."""
    tracked = tracked_files(REPO)
    names: set[str] = set()
    for relative in tracked:
        path = REPO / relative
        if relative.startswith("third_party/") or path == pathlib.Path(__file__).resolve() or not path.is_file():
            continue
        try:
            found = set(re.findall(r"HPCAGENT_BENCH_[A-Z0-9_]*[A-Z0-9]\b", path.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            continue
        # The table of renamed names and the changelog name an old name only to refuse or record it.
        names |= found - set(config.RENAMED_ENV) if relative in RENAMED_HOMES else found
    config.check_env(dict.fromkeys(names, ""))


# precedence


def test_an_override_wins_over_the_scoped_environment_which_wins_over_os_environ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HPCAGENT_BENCH_RECORD_STUDY", "process")
    with config.scoped_environment({"HPCAGENT_BENCH_RECORD_STUDY": "scoped"}):
        assert config.current().record.study == "scoped"
        with config.overridden("record.study", "override"):
            assert config.current().record.study == "override"
    assert config.current().record.study == "process"
    # The raw env read before the config made the env win over set_override and the scoped env.
    monkeypatch.setenv("HPCAGENT_BENCH_RUNTIME_BACKEND", "apptainer")
    with config.overridden("runtime.backend", ContainerRuntime.PODMAN):
        assert containers.resolve_backend() is ContainerRuntime.PODMAN


def test_the_agent_rootfs_reads_the_config_env_and_defaults_to_writable(monkeypatch: pytest.MonkeyPatch) -> None:
    """``writable`` is the default (today's launch behaviour); ``read-only`` is the one opt-in spelling, and the
    cluster seam (container_runtime.sh) validates the same variable's same spellings."""
    monkeypatch.delenv("HPCAGENT_BENCH_ISOLATION_AGENT_ROOTFS", raising=False)
    assert config.current().isolation.agent_rootfs is AgentRootfs.WRITABLE
    assert AgentRootfs.READ_ONLY.text == "read-only"
    monkeypatch.setenv("HPCAGENT_BENCH_ISOLATION_AGENT_ROOTFS", "read-only")
    assert config.current().isolation.agent_rootfs is AgentRootfs.READ_ONLY


def test_the_agent_worker_count_honours_an_override_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HPCAGENT_BENCH_AGENT_WORKERS", "7")
    with config.overridden("agent.workers", 3):
        assert pipeline.agent_workers([None], ["http://judge"]) == 3
    # A null count is one worker per endpoint.
    monkeypatch.delenv("HPCAGENT_BENCH_AGENT_WORKERS")
    assert pipeline.agent_workers([None, None], ["a", "b", "c"]) == 3


def test_harbor_images_follow_an_override_made_after_import() -> None:
    """The images were bound into module constants at import, so an override never reached them."""
    from hpcagent_bench.export import harbor

    with config.overridden("images.cpu.agent", "registry.example/agent:test"):
        assert harbor.images_for(Hardware.CPU)[0] == "registry.example/agent:test"


# readers take the model defaults


@pytest.mark.parametrize(
    ("read", "want"),
    [
        (timing.measurement_repeat, 20),
        (lambda: timing.quiescence_residual_limit(0.0), 11340.0),
        (lambda: timing.clocks_agree(1000.0, 3470.0 + 77164.0), True),
        (lambda: timing.clocks_agree(1000.0, 3470.0 + 77165.0), False),
        (lambda: scoring.guillotine_seconds(10**9, 100.0), 2.0),
        (lambda: mpi_gang.launch_timeout({}), 120.0),
        (spec.bsr_block_sizes, (2, 4, 8)),
        (lambda: config.default_value("record.db_path"), "results/hpcagent_bench.db"),
        (lambda: config.current().limits.kernel_memory_gb, 20.0),
        (lambda: config.current().limits.oracle_cache_gb, 12.0),
        (lambda: config.current().ml.rank_counts, (1, 2, 4)),
        (lambda: config.current().ml.grade_rank_counts, (1, 2, 4, 8, 16)),
        (lambda: config.current().runtime.backend, ContainerRuntime.OCI),
    ],
)
def test_a_reader_without_an_override_returns_the_model_default(read: Callable[[], object], want: object) -> None:
    """Every reader used to carry its own fallback; several disagreed with config.yaml (repeat 50, guillotine
    off, quiescence gates off, launch timeout 1800 s)."""
    assert read() == want


def test_a_level_free_budget_still_honours_the_overrides() -> None:
    """A kernel whose spec does not load fell back to kernel_s / token_budget, past both overrides."""
    with config.overridden("timeouts.kernel_s_override", 60.0), config.overridden("attempts.token_budget_override", 9):
        assert (scoring.resolve_kernel_timeout(None), scoring.resolve_token_budget(None)) == (60.0, 9)
    assert (scoring.resolve_kernel_timeout(None), scoring.resolve_token_budget(None)) == (
        config.current().timeouts.kernel_s,
        config.current().attempts.token_budget,
    )


def test_a_null_gpu_count_splits_the_node_by_the_detected_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scheduler ran one slot per detected GPU while the cpuset split read null as 0 slots, so every
    slot was handed the whole node."""
    detected = 4
    monkeypatch.setattr(judge_scheduler, "local_gpu_count", lambda: detected)
    assert config.current().judge.gpus_per_node is None
    assert judge_scheduler.judge_gpu_slots() == detected
    assert languages.grading_ncores() == max(1, flags.ncores() // detected)
    whole = native_call.grading_cpus(None)
    if len(whole) >= detected:
        assert len(native_call.grading_cpus(0)) == len(whole) // detected


@pytest.mark.parametrize(
    ("device", "gate"), [pytest.param(None, None, id="undeclared"), pytest.param("cpu", True, id="cpu")]
)
def test_an_undeclared_device_does_not_gate_and_is_recorded_as_cpu(device: str | None, gate: bool | None) -> None:
    """``record.device`` null: no host-only gate (the yaml default ``cpu`` gated every undeclared setup when
    read through the config), and the row is still stamped cpu; a declared ``cpu`` gates."""
    with config.overridden("record.device", device) if device else contextlib.nullcontext():
        if device is None:
            assert config.current().record.device is None
        assert task.setup_declared_host_only() is gate
        assert recording.device_tag() == recording.RecordDevice.CPU


def test_the_seal_hides_the_configured_generated_reference_cache(tmp_path: pathlib.Path) -> None:
    """The seal hid only ``$HPCAGENT_BENCH_GENERATED_CACHE``, so a cache named through the config key stayed visible."""
    cache = tmp_path / "generated"
    cache.mkdir()
    with config.overridden("references.generated_cache_dir", str(cache)), config.overridden("grading.seal", True):
        plan = seal.grading_plan([])
        if plan is None:
            pytest.skip("not Linux: no seal")
        assert str(cache) in plan.hide
        assert agent.generated_cache_root() == cache


# the model and the registries it names


@pytest.mark.parametrize(
    ("dotted", "values"),
    [
        ("service.oracle", set(Oracle)),
        ("service.input_mode", set(InputMode)),
        ("measurement.baseline", {*grading.BASELINE_CHOICES, Baseline.TORCH_AUTOTUNE}),
        ("measurement.denominator.scientific_computing", set(Denominator)),
        ("measurement.denominator.machine_learning", {Denominator.TORCH_AUTOTUNE}),
        ("measurement.timing_backend", set(timing.REDUCTIONS)),
        ("measurement.baseline_race", {BaselineRace.LEADER_FIRST, BaselineRace.COMPLETE}),
        ("record.device", set(RecordDevice)),
        ("agent.execution", set(Execution)),
        ("isolation.agent_rootfs", set(AgentRootfs)),
        ("agent.network", set(AgentNetwork)),
        ("agent.network_unsealed", set(Unsealed)),
        ("mpi.mode", set(ScalingLaw)),
        ("mpi.residency", {Residency.parse(member.value) for member in Location}),
        ("runtime.backend", set(ContainerRuntime)),
    ],
)
def test_a_choice_key_offers_exactly_what_its_reader_accepts(dotted: str, values: set[object]) -> None:
    assert field_values(dotted) == values


def test_the_agent_driver_defaults_match_the_model() -> None:
    """The agent driver imports the standard library only, so it restates two defaults."""
    from hpcagent_agent.driver import agent_driver

    assert config.default_value("grading.allow_agent_build_tokens") is agent_driver.BUILD_TOKENS_DEFAULT
    assert RecordDevice.CPU.text == agent_driver.RECORD_DEVICE_DEFAULT


def test_config_yaml_names_exactly_the_model_keys() -> None:
    """``config.yaml`` alone validates: every key the models declare is present and no other key is."""
    config.Settings.model_validate(config.yaml_defaults())


@pytest.mark.smoke
def test_an_env_overlay_reaches_readers(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end: env layer -> validated model -> readers."""
    monkeypatch.setenv("HPCAGENT_BENCH_MEASUREMENT_REPEAT", "7")
    monkeypatch.setenv("HPCAGENT_BENCH_RECORD_LOG_ATTEMPTS", "off")
    monkeypatch.setenv("HPCAGENT_BENCH_MPI_LAUNCHER", "srun,--mpi=pmi2,-n")
    assert timing.measurement_repeat() == 7
    assert config.current().record.log_attempts is False
    assert scoring.mpi_launch_cfg().launcher == ["srun", "--mpi=pmi2", "-n"]
    assert config.get("measurement.repeat") == 7
    assert config.env_name("measurement.quiescence.residual_ns") == "HPCAGENT_BENCH_MEASUREMENT_QUIESCENCE_RESIDUAL_NS"
