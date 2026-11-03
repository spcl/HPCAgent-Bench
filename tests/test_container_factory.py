# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the backend table: backend resolution, Harbor provider name, and the Apptainer installer."""

import os
import pathlib
import subprocess

import pydantic
import pytest

from hpcagent_bench import config, containers
from hpcagent_bench.config import ContainerRuntime as R


@pytest.fixture(autouse=True)
def clean_backend_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop every ambient container/runtime var so a developer's shell cannot skew the argv assertions."""
    for key in list(os.environ):
        if key.startswith("HPCAGENT_BENCH_") or key == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(key, raising=False)


def test_load_backends_lists_every_backend() -> None:
    """Two OCI implementations that consume the shipped image directly, the SIF conversion
    target, the Alps container engine, and the no-container path."""
    assert list(containers.load_backends()) == [R.DOCKER, R.PODMAN, R.APPTAINER, R.CE, R.NATIVE]


def test_oci_is_a_standard_not_a_program() -> None:
    """docker and podman are two IMPLEMENTATIONS of one standard, which is why `oci` is an alias over both rather than a
    runtime of its own. Nothing is ever launched as `oci`: selecting it always resolves to a program."""
    assert R.OCI not in containers.load_backends()  # a standard has no row: it is not a thing you exec
    assert containers.family_members(R.OCI) == (R.DOCKER, R.PODMAN)
    assert containers.family_members(R.SIF) == (R.APPTAINER,)
    assert containers.family_members(R.CE) == (R.CE,)
    assert containers.family_members(R.NATIVE) == (R.NATIVE,)


def test_a_family_name_resolves_to_whichever_flavour_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting `oci` says WHICH INTERFACE, not which program. It must ask the machine."""
    monkeypatch.setattr(containers.shutil, "which", lambda name: "/usr/bin/" + name if name == "docker" else None)
    assert containers.resolve_backend(R.OCI) == R.DOCKER
    monkeypatch.setattr(containers.shutil, "which", lambda name: "/usr/bin/" + name)
    assert containers.resolve_backend(R.OCI) == R.DOCKER  # both present -> the one most users have
    monkeypatch.setattr(containers.shutil, "which", lambda name: "/usr/bin/" + name if name == "podman" else None)
    assert containers.resolve_backend(R.OCI) == R.PODMAN  # docker absent -> the other implementation
    monkeypatch.setattr(containers.shutil, "which", lambda name: None)
    assert containers.resolve_backend(R.OCI) == R.DOCKER  # none present -> deterministic fallback
    assert containers.resolve_backend(R.SIF) == R.APPTAINER
    assert containers.resolve_backend(R.CE) == R.CE


def test_every_family_has_at_least_one_runtime() -> None:
    """A family nobody implements would resolve to nothing and fail far from its cause."""
    for family in containers.FAMILIES:
        assert containers.family_members(family), family


def test_detect_backend_probes_path_rather_than_assuming(monkeypatch: pytest.MonkeyPatch) -> None:
    """A login node has podman and no dockerd; a laptop often has the reverse. Detection asks
    the machine instead of trusting the constant."""
    monkeypatch.setattr(containers.shutil, "which", lambda name: "/usr/bin/" + name if name == "docker" else None)
    assert containers.detect_backend() == R.DOCKER
    monkeypatch.setattr(containers.shutil, "which", lambda name: None)
    assert containers.detect_backend() is None


def test_resolve_backend_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    # PATH is pinned so the family fallback is deterministic rather than a property of whichever
    # runtime this developer happens to have installed.
    monkeypatch.setattr(containers.shutil, "which", lambda name: "/usr/bin/" + name)
    assert containers.resolve_backend(R.PODMAN) == R.PODMAN  # explicit wins
    monkeypatch.setenv("HPCAGENT_BENCH_RUNTIME_BACKEND", "docker")
    assert containers.resolve_backend() == R.DOCKER  # canonical env next
    monkeypatch.delenv("HPCAGENT_BENCH_RUNTIME_BACKEND")
    # config ships the STANDARD `oci`, which resolves to docker when both are installed
    assert containers.resolve_backend() == R.DOCKER


def test_the_config_rejects_an_unknown_backend() -> None:
    # "singularity" is a Harbor PROVIDER name, not a backend this factory spells (apptainer is);
    # neither it nor an unsupported runtime may silently resolve to a neighbouring one.
    for dropped in ("singularity", "udocker", "shifter"):
        with pytest.raises(pydantic.ValidationError):
            config.set_override("runtime.backend", dropped)


def test_harbor_provider_names_docker_podman_and_singularity() -> None:
    """Harbor (>= 0.23) drives docker, podman and singularity. ce has no provider, so it must raise
    rather than emit one Harbor would reject."""
    assert containers.harbor_env_for(R.DOCKER) == "docker"
    assert containers.harbor_env_for(R.PODMAN) == "podman"
    assert containers.harbor_env_for(R.APPTAINER) == "singularity"
    with pytest.raises(ValueError, match="Harbor"):
        containers.harbor_env_for(R.CE)


# install_apptainer retry: both fetches are live-network; subprocess/sleep stubbed, stays pure-unit


def stub_installer(
    monkeypatch: pytest.MonkeyPatch, bash_returncodes: list[int], curl_error: str | None = None
) -> tuple[list[list[str]], list[float]]:
    """Stub curl+bash; only intercepts those two argv (patching subprocess.run is process-global)."""
    calls, sleeps, pending = [], [], list(bash_returncodes)
    real_run = subprocess.run

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if not argv or argv[0] not in ("curl", "bash"):
            return real_run(argv, **kwargs)  # not ours -- never consume a queued returncode
        calls.append(argv[0])
        if argv[0] == "curl":
            failed = curl_error is not None and calls.count("curl") == 1
            returncode = curl_error if failed else 0
            stdout = "" if failed else "#!/bin/bash\ntrue\n"
        else:
            returncode, stdout = pending.pop(0), ""
        # Honour subprocess.run's real contract: check=True turns a nonzero rc into CalledProcessError.
        if kwargs.get("check") and returncode != 0:
            raise subprocess.CalledProcessError(returncode, argv)
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout)

    monkeypatch.setattr(containers.subprocess, "run", fake_run)
    monkeypatch.setattr(containers.time, "sleep", sleeps.append)
    return calls, sleeps


def test_install_apptainer_retries_a_transient_mirror_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mirror blip is retried in a FRESH process; upstream's own loop cannot recover from this."""
    calls, sleeps = stub_installer(monkeypatch, bash_returncodes=[2, 2, 0])
    assert containers.install_apptainer("/tmp/apptainer-prefix", attempts=4) == 0
    assert calls.count("bash") == 3, "each attempt must re-run the installer in a fresh process"
    assert sleeps == [5, 10], "backoff must grow, and must NOT sleep after the attempt that succeeded"


def test_install_apptainer_gives_up_and_reports_the_installer_returncode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exhausting the attempts still surfaces the real failure -- never a false success."""
    calls, sleeps = stub_installer(monkeypatch, bash_returncodes=[2, 2, 2])
    assert containers.install_apptainer("/tmp/apptainer-prefix", attempts=3) == 2
    assert calls.count("bash") == 3
    assert sleeps == [5, 10], "no trailing sleep after the final attempt"


def test_install_apptainer_retries_a_failed_installer_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """The installer download is live-network too, so a curl failure retries rather than raising."""
    calls, unused = stub_installer(monkeypatch, bash_returncodes=[0], curl_error=6)
    assert containers.install_apptainer("/tmp/apptainer-prefix", attempts=3) == 0
    assert calls.count("curl") == 2, "the failed download must be re-fetched, not raised to the caller"


def test_install_apptainer_succeeds_first_try_without_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    """The happy path must not pay any backoff (guards against an off-by-one in the loop)."""
    calls, sleeps = stub_installer(monkeypatch, bash_returncodes=[0])
    assert containers.install_apptainer("/tmp/apptainer-prefix") == 0
    assert calls.count("bash") == 1
    assert sleeps == []


def test_install_apptainer_clears_a_partial_tree_between_attempts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A failed attempt's leftovers must be gone before the retry runs, or upstream hard-refuses on retry."""
    prefix = tmp_path / "apptainer"
    prefix.mkdir()
    seen_dirty = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if argv[0] == "curl":
            return subprocess.CompletedProcess(argv, 0, stdout="#!/bin/bash\ntrue\n")
        # Record whether upstream would refuse, then leave a partial tree as a dead mirror does.
        seen_dirty.append((prefix / "x86_64").exists())
        (prefix / "x86_64").mkdir(exist_ok=True)
        (prefix / "x86_64" / "partial.rpm").write_text("half-unpacked")
        return subprocess.CompletedProcess(argv, 0 if len(seen_dirty) == 3 else 2)

    monkeypatch.setattr(containers.subprocess, "run", fake_run)
    monkeypatch.setattr(containers.time, "sleep", lambda s: None)
    assert containers.install_apptainer(str(prefix), attempts=4) == 0
    assert seen_dirty == [False, False, False], (
        f"a retry started against a dirty prefix {seen_dirty} -- upstream would refuse it outright"
    )


def test_clean_partial_install_never_touches_a_preexisting_path(tmp_path: pathlib.Path) -> None:
    """Only paths the attempt created may be removed; `prefix` is caller-supplied (often ~/.local)."""
    prefix = tmp_path / "local"
    (prefix / "share").mkdir(parents=True)
    (prefix / "share" / "user_data.txt").write_text("do not delete me")
    preexisting = {entry.name for entry in prefix.iterdir()}
    (prefix / "x86_64").mkdir()
    (prefix / "bin").mkdir()

    containers.clean_partial_install(str(prefix), preexisting)

    assert sorted(entry.name for entry in prefix.iterdir()) == ["share"]
    assert (prefix / "share" / "user_data.txt").read_text() == "do not delete me"


def test_clean_partial_install_tolerates_a_missing_prefix(tmp_path: pathlib.Path) -> None:
    """The very first attempt can fail before the prefix exists at all."""
    containers.clean_partial_install(str(tmp_path / "never-created"), set())


def test_ce_stays_its_own_family_even_though_it_is_podman_underneath() -> None:
    """CSCS Alps' container engine is podman with SquashFS layers and Cray-tuned OCI hooks, so on
    runtime alone it belongs in the ``oci`` family. It is kept separate on purpose: ``oci`` is
    what a user selects to mean "whatever this machine has", and it must never resolve to the one
    backend with no local launch form, which fails without an EDF and only inside an allocation.
    The runtime is shared; the launch contract is not."""
    spellings = containers.load_backends()
    assert containers.family_members(R.OCI) == (R.DOCKER, R.PODMAN)
    assert containers.family_members(R.CE) == (R.CE,)
    assert spellings[R.CE].kind == "srun_env"
    assert all(spellings[name].kind == "exec" for name in containers.family_members(R.OCI))
    # The safety property itself: asking for the OCI family never lands on the Slurm-only one.
    assert containers.resolve_backend(R.OCI) in containers.family_members(R.OCI)
