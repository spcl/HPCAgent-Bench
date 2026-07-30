# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the container-launch factory: argv assembly, backend resolution, Harbor provider name."""
import os
import pathlib
import subprocess

import pytest

from hpcagent_bench import containers


@pytest.fixture(autouse=True)
def clean_backend_env(monkeypatch):
    """Drop every ambient container/runtime var so a developer's shell cannot skew the argv assertions."""
    for key in list(os.environ):
        if key.startswith("HPCAGENT_BENCH_") or key in ("OLLAMA_HOST", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(key, raising=False)
    yield


def test_load_backends_lists_every_backend():
    """Two OCI implementations that consume the shipped image directly, the SIF conversion
    target, the Alps container engine, and the no-container path."""
    spellings, passthrough = containers.load_backends()
    assert set(spellings) == {"docker", "podman", "apptainer", "ce", "native"}
    assert "ANTHROPIC_API_KEY" in passthrough
    assert spellings["apptainer"].verb == ("exec", )
    assert spellings["podman"].verb == ("run", "--rm", "--network", "host")
    assert spellings["docker"].verb == ("run", "--rm", "--network", "host")


def test_oci_is_a_standard_not_a_program():
    """docker and podman are two IMPLEMENTATIONS of one standard: same verb, same
    bind/workdir/env flags, same image form. Only the NVIDIA flag and the rootless property
    differ, which is why `oci` is an alias over both rather than a runtime of its own. Nothing is
    ever launched as `oci` -- selecting it always resolves to a program."""
    spellings, _ = containers.load_backends()
    assert "oci" not in spellings  # a standard has no row: it is not a thing you exec
    assert containers.family_members("oci") == ("docker", "podman")
    assert containers.family_members("sif") == ("apptainer", )
    assert containers.family_members("ce") == ("ce", )
    assert containers.family_members("native") == ("native", )
    podman, docker = spellings["podman"], spellings["docker"]
    assert podman.verb == docker.verb
    assert (podman.bind_flag, podman.workdir_flag, podman.env_flag) == (docker.bind_flag, docker.workdir_flag,
                                                                        docker.env_flag)
    assert podman.image_form == docker.image_form == "tag"
    assert podman.gpu["nvidia"] != docker.gpu["nvidia"]  # the one flag spelling that differs
    assert podman.rootless and not docker.rootless  # and the one property that decides defaults


def test_a_family_name_resolves_to_whichever_flavour_is_installed(monkeypatch):
    """Selecting `oci` says WHICH INTERFACE, not which program. It must ask the machine."""
    monkeypatch.setattr(containers.shutil, "which", lambda name: "/usr/bin/" + name if name == "docker" else None)
    assert containers.resolve_backend("oci") == "docker"
    monkeypatch.setattr(containers.shutil, "which", lambda name: "/usr/bin/" + name)
    assert containers.resolve_backend("oci") == "docker"  # both present -> the one most users have
    monkeypatch.setattr(containers.shutil, "which", lambda name: "/usr/bin/" + name if name == "podman" else None)
    assert containers.resolve_backend("oci") == "podman"  # docker absent -> the other implementation
    monkeypatch.setattr(containers.shutil, "which", lambda name: None)
    assert containers.resolve_backend("oci") == "docker"  # none present -> deterministic fallback
    assert containers.resolve_backend("sif") == "apptainer"
    assert containers.resolve_backend("ce") == "ce"


def test_every_family_has_at_least_one_runtime():
    """A family nobody implements would resolve to nothing and fail far from its cause."""
    for family in containers.FAMILIES:
        assert containers.family_members(family), family


def test_the_default_backend_is_rootless_and_daemonless():
    """The fallback has to be something an unprivileged user can actually invoke. docker needs a
    running daemon and a root-equivalent group, which no HPC login node grants; apptainer and ce
    need the OCI image converted first. Only podman is both OCI-native and rootless."""
    spellings, _ = containers.load_backends()
    assert containers.DEFAULT_BACKEND == "podman"
    assert spellings[containers.DEFAULT_BACKEND].image_form == "tag"  # consumes OCI unconverted
    assert spellings["apptainer"].image_form == "sif"  # a conversion, so never the default
    assert spellings["ce"].image_form == "edf"


def test_ce_is_a_different_shape_of_backend_not_just_different_flags():
    """Alps' container engine has NO wrapper argv: the image comes from an EDF on the srun line
    and the command runs unwrapped. Synthesising a wrapper it does not have would emit an argv
    that cannot run, so local_run_command must hand the command back untouched."""
    spellings, _ = containers.load_backends()
    assert spellings["ce"].kind == "srun_env"
    assert spellings["ce"].verb == () and spellings["ce"].bind_flag == ""
    assert containers.local_run_command(["hpcagent-bench", "run"], backend="ce") == ["hpcagent-bench", "run"]
    assert all(spellings[name].kind == "exec" for name in containers.EXEC_BACKENDS)


def test_native_is_a_supported_backend_not_a_missing_one():
    """A site with no container runtime still has to run. Saying so as a backend keeps that path
    on the same seam as the others; it consumes no image, so asking it for one is an error."""
    spellings, _ = containers.load_backends()
    assert spellings["native"].kind == "none"
    assert spellings["native"].image_form == ""
    assert containers.local_run_command(["hpcagent-bench", "run"], backend="native") == ["hpcagent-bench", "run"]
    assert containers.srun_container_flags("native") == []
    with pytest.raises(ValueError, match="consumes no image"):
        containers.default_image("native")


def test_a_sif_is_never_the_distributed_artifact():
    """An OCI image converts INTO a SIF or a SquashFS and neither converts back, so shipping a
    converted form would strand every user of the other runtimes. Exactly one backend family
    consumes the shipped image unconverted, and the shipped image is OCI."""
    spellings, _ = containers.load_backends()
    unconverted = {name for name, s in spellings.items() if s.image_form == "tag"}
    assert unconverted == {"docker", "podman"}
    assert spellings["apptainer"].image_form == "sif"
    assert spellings["ce"].image_form == "edf"


def test_ce_contributes_an_srun_flag_and_refuses_to_be_silent_without_one():
    """On Alps a step without --environment runs OUTSIDE the image, on the bare node, which
    looks like a broken environment rather than a missing flag. So a missing EDF raises."""
    assert containers.srun_container_flags(
        "ce", edf="/scratch/foundation.toml") == ["--environment=/scratch/foundation.toml"]
    assert containers.srun_container_flags("podman") == []  # an exec wrapper needs no srun flag
    with pytest.raises(ValueError, match="HPCAGENT_BENCH_EDF"):
        containers.srun_container_flags("ce")


def test_ce_has_no_image_reference_of_its_own():
    """Its EDF names the image, so asking this factory for one is a category error, not a
    default to invent."""
    with pytest.raises(ValueError, match="no image reference"):
        containers.default_image("ce", "cpu")


def test_detect_backend_probes_path_rather_than_assuming(monkeypatch):
    """A login node has podman and no dockerd; a laptop often has the reverse. Detection asks
    the machine instead of trusting the constant."""
    monkeypatch.setattr(containers.shutil, "which", lambda name: "/usr/bin/" + name if name == "docker" else None)
    assert containers.detect_backend() == "docker"
    monkeypatch.setattr(containers.shutil, "which", lambda name: None)
    assert containers.detect_backend() is None


def test_resolve_backend_precedence(monkeypatch):
    # PATH is pinned so the family fallback is deterministic rather than a property of whichever
    # runtime this developer happens to have installed.
    monkeypatch.setattr(containers.shutil, "which", lambda name: "/usr/bin/" + name)
    assert containers.resolve_backend("podman") == "podman"  # explicit wins
    monkeypatch.setenv("HPCAGENT_BENCH_RUNTIME_BACKEND", "docker")
    assert containers.resolve_backend() == "docker"  # canonical env next
    monkeypatch.delenv("HPCAGENT_BENCH_RUNTIME_BACKEND")
    # config ships the STANDARD `oci`, which resolves to docker when both are installed
    assert containers.resolve_backend() == "docker"


def test_resolve_backend_ignores_the_legacy_bash_var(monkeypatch):
    # $HPCAGENT_BENCH_CONTAINER_RUNTIME is the shell launcher's own knob; only $HPCAGENT_BENCH_RUNTIME_BACKEND is shared.
    monkeypatch.setattr(containers.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setenv("HPCAGENT_BENCH_CONTAINER_RUNTIME", "apptainer")
    assert containers.resolve_backend() == "docker"  # config's `oci`, not the bash-only var


def test_resolve_backend_rejects_unknown():
    # "singularity" is a Harbor PROVIDER name, not a backend this factory spells (apptainer is);
    # neither it nor an unsupported runtime may silently resolve to a neighbouring one.
    for dropped in ("singularity", "udocker", "enroot", "shifter"):
        with pytest.raises(ValueError):
            containers.resolve_backend(dropped)


def test_local_run_command_apptainer_cpu():
    argv = containers.local_run_command(["python", "-m", "hpcagent_bench.cli", "agent"],
                                        backend="apptainer",
                                        hardware="cpu",
                                        repo_root="/repo")
    assert argv == [
        "apptainer", "exec", "--env", "HPCAGENT_BENCH_IMAGE=cpu", "--bind", "/repo:/repo", "--pwd", "/repo",
        "/repo/hpcagent_bench-cpu.sif", "python", "-m", "hpcagent_bench.cli", "agent"
    ]


def test_local_run_command_podman_nvidia_gpu_tokens():
    argv = containers.local_run_command(["run"], backend="podman", hardware="nvidia", repo_root="/r")
    # podman run --rm --network host --device nvidia.com/gpu=all ...
    assert argv[:5] == ["podman", "run", "--rm", "--network", "host"]
    assert "--device" in argv and "nvidia.com/gpu=all" in argv
    assert argv[-2:] == ["hpcagent_bench:nvidia", "run"]


def test_local_run_command_podman_amd_gpu_tokens():
    argv = containers.local_run_command(["x"], backend="podman", hardware="amd", repo_root="/r")
    assert "/dev/kfd" in argv and "--group-add" in argv and "keep-groups" in argv


def test_local_run_command_rejects_dropped_backend():
    for dropped in ("singularity", "udocker", "enroot", "shifter"):
        with pytest.raises(ValueError):
            containers.local_run_command(["x"], backend=dropped)


def test_local_run_command_docker_nvidia_uses_the_docker_gpu_spelling():
    """docker and podman differ on exactly one thing that matters here: the NVIDIA flag."""
    argv = containers.local_run_command(["run"], backend="docker", hardware="nvidia", repo_root="/r")
    assert argv[:5] == ["docker", "run", "--rm", "--network", "host"]
    assert "--gpus" in argv and "all" in argv
    assert "nvidia.com/gpu=all" not in argv  # that is podman's spelling, not docker's
    assert argv[-2:] == ["hpcagent_bench:nvidia", "run"]


def test_harbor_provider_names_docker_and_singularity():
    """Harbor drives docker and singularity. podman and ce have no provider, so they must raise
    rather than emit one Harbor would reject."""
    assert containers.harbor_env_for("docker") == "docker"
    assert containers.harbor_env_for("apptainer") == "singularity"
    for without in ("podman", "ce"):
        with pytest.raises(ValueError, match="Harbor"):
            containers.harbor_env_for(without)


def test_default_image_sif_tag_and_overrides(monkeypatch):
    assert containers.default_image("apptainer", "cpu", repo_root="/r") == "/r/hpcagent_bench-cpu.sif"
    assert containers.default_image("podman", "nvidia") == "hpcagent_bench:nvidia"
    monkeypatch.setenv("HPCAGENT_BENCH_SIF", "/scratch/my.sif")
    assert containers.default_image("apptainer", "cpu", repo_root="/r") == "/scratch/my.sif"
    monkeypatch.setenv("HPCAGENT_BENCH_DOCKER_IMAGE", "reg/img:tag")
    assert containers.default_image("podman", "cpu") == "reg/img:tag"


def test_collect_env_order_is_pinned(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk")  # a passthrough (non-HPCAGENT_BENCH) var
    monkeypatch.setenv("HPCAGENT_BENCH_ZED", "z")  # dynamic HPCAGENT_BENCH_*, sorts last
    monkeypatch.setenv("HPCAGENT_BENCH_ABC", "a")  # dynamic HPCAGENT_BENCH_*, sorts before ZED
    pairs = containers.collect_env("cpu")
    assert pairs[0] == ("HPCAGENT_BENCH_IMAGE", "cpu")  # image first
    assert ("ANTHROPIC_API_KEY", "sk") in pairs
    keys = [k for k, _ in pairs]
    assert keys.index("HPCAGENT_BENCH_ABC") < keys.index("HPCAGENT_BENCH_ZED")  # sorted
    assert keys.count("HPCAGENT_BENCH_IMAGE") == 1  # no duplicate


def test_collect_env_rejects_a_newline_value(monkeypatch):
    monkeypatch.setenv("HPCAGENT_BENCH_BAD", "line1\nline2")
    with pytest.raises(ValueError):
        containers.collect_env("cpu")


def test_harbor_env_for_maps_and_raises():
    assert containers.harbor_env_for("apptainer") == "singularity"
    with pytest.raises(ValueError):
        containers.harbor_env_for("podman")  # podman is launched directly, not via Harbor


# --- install_apptainer retry: both fetches are live-network; subprocess/sleep stubbed, stays pure-unit ---


def _stub_installer(monkeypatch, bash_returncodes, curl_error=None):
    """Stub curl+bash; only intercepts those two argv (patching subprocess.run is process-global)."""
    calls, sleeps, pending = [], [], list(bash_returncodes)
    real_run = subprocess.run

    def fake_run(argv, **kwargs):
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
    monkeypatch.setattr(containers.time, "sleep", lambda s: sleeps.append(s))
    return calls, sleeps


def test_install_apptainer_retries_a_transient_mirror_failure(monkeypatch):
    """A mirror blip is retried in a FRESH process; upstream's own loop cannot recover from this."""
    calls, sleeps = _stub_installer(monkeypatch, bash_returncodes=[2, 2, 0])
    assert containers.install_apptainer("/tmp/apptainer-prefix", attempts=4) == 0
    assert calls.count("bash") == 3, "each attempt must re-run the installer in a fresh process"
    assert sleeps == [5, 10], "backoff must grow, and must NOT sleep after the attempt that succeeded"


def test_install_apptainer_gives_up_and_reports_the_installer_returncode(monkeypatch):
    """Exhausting the attempts still surfaces the real failure -- never a false success."""
    calls, sleeps = _stub_installer(monkeypatch, bash_returncodes=[2, 2, 2])
    assert containers.install_apptainer("/tmp/apptainer-prefix", attempts=3) == 2
    assert calls.count("bash") == 3
    assert sleeps == [5, 10], "no trailing sleep after the final attempt"


def test_install_apptainer_retries_a_failed_installer_download(monkeypatch):
    """The installer download is live-network too, so a curl failure retries rather than raising."""
    calls, _ = _stub_installer(monkeypatch, bash_returncodes=[0], curl_error=6)
    assert containers.install_apptainer("/tmp/apptainer-prefix", attempts=3) == 0
    assert calls.count("curl") == 2, "the failed download must be re-fetched, not raised to the caller"


def test_install_apptainer_succeeds_first_try_without_sleeping(monkeypatch):
    """The happy path must not pay any backoff (guards against an off-by-one in the loop)."""
    calls, sleeps = _stub_installer(monkeypatch, bash_returncodes=[0])
    assert containers.install_apptainer("/tmp/apptainer-prefix") == 0
    assert calls.count("bash") == 1
    assert sleeps == []


def test_install_apptainer_clears_a_partial_tree_between_attempts(monkeypatch, tmp_path):
    """A failed attempt's leftovers must be gone before the retry runs, or upstream hard-refuses on retry."""
    prefix = tmp_path / "apptainer"
    prefix.mkdir()
    seen_dirty = []

    def fake_run(argv, **kwargs):
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
    assert seen_dirty == [False, False, False], \
        f"a retry started against a dirty prefix {seen_dirty} -- upstream would refuse it outright"


def test_clean_partial_install_never_touches_a_preexisting_path(tmp_path):
    """Only paths the attempt created may be removed; `prefix` is caller-supplied (often ~/.local)."""
    prefix = tmp_path / "local"
    (prefix / "share").mkdir(parents=True)
    (prefix / "share" / "user_data.txt").write_text("do not delete me")
    preexisting = set(os.listdir(prefix))
    (prefix / "x86_64").mkdir()
    (prefix / "bin").mkdir()

    containers.clean_partial_install(str(prefix), preexisting)

    assert sorted(os.listdir(prefix)) == ["share"]
    assert (prefix / "share" / "user_data.txt").read_text() == "do not delete me"


def test_clean_partial_install_tolerates_a_missing_prefix(tmp_path):
    """The very first attempt can fail before the prefix exists at all."""
    containers.clean_partial_install(str(tmp_path / "never-created"), set())


def test_ce_stays_its_own_family_even_though_it_is_podman_underneath():
    """CSCS Alps' container engine is podman with SquashFS layers and Cray-tuned OCI hooks, so on
    runtime alone it belongs in the ``oci`` family. It is kept separate on purpose: ``oci`` is
    what a user selects to mean "whatever this machine has", and it must never resolve to the one
    backend with no local launch form, which fails without an EDF and only inside an allocation.
    The runtime is shared; the launch contract is not."""
    spellings, _ = containers.load_backends()
    assert containers.family_members("oci") == ("docker", "podman")
    assert containers.family_members("ce") == ("ce", )
    assert spellings["ce"].kind == "srun_env"
    assert all(spellings[name].kind == "exec" for name in containers.family_members("oci"))
    # The safety property itself: asking for the OCI family never lands on the Slurm-only one.
    assert containers.resolve_backend("oci") in containers.family_members("oci")


def test_the_alps_script_reads_the_ce_flag_from_the_spelling_file():
    """``--environment`` is declared once, in container_backends.txt. The Alps submission script
    derives it from there rather than spelling it again, so a change to how CE is invoked cannot
    leave the cluster path behind."""
    script = pathlib.Path(__file__).resolve().parents[1] / "scripts/cscs/submit_foundation_alps.sbatch"
    text = script.read_text()
    assert "ce.srun_flag" in text, "the Alps script must derive the flag from the spelling file"
    launches = [line for line in text.splitlines() if line.lstrip().startswith(("srun ", 'eval "$(srun '))]
    assert launches, "no srun steps found"
    for line in launches:
        assert '"${CE[@]}"' in line, f"srun step does not carry the derived CE flag: {line.strip()}"
        assert "--environment=" not in line, f"CE flag hardcoded instead of derived: {line.strip()}"
