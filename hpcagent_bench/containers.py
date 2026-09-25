# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Container launch factory + the unprivileged Apptainer installer.

One factory (:func:`local_run_command`) turns a ``(backend, image, command)`` into a launch
argv. There is ONE image and it is OCI; the four backends differ only in how they consume it.
``podman`` and ``docker`` run the OCI tag directly, ``apptainer`` builds a SIF from it, and
``ce`` (the CSCS Alps container engine) is podman again -- SquashFS layers plus Cray-tuned OCI
hooks -- reached through Slurm rather than a wrapper. Shipping OCI is what lets a laptop, a
cloud VM and an HPC site start from the same artifact and the same runtime.

The default is ``podman``, not ``docker``: docker needs a running daemon and membership of a
root-equivalent group, neither of which an HPC login node grants, so it cannot be the thing a
user falls back to. podman is the OCI runtime that is rootless and daemonless and therefore
runs in both places. :func:`detect_backend` probes PATH when the answer should come from the
machine rather than from a default.

``ce`` is a different SHAPE of backend, not just different flags: it has no wrapper argv at
all. The container is selected by ``srun --environment=<edf>`` and the command runs unwrapped, which is why
:func:`local_run_command` returns it untouched.

The per-backend flag SPELLINGS live in the language-neutral ``container_backends.txt`` (this
directory), read here by Python and by ``scripts/run_agent_in_container.sh`` in pure bash --
one source of truth for both the Python callers and the python-less HPC login host.

Harbor is an orchestrator, not a wrapper; :func:`harbor_env_for` only supplies its provider
name (apptainer -> singularity, docker -> docker, podman -> podman; ce has no Harbor provider).

Apptainer itself is a Go binary (not pip-installable); :func:`install_apptainer` runs its
official unprivileged install into a user prefix, exposed as the ``hpcagent-bench-install-apptainer``
entry point.
"""

import os
import pathlib
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from hpcagent_bench import config

#: Apptainer's official unprivileged (no-root) installer.
APPTAINER_INSTALLER = "https://raw.githubusercontent.com/apptainer/apptainer/main/tools/install-unprivileged.sh"

#: The single-source spelling file, read by BOTH this module and the bash launcher.
BACKENDS_PATH = pathlib.Path(__file__).parent / "container_backends.txt"

#: The FAMILIES the benchmark supports, in preference order. A family is an image form plus a
#: launch contract, NOT a program: ``oci`` is a standard that docker and podman each implement,
#: so selecting it means "any OCI-compliant runtime this machine has" and always resolves to one
#: of them. CSCS Alps' container engine is podman underneath -- SquashFS layers plus the Cray
#: OCI hooks that give correct GPU and NIC access -- so ``ce`` is very nearly ``oci``. It is kept
#: SEPARATE deliberately: ``oci`` must never resolve to the one implementation with no local
#: launch form, which fails without an EDF and only inside an allocation. The runtime is shared;
#: the contract is not, and the contract is what a caller has to get right. ``native`` is a real
#: member, not the absence of one -- a site with no runtime installed still has to run.
FAMILIES = ("oci", "sif", "ce", "native")
#: The concrete implementations, in preference order within the whole set. docker leads: it is
#: the OCI implementation most users have, and ``oci`` resolves to it wherever it is installed.
KNOWN_BACKENDS = ("docker", "podman", "apptainer", "ce", "native")
#: Names accepted from a user: a concrete implementation, or a FAMILY name that resolves to
#: whichever of its implementations is installed. Selecting ``oci`` is the honest way to say
#: "any OCI runtime" -- it names a standard, and the resolution names the program.
SELECTABLE = KNOWN_BACKENDS + FAMILIES
#: What a run falls back to when nothing selects a backend and nothing is detectable on PATH.
#: Rootless and daemonless, so it is the one an unprivileged user can always invoke -- docker
#: leads the ``oci`` family but cannot be the last resort, because it needs a daemon and a
#: root-equivalent group that a login node does not grant.
DEFAULT_BACKEND = "podman"
#: Backends launched by WRAPPING the command (``docker run ... image cmd``). Everything outside
#: this set selects its image by another mechanism and contributes no wrapper prefix.
EXEC_BACKENDS = ("docker", "podman", "apptainer")


def family_members(family: str) -> tuple[str, ...]:
    """The runtimes implementing ``family``, in :data:`KNOWN_BACKENDS` preference order."""
    return tuple(name for name in KNOWN_BACKENDS if SPELLINGS[name].family == family)


@dataclass(frozen=True)
class WrapperSpelling:
    """How one backend spells its launch flags (one row of the file).

    Two shapes share this record. An ``exec`` backend wraps the command and every flag field is
    meaningful. An ``srun_env`` backend (CSCS Alps' container engine) has no wrapper argv at all:
    it contributes :attr:`srun_flag` to the ``srun`` line and the command runs unwrapped, so its
    flag fields are empty and reading them would be a category error.
    """

    name: str
    family: str  # "oci" (the shipped image, unconverted) | "sif" | "ce" (conversions of it)
    rootless: bool  # invocable unprivileged, with no daemon and no root-equivalent group
    kind: str  # "exec" (wraps the command) | "srun_env" (selected by an srun flag)
    verb: tuple[str, ...]  # ("exec",) | ("run", "--rm", "--network", "host") | ()
    bind_flag: str  # "--bind" | "-v" | "" for srun_env (the EDF declares its own mounts)
    workdir_flag: str  # "--pwd" | "-w" | "" for srun_env (the EDF declares its own workdir)
    env_flag: str  # "--env" | "-e" | "" for srun_env (the EDF declares its own [env])
    gpu: Mapping[str, tuple[str, ...]]  # {"nvidia": (...), "amd": (...)}; a cpu run adds nothing
    image_form: str  # "sif" | "tag" | "edf"
    image_default: str  # "hpcagent_bench-{hw}.sif" | "hpcagent_bench:{hw}" | "" (EDF is supplied)
    harbor_env: str  # "singularity" | "docker" | "" (empty = not a Harbor backend)
    srun_flag: str  # "--environment" for srun_env; "" for an exec wrapper


def load_backends(path: pathlib.Path = BACKENDS_PATH) -> tuple[dict, tuple[str, ...]]:
    """Parse the spelling file into ``({backend: WrapperSpelling}, passthrough_env)``.

    Both the Python fold and the bash fold read this one file, so the launch argv is
    byte-identical across the language boundary."""
    rows: dict = {}
    passthrough: tuple[str, ...] = ()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        head, _, field = key.strip().partition(".")
        if head == "global":
            if field == "passthrough":
                passthrough = tuple(value.split())
            continue
        rows.setdefault(head, {})[field] = value
    spellings = {
        name: WrapperSpelling(
            name=name,
            family=f["family"].strip(),
            rootless=f.get("rootless", "0").strip() == "1",
            kind=f.get("kind", "exec").strip(),
            verb=tuple(f.get("verb", "").split()),
            bind_flag=f.get("bind", "").strip(),
            workdir_flag=f.get("workdir", "").strip(),
            env_flag=f.get("env", "").strip(),
            gpu={
                "nvidia": tuple(f.get("gpu.nvidia", "").split()),
                "amd": tuple(f.get("gpu.amd", "").split()),
            },
            image_form=f["image_form"].strip(),
            image_default=f.get("image_default", "").strip(),
            harbor_env=f.get("harbor_env", "").strip(),
            srun_flag=f.get("srun_flag", "").strip(),
        )
        for name, f in rows.items()
    }
    return spellings, passthrough


SPELLINGS, PASSTHROUGH_ENV = load_backends()


def resolve_backend(explicit: str | None = None) -> str:
    """The active container RUNTIME: ``explicit`` arg > ``$HPCAGENT_BENCH_RUNTIME_BACKEND`` >
    ``config.get("runtime.backend")`` > :data:`DEFAULT_BACKEND`.

    Accepts a family name as well as a runtime name. ``"oci"`` means "whichever OCI runtime this
    machine has", and resolves by probing PATH -- which is what a user selecting an interface
    rather than a program actually wants. The return is always a concrete runtime, so every
    caller downstream keeps working with one.

    The fallback is the ROOTLESS OCI runtime: the shipped artifact is an OCI image, docker needs a
    daemon and a root-equivalent group, and apptainer/ce need the image converted first. Use
    :func:`detect_backend` to pick from what is installed. The shell launcher's own
    ``$HPCAGENT_BENCH_CONTAINER_RUNTIME`` is deliberately not read here."""
    backend = (
        explicit
        or os.environ.get("HPCAGENT_BENCH_RUNTIME_BACKEND")
        or config.get("runtime.backend", DEFAULT_BACKEND)
        or DEFAULT_BACKEND
    ).strip()
    if backend in FAMILIES:
        members = family_members(backend)
        # A family names the interface, so resolving it must ask the machine which flavour is
        # present. Falling back to the family's first member keeps the answer deterministic on a
        # host with none installed, where the caller's real error is the missing runtime.
        return detect_backend(members) or members[0]
    if backend not in KNOWN_BACKENDS:
        raise ValueError(f"unknown container backend {backend!r}; selectable: {list(SELECTABLE)}")
    return backend


def detect_backend(candidates: Sequence[str] = KNOWN_BACKENDS) -> str | None:
    """The first backend in ``candidates`` whose launcher is actually on PATH, or ``None``.

    Probing beats assuming: a login node has podman and no dockerd, a laptop usually has the
    reverse, and an Alps allocation has neither but does have ``srun``. Only the CLI's presence
    is checked, never whether an image is built -- that is the caller's error to report, and it
    is a different error.
    """
    for name in candidates:
        if name not in SPELLINGS:
            continue
        probe = "srun" if SPELLINGS[name].kind == "srun_env" else name
        if shutil.which(probe):
            return name
    return None


def default_image(backend: str, hardware: str = "cpu", repo_root: str | None = None) -> str:
    """The image reference for ``backend`` on ``hardware`` -- an ``$HPCAGENT_BENCH_SIF`` /
    ``$HPCAGENT_BENCH_DOCKER_IMAGE`` override, else the file's default (a sif path under
    ``repo_root``, or an ``hpcagent_bench:<hw>`` tag)."""
    spelling = SPELLINGS[backend]
    if spelling.image_form == "edf":
        raise ValueError(f"{backend!r} has no image reference of its own: its EDF names the image (srun --environment)")
    if not spelling.image_form:
        raise ValueError(
            f"{backend!r} runs on the host and consumes no image; asking it for one is a "
            "category error, not a missing default"
        )
    if spelling.image_form == "sif":
        override = os.environ.get("HPCAGENT_BENCH_SIF")
        if override:
            return override
        name = spelling.image_default.format(hw=hardware)
        return os.path.join(repo_root, name) if repo_root else name
    return os.environ.get("HPCAGENT_BENCH_DOCKER_IMAGE") or spelling.image_default.format(hw=hardware)


def collect_env(hardware: str) -> list[tuple[str, str]]:
    """The ``(key, value)`` env pairs to forward into the image, in a PINNED order so the
    bash fold matches byte-for-byte: ``HPCAGENT_BENCH_IMAGE=<hw>`` first, then
    :data:`PASSTHROUGH_ENV` (present, in file order), then every other ``HPCAGENT_BENCH_*`` var
    sorted (Python's str sort == ``LC_ALL=C sort``). Reads only the environment -- there is no
    caller-supplied extra, because the bash fold has no such channel and any divergence would
    silently break the byte-for-byte parity."""
    pairs: list[tuple[str, str]] = [("HPCAGENT_BENCH_IMAGE", hardware)]
    seen = {"HPCAGENT_BENCH_IMAGE"}
    for key in PASSTHROUGH_ENV:
        value = os.environ.get(key)
        if value and key not in seen:
            pairs.append((key, value))
            seen.add(key)
    for key in sorted(k for k in os.environ if k.startswith("HPCAGENT_BENCH_") and k not in seen):
        value = os.environ.get(key)
        if value:
            pairs.append((key, value))
            seen.add(key)
    # Invariant (container_backends.txt): the fold is newline-delimited on the bash side, so a
    # value with a newline would split into extra argv tokens there while staying one token here
    # -- a silent parity break. Fail loud rather than emit a corrupt launch.
    for key, value in pairs:
        if "\n" in value:
            raise ValueError(
                f"env {key!r} contains a newline; the launch fold is newline-delimited "
                f"and cannot forward it (container_backends.txt token-list invariant)"
            )
    return pairs


def local_run_command(
    inner: Sequence[str],
    *,
    backend: str | None = None,
    hardware: str = "cpu",
    image: str | None = None,
    repo_root: str | None = None,
) -> list[str]:
    """THE factory: the full launch argv for running ``inner`` inside the image under an
    exec-wrapper backend -- ``prefix + [image] + inner`` in the fixed fold order the bash
    launcher mirrors. ``backend`` defaults to :func:`resolve_backend`.

    Two kinds return ``inner`` UNCHANGED, for the same reason: there is no wrapper argv to build.
    An ``srun_env`` backend (CSCS Alps' container engine) has its container chosen by the
    ``--environment`` flag on the ``srun`` line, and a ``none``
    backend (``native``) is not a container at all. Returning the bare command is the honest
    answer -- synthesising a wrapper a backend does not have would produce an argv that cannot
    run.
    """
    chosen = resolve_backend(backend)
    spelling = SPELLINGS[chosen]
    if spelling.kind in ("srun_env", "none"):
        return list(inner)
    repo = repo_root or os.getcwd()
    argv: list[str] = [chosen, *spelling.verb, *spelling.gpu.get(hardware, ())]
    for key, value in collect_env(hardware):
        argv += [spelling.env_flag, f"{key}={value}"]
    argv += [spelling.bind_flag, f"{repo}:{repo}", spelling.workdir_flag, repo]
    argv.append(image or default_image(chosen, hardware, repo))
    argv += list(inner)
    return argv


def harbor_env_for(backend: str | None = None) -> str:
    """Harbor's ``--env`` provider name for the resolved backend (``docker``, ``podman``,
    ``apptainer -> singularity``). Raises for ``ce`` and ``native``, which Harbor cannot drive."""
    chosen = resolve_backend(backend)
    name = SPELLINGS[chosen].harbor_env
    if not name:
        raise ValueError(
            f"{chosen!r} is not a Harbor backend (Harbor provides docker, podman, singularity); "
            "run it directly via local_run_command / scripts/run_agent_in_container.sh"
        )
    return name


def install_apptainer(prefix: str = "~/.local", attempts: int = 4) -> int:
    """Install Apptainer unprivileged (no sudo) into ``prefix`` via its official installer;
    returns the installer's return code.

    The installer is piped to ``bash`` on stdin with ``prefix`` as a real argv element, never
    interpolated into a shell string. The whole script is retried in a FRESH process with backoff:
    it scrapes an EPEL listing behind the ``download.fedoraproject.org`` redirector, its own retry
    loop never sleeps, and it caches the listing per process, so only a new process can land on a
    different mirror. Each failed attempt's partial tree is removed first
    (:func:`clean_partial_install`), because the installer refuses a non-empty ``<prefix>/<arch>``."""
    prefix = os.path.expanduser(prefix)
    preexisting = set(os.listdir(prefix)) if os.path.isdir(prefix) else set()
    returncode = 1
    for attempt in range(1, attempts + 1):
        try:
            script = subprocess.run(
                ["curl", "-fsSL", APPTAINER_INSTALLER], check=True, capture_output=True, text=True
            ).stdout
            returncode = subprocess.run(["bash", "-s", "-", prefix], input=script, text=True).returncode
            if returncode == 0:
                return 0
        except subprocess.CalledProcessError as exc:
            returncode = exc.returncode
        if attempt < attempts:
            clean_partial_install(prefix, preexisting)
            delay = 5 * attempt
            print(
                f"apptainer install attempt {attempt}/{attempts} failed (rc={returncode}); retrying in {delay}s",
                file=sys.stderr,
            )
            time.sleep(delay)
    return returncode


def clean_partial_install(prefix: str, preexisting: Sequence[str]) -> None:
    """Remove what a failed :func:`install_apptainer` attempt left in ``prefix`` -- and ONLY that.

    ``preexisting`` is the prefix's entries from before the first attempt, left alone: ``prefix``
    defaults to ``~/.local``, and a blanket wipe would delete a user's unrelated installs."""
    if not os.path.isdir(prefix):
        return
    for name in os.listdir(prefix):
        if name in preexisting:
            continue
        path = os.path.join(prefix, name)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                os.remove(path)
            except OSError:
                pass


def install_apptainer_main(argv: Sequence[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    prefix = argv[0] if argv else "~/.local"
    return install_apptainer(prefix)


if __name__ == "__main__":
    sys.exit(install_apptainer_main())
