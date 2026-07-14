# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Container launch factory + the unprivileged Apptainer installer.

One factory turns a ``(backend, image, command)`` into a launch argv. The per-backend
flag SPELLINGS live in the language-neutral ``container_backends.txt`` (this directory),
read here by Python and by ``scripts/run_agent_in_container.sh`` in pure bash -- one
source of truth for both the Python callers and the python-less HPC login host. See
``docs/DESIGN_container_launch_and_submission.md``.

Two shapes: an EXEC-WRAPPER (apptainer/docker/podman/udocker) built by
:func:`local_run_command`, and the CSCS Container Engine, whose image is a flag
(``--environment=<edf>``) not a positional -- built by :func:`ce_launcher`. Harbor is an
orchestrator, not a wrapper; :func:`harbor_env_for` only supplies its provider name.

Apptainer itself is a Go binary (not pip-installable); :func:`install_apptainer` runs its
official unprivileged install into a user prefix, exposed as the ``optarena-install-apptainer``
entry point.
"""
import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

from optarena import config

#: Apptainer's official unprivileged (no-root) installer.
APPTAINER_INSTALLER = "https://raw.githubusercontent.com/apptainer/apptainer/main/tools/install-unprivileged.sh"

#: The single-source spelling file, read by BOTH this module and the bash launcher.
BACKENDS_PATH = pathlib.Path(__file__).parent / "container_backends.txt"

#: The selectable backends. ``ce`` is a valid selection but is launched by
#: :func:`ce_launcher` (not an exec-wrapper fold row, so not in :data:`SPELLINGS`).
KNOWN_BACKENDS = ("apptainer", "docker", "podman", "udocker", "ce")


@dataclass(frozen=True)
class WrapperSpelling:
    """How one exec-wrapper backend spells its launch flags (one row of the file)."""
    name: str
    verb: Tuple[str, ...]  # ("exec",) | ("run", "--rm", "--network", "host") | ("run", "--rm")
    bind_flag: str  # "--bind" | "-v"
    workdir_flag: str  # "--pwd" | "-w"
    env_flag: str  # "--env" | "-e"
    gpu: Mapping[str, Tuple[str, ...]]  # {"nvidia": (...), "amd": (...)}; a cpu run adds nothing
    image_form: str  # "sif" | "tag"
    image_default: str  # "optarena-{hw}.sif" | "optarena:{hw}"
    harbor_env: str  # "singularity" | "docker" | "" (empty = not a Harbor backend)


def load_backends(path: pathlib.Path = BACKENDS_PATH) -> Tuple[dict, Tuple[str, ...]]:
    """Parse the spelling file into ``({backend: WrapperSpelling}, passthrough_env)``.

    Both the Python fold and the bash fold read this one file, so the launch argv is
    byte-identical across the language boundary."""
    rows: dict = {}
    passthrough: Tuple[str, ...] = ()
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
        name:
        WrapperSpelling(name=name,
                        verb=tuple(f["verb"].split()),
                        bind_flag=f["bind"].strip(),
                        workdir_flag=f["workdir"].strip(),
                        env_flag=f["env"].strip(),
                        gpu={
                            "nvidia": tuple(f.get("gpu.nvidia", "").split()),
                            "amd": tuple(f.get("gpu.amd", "").split()),
                        },
                        image_form=f["image_form"].strip(),
                        image_default=f["image_default"].strip(),
                        harbor_env=f.get("harbor_env", "").strip())
        for name, f in rows.items()
    }
    return spellings, passthrough


SPELLINGS, PASSTHROUGH_ENV = load_backends()


def resolve_backend(explicit: Optional[str] = None) -> str:
    """The active container backend: ``explicit`` arg > ``$OPTARENA_RUNTIME_BACKEND`` >
    ``config.get("runtime.backend")`` > ``apptainer``.

    Note: the legacy bash-only ``$OPTARENA_CONTAINER_RUNTIME`` is DELIBERATELY not read
    here -- the shell launcher still honors it locally, but wiring it into the Python
    path would make a Harbor run crash whenever a user had set it for a local bash run.
    Both paths share the one canonical ``$OPTARENA_RUNTIME_BACKEND``."""
    backend = (explicit or os.environ.get("OPTARENA_RUNTIME_BACKEND") or config.get("runtime.backend", "apptainer")
               or "apptainer").strip()
    if backend not in KNOWN_BACKENDS:
        raise ValueError(f"unknown container backend {backend!r}; known: {list(KNOWN_BACKENDS)}")
    return backend


def default_image(backend: str, hardware: str = "cpu", repo_root: Optional[str] = None) -> str:
    """The image reference for ``backend`` on ``hardware`` -- an ``$OPTARENA_SIF`` /
    ``$OPTARENA_DOCKER_IMAGE`` override, else the file's default (a sif path under
    ``repo_root``, or an ``optarena:<hw>`` tag)."""
    spelling = SPELLINGS[backend]
    if spelling.image_form == "sif":
        override = os.environ.get("OPTARENA_SIF")
        if override:
            return override
        name = spelling.image_default.format(hw=hardware)
        return os.path.join(repo_root, name) if repo_root else name
    return os.environ.get("OPTARENA_DOCKER_IMAGE") or spelling.image_default.format(hw=hardware)


def collect_env(hardware: str) -> List[Tuple[str, str]]:
    """The ``(key, value)`` env pairs to forward into the image, in a PINNED order so the
    bash fold matches byte-for-byte: ``OPTARENA_IMAGE=<hw>`` first, then
    :data:`PASSTHROUGH_ENV` (present, in file order), then every other ``OPTARENA_*`` var
    sorted (Python's str sort == ``LC_ALL=C sort``). Reads only the environment -- there is no
    caller-supplied extra, because the bash fold has no such channel and any divergence would
    silently break the byte-for-byte parity."""
    pairs: List[Tuple[str, str]] = [("OPTARENA_IMAGE", hardware)]
    seen = {"OPTARENA_IMAGE"}
    for key in PASSTHROUGH_ENV:
        value = os.environ.get(key)
        if value and key not in seen:
            pairs.append((key, value))
            seen.add(key)
    for key in sorted(k for k in os.environ if k.startswith("OPTARENA_") and k not in seen):
        value = os.environ.get(key)
        if value:
            pairs.append((key, value))
            seen.add(key)
    # Invariant (container_backends.txt): the fold is newline-delimited on the bash side, so a
    # value with a newline would split into extra argv tokens there while staying one token here
    # -- a silent parity break. Fail loud rather than emit a corrupt launch.
    for key, value in pairs:
        if "\n" in value:
            raise ValueError(f"env {key!r} contains a newline; the launch fold is newline-delimited "
                             f"and cannot forward it (container_backends.txt token-list invariant)")
    return pairs


def local_run_command(inner: Sequence[str],
                      *,
                      backend: Optional[str] = None,
                      hardware: str = "cpu",
                      image: Optional[str] = None,
                      repo_root: Optional[str] = None) -> List[str]:
    """THE factory: the full launch argv for running ``inner`` inside the image under an
    exec-wrapper backend -- ``prefix + [image] + inner`` in the fixed fold order the bash
    launcher mirrors. ``backend`` defaults to :func:`resolve_backend`; ``ce`` is rejected
    (use :func:`ce_launcher`)."""
    chosen = resolve_backend(backend)
    if chosen not in SPELLINGS:
        raise ValueError(f"{chosen!r} is not a local exec-wrapper backend; use ce_launcher() for CE")
    spelling = SPELLINGS[chosen]
    repo = repo_root or os.getcwd()
    argv: List[str] = [chosen, *spelling.verb, *spelling.gpu.get(hardware, ())]
    for key, value in collect_env(hardware):
        argv += [spelling.env_flag, f"{key}={value}"]
    argv += [spelling.bind_flag, f"{repo}:{repo}", spelling.workdir_flag, repo]
    argv.append(image or default_image(chosen, hardware, repo))
    argv += list(inner)
    return argv


def harbor_env_for(backend: Optional[str] = None) -> str:
    """Harbor's ``--env`` provider name for the resolved backend
    (``apptainer -> singularity``, ``docker -> docker``). Raises for a backend Harbor
    cannot drive (``podman`` / ``udocker`` / ``ce`` -- Harbor provides only docker +
    singularity), so the caller never emits an invalid provider."""
    chosen = resolve_backend(backend)
    name = SPELLINGS[chosen].harbor_env if chosen in SPELLINGS else ""
    if not name:
        raise ValueError(f"{chosen!r} is not a Harbor backend (Harbor provides docker + singularity)")
    return name


def ce_launcher(environment: Optional[str] = None,
                gpus: str = "1",
                ntasks: str = "1",
                overlap: bool = False,
                node: str = "{node}") -> Tuple[str, ...]:
    """The CSCS Container Engine srun prefix (image is the ``--environment=<edf>`` flag,
    not a positional). ``{node}`` is left un-substituted by default so
    ``judge_scheduler.srun_wrap`` keeps sole ownership of the per-slot replacement. With
    no args this is the plain judge launcher; ``environment`` adds the glued CE flag."""
    argv: List[str] = ["srun", "--nodelist", node, "--gpus", gpus, "-n", ntasks]
    if overlap:
        argv.append("--overlap")
    if environment:
        argv.append(f"--environment={environment}")
    return tuple(argv)


def require_ce_environment(prefix: Sequence[str], image: Optional[str] = None) -> None:
    """Fail loud if a CE launch would carry neither an ``--environment=<edf>`` (glued or
    space form) nor an explicit ``image`` -- i.e. it would grade on the bare host instead
    of inside the CE image."""
    if image:
        return
    for token in prefix:
        if token == "--environment" or token.startswith("--environment="):
            return
    raise ValueError("CE launch carries no --environment=<edf> and no explicit image; "
                     "the grade would run on the bare host, not inside the CE image")


def install_apptainer(prefix="~/.local"):
    """Install Apptainer unprivileged (no sudo) into ``prefix`` via its official
    installer. Returns the subprocess return code.

    The installer is downloaded then piped to ``bash`` over stdin, with ``prefix``
    passed as a real argv element -- NOT interpolated into a ``shell=True`` string
    (which would let a crafted ``prefix`` inject arbitrary commands)."""
    prefix = os.path.expanduser(prefix)
    script = subprocess.run(["curl", "-fsSL", APPTAINER_INSTALLER], check=True, capture_output=True, text=True).stdout
    return subprocess.run(["bash", "-s", "-", prefix], input=script, text=True).returncode


def install_apptainer_main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    prefix = argv[0] if argv else "~/.local"
    return install_apptainer(prefix)


if __name__ == "__main__":
    sys.exit(install_apptainer_main())
