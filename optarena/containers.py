# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Container backends for running the OptArena images.

Four backends, selected by ``config.yaml`` ``runtime.backend``:

* ``docker`` / ``singularity`` -- run THROUGH Harbor (its ``--env`` flag picks the
  provider). :func:`harbor_env_type` maps the backend to that value.
* ``apptainer`` / ``udocker`` -- run an image LOCALLY without Harbor and without
  sudo (Apptainer rootless; udocker is pure user space, pip-installable).
  :func:`local_run_command` builds the argv.

Apptainer is not pip-installable (a Go binary); :func:`install_apptainer` runs its
unprivileged install into a user prefix (exposed as ``optarena-install-apptainer``).
udocker is `pip install udocker`.
"""
import os
import subprocess
import sys

from optarena import config

HARBOR_BACKENDS = ("docker", "singularity")  # via `harbor run --env <backend>`
LOCAL_BACKENDS = ("apptainer", "udocker")  # run an image directly, sudoless
BACKENDS = HARBOR_BACKENDS + LOCAL_BACKENDS

#: Apptainer's official unprivileged (no-root) installer.
APPTAINER_INSTALLER = "https://raw.githubusercontent.com/apptainer/apptainer/main/tools/install-unprivileged.sh"


def backend(name=None):
    name = name or config.get("runtime.backend", "singularity")
    if name not in BACKENDS:
        raise ValueError(f"runtime.backend must be one of {BACKENDS}; got {name!r}")
    return name


def harbor_env_type(name=None):
    """The Harbor ``--env`` value for a Harbor-driven backend (``docker`` /
    ``singularity``). Raises for a local backend, which Harbor does not provide."""
    name = backend(name)
    if name not in HARBOR_BACKENDS:
        raise ValueError(f"{name!r} is a local backend; pass it to local_run_command, not Harbor "
                         f"(Harbor provides only {HARBOR_BACKENDS})")
    return name


def local_run_command(image, *cmd, name=None):
    """Argv to run ``image`` locally + sudoless with ``apptainer`` or ``udocker``."""
    name = backend(name)
    if name == "apptainer":
        return ["apptainer", "run", image, *cmd]
    if name == "udocker":
        return ["udocker", "run", image, *cmd]
    raise ValueError(f"{name!r} is a Harbor backend; run it via Harbor (--env {name})")


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
