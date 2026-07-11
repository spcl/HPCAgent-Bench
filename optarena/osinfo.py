# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Host-OS facts that keep the build + runtime portable across Linux, macOS, and WSL2.

Stdlib-only (``sys`` + ``platform``) so the lowest layers -- the flag matrix, the
fork primitive -- can import it without pulling in config/yaml. The one config-aware
helper (:func:`mp_context`) is the exception and reads the runtime config.

WSL2 is a real Linux kernel, so it is ``IS_LINUX`` and needs no special casing.
"""
import platform
import sys

from optarena import config

#: True on macOS (Darwin). fork-after-threads is unsafe here and the glibc-only
#: build flags (``libgomp``/``libmvec``) do not exist.
IS_MACOS = sys.platform == "darwin"
#: True on Linux, INCLUDING WSL2 (a genuine Linux kernel).
IS_LINUX = sys.platform.startswith("linux")


def machine() -> str:
    """The host CPU architecture (``platform.machine()``: ``x86_64`` / ``arm64`` /
    ``aarch64`` / ...)."""
    return platform.machine()


def is_arm() -> bool:
    """True on 64-bit ARM (Apple Silicon ``arm64`` or Linux ``aarch64``)."""
    return machine().lower() in ("arm64", "aarch64")


def default_mp_context() -> str:
    """The safe multiprocessing start method for this OS.

    ``fork`` on Linux/WSL2 (cheap -- the child inherits the parent's inputs). ``spawn``
    on macOS: forking a process that has already spawned numpy/BLAS/Accelerate threads
    (or initialised an Objective-C runtime) can abort or deadlock the child -- which is
    exactly why Python made ``spawn`` the macOS default at 3.8. A concrete config/env
    value overrides this (see :func:`mp_context`)."""
    return "spawn" if IS_MACOS else "fork"


def mp_context() -> str:
    """The multiprocessing start method to use, resolving the ``auto`` default to
    :func:`default_mp_context`. A concrete ``runtime.mp_context`` (``fork`` / ``spawn``
    / ``forkserver``, or ``OPTARENA_RUNTIME_MP_CONTEXT``) wins -- e.g. the threaded judge
    service pins ``forkserver`` (fork-from-a-thread is unsafe)."""
    value = config.get("runtime.mp_context", "auto")
    return default_mp_context() if value == "auto" else value
