# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Host-OS facts that keep the build + runtime portable across Linux, macOS, and WSL2.

Stdlib-only (``sys`` + ``platform``) so the lowest layers -- the flag matrix, the
fork primitive -- can import it without pulling in config/yaml. The one config-aware
helper (:func:`mp_context`) is the exception and reads the runtime config.

WSL2 is a real Linux kernel, so it is ``IS_LINUX`` and needs no special casing.
"""
import platform
import sys
from functools import lru_cache

from hpcagent_bench import config

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


def cpu_model() -> str:
    """Best-effort CPU model string; honors ``$HPCAGENT_BENCH_CPU``, else falls back to platform info.

    Identifies the host well enough to tell two machines apart, which is what the recording
    tables' ``cpu`` column and the compiler-cache namespace both need -- ``-march=native``
    means a cached object is only valid on the CPU that produced it.
    """
    import os
    env = os.environ.get("HPCAGENT_BENCH_CPU")
    if env:
        return env
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


@lru_cache(maxsize=1, typed=True)
def gpu_model() -> str:
    """Best-effort GPU model string; honors ``$HPCAGENT_BENCH_GPU``, else asks ``nvidia-smi``.

    ``""`` when the host has no discoverable device -- not an error, just a CPU-only box.

    Pairs with :func:`cpu_model` to name the NODE a measurement came from. Two nodes are two
    experiments: a baseline timed on one machine against a candidate timed on another is a hardware
    comparison wearing a software label. Cached, because this is read once per recorded row and a
    subprocess per row would cost more than the measurement.
    """
    import os
    import subprocess
    env = os.environ.get("HPCAGENT_BENCH_GPU")
    if env:
        return env
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                      timeout=10).decode().strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return ""
    return out[0].strip() if out else ""


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
    / ``forkserver``, or ``HPCAGENT_BENCH_RUNTIME_MP_CONTEXT``) wins -- e.g. the threaded judge
    service pins ``forkserver`` (fork-from-a-thread is unsafe)."""
    value = config.get("runtime.mp_context", "auto")
    return default_mp_context() if value == "auto" else value
