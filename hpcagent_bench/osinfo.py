# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Host-OS facts that keep the build and runtime portable across Linux (WSL2 included), and macOS."""

import pathlib
import platform
import re
import signal
import sys
from collections.abc import Mapping
from functools import lru_cache

from hpcagent_bench import config
from hpcagent_bench.config import MpContext

__all__ = [
    "IS_LINUX",
    "IS_MACOS",
    "VISIBLE_DEVICE_ENV",
    "cpu_model",
    "default_mp_context",
    "gpu_model",
    "inherited_visible",
    "is_arm",
    "machine",
    "mp_context",
    "node_name",
    "rocminfo_gpu_name",
    "unblock_sigchld",
]

#: True on macOS: fork-after-threads is unsafe and the glibc-only flags (``libgomp``/``libmvec``) do not exist.
IS_MACOS = sys.platform == "darwin"
#: True on Linux, INCLUDING WSL2 (a genuine Linux kernel).
IS_LINUX = sys.platform.startswith("linux")

#: The device-visibility variables, in the order read. ROCr and HIP compose (setting both is
#: ``hipErrorNoDevice``), so this harness sets ROCr only.
VISIBLE_DEVICE_ENV: tuple[str, ...] = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


def inherited_visible(env: Mapping[str, str]) -> str:
    """The first non-blank :data:`VISIBLE_DEVICE_ENV` value of ``env`` (a comma list), or ``""``."""
    return next((env[name] for name in VISIBLE_DEVICE_ENV if env.get(name, "").strip()), "")


def machine() -> str:
    """The host CPU architecture (``platform.machine()``)."""
    return platform.machine()


def is_arm() -> bool:
    """True on 64-bit ARM (Apple Silicon ``arm64`` or Linux ``aarch64``)."""
    return machine().lower() in ("arm64", "aarch64")


def cpu_model() -> str:
    """Best-effort CPU model string (``$HPCAGENT_BENCH_CPU`` overrides); also namespaces ``-march=native`` caches."""
    import os

    env = os.environ.get("HPCAGENT_BENCH_CPU")
    if env:
        return env
    try:
        with pathlib.Path("/proc/cpuinfo").open() as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def node_name() -> str:
    """The node a measurement runs on: ``$HPCAGENT_BENCH_HOST``, else ``$SLURMD_NODENAME``, else the hostname.

    Recorded per ROW: the only column telling two nodes of the same hardware model apart."""
    import os
    import socket

    return os.environ.get("HPCAGENT_BENCH_HOST") or os.environ.get("SLURMD_NODENAME") or socket.gethostname()


#: A ``rocminfo`` agent's device type and marketing name lines.
ROCMINFO_DEVICE_TYPE = re.compile(r"^\s*Device Type:\s+(\S+)\s*$", re.MULTILINE)
ROCMINFO_MARKETING_NAME = re.compile(r"^\s*Marketing Name:\s+(.+?)\s*$", re.MULTILINE)


def rocminfo_gpu_name(text: str) -> str:
    """The marketing name of the first GPU agent in ``rocminfo`` output ``text``, or ``""``.

    The agent's Device Type picks the GPU: an APU's CPU agent (MI300A) carries a marketing name too."""
    for agent in re.split(r"^\*+\s*$", text, flags=re.MULTILINE):
        kind, name = ROCMINFO_DEVICE_TYPE.search(agent), ROCMINFO_MARKETING_NAME.search(agent)
        if kind and kind.group(1) == "GPU" and name:
            return name.group(1)
    return ""


@lru_cache(maxsize=1, typed=True)
def gpu_model() -> str:
    """Best-effort GPU model string (``$HPCAGENT_BENCH_GPU``, else ``nvidia-smi``, else ``rocminfo``); ``""`` on a
    CPU-only host. Cached (read once per recorded row)."""
    import os
    import subprocess

    env = os.environ.get("HPCAGENT_BENCH_GPU")
    if env:
        return env
    try:
        out = (
            subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], timeout=10)
            .decode()
            .strip()
            .splitlines()
        )
        if out:
            return out[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        return rocminfo_gpu_name(subprocess.check_output(["rocminfo"], timeout=30).decode(errors="replace"))
    except (OSError, subprocess.SubprocessError):
        return ""


def default_mp_context() -> str:
    """The safe multiprocessing start method: ``fork`` on Linux, ``spawn`` on macOS (forking after BLAS threads can
    deadlock); config/env overrides (:func:`mp_context`)."""
    return "spawn" if IS_MACOS else "fork"


def mp_context() -> str:
    """``runtime.mp_context`` as :mod:`multiprocessing` spells it, ``auto`` resolved by :func:`default_mp_context`."""
    value = config.current().runtime.mp_context
    return default_mp_context() if value is MpContext.AUTO else value.text


def unblock_sigchld() -> None:
    """Unblock SIGCHLD, so a build sees its own children exit.

    srun/mpirun start tasks with SIGCHLD blocked, and the mask survives fork and exec (cmake's configure then waits
    forever). Masks are per-thread and inherited, so :func:`hpcagent_bench.cli.main` calls this before anything
    builds or starts a thread."""
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})
