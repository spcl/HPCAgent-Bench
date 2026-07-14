# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Framework binding for the Pluto polyhedral native backend.

Pluto is kept SEPARATE from :class:`~optarena.frameworks.native_framework.NativeFramework`
because it is a distinct toolchain, not a compiler flag: polycc applies a
polyhedral source-to-source transform (tiling + auto-parallelization) offline,
producing a different generated source (``<bench>_pluto_nb.cpp``) that clang then
compiles. This is unlike ``polly``, which is just a clang compile flag on the SAME
C++ source as the ``llvm`` flavor. Pluto reuses all of the native backend's wrapper
/ C-ABI machinery by subclassing it.
"""

from optarena.frameworks.native_framework import NativeFramework


class PlutoFramework(NativeFramework):
    """The Pluto polyhedral native backend (base ``pluto``). A thin subclass of
    :class:`NativeFramework`: the C-ABI wrapper, on-demand source generation, and
    host-side timing are all inherited, and the inherited ``kernel_<framework>``
    rule dispatches to the wrapper's ``kernel_pluto`` entry point. Kept a first-class
    framework (its own class + ``base``) rather than a flavor of the base-language
    native backend, because the polyhedral source-to-source transform is a genuinely
    different toolchain."""
