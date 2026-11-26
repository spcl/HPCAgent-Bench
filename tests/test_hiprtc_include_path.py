# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The include list cupy hands HIPRTC must not carry clang's ``cuda_wrappers`` directory (a measured rule), and
a guard fires when the cupy hook moves."""

import pytest

from hpcagent_bench import modules
from hpcagent_bench.harness.native_call import CLANG_CUDA_WRAPPERS, hiprtc_include_dirs, repair_hiprtc_include_path

# The list as cupy really scraped it on MI300A (ROCm 7.2.3, gcc 16), wrapper dir at index 1.
SCRAPED = (
    "/opt/spack/intel-tbb-2023.1.0/include",
    "/opt/rocm-7.2.3/lib/llvm/lib/clang/22/include/cuda_wrappers",
    "/usr/lib/gcc/x86_64-linux-gnu/16/../../../../include/c++/16",
    "/usr/lib/gcc/x86_64-linux-gnu/16/../../../../include/x86_64-linux-gnu/c++/16",
    "/opt/rocm-7.2.3/lib/llvm/lib/clang/22/include",
    "/usr/include",
)


def test_wrapper_directory_is_removed_and_every_other_directory_survives_in_order() -> None:
    # Only the one entry goes. Dropping more would take libstdc++ or the ROCm headers with it,
    # and REORDERING is not a fix here -- the wrapper dir fails from any position -- so the
    # surviving order must be the scraped order.
    kept = hiprtc_include_dirs(SCRAPED)
    assert not any(CLANG_CUDA_WRAPPERS in d for d in kept)
    assert kept == tuple(d for d in SCRAPED if CLANG_CUDA_WRAPPERS not in d)
    once = hiprtc_include_dirs(SCRAPED)
    assert hiprtc_include_dirs(once) == once


class FakeRuntime:
    def __init__(self, is_hip: bool) -> None:
        self.is_hip = is_hip


class FakeCupy:
    def __init__(self, is_hip: bool) -> None:
        self.cuda = type("cuda", (), {"runtime": FakeRuntime(is_hip)})


def test_missing_cupy_hook_raises_rather_than_skipping_silently() -> None:
    # The repair hangs on a cupy PRIVATE name. If it disappears, a silent skip would come back
    # as an inscrutable HIPRTC error hours later inside a device grade, so it must fail loudly.
    environment = pytest.importorskip("cupy._environment", reason="needs a cupy install")
    saved = modules.optional(environment, "_get_hipcc_include_dirs")
    if saved is None:
        pytest.skip("cupy build has no _get_hipcc_include_dirs to remove")
    del environment._get_hipcc_include_dirs
    try:
        with pytest.raises(RuntimeError, match="cuda_wrappers"):
            repair_hiprtc_include_path(FakeCupy(is_hip=True))
    finally:
        environment._get_hipcc_include_dirs = saved
