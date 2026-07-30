# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-kernel framework-autogen cache + its freshness guard (hpcagent_bench.framework_cache).

Four features, one contract -- a cache entry is served ONLY for the source it was built from:

* the autogen SOURCE cache: :func:`hpcagent_bench.autogen.ensure` emits a sibling once, reuses it on the
  next call WITHOUT re-emitting, and REGENERATES it the moment the numpy reference changes (the
  emit-counter test -- the whole point, since a stale sibling would silently miscompile a graded run);
* the DaCe SDFG cache: a base SDFG round-trips through a compressed ``.sdfgz`` (save + release-agnostic
  ``from_file``), a hit reloads the byte-identical graph, a fingerprint mismatch or a corrupt file MISSES
  (degrades to a rebuild, never a crash);
* the base-class hook: :meth:`Framework.build_with_cache` is a clean no-op default -- it just calls the
  builder, caching nothing -- so the interface is uniform while only DaCe persists anything;
* the ``.gitkeep``/gitignore rule: each kernel's ``.cache/`` dir is kept but its contents are ignored.

The DaCe tests ``importorskip('dace')``; the rest are pure pathlib/hashlib and always run.
"""
import subprocess

import pytest

from hpcagent_bench import framework_cache as fc

# --- freshness key --------------------------------------------------------------------------


def test_source_fingerprint_tracks_source_and_bench_info(tmp_path):
    numpy_py = tmp_path / "x_numpy.py"
    numpy_py.write_text("def kernel(A):\n    return A\n")
    base = fc.source_fingerprint(numpy_py, b"bench-info-1")
    assert fc.source_fingerprint(numpy_py, b"bench-info-1") == base  # deterministic
    numpy_py.write_text("def kernel(A):\n    return A + 1\n")
    assert fc.source_fingerprint(numpy_py, b"bench-info-1") != base  # source change -> new key
    assert fc.source_fingerprint(numpy_py, b"bench-info-2") != fc.source_fingerprint(numpy_py, b"bench-info-1")


def test_kernel_cache_dir_creates_dir_with_gitkeep(tmp_path):
    kdir = tmp_path / "kern"
    kdir.mkdir()
    cache = fc.kernel_cache_dir(kdir)
    assert cache == kdir / ".cache" and cache.is_dir()
    assert (cache / ".gitkeep").exists(), "the .cache/ dir must be kept via a .gitkeep"


# --- autogen SOURCE cache: hit / restore / INVALIDATE ---------------------------------------


def test_generated_source_cache_hit_restore_and_invalidation(tmp_path):
    kdir = tmp_path / "k"
    kdir.mkdir()
    cache = fc.kernel_cache_dir(kdir)
    canonical = kdir / "k_dace.py"
    canonical.write_text("# gen v1\nX = 1\n")
    fp1 = fc.fingerprint_bytes(b"source-v1")

    assert fc.load_generated(cache, canonical, fp1) is False, "empty cache must MISS"
    fc.save_generated(cache, canonical, fp1)
    assert fc.load_generated(cache, canonical, fp1) is True, "unchanged source must HIT"

    canonical.unlink()  # a hit RESTORES the sibling from the cache without re-emitting
    assert fc.load_generated(cache, canonical, fp1) is True
    assert canonical.read_text() == "# gen v1\nX = 1\n"

    # The guard: a changed source fingerprint MISSES -- a stale entry is never served.
    assert fc.load_generated(cache, canonical, fc.fingerprint_bytes(b"source-v2")) is False


def _widget_kernel(benchmarks_root):
    """Write a minimal valid kernel (manifest + numpy reference) under a tmp benchmarks root."""
    kdir = benchmarks_root / "widget"
    kdir.mkdir(parents=True)
    (kdir / "widget.yaml").write_text("name: widget\n"
                                      "relative_path: widget\n"
                                      "kind: microkernel\n"
                                      "parameters:\n"
                                      "  S:\n"
                                      "    N: 8\n"
                                      "output_args:\n"
                                      "- C\n"
                                      "init:\n"
                                      "  input_args:\n"
                                      "  - N\n"
                                      "  func_name: initialize\n"
                                      "  arrays:\n"
                                      "    C:\n"
                                      "      shape: (N,)\n"
                                      "    A:\n"
                                      "      shape: (N,)\n"
                                      "array_args:\n"
                                      "- C\n"
                                      "- A\n")
    (kdir / "widget_numpy.py").write_text("def kernel(C, A):\n    C[:] = A\n")
    return kdir


def test_ensure_emits_once_reuses_then_reemits_on_source_change(tmp_path, monkeypatch):
    """The maintainer's autogen-cache proof: first ensure() emits + caches, a second reuses the cache
    WITHOUT re-emitting (emit counter unchanged), and mutating the numpy source re-emits (invalidation)."""
    from hpcagent_bench import autogen, paths
    from hpcagent_bench.spec import KERNELS
    from numpyto_common.emit_io import write_generated

    benchmarks = tmp_path / "benchmarks"
    kdir = _widget_kernel(benchmarks)

    emits = {"n": 0}

    def fake_emit_targets(spec, targets):
        emits["n"] += 1
        out = {}
        for t in targets:
            canonical = paths.BENCHMARKS / spec.relative_path / f"{spec.module_name}_{t}.py"
            write_generated(canonical, f"# body for {t}\n", source=f"{spec.module_name}_numpy.py")
            out[t] = "ok"
        return out

    original_root = paths.BENCHMARKS
    monkeypatch.setattr(autogen, "emit_targets", fake_emit_targets)
    try:
        paths.BENCHMARKS = benchmarks
        KERNELS.refresh()

        autogen.ensure("widget", ["dace"])  # cold: emit + cache
        assert emits["n"] == 1
        assert (kdir / "widget_dace.py").exists()

        autogen.ensure("widget", ["dace"])  # warm: cache HIT, no re-emit
        assert emits["n"] == 1

        # Mutate the numpy reference -> fingerprint changes -> the cache is INVALIDATED, not stale-loaded.
        (kdir / "widget_numpy.py").write_text("def kernel(C, A):\n    C[:] = A * 2\n")
        KERNELS.refresh()
        autogen.ensure("widget", ["dace"])
        assert emits["n"] == 2, "a changed numpy source must re-emit, never serve a stale sibling"
    finally:
        paths.BENCHMARKS = original_root
        KERNELS.refresh()


def test_ensure_never_touches_a_hand_override(tmp_path, monkeypatch):
    """A hand-written override (no autogen marker) is never emitted and never cached -- left byte-for-byte."""
    from hpcagent_bench import autogen, paths
    from hpcagent_bench.spec import KERNELS

    benchmarks = tmp_path / "benchmarks"
    kdir = _widget_kernel(benchmarks)
    override = kdir / "widget_dace.py"
    override.write_text("# my own hand impl\nX = 42\n")

    called = {"n": 0}

    def fake_emit_targets(spec, targets):
        called["n"] += 1
        return {t: "ok" for t in targets}

    original_root = paths.BENCHMARKS
    monkeypatch.setattr(autogen, "emit_targets", fake_emit_targets)
    try:
        paths.BENCHMARKS = benchmarks
        KERNELS.refresh()
        autogen.ensure("widget", ["dace"])
        assert called["n"] == 0, "an override must never trigger an emit"
        assert override.read_text() == "# my own hand impl\nX = 42\n"
        assert not (kdir / ".cache" / "widget_dace.py").exists(), "an override must never be cached"
    finally:
        paths.BENCHMARKS = original_root
        KERNELS.refresh()


# --- base-class hook: a clean no-op default -------------------------------------------------


def test_base_framework_cache_hook_is_a_noop():
    from hpcagent_bench.frameworks import Framework

    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return "ARTIFACT"

    fw = Framework("numpy")
    assert fw.build_with_cache(None, "cpu", build) == "ARTIFACT"
    assert fw.build_with_cache(None, "gpu", build) == "ARTIFACT"
    # No cache on the base: it just calls the builder every time, and touches no filesystem.
    assert calls["n"] == 2


# --- DaCe SDFG cache: .sdfgz round-trip, invalidation, corruption tolerance ------------------


def test_sdfg_cache_roundtrip_invalidation_and_corruption(tmp_path, monkeypatch):
    pytest.importorskip("dace")
    from hpcagent_bench.frameworks import Benchmark, generate_framework

    # implementations() -> ensure() populates a SOURCE cache; redirect it into tmp so the real
    # gemm/.cache stays clean, then use a plain tmp dir for the SDFG cache calls below.
    monkeypatch.setattr(fc, "kernel_cache_dir", lambda kdir: tmp_path / "src")
    (tmp_path / "src").mkdir()
    fw = generate_framework("dace_cpu")
    fw.set_datatype("float64")
    bench = Benchmark("gemm")
    ct_impl = fw.implementations(bench)[0][0]
    fresh = ct_impl.to_sdfg(simplify=False)

    cache = tmp_path / "sdfg"
    cache.mkdir()
    fp = fc.fingerprint_bytes(b"gemm-fp64-v1")

    assert fc.load_sdfg(cache, "gemm", "cpu", fp) is None, "empty cache must MISS"
    fc.save_sdfg(cache, "gemm", "cpu", fp, fresh)
    saved = cache / "gemm_cpu.sdfgz"
    assert saved.exists(), "the base SDFG must be saved as <module>_cpu.sdfgz"

    loaded = fc.load_sdfg(cache, "gemm", "cpu", fp)
    assert loaded is not None
    # A hit reloads the SAME graph a fresh parse produces -> identical pipelines -> identical grading.
    assert loaded.hash_sdfg() == fresh.hash_sdfg()

    # A changed fingerprint MISSES even though the file is present (source/precision moved on).
    assert fc.load_sdfg(cache, "gemm", "cpu", fc.fingerprint_bytes(b"gemm-fp64-v2")) is None

    # A corrupt/incompatible .sdfgz degrades to a rebuild (None), never a crash.
    saved.write_bytes(b"this is not a valid sdfgz")
    assert fc.load_sdfg(cache, "gemm", "cpu", fp) is None


def test_dace_build_with_cache_bypasses_build_on_hit_and_invalidates_on_precision(tmp_path, monkeypatch):
    pytest.importorskip("dace")
    from hpcagent_bench.frameworks import Benchmark, generate_framework

    cache = tmp_path / ".cache"
    cache.mkdir()
    # Redirect BOTH the source cache and the SDFG cache into tmp so the real gemm/.cache stays clean.
    monkeypatch.setattr(fc, "kernel_cache_dir", lambda kdir: cache)

    fw = generate_framework("dace_cpu")
    fw.set_datatype("float64")
    bench = Benchmark("gemm")
    ct_impl = fw.implementations(bench)[0][0]

    builds = {"n": 0}

    def build():
        builds["n"] += 1
        return ct_impl.to_sdfg(simplify=False)

    first = fw.build_with_cache(bench, "cpu", build)  # MISS -> build + save
    assert builds["n"] == 1
    assert (cache / "gemm_cpu.sdfgz").exists()

    def must_not_run():
        raise AssertionError("build ran on a cache hit -- the load path was not taken")

    second = fw.build_with_cache(bench, "cpu", must_not_run)  # HIT -> build bypassed
    assert builds["n"] == 1
    assert second.hash_sdfg() == first.hash_sdfg()

    # A precision change moves the fingerprint -> MISS -> rebuild (never a stale fp64 SDFG for fp32).
    fw.set_datatype("float32")
    fw.build_with_cache(bench, "cpu", build)
    assert builds["n"] == 2


# --- .gitignore: keep the dir, ignore the contents ------------------------------------------


def test_cache_tree_is_fully_gitignored():
    """The whole per-kernel ``.cache/`` is ignored -- artifacts, sidecars AND the ``.gitkeep``.

    Nothing needs to be tracked to keep the directory alive (``kernel_cache_dir`` mkdirs on demand),
    and re-including the ``.gitkeep`` would add one empty file per kernel (356 of them) plus an
    untracked ``.cache/`` in ``git status`` for every kernel ever run."""
    from hpcagent_bench import paths

    repo = paths.ROOT
    if subprocess.run(["git", "-C", str(repo), "rev-parse", "--is-inside-work-tree"],
                      capture_output=True).returncode != 0:
        pytest.skip("not a git checkout")

    base = "hpcagent_bench/benchmarks/hpc/dense_linear_algebra/gemm/.cache"

    def ignored(rel):
        return subprocess.run(["git", "-C", str(repo), "check-ignore", "-q", rel], capture_output=True).returncode == 0

    assert ignored(base + "/gemm_cpu.sdfgz"), "cache artifacts must be gitignored"
    assert ignored(base + "/gemm_cpu.sdfgz.fp"), "fingerprint sidecars must be gitignored"
    assert ignored(base + "/gemm_dace.py"), "cached generated siblings must be gitignored"
    assert ignored(base + "/.gitkeep"), "the .gitkeep must be ignored too -- 356 kernels, dir is made on demand"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
