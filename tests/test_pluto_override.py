# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A tracked ORIGINAL-PolyBench scop (``<base>_pluto_reference.c``) replaces the translator's generated
one -- and does so by skipping generation entirely, not by generating then swapping. Every consumer
(``pluto_transform.scop_inputs``, the timed build, the numerical oracle, the affine survey) goes
through the ONE choke point pinned here."""
import pathlib

from hpcagent_bench import pluto_transform
from hpcagent_bench.support.collect import pluto_survey


def _generated(cpp_backend: pathlib.Path, base: str) -> pathlib.Path:
    cpp_backend.mkdir(parents=True, exist_ok=True)
    scop = cpp_backend / f"{base}_fp64_pluto_input.c"
    scop.write_text("#pragma scop\nx = 1;\n#pragma endscop\n")
    return scop


def test_override_replaces_the_generated_scop_and_never_touches_it(tmp_path) -> None:
    """A ``<base>_pluto_reference.c`` beside the kernel wins over the generated ``fp64`` scop, and the
    generated file is neither read into a copy nor overwritten -- it is not even opened."""
    bench_dir = tmp_path / "kern"
    cpp_backend = bench_dir / "cpp_backend"
    generated = _generated(cpp_backend, "kern")
    before_mtime, before_text = generated.stat().st_mtime, generated.read_text()
    override = bench_dir / "kern_pluto_reference.c"
    override.write_text("#pragma scop\ny = 2;\n#pragma endscop\n")

    resolved = pluto_transform.scop_inputs(cpp_backend, "kern", bench_dir=bench_dir)

    assert resolved == [override]
    assert generated.stat().st_mtime == before_mtime, "the generated scop was rewritten"
    assert generated.read_text() == before_text, "the generated scop's content changed"
    assert sorted(p.name for p in cpp_backend.iterdir()) == [generated.name], "generation ran anyway"


def test_scop_inputs_falls_back_to_generated_without_an_override(tmp_path) -> None:
    """A kernel with no tracked override still resolves to the translator's generated scop."""
    bench_dir = tmp_path / "kern2"
    cpp_backend = bench_dir / "cpp_backend"
    generated = _generated(cpp_backend, "kern2")

    resolved = pluto_transform.scop_inputs(cpp_backend, "kern2", bench_dir=bench_dir)

    assert resolved == [generated]


def test_override_transform_output_stays_out_of_the_tracked_source_dir(tmp_path) -> None:
    """polycc's output for an override lands under the kernel's gitignored ``cpp_backend``, not
    beside the tracked ``_pluto_reference.c`` -- a build artifact must never dirty ``git status``."""
    bench_dir = tmp_path / "kern3"
    bench_dir.mkdir()
    override = bench_dir / "kern3_pluto_reference.c"
    override.write_text("#pragma scop\nz = 3;\n#pragma endscop\n")

    out = pluto_transform.transformed_path(override)

    assert out.parent == bench_dir / "cpp_backend"
    assert out.name == "kern3_fp64_pluto.c"


def test_classify_affine_never_invokes_the_emitter_for_an_override_backed_kernel(monkeypatch) -> None:
    """gemm carries a tracked override (see the ``gemm`` benchmark dir). Classifying its affine
    status must resolve straight from that file -- the translator is never asked to emit anything,
    proven by making the emit call itself fail loudly if reached."""

    def must_not_run(*args, **kwargs):
        raise AssertionError("the translator was invoked for an override-backed kernel")

    monkeypatch.setattr(pluto_survey, "_emit", must_not_run)

    has_scop, affine, reason = pluto_survey.classify_affine("gemm")

    assert has_scop is True
    assert affine is True
    assert reason is None


def test_oracle_pluto_leg_transforms_the_override_path_not_a_generated_copy(monkeypatch) -> None:
    """The numerical oracle's pluto leg feeds polycc the override file ITSELF -- captured off the
    real ``run_polycc`` call -- rather than any file ``scop_inputs`` might have materialized."""
    import tests.numerical_oracle as oracle
    from hpcagent_bench.emit_bridge import legacy_bench_info_dict
    from hpcagent_bench.spec import BenchSpec

    info = legacy_bench_info_dict(BenchSpec.load("gemm"))["benchmark"]
    bench_dir = oracle.REPO / "hpcagent_bench" / "benchmarks" / info["relative_path"]
    override = pluto_transform.override_source(bench_dir, "gemm")
    assert override is not None, "gemm's tracked override is missing -- fixture assumption broken"
    seen = {}

    def capture(scop, out, **kwargs):
        seen["scop"] = scop
        return [], type("R", (), {"returncode": 1, "stderr": "stop-here"})()

    monkeypatch.setattr(oracle.pluto_transform, "run_polycc", capture)
    monkeypatch.setattr(oracle.pluto_transform, "polycc_exe", lambda: "/nonexistent/polycc")

    oracle._run_pluto(pathlib.Path("/nonexistent-tdp"), "gemm", "fp64", {}, {}, {}, {}, (), 0.0, 0.0, "ok", bench_dir,
                      "gemm")

    assert seen["scop"] == override
