# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A shard's rank must finish and record a verdict for every kernel it owns, never kill its siblings.

Job 633849 died because a rank returned nonzero for a per-kernel render failure inside its own
shard; srun's kill-on-bad-exit took the other ranks down mid-render, and the roster-wide check that
ran afterward mistook their unfinished kernels for misses. The fix moves failure reporting to the
recorded verdict (:mod:`hpcagent_bench.cpf_cache`) and leaves the rank's own exit status to signal
only an internal error.
"""

from __future__ import annotations

import argparse
import pathlib
import re

import pytest

from hpcagent_bench import cpf_bridge, cpf_cache, cpf_prerender

SBATCH = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "prerender_cpf.sbatch"


class FakeSpec:
    """Stands in for a loaded BenchSpec: prerender() only reads short_name off it."""

    def __init__(self, short_name: str) -> None:
        self.short_name = short_name


def args_for(cache: pathlib.Path, view: pathlib.Path, kernels: str) -> argparse.Namespace:
    return argparse.Namespace(
        cache=cache, view=view, kernels=kernels, target="cpu", precision="", rank=0, ranks=1, timeout=None
    )


def test_a_shard_with_a_load_failure_and_a_render_failure_still_exits_zero(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every kernel in the shard gets a verdict; the rank hands back 0 whatever those verdicts are."""
    package, before = tmp_path / "dace", "digest"
    package.mkdir()
    view, cache = tmp_path / "view", tmp_path / "cache"
    cpf_cache.open_view(view, cache, "cpu", before)
    monkeypatch.setattr(cpf_cache, "source_digest", lambda pkg: before)

    def fake_load(short_name: str) -> FakeSpec:
        if short_name == "missing_kernel":
            raise KeyError(short_name)
        return FakeSpec(short_name)

    monkeypatch.setattr(cpf_prerender.BenchSpec, "load", classmethod(lambda cls, short_name: fake_load(short_name)))

    def fake_prerender_kernel(spec: FakeSpec, cache_root: pathlib.Path, **kwargs: object) -> dict[str, object]:
        if spec.short_name == "broken_render":
            bad = {"key": "badkey", "verdict": "timeout", "error": "render exceeded budget"}
            return {"results": {"c": {"form": bad, "dropin": bad}, "c++": {"form": bad, "dropin": bad}}}
        key = f"{spec.short_name}key"
        cpf_cache.publish(
            cache_root, key, {"kernel": spec.short_name}, (f"{spec.short_name}.c", "// ok\n"), ("binding.json", "{}\n")
        )
        ok = {"key": key, "verdict": "ok", "cached": False}
        return {"results": {"c": {"form": ok, "dropin": ok}, "c++": {"form": ok, "dropin": ok}}}

    monkeypatch.setattr(cpf_bridge, "prerender_kernel", fake_prerender_kernel)

    args = args_for(cache, view, "missing_kernel,broken_render,ok_kernel")
    assert cpf_prerender.prerender(args, package, before) == 0

    assert cpf_cache.missing(view, ["ok_kernel"], "c", "fp64", "form") == []
    assert cpf_cache.missing(view, ["missing_kernel"], "c", "fp64", "form") != []
    assert cpf_cache.missing(view, ["broken_render"], "c", "fp64", "form") != []


def test_dace_edited_mid_run_still_withdraws_and_fails_the_rank(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed dace source is a real internal error, unlike a per-kernel render failure: it stays nonzero."""
    package, before = tmp_path / "dace", "before"
    package.mkdir()
    view, cache = tmp_path / "view", tmp_path / "cache"
    cpf_cache.open_view(view, cache, "cpu", before)
    monkeypatch.setattr(cpf_cache, "source_digest", lambda pkg: "after")
    monkeypatch.setattr(cpf_prerender.BenchSpec, "load", classmethod(lambda cls, short_name: FakeSpec(short_name)))

    def fake_prerender_kernel(spec: FakeSpec, cache_root: pathlib.Path, **kwargs: object) -> dict[str, object]:
        ok = {"key": "okkey", "verdict": "ok", "cached": False}
        return {"results": {"c": {"form": ok, "dropin": ok}, "c++": {"form": ok, "dropin": ok}}}

    monkeypatch.setattr(cpf_bridge, "prerender_kernel", fake_prerender_kernel)

    args = args_for(cache, view, "ok_kernel")
    assert cpf_prerender.prerender(args, package, before) == 3


def test_srun_cannot_kill_a_sibling_shard_on_a_bad_exit() -> None:
    """A rank exiting nonzero (an internal error) must never take the other shards down with it."""
    text = SBATCH.read_text()
    assert re.search(r"^srun\b.*--kill-on-bad-exit=0", text, re.MULTILINE)


def test_the_roster_check_runs_once_after_srun_returns() -> None:
    """The 40-kernel view check belongs to the batch script, after every shard, never inside a rank."""
    text = SBATCH.read_text()
    srun_at = text.index("\nsrun ")
    check_at = text.index("cpf_cache check")
    assert check_at > srun_at
    assert text.count("cpf_cache check") == 1
    assert "cpf_prerender" in text[:check_at]


def test_the_job_exit_status_follows_the_roster_check_not_the_raw_srun_status() -> None:
    """A rank's fail verdicts must not, by themselves, decide whether the job is reported healthy."""
    text = SBATCH.read_text()
    tail = text[text.index("cpf_cache check") :]
    assert re.search(r"exit\s+\$\(\(.*checks_status", tail)
