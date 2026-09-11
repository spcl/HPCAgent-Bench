# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pre-identity shards rewritten into the current schema.

The migration derives an arm's experiment, model, language, device and packet ONCE so that nothing
downstream parses a run_id again. Two things have to hold or a campaign's numbers move: every arm
resolves to exactly one identity, and a row that ALREADY carries an identity is copied untouched.
"""

import sys
import importlib.util
import pathlib
import sqlite3

import pytest

from hpcagent_bench.harness import recording

SPEC = importlib.util.spec_from_file_location(
    "migrate_db", pathlib.Path(recording.__file__).parents[2] / "scripts" / "migrate_db.py"
)
migrate = importlib.util.module_from_spec(SPEC)
# Registered BEFORE exec: dataclasses resolves a string annotation through
# sys.modules[cls.__module__], which is None for a module loaded by path alone.
sys.modules[SPEC.name] = migrate
SPEC.loader.exec_module(migrate)


@pytest.mark.parametrize(
    "arm, want",
    [
        ("cpf-llr-focus40-qwen38-c", ("llr-focus40", "qwen38", "c", "cpu", "")),
        ("cpf-llr-focus40-qwen38-c-cpfsrc", ("llr-focus40", "qwen38", "c", "cpu", "cpfsrc")),
        ("cpf-llr-focus40-oss120b-fortran-skills", ("llr-focus40", "oss120b", "fortran", "cpu", "lang-skills")),
        ("gpu-llr-focus40-qwen38-hip", ("llr-focus40", "qwen38", "hip", "gpu", "")),
        ("gpu-llr-focus40-qwen38-triton-skills", ("llr-focus40", "qwen38", "triton", "gpu", "lang-skills")),
        # an offload arm carries NO packet: device=gpu with language=c already says offload, so
        # naming it again would split one condition into two group keys
        ("gpu-llr-focus40-qwen38-c-openmp", ("llr-focus40", "qwen38", "c", "gpu", "")),
        ("gpu-llr-focus40-oss120b-c-openmp-skills", ("llr-focus40", "oss120b", "c", "gpu", "lang-skills")),
        ("gpuv4-llr40-qwen38-pytriton", ("llr-focus40-v11", "qwen38", "triton", "gpu", "")),
        ("git-scicomp-kimi27sglang-repo", ("git-scicomp", "kimi27sglang", "c", "cpu", "repo")),
        # campaign VERSIONS are separate experiments; WAVES of one version share it
        ("llr40v11-qwen38-c", ("llr-focus40-v11", "qwen38", "c", "cpu", "")),
        ("v11w2-qwen38-c", ("llr-focus40-v11", "qwen38", "c", "cpu", "")),
    ],
)
def test_an_arm_resolves_to_one_identity(arm, want):
    assert migrate.parse_arm(arm) == want


@pytest.mark.parametrize("arm", ["adhoc", "test-run", "gpusmoke5-hip", "gpusmoke5-hip-cpf"])
def test_a_run_that_belongs_to_no_experiment_is_dropped(arm):
    """A smoke is not an experiment, and an ad-hoc run names no arm; neither may join a campaign."""
    assert migrate.parse_arm(arm) is None


@pytest.mark.parametrize("arm", ["cpf-llr-focus40-nosuchmodel-c", "nosuchcampaign-qwen38-c"])
def test_an_unrecognised_arm_raises_rather_than_guessing(arm):
    """A campaign or a model the table does not know is a migration that would invent an identity."""
    with pytest.raises(ValueError):
        migrate.parse_arm(arm)


def test_an_arm_that_names_no_language_takes_its_campaign_default(tmp_path):
    """git-scicomp's arms are `-kernel` and `-repo`: they name the treatment and leave the language
    implicit, so without a default the control of that A/B has no language while its treatment does
    and the pair stops joining."""
    assert migrate.parse_arm("cpf-llr-focus40-qwen38") == ("llr-focus40", "qwen38", "c", "cpu", "")


def _old_shard(path, rows):
    """A shard as the judge wrote it before the identity columns: no model, language is the claim."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE calls (id INTEGER PRIMARY KEY, run_id TEXT, ts INTEGER, benchmark TEXT, preset TEXT, "
        "datatype TEXT, language TEXT, delivered_language TEXT, source_mode TEXT, round INTEGER, "
        "tokens INTEGER, speedup REAL, correct INTEGER)"
    )
    conn.executemany(
        "INSERT INTO calls(run_id, ts, benchmark, preset, datatype, language, delivered_language, source_mode, "
        "round, tokens, speedup, correct) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


def test_the_bodys_claim_is_dropped_rather_than_carried_across(tmp_path):
    """A pre-identity row's `language` is what the AGENT said, and bodies arrived naming py, zzz and
    a file path. It is not migrated anywhere: the column a figure groups by comes from the run, and
    the claim recorded a naming artifact with no consequence that is not already in `status`."""
    shard = str(tmp_path / "old0.db")
    _old_shard(
        shard,
        [("gpu-llr-focus40-qwen38-triton.0", 1, "gemm", "XL", "float64", "zzz", "", "any", 1, 0, 2.0, 1)],
    )
    dest = recording.connect(str(tmp_path / "out.db"))
    src = sqlite3.connect(shard)
    try:
        identity = lambda r: migrate.parse_arm(migrate.arm_of(r))  # noqa: E731 -- one expression
        run_ids = [r[0] for r in src.execute("SELECT DISTINCT run_id FROM calls")]
        migrate.write_runs(dest, run_ids, identity)
        migrate.copy_table(dest, src, "calls", identity)
        dest.commit()
        columns = {r[1] for r in dest.execute("PRAGMA table_info(calls)")}
        assert "delivered_language" not in columns and "language" not in columns
        assert list(dest.execute("SELECT language FROM runs")) == [("triton",)]
    finally:
        src.close()
        dest.close()


def test_the_identity_is_written_once_per_run_not_onto_every_row(tmp_path):
    """The same fact on every measurement row could disagree between the three tables for one run."""
    shard = str(tmp_path / "old0.db")
    rows = [
        (f"gpu-llr-focus40-qwen38-triton.n0.p0.w{w}", 1, "gemm", "XL", "float64", "zzz", "", "any", 1, 0, 2.0, 1)
        for w in (0, 0, 1)
    ]
    _old_shard(shard, rows)
    dest = recording.connect(str(tmp_path / "out.db"))
    src = sqlite3.connect(shard)
    try:
        migrate.copy_table(dest, src, "calls", lambda r: migrate.parse_arm(migrate.arm_of(r)))
        run_ids = [r[0] for r in src.execute("SELECT run_id FROM calls")]
        assert migrate.write_runs(dest, run_ids, lambda r: migrate.parse_arm(migrate.arm_of(r))) == 2
        dest.commit()
        assert sorted(dest.execute("SELECT run_id, model, language, device, packet, rep FROM runs")) == [
            ("gpu-llr-focus40-qwen38-triton.n0.p0.w0", "qwen38", "triton", "gpu", "", 1),
            ("gpu-llr-focus40-qwen38-triton.n0.p0.w1", "qwen38", "triton", "gpu", "", 1),
        ]
        assert list(dest.execute("SELECT COUNT(*) FROM calls"))[0][0] == 3
    finally:
        src.close()
        dest.close()


def test_a_migrated_run_claims_the_first_repetition_rather_than_inventing_one(tmp_path):
    """A run id carries no repetition, so recovering one would be making it up."""
    dest = recording.connect(str(tmp_path / "out.db"))
    try:
        migrate.write_runs(dest, ["cpf-llr-focus40-qwen38-c.n0.p0.w0"], lambda r: migrate.parse_arm(migrate.arm_of(r)))
        dest.commit()
        assert list(dest.execute("SELECT rep FROM runs")) == [(1,)]
    finally:
        dest.close()


def test_a_row_whose_run_has_no_identity_is_dropped(tmp_path):
    """An ad-hoc or smoke run belongs to no experiment; keeping its rows would pool them with one."""
    shard = str(tmp_path / "old0.db")
    _old_shard(shard, [("adhoc", 1, "gemm", "XL", "float64", "c", "", "any", 1, 0, 2.0, 1)])
    dest = recording.connect(str(tmp_path / "out.db"))
    src = sqlite3.connect(shard)
    try:
        copied, dropped = migrate.copy_table(dest, src, "calls", lambda r: migrate.parse_arm(migrate.arm_of(r)))
        assert (copied, dropped) == (0, 1)
    finally:
        src.close()
        dest.close()
