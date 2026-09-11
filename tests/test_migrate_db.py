# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pre-identity shards rewritten into the current schema.

The migration derives an arm's experiment, model, language, device and packet ONCE so that nothing
downstream parses a run_id again. Two things have to hold or a campaign's numbers move: every arm
resolves to exactly one identity, and a row that ALREADY carries an identity is copied untouched.
"""

import importlib.util
import pathlib
import sqlite3

import pytest

from hpcagent_bench.harness import recording

SPEC = importlib.util.spec_from_file_location(
    "migrate_db", pathlib.Path(recording.__file__).parents[2] / "scripts" / "migrate_db.py"
)
migrate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migrate)


@pytest.mark.parametrize(
    "arm, want",
    [
        ("cpf-llr-focus40-qwen38-c", ("llr-focus40", "qwen38", "c", "cpu", "")),
        ("cpf-llr-focus40-qwen38-c-cpfsrc", ("llr-focus40", "qwen38", "c", "cpu", "cpfsrc")),
        ("cpf-llr-focus40-oss120b-fortran-skills", ("llr-focus40", "oss120b", "fortran", "cpu", "lang-skills")),
        ("gpu-llr-focus40-qwen38-hip", ("llr-focus40", "qwen38", "hip", "gpu", "")),
        ("gpu-llr-focus40-qwen38-triton-skills", ("llr-focus40", "qwen38", "triton", "gpu", "lang-skills")),
        # an offload arm's language is c; the directive model is what tells it from a CPU C arm
        ("gpu-llr-focus40-qwen38-c-openmp", ("llr-focus40", "qwen38", "c", "gpu", "openmp-offload")),
        (
            "gpu-llr-focus40-oss120b-c-openmp-skills",
            ("llr-focus40", "oss120b", "c", "gpu", "lang-skills+openmp-offload"),
        ),
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


@pytest.mark.parametrize("arm", ["cpf-llr-focus40-nosuchmodel-c", "nosuchcampaign-qwen38-c", "cpf-llr-focus40-qwen38"])
def test_an_unrecognised_arm_raises_rather_than_guessing(arm):
    with pytest.raises(ValueError):
        migrate.parse_arm(arm)


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


def test_the_arm_language_replaces_the_bodys_claim(tmp_path):
    """An old row's `language` is what the agent said; it moves aside so the column can be grouped."""
    shard = str(tmp_path / "old0.db")
    _old_shard(
        shard, [("gpu-llr-focus40-qwen38-triton.0", 1, "gemm", "XL", "float64", "zzz", "zzz", "any", 1, 0, 2.0, 1)]
    )
    out = str(tmp_path / "out.db")
    dest = recording.connect(out)
    src = sqlite3.connect(shard)
    migrate.copy_table(dest, src, "calls", lambda r: migrate.parse_arm(migrate.arm_of(r)))
    dest.commit()
    assert list(dest.execute("SELECT language, delivered_language, model, device, packet FROM calls")) == [
        ("triton", "zzz", "qwen38", "gpu", "")
    ]
    src.close()
    dest.close()


def test_a_row_that_already_carries_an_identity_is_copied_untouched(tmp_path):
    """Rewriting one would put the arm's language into delivered_language and lose the claim."""
    shard = str(tmp_path / "new0.db")
    conn = recording.connect(shard)
    conn.execute(
        "INSERT INTO calls(run_id, ts, benchmark, preset, datatype, language, delivered_language, source_mode, "
        "round, tokens, speedup, correct, experiment, model, device, packet, arm) "
        "VALUES ('a.0',1,'gemm','XL','float64','fortran','python','restricted',1,0,2.0,1,"
        "'llr-focus40','qwen38','cpu','lang-skills','cpf-llr-focus40-qwen38-fortran-skills')"
    )
    conn.commit()
    conn.close()
    out = str(tmp_path / "out.db")
    dest = recording.connect(out)
    src = sqlite3.connect(shard)

    def refuse(_run_id):
        raise AssertionError("a tagged row must not be re-derived")

    migrate.copy_table(dest, src, "calls", refuse)
    dest.commit()
    assert list(dest.execute("SELECT language, delivered_language, packet FROM calls")) == [
        ("fortran", "python", "lang-skills")
    ]
    src.close()
    dest.close()
