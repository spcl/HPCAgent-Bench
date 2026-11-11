# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""F18, resumed-episode cost: what a task's token cost counts when an episode crashed and resumed.

Since v0.4.0 a crashed episode resumes in place, and the driver bills every attempt's tokens
(``token_cost.episode_totals`` sums the earlier ones into ``tokens_effective_crashed``; the results DB
carries them in ``episodes.crashed_effective_tokens``, extracted as the task row's ``tokens_crashed``).
The cost aggregation read the task row's ``tokens`` alone -- the FINAL attempt's -- so a resumed
episode was under-costed. ``cost.attempts`` picks the reading: ``all`` (the default) or ``final`` (the
reading the paper's published numbers used).

THE FIXTURE IS THE PRODUCTION SHAPE end to end once (sqlite): a driver record with two crashed
attempts (30 + 20 tokens, summed upstream into one field), merged into a results DB, extracted, and
read back as an observations frame; the switch matrix then runs on frames of the same shape.
"""

import contextlib
import json
import pathlib
import types

import pytest

pytest.importorskip("pandas")
pytest.importorskip("numpy")

import pandas as pd

from hpcagent_bench import config
from hpcagent_bench.results import extract, studies
from hpcagent_bench.stats import population
from tests.fresh_module import module_at

KERNEL = "fuse_stencil_through_transient"
JOB = 636501
LABEL = "setup-a.n0.p0.w0"
START_MS = 1_789_000_000_000

#: The two crashed attempts spent 30 and 20 tokens; the driver sums them into the record's one
#: ``tokens_effective_crashed`` field (token_cost.episode_totals), the DB stores the sum, and ``all``
#: must add the sum back in: 100 + 50 = 150.
FINAL_TOKENS = 100
CRASHED_TOKENS = 50
ALL_TOKENS = FINAL_TOKENS + CRASHED_TOKENS

STATISTICS = pathlib.Path(extract.__file__).parents[2] / "statistics"


@pytest.fixture(name="paired_setups")
def paired_setups_fixture() -> types.ModuleType:
    return module_at(STATISTICS / "paired_setups.py", "paired_setups")


def all_attempts() -> contextlib.AbstractContextManager[None]:
    """``cost.attempts = all`` for one test, whatever the environment says (restored on exit)."""
    return config.overridden("cost.attempts", config.CostAttempts.ALL)


def final_attempt() -> contextlib.AbstractContextManager[None]:
    """``cost.attempts = final`` for one test (restored on exit; the default is ``all``)."""
    return config.overridden("cost.attempts", config.CostAttempts.FINAL)


# -- the config surface -------------------------------------------------------


def test_the_switch_reads_all_by_default_and_spells_its_two_modes() -> None:
    """The default counts every attempt, what an answer cost; the spellings are what config.yaml,
    the environment and a recorded ``cost_attempts`` column carry."""
    assert population.cost_attempts() is config.CostAttempts.ALL
    assert config.CostAttempts.texts() == ("final", "all")
    assert config.default_value("cost.attempts") is config.CostAttempts.ALL


def test_the_environment_overlay_flips_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.env_name("cost.attempts"), "final")
    assert population.cost_attempts() is config.CostAttempts.FINAL


# -- the sqlite fixture: the production path ----------------------------------


def write_record(job_dir: pathlib.Path, worker: int, **fields: object) -> pathlib.Path:
    """The ``tokens.json`` the driver leaves in ``agents/node-0/problem-<w>-worker-<w>/``."""
    worker_dir = job_dir / "agents" / "node-0" / f"problem-{worker}-worker-{worker}"
    worker_dir.mkdir(parents=True)
    record: dict[str, object] = {
        "episode_id": f"setup-a.n0.p{worker}.w{worker}",
        "kernel": f"loop_level_reasoning/{KERNEL}/{KERNEL}",
        "result": "success",
        "turns": 7,
        "tokens_effective": FINAL_TOKENS,
        "fresh_input": FINAL_TOKENS,
        "cached_input": 0,
        "output": 0,
        "attempts": 1,
        "final_attempt_start_ms": START_MS,
        **fields,
    }
    path = worker_dir / "tokens.json"
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    return path


def shard(job_dir: pathlib.Path, rank: int, label: str, ts_ms: int) -> None:
    """Rank ``rank``'s judge shard: one credited submission of ``label``."""
    from hpcagent_bench.harness import results_db
    from tests.results_seed import STAMP

    path = job_dir / "judge" / f"rank-{rank}" / f"hpcagent_bench{rank}.db"
    path.parent.mkdir(parents=True)
    setup = label.split(".", maxsplit=1)[0]
    with contextlib.closing(results_db.open_db(path)) as conn:
        results_db.ensure_setup(conn, results_db.Setup(setup, "c", "cpu", study="llr40", model="stub-model"))
        run = results_db.ensure_episode(conn, setup, label, JOB)
        results_db.add_grade(
            conn,
            run,
            KERNEL,
            "submit",
            ts_ms=ts_ms,
            values=STAMP
            | {
                "preset": "XL",
                "datatype": "float64",
                "source_mode": "restricted",
                "baseline": "c",
                "build_ok": 1,
                "correct": 1,
                "speedup": 2.0,
                "credited_speedup": 2.0,
            },
        )
        conn.commit()


@pytest.fixture(name="task_frame")
def task_frame_fixture(tmp_path: pathlib.Path) -> pd.DataFrame:
    """One resumed episode through the production path: record (100 final + 30 + 20 crashed) -> job DB
    -> extracted observations, read back from the observations sqlite file."""
    from hpcagent_bench.cluster import merge_results

    job_dir = tmp_path / str(JOB)
    shard(job_dir, 0, LABEL, 1)
    write_record(job_dir, 0, attempts=3, tokens_effective_crashed=CRASHED_TOKENS)
    out = job_dir / "results.db"
    merge_results.merge(job_dir, out)

    database = extract.Database(out, "r", out.parent, str(JOB))
    rows = extract.read_db(database, "", frozenset(), 0).observations
    observations = tmp_path / "observations.db"
    assert extract.write_db(observations, extract.OBSERVATION_FIELDS, rows) == 2  # the submission and the task
    return studies.read_observations(observations)


def test_the_results_db_stores_the_crashed_attempts_spend_on_the_episode_row(
    tmp_path: pathlib.Path,
) -> None:
    """The schema column ``episodes.crashed_effective_tokens`` is the crashed attempts' spend, summed by
    the driver (30 + 20 here); extraction turns it into the task row's ``tokens_crashed``."""
    from hpcagent_bench.cluster import merge_results
    from hpcagent_bench.harness import results_db

    job_dir = tmp_path / str(JOB)
    shard(job_dir, 0, LABEL, 1)
    write_record(job_dir, 0, attempts=3, tokens_effective_crashed=CRASHED_TOKENS)
    out = job_dir / "results.db"
    merge_results.merge(job_dir, out)

    with contextlib.closing(results_db.open_db(out)) as conn:
        row = conn.execute("SELECT effective_tokens, crashed_effective_tokens FROM episodes").fetchone()
    assert tuple(row) == (FINAL_TOKENS, CRASHED_TOKENS)


def test_the_extracted_task_row_keeps_the_two_conventions_apart(task_frame: pd.DataFrame) -> None:
    """The task row states the FINAL attempt's total and carries the crashed spend beside it; the
    aggregation reads one or the other reading, never half of each."""
    (row,) = task_frame[task_frame.row_kind == "episode"].to_dict("records")
    assert row["tokens"] == FINAL_TOKENS
    assert row["tokens_crashed"] == CRASHED_TOKENS

    with final_attempt():
        default = population.episode_tokens(task_frame)
        assert default.tokens.tolist() == [FINAL_TOKENS]
        assert population.CRASHED_COLUMN not in default.columns
    with all_attempts():
        merged = population.episode_tokens(task_frame)
        assert merged.tokens.tolist() == [ALL_TOKENS]
        assert merged[population.CRASHED_COLUMN].tolist() == [CRASHED_TOKENS]
        assert merged[population.CRASHED_SHARE_COLUMN].tolist() == [CRASHED_TOKENS / ALL_TOKENS]


# -- the switch on frames of the production shape ------------------------------


def submissions(rows: list[dict[str, object]]) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    defaults: dict[str, object] = {
        "run_root": "r",
        "job": "1",
        "baseline": "c",
        "setup": "a",
        "kernel": "k",
        "episode_id": "w0",
        "attempt_index": 0,
        "speedup": 2.0,
        "timing_suspect": 0,
        "timing_reduction": "mw4x5",
        "denominator": "best-of(numba,c)",
        "baseline_ns": 0.0,
        "native_ns": 0.0,
        "slot": 1,
        "ts_ms": 1,
    }
    for column, value in defaults.items():
        if column not in out:
            out[column] = value
    return out


def episode_frame(tokens: float = FINAL_TOKENS, crashed: object = CRASHED_TOKENS) -> pd.DataFrame:
    """One resumed episode: its graded submission, a call row, and the task row carrying the FINAL
    attempt's total with the crashed spend beside it."""
    return submissions(
        [
            {"row_kind": "submission", "tokens": "", "ts_ms": 2},
            {"row_kind": "call", "tokens": 40.0, "speedup": "", "ts_ms": 3},
            {
                "row_kind": "episode",
                "tokens": tokens,
                "tokens_crashed": crashed,
                "episode_attempts": 3,
                "speedup": "",
                "attempt_index": "",
                "ts_ms": 1,
            },
        ]
    )


def test_final_counts_the_final_attempt_only() -> None:
    """``cost.attempts = final``: the task row's ``tokens`` stands alone, and no crashed column or
    marking appears -- byte-for-byte today's outputs."""
    with final_attempt():
        frame = episode_frame()
        episodes = population.episode_tokens(frame)
        assert episodes.tokens.tolist() == [FINAL_TOKENS]
        assert population.CRASHED_COLUMN not in episodes.columns
        assert population.CRASHED_SHARE_COLUMN not in episodes.columns
        assert population.kernel_tokens(frame).to_dict() == {"k": FINAL_TOKENS}
        point = population.kernel_medians(frame)
        assert point is not None and point["tokens"] == pytest.approx(FINAL_TOKENS)
        assert "cost_attempts" not in point


def test_all_adds_every_crashed_attempt_in_and_marks_the_point() -> None:
    """``cost.attempts = all``: 100 + (30 + 20) = 150, the crashed part and its share reported beside
    the total, and the point names its convention so no figure mixes the two readings."""
    frame = episode_frame()
    with all_attempts():
        assert population.kernel_tokens(frame).to_dict() == {"k": ALL_TOKENS}
        assert population.kernel_tokens_crashed(frame).to_dict() == {"k": CRASHED_TOKENS}
        point = population.kernel_medians(frame)
        assert point is not None
        assert point["tokens"] == pytest.approx(ALL_TOKENS)
        assert point["cost_attempts"] == "all"
        assert point["tokens_crashed"] == pytest.approx(CRASHED_TOKENS)
        assert point[population.CRASHED_SHARE_COLUMN] == pytest.approx(CRASHED_TOKENS / ALL_TOKENS)


def test_all_keeps_the_reduction_rules_of_the_final_reading() -> None:
    """Adding the crashed spend in must not bend the reductions around it: a rerun supersedes the run
    it repeats (never a sum over reruns), call rows are still not a cost, and an unmeasured final
    attempt is still no measurement -- its crashed spend alone is not a total."""
    with all_attempts():
        rerun = submissions(
            [
                {"row_kind": "episode", "episode_id": "w0", "tokens": 400.0, "tokens_crashed": 10.0, "ts_ms": 10},
                {
                    "row_kind": "episode",
                    "episode_id": "w0",
                    "tokens": FINAL_TOKENS,
                    "tokens_crashed": CRASHED_TOKENS,
                    "ts_ms": 30,
                    "job": "2",
                    "run_root": "r2",
                },
            ]
        )
        assert population.kernel_tokens(rerun).to_dict() == {"k": ALL_TOKENS}
        calls = submissions([{"row_kind": "call", "tokens": 900.0, "speedup": "", "ts_ms": 5}])
        with pytest.raises(population.MixedPopulationError, match="no episode records"):
            population.kernel_tokens(calls)
        unmeasured = episode_frame(tokens="", crashed=CRASHED_TOKENS)
        assert population.episode_tokens(unmeasured).empty


def test_all_over_a_frame_without_crashed_columns_is_the_final_reading() -> None:
    """A frame extracted before task records carried ``tokens_crashed`` adds nothing in, rather than
    failing: its episodes had no recorded crash spend."""
    frame = episode_frame().drop(columns=["tokens_crashed"])
    with all_attempts():
        assert population.kernel_tokens(frame).to_dict() == {"k": FINAL_TOKENS}


# -- paired_setups aggregates the same way ------------------------------------


def paired_rows(setup: str, kernel: str, tokens: float, crashed: float, ts: int = 900) -> list[dict[str, object]]:
    """One episode in the rows extraction writes: a graded submission, a call, and the task record."""
    episode_id = f"{setup}.n0.p{kernel}.w0"
    graded = {
        "optimizer": "a-model",
        "run_root": "stamp",
        "job": "j1",
        "row_kind": "submission",
        "episode_id": episode_id,
        "setup": setup,
        "kernel": kernel,
        "speedup": 2.0,
        "tokens": "",
        "timing_suspect": 0,
        "baseline": "numba",
        "ts_ms": 1000,
        "attempt_index": 1,
        "timing_reduction": "mw4x5",
        "denominator": "best-of(numba,c)",
    }
    call = graded | {"row_kind": "call", "speedup": "", "tokens": tokens, "timing_suspect": "", "ts_ms": 1100}
    task = {
        "optimizer": "",
        "run_root": "stamp",
        "job": "j1",
        "row_kind": "episode",
        "episode_id": episode_id,
        "setup": setup,
        "kernel": kernel,
        "speedup": "",
        "tokens": tokens,
        "tokens_crashed": crashed,
        "episode_attempts": 2 if crashed else 1,
        "tokens_fresh_input": tokens,
        "tokens_cached_input": 0.0,
        "tokens_output": 0.0,
        "timing_suspect": "",
        "baseline": "",
        "ts_ms": ts,
        "attempt_index": "",
    }
    return [graded, call, task]


def test_paired_setups_aggregates_each_mode_exactly_like_the_population_module(
    paired_setups: types.ModuleType,
) -> None:
    """One treatment episode that resumed (100 + 30 + 20) and one control that did not (100): the
    script's per-(setup, kernel) totals, its usage row and its setup row must match the population
    module's reading in BOTH modes, and say which mode they used."""
    rows = paired_rows("treated", "k1", FINAL_TOKENS, CRASHED_TOKENS) + paired_rows("control", "k1", FINAL_TOKENS, 0.0)
    frame = pd.DataFrame(rows)
    setups = ["treated", "control"]
    best = paired_setups.best_by_setup_kernel(frame)
    served = paired_setups.served_by_setup(frame)
    table = paired_setups.setup_aggregates(best, served, "numba")

    with final_attempt():
        default_tokens = paired_setups.tokens_by_setup_kernel(frame)
        assert default_tokens == {("treated", "k1"): float(FINAL_TOKENS), ("control", "k1"): float(FINAL_TOKENS)}
        usage = paired_setups.episode_usage(frame)
        assert "share_tokens_crashed" not in usage.columns
        assert usage.loc["treated", "tokens_crashed"] == pytest.approx(CRASHED_TOKENS)
        setup_frame = pd.DataFrame(
            paired_setups.setup_rows(
                best, paired_setups.graded_rows(frame, setups), table, served, default_tokens, usage
            )
        )
        assert setup_frame.cost_attempts.tolist() == ["final", "final"]
        assert setup_frame.share_tokens_crashed.isna().all()
        assert setup_frame.gm_tokens.tolist() == [pytest.approx(FINAL_TOKENS)] * 2

    with all_attempts():
        all_tokens = paired_setups.tokens_by_setup_kernel(frame)
        assert all_tokens == {("treated", "k1"): float(ALL_TOKENS), ("control", "k1"): float(FINAL_TOKENS)}
        assert all_tokens == {
            (setup, kernel): float(value)
            for (setup, kernel), value in population.kernel_tokens(frame, ("setup", "kernel")).items()
        }
        usage = paired_setups.episode_usage(frame)
        assert usage.loc["treated", "tokens_billed"] == pytest.approx(ALL_TOKENS)
        assert usage.loc["treated", "share_tokens_crashed"] == pytest.approx(CRASHED_TOKENS / ALL_TOKENS)
        assert usage.loc["control", "share_tokens_crashed"] == 0.0
        setup_frame = pd.DataFrame(
            paired_setups.setup_rows(best, paired_setups.graded_rows(frame, setups), table, served, all_tokens, usage)
        )
        assert setup_frame.cost_attempts.tolist() == ["all", "all"]
        by_setup = setup_frame.set_index("setup")
        assert by_setup.loc["treated", "gm_tokens"] == pytest.approx(ALL_TOKENS)
        assert by_setup.loc["control", "gm_tokens"] == pytest.approx(FINAL_TOKENS)
        assert by_setup.loc["treated", "share_tokens_crashed"] == pytest.approx(CRASHED_TOKENS / ALL_TOKENS)
        assert by_setup.loc["control", "share_tokens_crashed"] == 0.0


def test_the_written_tables_carry_the_mode_they_were_built_under(
    paired_setups: types.ModuleType, tmp_path: pathlib.Path
) -> None:
    """End to end through ``main``: the pairs and setups CSVs record ``cost_attempts`` beside
    ``cost_model``/``kernel_policy``, and the tokens leg's rho follows the mode."""
    rows = paired_rows("treated", "k1", FINAL_TOKENS, CRASHED_TOKENS) + paired_rows("control", "k1", FINAL_TOKENS, 0.0)
    path = tmp_path / "observations.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    argv = [
        "--observations",
        str(path),
        "--pair",
        "treated,control",
        "--family",
        "f",
        "--out",
        str(tmp_path / "pairs.csv"),
        "--setups-out",
        str(tmp_path / "setups.csv"),
    ]

    with final_attempt():
        assert paired_setups.main(argv) == 0
    pairs = pd.read_csv(tmp_path / "pairs.csv")
    setups = pd.read_csv(tmp_path / "setups.csv")
    assert (pairs.cost_attempts == "final").all()
    assert (setups.cost_attempts == "final").all()
    final_rho = float(pairs[pairs.leg == "tokens"].rho.iloc[0])
    assert setups.set_index("setup").loc["treated", "gm_tokens"] == pytest.approx(FINAL_TOKENS)

    with all_attempts():
        assert paired_setups.main(argv) == 0
        pairs = pd.read_csv(tmp_path / "pairs.csv")
        setups = pd.read_csv(tmp_path / "setups.csv")
        assert (pairs.cost_attempts == "all").all()
        assert (setups.cost_attempts == "all").all()
        assert setups.set_index("setup").loc["treated", "gm_tokens"] == pytest.approx(ALL_TOKENS)
        assert setups.set_index("setup").loc["treated", "share_tokens_crashed"] == pytest.approx(
            CRASHED_TOKENS / ALL_TOKENS
        )
        # rho_C = control over treated: billing the resumed episode makes the treated one dearer
        assert float(pairs[pairs.leg == "tokens"].rho.iloc[0]) == pytest.approx(FINAL_TOKENS / ALL_TOKENS)
        assert float(pairs[pairs.leg == "tokens"].rho.iloc[0]) < final_rho


def test_one_table_states_the_cost_with_and_without_the_crashed_attempts(
    paired_setups: types.ModuleType, tmp_path: pathlib.Path
) -> None:
    """``--all-attempts-leg``: the table of a ``cost.attempts = final`` run, plus per pair the tokens row the
    same run writes under ``cost.attempts = all``, named ``tokens_all_attempts`` and stamped with its reading."""
    rows = [
        row
        for index in range(6)
        for row in (
            paired_rows("treated", f"k{index}", FINAL_TOKENS, CRASHED_TOKENS * index)
            + paired_rows("control", f"k{index}", FINAL_TOKENS, 0.0)
        )
    ]
    path = tmp_path / "observations.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    argv = ["--observations", str(path), "--pair", "treated,control", "--family", "f", "--out"]

    def table(name: str, *flags: str) -> pd.DataFrame:
        assert paired_setups.main([*argv, str(tmp_path / name), *flags]) == 0
        return pd.read_csv(tmp_path / name)

    with final_attempt():
        final, both = table("final.csv"), table("both.csv", "--all-attempts-leg")
    with all_attempts():
        every = table("all.csv")
        with pytest.raises(SystemExit, match="already counts every attempt"):
            paired_setups.main([*argv, str(tmp_path / "refused.csv"), "--all-attempts-leg"])

    assert both.leg.tolist() == ["speedup", "tokens", "tokens_all_attempts"]
    assert both.cost_attempts.tolist() == ["final", "final", "all"]
    pd.testing.assert_frame_equal(both.iloc[:2], final)
    added = both.iloc[2].drop("leg")
    pd.testing.assert_series_equal(added, every[every.leg == "tokens"].iloc[0].drop("leg"), check_names=False)
    assert added.rho < final[final.leg == "tokens"].rho.iloc[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
