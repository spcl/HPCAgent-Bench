# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""An experiment's observations read the same from its CSV and from its extracted ``.db``.

The reproducibility artifact ships one database per experiment and every figure reads it through
:func:`hpcagent_bench.experiments.read_observations`, so a row, a number or the row order moving
between the two files would change a published figure without anyone touching the data.
"""

import pathlib

import pandas as pd

from hpcagent_bench import experiments

#: The pair is written by the extractor that writes every shipped artifact, not by a hand-rolled
#: CREATE TABLE here: the two files have to agree on the COLUMN TYPES as well as on the rows, and a
#: writer invented in the test body agrees with nothing.
from hpcagent_bench import observations_extract as extract_llr40

FIELDS = ("run_root", "job", "row_kind", "arm", "benchmark", "speedup", "tokens", "tokens_billed", "packet")
ROWS = [
    {
        "run_root": "r1",
        "job": 636541,
        "row_kind": "submission",
        "arm": "a-c",
        "benchmark": "k2",
        "speedup": 3.5,
        "tokens": "",
        "tokens_billed": "",
        "packet": "",
    },
    {
        "run_root": "r1",
        "job": 636541,
        "row_kind": "task",
        "arm": "a-c",
        "benchmark": "k1",
        "speedup": "",
        "tokens": 1200,
        "tokens_billed": 48_000,
        "packet": "cpf",
    },
]


def write_pair(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    csv_path, db_path = tmp_path / "obs.csv", tmp_path / "obs.db"
    extract_llr40.write_csv(csv_path, FIELDS, ROWS)
    extract_llr40.write_db(db_path, FIELDS, ROWS)
    return csv_path, db_path


def test_a_db_and_its_csv_give_the_same_rows_in_the_same_order(tmp_path: pathlib.Path) -> None:
    csv_path, db_path = write_pair(tmp_path)
    from_csv = experiments.read_observations(csv_path)
    from_db = experiments.read_observations(db_path)
    # fill_arm_identity (run by read_observations on both paths) appends recorded_packet, the raw
    # value kept beside the filled "packet" column -- both paths must gain it identically.
    expected_columns = [*FIELDS, "recorded_packet"]
    assert list(from_db.columns) == expected_columns
    assert list(from_csv.columns) == expected_columns
    assert from_db["benchmark"].tolist() == from_csv["benchmark"].tolist() == ["k2", "k1"]
    pd.testing.assert_series_equal(from_db["speedup"], from_csv["speedup"], check_dtype=False)
    pd.testing.assert_series_equal(from_db["tokens"], from_csv["tokens"], check_dtype=False)


def test_the_token_columns_have_the_same_dtype_from_either_file(tmp_path: pathlib.Path) -> None:
    """Same rows AND the same dtype. An untyped CREATE TABLE gave the columns no affinity, so the
    "" the CSV writer spells a missing cell as was stored as TEXT and made the whole column object
    dtype on the DB path while the CSV path read float64 -- one table, two dtypes, and arithmetic
    that raised on exactly one of them."""
    csv_path, db_path = write_pair(tmp_path)
    from_csv = experiments.read_observations(csv_path)
    from_db = experiments.read_observations(db_path)
    for column in ("tokens", "tokens_billed", "speedup"):
        assert from_db[column].dtype == from_csv[column].dtype, column
        pd.testing.assert_series_equal(from_db[column], from_csv[column])
        # The arithmetic every cost table starts from, on BOTH paths.
        assert from_db[column].sum() == from_csv[column].sum()
    assert from_db["tokens"].sum() == 1200.0


def test_a_db_is_opened_read_only(tmp_path: pathlib.Path) -> None:
    _, db_path = write_pair(tmp_path)
    before = db_path.stat().st_mtime_ns
    experiments.read_observations(db_path)
    assert db_path.stat().st_mtime_ns == before
