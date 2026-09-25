# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``impact_markdown.py`` renders the committed impact CSV as the table a README carries.

A README that quotes a number typed by hand is a second copy of that number, and the two disagree
the first time the data is re-extracted. These fix what the rendering must not lose: the control
rows carry no ratio and do not appear, a withheld interval says so rather than printing NaN, and
the relaunch rate the final-attempt rule makes every token ratio conditional on is in the row.
"""

import importlib.util
import math
import pathlib
import sys

import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("impact_markdown", REPO / "scripts" / "impact_markdown.py")
assert SPEC is not None and SPEC.loader is not None
impact_markdown = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = impact_markdown
SPEC.loader.exec_module(impact_markdown)


def impact_frame(**overrides: object) -> pd.DataFrame:
    """One treatment row and its control, in the columns ``paired_arms.py --impact-out`` writes."""
    treated: dict[str, object] = {
        "model": "qwen38",
        "language": "c",
        "packet": "lang-skills",
        "arm": "x-qwen38-c-skills",
        "control": "x-qwen38-c",
        "attempts_per_task": 1.9,
        "share_relaunched": 0.45,
        "speedup_ratio": 1.206,
        "speedup_ci_low": 0.926,
        "speedup_ci_high": 1.571,
        "speedup_n": 14,
        "speedup_p_adjusted": 0.448,
        "speedup_verdict": "not-significant",
        "token_ratio": 1.6075,
        "token_ci_low": 1.0928,
        "token_ci_high": 2.3647,
        "token_n": 40,
        "token_p_adjusted": 0.0172,
        "token_verdict": "significant",
    }
    treated.update(overrides)
    control = dict.fromkeys(treated, math.nan) | {"model": "qwen38", "language": "c", "arm": "x-qwen38-c"}
    control["control"] = ""
    return pd.DataFrame([treated, control])


def test_only_the_treatment_rows_are_rendered() -> None:
    """A control row carries no ratio; printing it would read as a comparison against itself."""
    table = impact_markdown.markdown(impact_frame())
    assert "x-qwen38-c-skills" not in table  # arms are named by model and language, not by arm string
    assert len(table.splitlines()) == 3


def test_the_row_carries_both_legs() -> None:
    cells = impact_markdown.markdown(impact_frame()).splitlines()[2]
    assert "1.21 [0.93, 1.57]" in cells
    assert "1.61 [1.09, 2.36]" in cells


def test_a_significant_leg_is_starred_and_carries_its_corrected_p() -> None:
    """Spec M1: only a corrected verdict may be starred."""
    cells = impact_markdown.markdown(impact_frame()).splitlines()[2]
    assert "0.017*" in cells and "0.448" in cells and "0.448*" not in cells


def test_the_relaunch_rate_is_in_the_row_beside_the_token_ratio() -> None:
    """The cost is the FINAL attempt's (T2), so the share of tasks that relaunched is what says
    where that rule applied."""
    cells = impact_markdown.markdown(impact_frame()).splitlines()[2]
    assert "1.90" in cells and "45%" in cells


def test_a_withheld_interval_says_so_instead_of_printing_a_nan() -> None:
    """Spec P4: below the floor there is an estimate and no interval."""
    frame = impact_frame(
        speedup_ci_low=math.nan, speedup_ci_high=math.nan, speedup_p_adjusted=math.nan, speedup_verdict="underpowered"
    )
    cells = impact_markdown.markdown(frame).splitlines()[2]
    assert "1.21 (no interval)" in cells and "underpowered" in cells
    assert "nan" not in cells.lower()


def test_a_readme_marker_is_replaced_and_not_appended_to(tmp_path: pathlib.Path) -> None:
    """Re-running must give the same file: a block that grew on every run is how a README ends up
    carrying three versions of one table."""
    (tmp_path / "tables").mkdir()
    impact_frame().to_csv(tmp_path / "tables" / "impact_x.csv", index=False)
    readme = tmp_path / "README.md"
    readme.write_text("## Results\n\n<!--TABLE impact_x-->\n\n## Commands\n", encoding="utf-8")
    impact_markdown.main(["--readme", str(readme)])
    once = readme.read_text(encoding="utf-8")
    impact_markdown.main(["--readme", str(readme)])
    assert readme.read_text(encoding="utf-8") == once
    assert "<!--TABLE impact_x-->" in once and "| model | language |" in once
    assert once.endswith("## Commands\n")


def test_a_marker_naming_a_missing_table_fails(tmp_path: pathlib.Path) -> None:
    """A README that keeps a stale table while the data moved is the failure this prevents."""
    (tmp_path / "tables").mkdir()
    readme = tmp_path / "README.md"
    readme.write_text("<!--TABLE nothing-->\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        impact_markdown.main(["--readme", str(readme)])
