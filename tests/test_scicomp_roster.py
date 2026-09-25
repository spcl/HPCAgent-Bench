# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The scicomp launchers' arm tables."""

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_the_scicomp_launcher_gives_its_control_arm_no_skill_page() -> None:
    """The control's packet is "" only if it ships no page, and `skill_args_for` returns EVERY
    shipped page -- so an arm table built on it hands the control the treatment it is the control
    for, and the A/B reports a null it never measured."""
    for launcher in ("submit-scicomp-dc.sh", "submit-scicomp-perf-playbook.sh"):
        script = (REPO / "experiments" / launcher).read_text()
        table = re.search(r"^    case \"\$\{kind\}\" in\n(.*?)^    esac$", script, re.DOTALL | re.MULTILINE)
        assert table is not None, f"{launcher} no longer maps arm kinds to pages in a case block"
        control = [ln for ln in table.group(1).splitlines() if ln.strip().startswith("plain)")]
        assert control == ["        plain) ;;"], (launcher, control)
        assert "skill_args_for" not in script, f"{launcher}'s arm table is back on the every-page auto packet"
