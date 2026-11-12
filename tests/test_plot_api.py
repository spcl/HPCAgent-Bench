# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``hpcagent-bench plot <figure>`` and its Python API are one thing: each command's ``Options`` holds exactly its
parser's flags at the parser's defaults, and its parser refuses a single-valued flag given twice."""

import dataclasses
import importlib
import types

import pytest

from hpcagent_bench.stats import plot

#: The positional arguments each figure's command line cannot go without.
REQUIRED: dict[plot.Figure, list[str]] = {
    plot.Figure.EFFICACY: [],
    plot.Figure.KERNEL_SPEEDUPS: [],
    plot.Figure.KERNEL_RUNS: ["obs.db", "--out", "runs.pdf"],
    plot.Figure.SCALING: ["obs.db"],
}


def command(figure: plot.Figure) -> types.ModuleType:
    return importlib.import_module(plot.command(figure))


@pytest.mark.parametrize("figure", list(plot.Figure))
def test_options_are_the_flags_at_their_defaults(figure: plot.Figure) -> None:
    module = command(figure)
    parsed = vars(module.build_parser().parse_args(REQUIRED[figure]))
    filled = dataclasses.asdict(module.build_parser().parse_args(REQUIRED[figure], namespace=module.Options()))
    assert filled == parsed
    given = {name for name in ("observations", "out") if parsed[name] != dataclasses.asdict(module.Options())[name]}
    assert dataclasses.asdict(module.Options()) == {
        name: None if name in given else value for name, value in parsed.items()
    }


@pytest.mark.parametrize("figure", list(plot.Figure))
def test_a_single_valued_flag_given_twice_is_refused(figure: plot.Figure, capsys: pytest.CaptureFixture[str]) -> None:
    parser = command(figure).build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([*REQUIRED[figure], "--experiment", "a", "--experiment=b"])
    assert "--experiment given 2 times" in capsys.readouterr().err
    assert parser.prog == f"hpcagent-bench plot {figure.text}"


def test_efficacy_refuses_a_flag_its_route_never_reads() -> None:
    efficacy = command(plot.Figure.EFFICACY)
    with pytest.raises(SystemExit, match=r"--experiment, --include-incomplete: not read on the --pairs-csv route"):
        efficacy.main(["obs.csv", "--pairs-csv", "pairs.csv", "--experiment", "llr40", "--include-incomplete"])
    with pytest.raises(SystemExit, match=r"--treatment: not read on the --comparison route"):
        efficacy.main(["obs.csv", "--comparison", "treatment=lang-skills", "--treatment", "lang-skills"])
    with pytest.raises(SystemExit, match=r"--intervention: not read on the --experiment route"):
        efficacy.main(["obs.csv", "--experiment", "llr40", "--intervention", "lang-skills"])
