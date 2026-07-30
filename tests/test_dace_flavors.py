# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One DaCe flavor per SDFG pipeline, and the two things that make them meaningful.

A flavor that scores one pipeline is only useful if it scores THAT pipeline and pays for nothing
else, and if the columns that need spcl/dace@extended are exactly the ones that ask for it --
``dace_cpu_parallel`` exists to be runnable on upstream DaCe, and a gate that fires on every
``dace_*`` name would make that impossible while the column still looked fine locally.
"""
import pytest

from hpcagent_bench.frameworks.dace_framework import DACE_PIPELINES, DEFAULT_PIPELINES, needed_pipelines
from hpcagent_bench.frameworks.framework import (FRAMEWORK_META, check_flavor_registry, framework_flavors, split_flavor)
from hpcagent_bench.harness import preflight

#: (flavor, what it scores, what it must BUILD to get there).
EXPECTED = (
    ("dace_cpu_parallel", ("parallel", ), ["strict", "fusion", "parallel"]),
    ("dace_cpu_autoopt", ("autoopt", ), ["strict", "autoopt"]),
    ("dace_cpu_canonicalize", ("canonicalize", ), ["strict", "canonicalize"]),
    ("dace_gpu_parallel", ("parallel", ), ["strict", "fusion", "parallel"]),
    ("dace_gpu_autoopt", ("autoopt", ), ["strict", "autoopt"]),
    ("dace_gpu_canonicalize", ("canonicalize", ), ["strict", "canonicalize"]),
)


@pytest.mark.parametrize("flavor,scored,build", EXPECTED)
def test_a_flavor_scores_its_pipeline_and_builds_only_its_parents(flavor, scored, build):
    """``fusion`` is ``parallel``'s intermediate, so a canonicalize-only column must not run it."""
    assert FRAMEWORK_META[flavor]["pipelines"] == scored
    assert needed_pipelines(scored) == build


def test_parents_come_before_children():
    """A pipeline deepcopies from its parent's OUTPUT, so an order inversion silently optimizes the
    wrong graph rather than raising."""
    for pipe in DACE_PIPELINES:
        order = needed_pipelines((pipe.name, ))
        assert order[-1] == pipe.name
        if pipe.parent:
            assert order.index(pipe.parent) < order.index(pipe.name)


def test_unknown_pipeline_is_rejected():
    with pytest.raises(KeyError):
        needed_pipelines(("does_not_exist", ))


def test_only_canonicalize_columns_need_the_fork():
    """The gate is derived from what a flavor RUNS, not from a second hand-maintained list."""
    every = framework_flavors("dace")
    gated = set(preflight.needs_canonicalize(every))
    for name in every:
        wants = "canonicalize" in FRAMEWORK_META[name].get("pipelines", DEFAULT_PIPELINES)
        assert (name in gated) is wants, f"{name}: fork gate does not match its pipelines"
    assert "dace_cpu_parallel" not in gated, (
        "dace_cpu_parallel is upstream transformations end to end; gating it on the fork removes "
        "the only column that can be measured on both trees")
    assert "dace_cpu_canonicalize" in gated


def test_every_dace_flavor_is_a_deterministic_column():
    """A sweep refuses a column it cannot run, so a new flavor missing here fails at submission."""
    assert not preflight.check_deterministic(framework_flavors("dace"))


@pytest.mark.parametrize("flavor,expected", [
    ("dace_cpu_parallel", ("dace_cpu", "parallel")),
    ("dace_cpu_canonicalize", ("dace_cpu", "canonicalize")),
    ("dace_gpu_parallel", ("dace_gpu", "parallel")),
    ("dace_cpu", ("dace_cpu", None)),
    ("numpy", ("numpy", None)),
])
def test_the_flat_name_splits_into_framework_and_flavor(flavor, expected):
    """One name on the CLI, two columns in the DB -- so `GROUP BY framework` still gathers every
    DaCe row instead of scattering it across five names."""
    assert split_flavor(flavor) == expected


def test_the_split_is_declared_not_parsed():
    """``dace_cpu_parallel`` reads equally well as ``dace_cpu`` + ``parallel`` or ``dace`` +
    ``cpu_parallel``; no underscore rule can tell them apart, so the entry states both halves.

    Pinned because the tempting shortcut -- derive the column by stripping the flavor suffix -- is
    only unambiguous while no framework named ``dace`` exists, and it would start returning a
    different answer on the day one is registered."""
    meta = FRAMEWORK_META["dace_cpu_parallel"]
    assert (meta["column"], meta["flavor"]) == ("dace_cpu", "parallel")
    assert split_flavor("dace_cpu_parallel") == ("dace_cpu", "parallel")
    # The alternative reading composes to the same flat name, which is exactly why parsing cannot
    # decide between them -- and why declaring is the only safe answer.
    assert "dace_cpu_parallel" == "dace" + "_" + "cpu_parallel"


@pytest.mark.parametrize("broken,why", [
    ({
        "flavor": "parallel",
        "column": None
    }, "flavor without a column"),
    ({
        "flavor": None,
        "column": "dace_cpu"
    }, "column without a flavor"),
    ({
        "flavor": "parallel",
        "column": "not_a_framework"
    }, "column is not registered"),
    ({
        "flavor": "cpu_parallel",
        "column": "dace_cpu"
    }, "pair does not compose into the name"),
])
def test_a_malformed_flavor_entry_is_rejected_at_import(monkeypatch, broken, why):
    """Each of these writes a wrong GROUP BY key onto every row of a finished sweep."""
    entry = {k: v for k, v in {**FRAMEWORK_META["dace_cpu_parallel"], **broken}.items() if v is not None}
    monkeypatch.setitem(FRAMEWORK_META, "dace_cpu_parallel", entry)
    with pytest.raises(KeyError):
        check_flavor_registry()


def test_the_registry_as_shipped_is_valid():
    check_flavor_registry()
    for name in FRAMEWORK_META:
        column, flavor = split_flavor(name)
        assert column in FRAMEWORK_META
        assert (flavor is None) or name == f"{column}_{flavor}"


def test_ranks_per_node_splits_the_node():
    """Four co-resident ranks get a quarter of the threads each; one rank still gets the node."""
    whole = preflight.thread_env()
    quarter = preflight.thread_env(ranks_per_node=4)
    for name, value in whole.items():
        assert int(quarter[name]) == max(1, int(value) // 4)
    assert preflight.thread_env(ranks_per_node=1) == whole


def test_absent_shard_csvs_report_instead_of_tracebacking(tmp_path, capsys):
    """The rollup is handed a shell GLOB, which bash passes through verbatim when nothing matches.

    So "every rank died before writing a row" arrives as a path containing a `*`. It must say that
    and stay non-zero -- an empty summary read as a clean run is the failure mode this guards."""
    from hpcagent_bench.support.collect.sweep import summarize_csv

    missing = str(tmp_path / "shard-*.csv")
    assert summarize_csv([missing]) > 0
    out = capsys.readouterr().out
    assert "absent" in out and "nothing was measured" in out
