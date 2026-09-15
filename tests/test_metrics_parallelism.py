# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The SDFG-level parallelism taxonomy, over SYNTHETIC SDFGs built in process.

Ported from ``mpr-artifacts/experiments/parallelism/llr_parallelism.py`` and its own
``test_llr_parallelism.py``: each test hand-builds the exact shape a bucket is defined by, so a
bucket that stops firing fails here in milliseconds instead of silently changing a corpus-wide
percentage. None of these touch the hpcagent_bench corpus or the DaCe python frontend.
"""

import ast

import dace
import pytest
from dace.libraries.standard.nodes import Reduce
from dace.libraries.standard.nodes.scan import Scan
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from numpyto_common.parallelism import TIMESTEP_SYMBOLS, is_timestep_loop

from hpcagent_bench.metrics.parallelism import (
    BUCKETS,
    DEFAULT_RATE,
    NEUTRAL_ALWAYS,
    PARALLEL_BUCKETS,
    RATE_DEFINITIONS,
    classify,
    is_timestep_loop_region,
    loop_bound_symbols,
    rate,
    rates,
)


def fill_loop(loop: LoopRegion, array: str, index: str) -> None:
    """Give a LoopRegion a body that writes one element -- enough for a valid, countable loop."""
    state = loop.add_state("body", is_start_block=True)
    tasklet = state.add_tasklet("t", {}, {"b"}, "b = 1.0")
    state.add_edge(tasklet, "b", state.add_access(array), None, dace.Memlet(f"{array}[{index}]"))


def loop_sdfg(name: str, bound: str) -> dace.SDFG:
    """An SDFG holding one sequential loop ``for i in 0 .. bound``."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [dace.symbol(bound)], dace.float64)
    loop = LoopRegion("the_loop", f"i < {bound}", "i", "i = 0", "i = i + 1", sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    fill_loop(loop, "A", "i")
    return sdfg


def map_sdfg() -> dace.SDFG:
    """An SDFG holding one parallel Map and nothing else."""
    sdfg = dace.SDFG("plain_map")
    sdfg.add_array("A", [20], dace.float64)
    sdfg.add_array("B", [20], dace.float64)
    state = sdfg.add_state("s")
    state.add_mapped_tasklet(
        "m",
        dict(i="0:20"),
        dict(a=dace.Memlet("A[i]")),
        "b = a * 2.0",
        dict(b=dace.Memlet("B[i]")),
        external_edges=True,
    )
    return sdfg


def guarded_sdfg(fallback_bound: str = "N") -> dace.SDFG:
    """``if N > 0: <Map> else: <seq loop>`` -- the runtime-guarded parallelization."""
    sdfg = dace.SDFG("guarded")
    sdfg.add_array("A", [dace.symbol("N")], dace.float64)
    guard = ConditionalBlock("guard", sdfg=sdfg)
    sdfg.add_node(guard, is_start_block=True)
    parallel = ControlFlowRegion("par", sdfg=sdfg)
    guard.add_branch(CodeBlock("N > 0"), parallel)
    state = parallel.add_state("ps", is_start_block=True)
    state.add_mapped_tasklet("m", dict(i="0:N"), {}, "b = 1.0", dict(b=dace.Memlet("A[i]")), external_edges=True)
    sequential = ControlFlowRegion("seq", sdfg=sdfg)
    guard.add_branch(None, sequential)
    loop = LoopRegion("fallback", f"i < {fallback_bound}", "i", "i = 0", "i = i + 1", sdfg=sdfg)
    sequential.add_node(loop, is_start_block=True)
    fill_loop(loop, "A", "i")
    return sdfg


def loop_in_map_sdfg(bound: str = "10") -> dace.SDFG:
    """A Map whose body is a nested SDFG holding a sequential loop -- a tile / wavefront body."""
    inner = dace.SDFG("inner")
    inner.add_array("a", [10], dace.float64)
    loop = LoopRegion("inner_loop", f"k < {bound}", "k", "k = 0", "k = k + 1", sdfg=inner)
    inner.add_node(loop, is_start_block=True)
    fill_loop(loop, "a", "k")

    outer = dace.SDFG("outer")
    outer.add_array("A", [4, 10], dace.float64)
    state = outer.add_state("s")
    entry, exit_ = state.add_map("m", dict(i="0:4"))
    nested = state.add_nested_sdfg(inner, {}, {"a"})
    state.add_edge(entry, None, nested, None, dace.Memlet())
    exit_.add_in_connector("IN_A")
    exit_.add_out_connector("OUT_A")
    state.add_edge(nested, "a", exit_, "IN_A", dace.Memlet("A[i, 0:10]"))
    state.add_edge(exit_, "OUT_A", state.add_access("A"), None, dace.Memlet("A[0:4, 0:10]"))
    return outer


def libnode_sdfg() -> dace.SDFG:
    """One ``Reduce`` and one ``Scan``, the two lifted operators with their own buckets."""
    sdfg = dace.SDFG("libnodes")
    sdfg.add_array("A", [10], dace.float64)
    sdfg.add_array("B", [1], dace.float64)
    sdfg.add_array("C", [10], dace.float64)
    state = sdfg.add_state("s")
    reduce_node = Reduce("red", "lambda a, b: a + b", None, 0.0)
    state.add_node(reduce_node)
    state.add_edge(state.add_access("A"), None, reduce_node, "_in_reduce", dace.Memlet("A[0:10]"))
    state.add_edge(reduce_node, "_out_reduce", state.add_access("B"), None, dace.Memlet("B[0]"))
    scan_node = Scan("sc")
    state.add_node(scan_node)
    state.add_edge(state.add_access("A"), None, scan_node, "_scan_in", dace.Memlet("A[0:10]"))
    state.add_edge(scan_node, "_scan_out", state.add_access("C"), None, dace.Memlet("C[0:10]"))
    return sdfg


def only(record, bucket: str) -> None:
    """Assert exactly one construct landed in ``bucket`` and nothing landed anywhere else."""
    assert record.buckets[bucket] == 1, record.buckets
    assert sum(record.buckets.values()) == 1, record.buckets
    assert record.total == 1


def test_a_plain_map_lands_in_the_map_bucket() -> None:
    only(classify(map_sdfg()), "map")


def test_reduce_and_scan_are_separate_buckets_not_one() -> None:
    """Scan is a recognized sequential operator; folding it into reduce would count it as
    parallelized work it is not."""
    buckets = classify(libnode_sdfg()).buckets
    assert buckets["reduce"] == 1
    assert buckets["scan"] == 1
    assert sum(buckets.values()) == 2


def test_an_unaccounted_sequential_loop_is_residual_with_triage_detail() -> None:
    record = classify(loop_sdfg("residual", "N"))
    only(record, "residual")
    described = record.residual_loops[0]
    assert described.label == "the_loop"
    assert described.loop_variable == "i"
    assert "N" in described.condition
    assert described.bound_symbols == ("N",)


def test_a_timestep_bounded_loop_is_its_own_bucket_not_residual() -> None:
    record = classify(loop_sdfg("timestep", "TSTEPS"))
    only(record, "timestep")
    assert not record.residual_loops


def test_the_sequential_fallback_of_a_runtime_guard_counts_as_parallelized() -> None:
    """``if cond: <Map> else: <seq loop>`` -- the fallback loop is NOT residual: the kernel WAS
    parallelized, under a predicate."""
    record = classify(guarded_sdfg())
    assert record.buckets["parallel_under_contract"] == 1
    assert record.buckets["map"] == 1
    assert record.buckets["residual"] == 0


def test_a_loop_inside_a_map_body_is_parallel_work_not_residual() -> None:
    record = classify(loop_in_map_sdfg())
    assert record.buckets["inmap"] == 1
    assert record.buckets["map"] == 1
    assert record.buckets["residual"] == 0


def test_a_guarded_fallback_outranks_the_timestep_bucket() -> None:
    """A guarded fallback whose bound happens to be a timestep symbol was still parallelized."""
    record = classify(guarded_sdfg("TSTEPS"))
    assert record.buckets["parallel_under_contract"] == 1
    assert record.buckets["timestep"] == 0


def test_a_map_body_outranks_the_timestep_bucket() -> None:
    """A timestep-bounded loop that is the body of a Map is parallel work, not a deliberate miss."""
    record = classify(loop_in_map_sdfg("TSTEPS"))
    assert record.buckets["inmap"] == 1
    assert record.buckets["timestep"] == 0


def test_the_buckets_partition_the_total_for_every_shape() -> None:
    """The invariant every record must hold: the buckets sum to maps + reduces + scans + loops."""
    sdfg = dace.SDFG("mixed")
    sdfg.add_array("A", [dace.symbol("N")], dace.float64)
    residual = LoopRegion("residual_loop", "i < N", "i", "i = 0", "i = i + 1", sdfg=sdfg)
    sdfg.add_node(residual, is_start_block=True)
    fill_loop(residual, "A", "i")
    timestep = LoopRegion("timestep_loop", "t < TSTEPS", "t", "t = 0", "t = t + 1", sdfg=sdfg)
    sdfg.add_node(timestep)
    fill_loop(timestep, "A", "t")
    sdfg.add_edge(residual, timestep, dace.InterstateEdge())
    record = classify(sdfg)
    assert record.buckets["residual"] == 1
    assert record.buckets["timestep"] == 1
    assert sum(record.buckets.values()) == record.total == 2
    for other in (
        classify(map_sdfg()),
        classify(guarded_sdfg()),
        classify(loop_in_map_sdfg()),
        classify(libnode_sdfg()),
    ):
        assert sum(other.buckets[b] for b in BUCKETS) == other.total


#: Bound spellings the two predicates are cross-checked on. ``t_steps`` is on the list on purpose:
#: the substring rule does NOT match it (the underscore breaks ``TSTEP``), and both predicates must
#: be wrong about it in the same way or the SDFG port has drifted from the AST rule it carries over.
CROSS_CHECK_BOUNDS = list(TIMESTEP_SYMBOLS) + ["t_steps", "niter", "n_timesteps_total", "N", "M", "LEN_1D", "nx"]


@pytest.mark.parametrize("bound", CROSS_CHECK_BOUNDS)
def test_the_sdfg_timestep_predicate_matches_the_ast_predicate_on_the_same_bound(bound: str) -> None:
    source = ast.parse(f"for t in range({bound}):\n    pass").body[0]
    loop = LoopRegion("l", f"t < {bound}", "t", "t = 0", "t = t + 1")
    assert is_timestep_loop_region(loop) == is_timestep_loop(source)


def test_the_loop_variable_itself_is_never_a_bound_symbol() -> None:
    loop = LoopRegion("l", "i < N", "i", "i = 0", "i = i + 1")
    assert loop_bound_symbols(loop) == ["N"]


#: One set of raw counts, four readings: the whole point of raw records is that these are
#: re-derivable without re-measuring. P = 6+1+1+2 = 10 parallel-bucket constructs.
SAMPLE_COUNTS = dict(map=6, reduce=1, parallel_under_contract=1, inmap=2, timestep=4, residual=6, scan=3, libnode=5)


def sample_agg() -> dict[str, int]:
    agg = dict.fromkeys(BUCKETS, 0)
    agg["libnode"] = 0
    agg.update(SAMPLE_COUNTS)
    return agg


#: ``(definition, numerator, denominator)`` -- worked by hand off SAMPLE_COUNTS.
EXPECTED_RATES = [
    ("libnode_parallel", 15, 25),
    ("libnode_parallel_no_timestep", 15, 21),
    ("libnode_neutral", 10, 20),
    ("libnode_neutral_no_timestep", 10, 16),
]


def test_the_rate_definition_table_is_exactly_these_four_named_entries() -> None:
    """A definition added or renamed must land here, not silently change a headline number."""
    assert list(RATE_DEFINITIONS) == [name for name, _, _ in EXPECTED_RATES]
    assert DEFAULT_RATE in RATE_DEFINITIONS


@pytest.mark.parametrize(("name", "numerator", "denominator"), EXPECTED_RATES)
def test_the_same_raw_counts_read_four_different_ways(name: str, numerator: int, denominator: int) -> None:
    computed = rate(sample_agg(), RATE_DEFINITIONS[name])
    assert computed["numerator"] == numerator
    assert computed["denominator"] == denominator
    assert computed["value"] == pytest.approx(numerator / denominator)


def test_the_default_definition_counts_libnode_parallel_and_keeps_timestep_in_the_denominator() -> None:
    definition = RATE_DEFINITIONS[DEFAULT_RATE]
    assert "libnode" in definition.numerator
    assert "timestep" in definition.denominator
    assert set(PARALLEL_BUCKETS) <= set(definition.numerator)


@pytest.mark.parametrize("name", list(RATE_DEFINITIONS))
def test_scan_sits_on_neither_side_of_any_rate_definition(name: str) -> None:
    definition = RATE_DEFINITIONS[name]
    assert NEUTRAL_ALWAYS[0] not in definition.numerator
    assert NEUTRAL_ALWAYS[0] not in definition.denominator


@pytest.mark.parametrize("name", list(RATE_DEFINITIONS))
def test_a_rates_numerator_is_always_a_subset_of_its_denominator(name: str) -> None:
    """A rate whose numerator can exceed its denominator is not a rate."""
    definition = RATE_DEFINITIONS[name]
    assert set(definition.numerator) <= set(definition.denominator)


def test_rates_applies_every_definition_to_one_aggregate() -> None:
    computed = rates(sample_agg())
    assert list(computed) == list(RATE_DEFINITIONS)
    assert {name: (r["numerator"], r["denominator"]) for name, r in computed.items()} == {
        name: (n, d) for name, n, d in EXPECTED_RATES
    }


def test_an_empty_denominator_reports_none_not_a_zero_division() -> None:
    agg = dict.fromkeys(BUCKETS, 0)
    agg.update(libnode=0, scan=2)
    for name in RATE_DEFINITIONS:
        assert rate(agg, RATE_DEFINITIONS[name])["value"] is None


def test_metric_rows_has_one_row_per_bucket_plus_libnode_and_total() -> None:
    """The shape a ``kernel_metrics`` sink reads: one row per bucket named ``parallelism.<bucket>``,
    plus ``parallelism.libnode`` and ``parallelism.total`` -- no rate baked in."""
    record = classify(guarded_sdfg())
    rows = dict(record.metric_rows())
    assert set(rows) == {f"parallelism.{b}" for b in BUCKETS} | {"parallelism.libnode", "parallelism.total"}
    assert rows["parallelism.parallel_under_contract"] == 1
    assert rows["parallelism.total"] == record.total
    assert rows["parallelism.libnode"] == record.libnode
