# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""How much of a canonicalized SDFG is proven parallel -- the per-bucket taxonomy, on the SDFG.

A PORT of the taxonomy ``mpr-artifacts/experiments/parallelism/llr_parallelism.py`` built for the
CPF/MPR paper, onto hpcagent_bench as the framework's own metric. The structural PREDICATES
(``count``, ``in_parallel_scope``, ``guarded_fallback_loop_set``, ``cpu_params``) are imported from
``dace/tests/corpus/measure_parallelization.py``, never restated -- two copies of "what counts as a
guarded fallback" drift. The bucket NAMES and the rate-definition table are the same taxonomy,
re-declared here (they are literal tuples of strings, not logic, so there is nothing to drift).

Every loop-level construct of a canonicalized SDFG lands in exactly one bucket, and the buckets
sum to ``maps + reduces + scans + loops`` (:func:`classify` asserts the partition):

  map                      -- lifted to a parallel Map.
  reduce                   -- lifted to a Reduce library node.
  scan                     -- lifted to a Scan library node. A recognized SEQUENTIAL operator:
                              neither parallelized nor residual, reported on its own.
  parallel_under_contract  -- the sequential fallback of ``if cond: <Map> else: <seq loop>``.
                              PARALLELIZED: the kernel was parallelized under a runtime predicate.
  timestep                 -- a loop whose bound names a time-stepping symbol. Deliberately left
                              sequential: parallelizing it would change the program's semantics.
  inmap                    -- a loop that is the body of a Map (a tile / wavefront body):
                              parallel work, not residual.
  residual                 -- a sequential loop not accounted for by any of the above.

``libnode`` (a MatMul, a BLAS call -- neither Reduce nor Scan) is counted alongside the buckets,
not inside the taxonomy: which side of a rate it lands on is the rate DEFINITION's choice.

Records hold RAW counts only, nothing pre-combined. The rate is read off them at report time
through a named :class:`RateDefinition`, so a decision about what counts as parallelized costs a
re-read of records already on disk, not a re-measurement of the corpus.
"""

import collections
import dataclasses
import functools
import math
import pathlib
import sys
import types
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

from hpcagent_bench import config, osinfo
from hpcagent_bench.frameworks.schema import KernelMetric

if TYPE_CHECKING:
    import pandas as pd
    from dace import SDFG
    from dace.sdfg.state import LoopRegion

#: The taxonomy, in report order. Every loop-level construct lands in exactly one.
BUCKETS = ("map", "reduce", "scan", "parallel_under_contract", "timestep", "inmap", "residual")
#: Buckets that count as PARALLELIZED in every rate definition below.
PARALLEL_BUCKETS = ("map", "reduce", "parallel_under_contract", "inmap")
#: Buckets that count as UNPARALLELIZED. ``timestep`` is deliberate, ``residual`` is not.
SEQUENTIAL_BUCKETS = ("timestep", "residual")
#: Never on either side of any rate definition: a recognized SEQUENTIAL operator is neither a
#: parallelization nor a miss.
NEUTRAL_ALWAYS = ("scan",)


class RateDefinition(NamedTuple):
    """One named way of turning raw bucket counts into a rate.

    ``numerator``/``denominator`` name raw count keys (the seven buckets plus ``libnode``); the
    denominator is spelled out in full rather than derived from the numerator, so a definition
    cannot quietly drop a term.
    """

    numerator: tuple[str, ...]
    denominator: tuple[str, ...]
    note: str


_P = PARALLEL_BUCKETS
#: ``libnode`` is a loop the pipeline recognized as a BLAS operator. Two readings are defensible:
#: recognition is the strongest form of parallelization (default), or it is an abstention. Crossed
#: with whether ``timestep`` sits in the denominator -- deliberately unparallelized work is still
#: unparallelized work.
RATE_DEFINITIONS: dict[str, RateDefinition] = {
    "libnode_parallel": RateDefinition(
        _P + ("libnode",),
        _P + ("libnode", "residual", "timestep"),
        "libnode counts as parallelized; timestep is in the denominator",
    ),
    "libnode_parallel_no_timestep": RateDefinition(
        _P + ("libnode",),
        _P + ("libnode", "residual"),
        "libnode counts as parallelized; timestep is dropped from the denominator",
    ),
    "libnode_neutral": RateDefinition(
        _P, _P + ("residual", "timestep"), "libnode is on neither side; timestep is in the denominator"
    ),
    "libnode_neutral_no_timestep": RateDefinition(
        _P, _P + ("residual",), "libnode is on neither side; timestep is dropped from the denominator"
    ),
}

#: The headline: a loop recognized as a BLAS operator IS parallelized, and a loop we decline to
#: parallelize still belongs in the denominator.
DEFAULT_RATE = "libnode_parallel"

METRIC_PREFIX = "parallelism."


def enabled() -> bool:
    """Whether ``metrics.parallelism`` is on (default: NO)."""
    return config.get_bool("metrics.parallelism", False)


@dataclasses.dataclass(frozen=True, slots=True)
class LoopDetail:
    """What the SDFG can say about one residual loop, for the triage list."""

    label: str
    sdfg: str
    loop_variable: str
    condition: str
    bound_symbols: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class ParallelismRecord:
    """One kernel's RAW taxonomy: bucket counts and nothing pre-combined."""

    buckets: dict[str, int]
    total: int
    libnode: int
    residual_loops: tuple[LoopDetail, ...]

    def metric_rows(self) -> list[tuple[str, int]]:
        """``(metric_name, value)`` pairs: one per bucket, plus libnode and total -- what a
        ``kernel_metrics`` row's ``metric``/``value`` columns are filled from, one row each.
        """
        rows = [(f"{METRIC_PREFIX}{bucket}", self.buckets[bucket]) for bucket in BUCKETS]
        rows.append((f"{METRIC_PREFIX}libnode", self.libnode))
        rows.append((f"{METRIC_PREFIX}total", self.total))
        return rows


def rows(
    record: ParallelismRecord,
    *,
    timestamp: int,
    benchmark: str,
    framework: str,
    flavor: str | None,
    impl: str,
    datatype: str,
) -> list[KernelMetric]:
    """One ``kernel_metrics`` row per bucket plus ``libnode``/``total``, stamped like the ``results``
    rows of the same run (:func:`autovec.rows` is the sibling this mirrors).

    ``detail`` names the pipeline (the ``framework`` column, plus ``flavor`` when there is one) so
    rows measured under two different SDFG pipelines are never pooled by a query that groups on
    ``metric`` alone -- ``framework``/``flavor`` are separate columns already, but a raw SQL group-by
    on ``metric`` only would otherwise average across pipelines silently.
    """
    build = config.get_str("record.build", "") or None
    detail = f"framework={framework}" if flavor is None else f"framework={framework} flavor={flavor}"
    return [
        KernelMetric(
            timestamp=timestamp,
            benchmark=benchmark,
            framework=framework,
            flavor=flavor,
            impl=impl,
            datatype=datatype,
            metric=name,
            value=float(value),
            detail=detail,
            build=build,
            cpu=osinfo.cpu_model(),
            node=osinfo.node_name(),
        )
        for name, value in record.metric_rows()
    ]


def column_name(framework: str, flavor: str | None) -> str:
    """The toolchain-column name a ``kernel_metrics`` row's ``framework``/``flavor`` pair reads as
    elsewhere in the repo (``canon`` table columns, ``envs/registry.yaml`` frameworks): the flavor
    appended with an underscore, or the framework alone when there is none.
    """
    return framework if flavor is None else f"{framework}_{flavor}"


def record_from_counts(counts: dict[str, float]) -> ParallelismRecord:
    """The inverse of :meth:`ParallelismRecord.metric_rows`: one kernel's raw taxonomy rebuilt from
    its ``metric -> value`` rows. ``residual_loops`` is not stored in ``kernel_metrics`` (only the
    count is), so a record rebuilt this way always carries an empty triage list.
    """
    buckets = {bucket: int(counts.get(f"{METRIC_PREFIX}{bucket}", 0.0)) for bucket in BUCKETS}
    libnode = int(counts.get(f"{METRIC_PREFIX}libnode", 0.0))
    total = int(counts.get(f"{METRIC_PREFIX}total", 0.0))
    return ParallelismRecord(buckets=buckets, total=total, libnode=libnode, residual_loops=())


def normalize_flavor(flavor: object) -> str | None:
    """A SQL NULL flavor comes back from pandas as ``float('nan')``, not ``None`` -- and two NaN
    values are never equal, so using one straight as a dict key would split one "no flavor" group
    into as many groups as it had rows. Collapse both spellings of "absent" to ``None``.
    """
    if flavor is None or (isinstance(flavor, float) and math.isnan(flavor)):
        return None
    return str(flavor)


def read_records(frame: "pd.DataFrame") -> dict[str, dict[str, ParallelismRecord]]:
    """``column -> {benchmark: ParallelismRecord}``, rebuilt from long-format ``kernel_metrics``
    rows (a pruned copy such as ``scripts/collect_parallelism.py`` writes, or the full table --
    every row not starting with :data:`METRIC_PREFIX` is ignored either way). One record per
    (framework, flavor, benchmark) triple; :func:`column_name` turns the pair into the column name.
    """
    grouped: dict[tuple[str, str | None, str], dict[str, float]] = collections.defaultdict(dict)
    for row in frame.itertuples(index=False):
        metric = str(row.metric)
        if not metric.startswith(METRIC_PREFIX):
            continue
        key = (str(row.framework), normalize_flavor(row.flavor), str(row.benchmark))
        grouped[key][metric] = float(row.value)
    out: dict[str, dict[str, ParallelismRecord]] = collections.defaultdict(dict)
    for (framework, flavor, benchmark), counts in grouped.items():
        out[column_name(framework, flavor)][benchmark] = record_from_counts(counts)
    return out


def import_dace_tests_corpus() -> types.ModuleType:
    """dace's ``tests.corpus.measure_parallelization``, importable despite the name collision.

    hpcagent_bench has its OWN top-level ``tests`` package (this repo's test suite), and whatever
    imported it first already bound ``sys.modules['tests']`` to it -- pytest's rootdir import,
    typically. dace's corpus module does ``from tests.corpus import corpus_suite`` internally,
    which cannot be renamed (it is dace's own file), so the only way to reach dace's ``tests`` tree
    under its real name is to swap the ``tests`` entry out of ``sys.modules`` for the span of this
    one import and put hpcagent_bench's own back immediately after -- the two never coexist under
    one name, so they never coexist at all, only in sequence. Not thread-safe against a concurrent
    unrelated ``import tests``; :func:`load_predicates` calls this at most once per process.
    """
    saved_path = list(sys.path)
    saved_modules = {name: mod for name, mod in sys.modules.items() if name == "tests" or name.startswith("tests.")}
    try:
        sys.path.insert(0, str(pathlib.Path(dace_root_for_tests())))
        for name in list(saved_modules):
            del sys.modules[name]
        from tests.corpus import measure_parallelization as measure

        return measure
    finally:
        for name in list(sys.modules):
            if name == "tests" or name.startswith("tests."):
                del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


def dace_root_for_tests(root: pathlib.Path | None = None) -> pathlib.Path:
    """Checkout root of the imported ``dace`` (or ``root``, for a test that wants a controlled
    path without patching): not an editable install's own path, so ``tests.corpus`` is located
    relative to the package actually loaded rather than a hardcoded checkout path.

    Raises :class:`ModuleNotFoundError` naming the missing directory when ``tests/corpus`` is not
    there, instead of leaving the caller to hit a bare ``ModuleNotFoundError`` for ``tests.corpus``
    three frames down with no path in it. A WHEEL install of dace (the PyPI release, or pip
    resolving the ``dace`` extra's git+https reference into a build) never ships ``tests/``; only a
    checkout does, which is what ``.github/actions/setup/action.yml``'s "DaCe checkout" step
    installs on CI (``git clone`` + ``pip install -e``).
    """
    if root is None:
        import dace

        root = pathlib.Path(dace.__file__).resolve().parents[1]
    corpus = root / "tests" / "corpus"
    if not corpus.is_dir():
        raise ModuleNotFoundError(
            f"{corpus} is missing: the dace installed from {root} ships no tests/ directory. "
            "hpcagent_bench.metrics.parallelism imports dace's own "
            "tests/corpus/measure_parallelization.py predicates rather than restating them, so it "
            "needs a git CHECKOUT of dace, not a wheel. CI installs one via "
            ".github/actions/setup/action.yml's 'DaCe checkout' step; locally, "
            "`pip install -e /path/to/a/dace/checkout`."
        )
    return root


@functools.lru_cache(maxsize=1)
def load_predicates() -> types.ModuleType:
    """dace's ``tests.corpus.measure_parallelization`` module, imported once.

    Lazy and cached: importing dace costs seconds, and most callers of this module (the rate
    arithmetic, the CSV/report helpers) never need it.
    """
    # canonicalize FIRST: the clean entry that loads passes.vectorization + interstate in the
    # right order: dace.transformation.interstate ahead of it trips a circular import.
    from dace.transformation.passes.canonicalize import canonicalize  # noqa: F401

    return import_dace_tests_corpus()


def all_loop_regions(sdfg: "SDFG") -> "list[LoopRegion]":
    """Every ``LoopRegion`` at the scope :func:`load_predicates`'s ``count`` counts loops in --
    across nested SDFGs as well as nested regions, so a loop inside a nested SDFG cannot vanish
    from the taxonomy while its Maps still count.
    """
    from dace.sdfg.state import LoopRegion

    return [
        cfr for sd in sdfg.all_sdfgs_recursive() for cfr in sd.all_control_flow_regions() if isinstance(cfr, LoopRegion)
    ]


def loop_bound_symbols(loop: "LoopRegion") -> list[str]:
    """Symbol names in a loop's init/condition/update, minus the loop variable itself."""
    names: set[str] = set()
    for code in loop.get_meta_codeblocks():
        names |= set(code.get_free_symbols())
    names.discard(loop.loop_variable)
    return sorted(names)


def is_timestep_loop_region(loop: "LoopRegion", timestep_symbols: Sequence[str] | None = None) -> bool:
    """True when the loop's bound names a time-stepping symbol -- a loop deliberately left
    sequential. Matched case-insensitively as a SUBSTRING, the same rule
    ``numpyto_common.parallelism.is_timestep_loop`` applies to a python ``for t in range(TSTEPS)``,
    so the two predicates answer the same question about the same bound symbol on two different IRs.
    """
    from numpyto_common.parallelism import TIMESTEP_SYMBOLS

    syms = tuple(s.lower() for s in (timestep_symbols or TIMESTEP_SYMBOLS))
    return any(s in name.lower() for name in loop_bound_symbols(loop) for s in syms)


def describe_loop(loop: "LoopRegion") -> LoopDetail:
    return LoopDetail(
        label=loop.label,
        sdfg=loop.sdfg.name if loop.sdfg is not None else "",
        loop_variable=loop.loop_variable,
        condition=loop.loop_condition.as_string if loop.loop_condition is not None else "",
        bound_symbols=tuple(loop_bound_symbols(loop)),
    )


def classify(sdfg: "SDFG") -> ParallelismRecord:
    """Bucket every loop-level construct of ``sdfg``.

    Loop precedence is ``inmap`` > ``parallel_under_contract`` > ``timestep`` > ``residual``: a
    loop that ended up as parallel work, or as the fallback of a guard that WAS parallelized, is
    reported as such whatever its bound is named.
    """
    measure = load_predicates()
    counts = measure.count(sdfg)
    columns = dict(zip(measure.COUNTERS, counts, strict=True))
    guarded = {id(lp) for lp in measure.guarded_fallback_loop_set(sdfg)}
    buckets = dict.fromkeys(BUCKETS, 0)
    buckets["map"] = columns["maps"]
    buckets["reduce"] = columns["reduce"]
    buckets["scan"] = columns["scan"]
    residual: list[LoopDetail] = []
    for loop in all_loop_regions(sdfg):
        bucket = bucket_for_loop(loop, guarded, measure)
        if bucket == "residual":
            residual.append(describe_loop(loop))
        buckets[bucket] += 1
    total = columns["loops"] + columns["maps"] + columns["reduce"] + columns["scan"]
    if sum(buckets.values()) != total:
        raise AssertionError(f"taxonomy does not partition: {buckets} sums to {sum(buckets.values())}, not {total}")
    if len(guarded) != measure.guarded_fallback_loops(sdfg):
        raise AssertionError("the guarded set and the guarded count disagree; the predicate drifted")
    return ParallelismRecord(buckets=buckets, total=total, libnode=columns["libnode"], residual_loops=tuple(residual))


def bucket_for_loop(loop: "LoopRegion", guarded_ids: set[int], measure: types.ModuleType) -> str:
    """One loop's bucket, applying the stated precedence."""
    if measure.in_parallel_scope(loop):
        return "inmap"
    if id(loop) in guarded_ids:
        return "parallel_under_contract"
    if is_timestep_loop_region(loop):
        return "timestep"
    return "residual"


def totals(records: Sequence[ParallelismRecord]) -> dict[str, int]:
    """Sum RAW per-kernel counts. Nothing combined beyond addition -- which counts land on which
    side of a rate is :data:`RATE_DEFINITIONS`' business, applied afterward by :func:`rate`.
    """
    agg = dict.fromkeys(BUCKETS, 0)
    agg["total"] = 0
    agg["libnode"] = 0
    for rec in records:
        for bucket, value in rec.buckets.items():
            agg[bucket] += value
        agg["total"] += rec.total
        agg["libnode"] += rec.libnode
    return agg


def rate(agg: dict[str, int], definition: RateDefinition) -> dict[str, Any]:
    """Apply one definition to raw counts. Carries its numerator/denominator terms with it, so a
    percentage is never quoted without them.
    """
    numerator = sum(agg.get(k, 0) for k in definition.numerator)
    denominator = sum(agg.get(k, 0) for k in definition.denominator)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "numerator_terms": " + ".join(definition.numerator),
        "denominator_terms": " + ".join(definition.denominator),
        "value": (numerator / denominator) if denominator else None,
        "note": definition.note,
    }


def rates(agg: dict[str, int]) -> dict[str, dict[str, Any]]:
    """Every named definition applied to the same raw counts, in table order."""
    return {name: rate(agg, defn) for name, defn in RATE_DEFINITIONS.items()}


def record_term(record: ParallelismRecord, term: str) -> int:
    """One raw count off ``record`` by its rate-definition name (a bucket, or ``libnode``/``total``)."""
    if term == "libnode":
        return record.libnode
    if term == "total":
        return record.total
    return record.buckets.get(term, 0)


class BenchmarkClass(NamedTuple):
    """One kernel's PER-BENCHMARK reading under one rate definition, read straight off its raw
    record -- never off a stored, pre-combined rate."""

    parallelized: bool
    fully_parallelized: bool


def classify_benchmark(record: ParallelismRecord, definition: RateDefinition) -> BenchmarkClass:
    """``parallelized`` = the kernel has at least one construct in ``definition``'s numerator.
    ``fully_parallelized`` = that, AND no residual loop is left over (``residual == 0``)."""
    numerator = sum(record_term(record, term) for term in definition.numerator)
    parallelized = numerator > 0
    fully_parallelized = parallelized and record.buckets["residual"] == 0
    return BenchmarkClass(parallelized=parallelized, fully_parallelized=fully_parallelized)


@dataclasses.dataclass(frozen=True, slots=True)
class BenchmarkCounts:
    """How many kernels of a corpus read as parallelized, fully parallelized, or neither, under one
    rate definition -- a count of :class:`BenchmarkClass` values, not a rate."""

    parallelized: int
    fully_parallelized: int
    neither: int
    total: int


def benchmark_counts(records: Sequence[ParallelismRecord], definition: RateDefinition) -> BenchmarkCounts:
    """:func:`classify_benchmark` applied to every record, tallied. Recomputed from the raw records
    every call -- nothing pre-combined is stored between a rate reading and this one."""
    classified = [classify_benchmark(rec, definition) for rec in records]
    parallelized = sum(1 for c in classified if c.parallelized)
    fully_parallelized = sum(1 for c in classified if c.fully_parallelized)
    return BenchmarkCounts(
        parallelized=parallelized,
        fully_parallelized=fully_parallelized,
        neither=len(classified) - parallelized,
        total=len(classified),
    )


def report_lines(records: Sequence[ParallelismRecord]) -> list[str]:
    """Every named rate definition applied to the SAME raw ``records``, each line spelling out its own
    numerator/denominator terms plus the per-kernel parallelized/fully_parallelized counts under that
    same definition -- the report reading :data:`RATE_DEFINITIONS` was built to support: a rate and
    the kernel counts behind it are never allowed to silently use different definitions."""
    agg = totals(records)
    lines: list[str] = []
    for name, definition in RATE_DEFINITIONS.items():
        computed = rate(agg, definition)
        counts = benchmark_counts(records, definition)
        tag = " (default)" if name == DEFAULT_RATE else ""
        value = "n/a" if computed["value"] is None else f"{computed['value']:.3f}"
        lines.append(f"{name}{tag}: ({computed['numerator_terms']}) / ({computed['denominator_terms']}) = {value}")
        lines.append(
            f"  parallelized [kernel has >=1 of: {computed['numerator_terms']}]: "
            f"{counts.parallelized}/{counts.total} kernels"
        )
        lines.append(
            f"  fully_parallelized [parallelized AND residual == 0]: {counts.fully_parallelized}/{counts.total} kernels"
        )
    return lines


def print_report(records: Sequence[ParallelismRecord]) -> None:
    """:func:`report_lines`, printed one line at a time."""
    for line in report_lines(records):
        print(line)
