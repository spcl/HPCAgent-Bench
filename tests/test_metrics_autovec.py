# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Auto-vectorization counts read off a compiler's optimization report.

The CPF/MPR comparison divides these counts between two pipelines, so a loop counted twice, a loop of the other
precision counted in, or a vanished source counted as a kernel without loops, moves the rate that paper reports.
"""

import json
import pathlib
import sqlite3

import pytest
from sqlmodel import Session

from hpcagent_bench import config, cpf_cache, languages
from hpcagent_bench.benchmarks import cpp_runtime
from hpcagent_bench.frameworks.schema import KERNEL_METRICS_TABLE, results_engine
from hpcagent_bench.harness import recording
from hpcagent_bench.metrics import autovec

KNOB = "HPCAGENT_BENCH_METRICS_AUTOVEC"

#: One vectorizable loop, a nest whose inner loop vectorizes, a loop around an opaque call, and four
#: statements SLP can pack: every count has something to count.
SOURCE = """#include <stddef.h>
extern double opaque(double);
void kernel(double *restrict a, const double *restrict b, double *restrict c, size_t n, size_t m) {
  for (size_t i = 0; i < n; i++) {
    a[i] = 2.0 * b[i];
  }
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      c[i * m + j] = b[j] + 1.0;
    }
  }
  for (size_t i = 0; i < n; i++) {
    a[i] = opaque(b[i]);
  }
}
void pair(double *restrict a, const double *restrict b, const double *restrict c) {
  a[0] = b[0] + c[0];
  a[1] = b[1] + c[1];
  a[2] = b[2] + c[2];
  a[3] = b[3] + c[3];
}
"""

ONE_LOOP = """#include <stddef.h>
void single(float *restrict a, const float *restrict b, size_t n) {
  for (size_t i = 0; i < n; i++) {
    a[i] = b[i];
  }
}
"""


def report_of(tmp_path: pathlib.Path, compiler: str, *sources: tuple[str, str]) -> str:
    """The report the production report compile writes for ``sources`` (name, text) under ``compiler``."""
    paths = []
    for name, text in sources:
        path = tmp_path / name
        path.write_text(text)
        paths.append(("c", path))
    report = cpp_runtime.report_compile(
        paths, tmp_path / f"build-{compiler}", compiler, languages.report_flags("c", compiler=compiler)
    )
    assert report is not None, compiler
    return report


def test_banners_split_the_report_and_text_before_the_first_belongs_to_no_compile() -> None:
    report = "pipeline: gemm\n$ gcc -O3 -c /a/k.c -o k.o\nfirst\n$ 'g++' -c b.cpp\nsecond\nthird"
    got = autovec.compiles(report)
    assert got == (
        autovec.Compile(("gcc", "-O3", "-c", "/a/k.c", "-o", "k.o"), "first"),
        autovec.Compile(("g++", "-c", "b.cpp"), "second\nthird"),
    ), got


def test_a_compiles_sources_are_its_source_arguments_but_never_the_output_target() -> None:
    unit = autovec.Compile(("gcc", "-include", "vecmath.h", "-c", "/a/k_fp64.c", "-o", "/a/out.c"), "")
    assert unit.sources() == (pathlib.Path("/a/k_fp64.c"),)


@pytest.mark.parametrize(
    ("name", "datatype", "counted"),
    [
        ("gemm_fp64.c", "float64", True),
        ("gemm_fp32.c", "float64", False),
        ("gemm_fp32.c", "float32", True),
        ("argmax_value_fp64_cpf.cpp", "float64", True),
        ("argmax_value_fp64_cpf.cpp", "float32", False),
        ("gemm.cpp", "float32", True),
    ],
)
def test_a_tagged_source_counts_for_its_own_precision_only(name: str, datatype: str, counted: bool) -> None:
    assert autovec.of_precision(pathlib.Path(name), datatype) is counted


@pytest.mark.parametrize(
    ("compiler", "want"),
    [
        # gcc refuses the nest's outer loop out loud ("complicated access pattern") ...
        (
            "gcc",
            {
                "loops": 4,
                "loops_vectorized": 2,
                "loops_missed": 2,
                "loops_unreported": 0,
                "inner_loops": 3,
                "inner_loops_vectorized": 2,
            },
        ),
        # ... clang says nothing about it, so the same loop is unreported rather than missed.
        (
            "clang",
            {
                "loops": 4,
                "loops_vectorized": 2,
                "loops_missed": 1,
                "loops_unreported": 1,
                "inner_loops": 3,
                "inner_loops_vectorized": 2,
            },
        ),
    ],
)
def test_every_loop_of_the_compiled_source_lands_in_exactly_one_verdict(
    tmp_path: pathlib.Path, compiler: str, want: dict[str, int]
) -> None:
    counts = autovec.count(report_of(tmp_path, compiler, ("k_fp64.c", SOURCE)), "float64").counts
    assert {name: counts[name] for name in want} == want, counts


@pytest.mark.parametrize("compiler", ["gcc", "clang"])
def test_a_nest_counts_once_and_a_vectorized_statement_group_outside_loops_is_slp(
    tmp_path: pathlib.Path, compiler: str
) -> None:
    counts = autovec.count(report_of(tmp_path, compiler, ("k_fp64.c", SOURCE)), "float64").counts
    assert (counts["nests"], counts["nests_vectorized"], counts["slp_vectorized"]) == (3, 2, 1), counts


#: The shapes CPF and DaCe emit: a parallel loop, a simd loop, and a guarded parallel loop with its plain fallback.
OPENMP_SOURCE = """#include <stddef.h>
void par(double *restrict a, const double *restrict b, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        a[i] = 2.0 * b[i];
    }
}
void simd(double *restrict a, const double *restrict b, size_t n) {
    #pragma omp simd
    for (size_t i = 0; i < n; i++) {
        a[i] = b[i] + 1.0;
    }
}
void guarded(double *restrict a, const double *restrict b, size_t n) {
    if (n > 64) {
        #pragma omp parallel for simd \\
            schedule(static)
        for (size_t i = 0; i < n; i++) {
            a[i] = b[i] * b[i];
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            a[i] = b[i] * b[i];
        }
    }
}
"""


@pytest.mark.parametrize("compiler", ["gcc", "clang"])
def test_an_openmp_loop_is_counted_whichever_line_its_compiler_reports_it_on(
    tmp_path: pathlib.Path, compiler: str
) -> None:
    """clang reports an OpenMP loop on its pragma and gcc inside its body; both are the loop, or every CPF
    loop under clang reads as unreported and the CPF rate is a compiler artifact."""
    counts = autovec.count(report_of(tmp_path, compiler, ("k_fp64.c", OPENMP_SOURCE)), "float64").counts
    assert (counts["loops"], counts["loops_vectorized"], counts["loops_unreported"]) == (4, 4, 0), counts


def test_a_statement_group_packed_inside_a_loop_is_slp_and_not_the_loop_vectorizing(tmp_path: pathlib.Path) -> None:
    """gcc's report on a CPF argmax form: the loop is refused, and only the reduction combiner outlined at the
    pragma is SLP packed. Counting that as the loop would credit CPF with a loop no compiler vectorized."""
    source = tmp_path / "argmax_fp64_cpf.c"
    source.write_text(
        "void k(const double *a, long n) {\n"
        "    double best = a[0];\n"
        "    #pragma omp parallel for reduction(max : best)\n"
        "    for (long i = 1; i < n; ++i) {\n"
        "        if (a[i] > best) { best = a[i]; }\n"
        "    }\n"
        "}\n"
    )
    report = (
        f"$ gcc -O3 -fopt-info-vec-optimized -fopt-info-vec-missed -c {source} -o k.o\n"
        f"{source}:4:39: missed: couldn't vectorize loop\n"
        f"{source}:5:20: missed: not vectorized: unsupported use in stmt.\n"
        f"{source}:3:21: optimized: basic block part vectorized using 16 byte vectors\n"
    )
    counts = autovec.count(report, "float64").counts
    assert (counts["loops_vectorized"], counts["loops_missed"], counts["slp_vectorized"]) == (0, 1, 1), counts


#: A dead multi-line helper with a loop (lines 2-9), a live one-liner, a dead one-liner (11), and a static function
#: declared before it is defined and called by the kernel: the shapes of the translator's C prelude.
HELPERS = """#include <stddef.h>
static inline long dead_pow(long base, long exp) {
    long result = 1;
    while (exp > 0) {
        result *= base;
        exp -= 1;
    }
    return result;
}
static inline double twice(double x) { return 2.0 * x; }
static inline double unused(double x) { return x; }
static void fill(double *a, size_t n);
static void fill(double *a, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] = twice(1.0);
    }
}
void kernel(double *a, size_t n) {
    fill(a, n);
}
"""


def test_only_a_static_function_no_other_line_names_is_dead_code() -> None:
    assert autovec.dead_ranges(HELPERS) == ((2, 9), (11, 11))


def test_a_loop_in_dead_code_is_not_counted(tmp_path: pathlib.Path) -> None:
    """Every translated baseline carries an uncalled integer-power helper with a loop; counting it adds a loop
    the compiler never sees to all 248 kernels."""
    counts = autovec.count(report_of(tmp_path, "gcc", ("k_fp64.c", HELPERS)), "float64").counts
    assert (counts["loops"], counts["nests"]) == (1, 1), counts


def test_the_other_precisions_source_in_the_same_report_is_not_counted(tmp_path: pathlib.Path) -> None:
    """A native column compiles its fp64 and fp32 sources in one report; counting both doubles every loop."""
    report = report_of(tmp_path, "gcc", ("k_fp64.c", SOURCE), ("k_fp32.c", ONE_LOOP))
    assert autovec.count(report, "float64").counts["loops"] == 4
    assert autovec.count(report, "float32").counts["loops"] == 1


def test_a_report_that_compiles_no_source_of_the_precision_is_refused(tmp_path: pathlib.Path) -> None:
    report = report_of(tmp_path, "gcc", ("k_fp32.c", ONE_LOOP))
    with pytest.raises(ValueError, match="float64"):
        autovec.count(report, "float64")


def test_a_compiled_source_that_is_gone_is_refused_by_name(tmp_path: pathlib.Path) -> None:
    report = report_of(tmp_path, "gcc", ("k_fp64.c", SOURCE))
    (tmp_path / "k_fp64.c").unlink()
    with pytest.raises(FileNotFoundError, match="k_fp64.c"):
        autovec.count(report, "float64")


def test_the_detail_names_the_family_and_the_cost_model_the_counts_were_taken_under(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HPCAGENT_BENCH_PERF_REPORTS_VECT_COST_MODEL", "unlimited")
    detail = autovec.count(report_of(tmp_path, "clang", ("k_fp64.c", SOURCE)), "float64").detail
    assert detail.startswith("family=clang cost_model=unlimited "), detail


def test_the_count_is_off_unless_switched_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KNOB, raising=False)
    assert not autovec.enabled()
    monkeypatch.setenv(KNOB, "1")
    assert autovec.enabled()


def test_every_count_is_one_row_that_round_trips_through_the_results_db(tmp_path: pathlib.Path) -> None:
    measured = autovec.Measured(counts={name: i for i, name in enumerate(autovec.COUNTS)}, detail="family=gcc")
    made = autovec.rows(
        measured, timestamp=7, benchmark="gemm", framework="cc", flavor=None, impl="default", datatype="float64"
    )
    db = tmp_path / "results.db"
    with Session(results_engine(str(db))) as session:
        session.add_all(made)
        session.commit()
    with sqlite3.connect(db) as conn:
        stored = conn.execute(f"SELECT metric, value, detail FROM {KERNEL_METRICS_TABLE} ORDER BY id").fetchall()
    assert stored == [(f"autovec.{name}", float(i), "family=gcc") for i, name in enumerate(autovec.COUNTS)], stored


def test_a_sweep_with_the_count_on_stores_it_beside_its_results_under_the_same_timestamp(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep is where baseline counts come from; a row that cannot be joined to its run's results is lost."""
    from hpcagent_bench.frameworks import Benchmark, Test, generate_framework

    monkeypatch.setenv(KNOB, "1")
    db = str(tmp_path / "hpcagent_bench.db")
    config.set_override("record.db_path", db)
    config.set_override("record.allow_memory_db", True)
    try:
        test = Test(Benchmark("tsvc_2_s212"), generate_framework("cc"), generate_framework("numpy"))
        test.run("S", validate=True, repeat=1, ignore_errors=True, datatype="float64")
    finally:
        config.clear_override("record.db_path")
        config.clear_override("record.allow_memory_db")
    with sqlite3.connect(recording.ensure_aggregated(db)) as conn:
        results = set(conn.execute("SELECT timestamp, framework FROM results").fetchall())
        metrics = conn.execute(f"SELECT timestamp, framework, metric, detail FROM {KERNEL_METRICS_TABLE}").fetchall()
    assert results, "the cc run wrote no results, so there is nothing to store counts beside"
    assert sorted(metric for _, _, metric, _ in metrics) == sorted(f"autovec.{name}" for name in autovec.COUNTS)
    assert {(stamp, framework) for stamp, framework, _, _ in metrics} == results, (metrics, results)
    assert all(detail.startswith("family=gcc ") for _, _, _, detail in metrics), metrics


def test_a_cpf_form_is_counted_from_its_view_under_the_columns_line(tmp_path: pathlib.Path) -> None:
    """No column compiles a CPF form, so the command line is the only way its loops reach kernel_metrics."""
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "gemm_fp64_cpf.c").write_text(SOURCE)
    (flat / "gemm_fp64_cpf_binding.json").write_text(json.dumps({"symbol": "kernel"}))
    view = tmp_path / "view"
    misses = cpf_cache.adopt(flat, tmp_path / "cache", view, ["gemm"], "form", "cpu", "fp64")
    assert misses == [f"gemm: {flat} has no gemm_fp64_cpf.cpp with gemm_fp64_cpf_binding.json"], misses
    db = tmp_path / "autovec.db"

    status = autovec.main(["--column", "cc", "--select", "gemm", "--view", str(view), "--db", str(db)])

    assert status == 0
    with sqlite3.connect(db) as conn:
        got = conn.execute(
            f"SELECT framework, flavor, value FROM {KERNEL_METRICS_TABLE} WHERE metric = 'autovec.loops'"
        ).fetchall()
    assert got == [("cc", "cpf", 4.0)], got
