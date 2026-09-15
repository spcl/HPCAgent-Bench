# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A "parallelized" kernel is not just "contains #pragma omp parallel for".

Contract-based parallelization emits ``if (contract holds) { parallel loop } else { sequential
loop }``: the pragma is textually present, but whether it runs matters on the branch taken. These
tests pin the taxonomy the CPF/MPR paper's parallel-nest-fraction metric is built on -- each shape
is a trimmed, real excerpt from a rendered CPF kernel (named in its docstring), not an invented
example, so a change to the classifier that breaks a real kernel's shape breaks a named test here.
"""

import pathlib

from hpcagent_bench.metrics import source_text as pm


def test_an_unconditional_parallel_for_is_parallel_with_no_guard() -> None:
    """tsvc_2_s318: a plain ``#pragma omp parallel for`` with no surrounding branch."""
    source = """
    extern "C" void k(const double* a, double* out, int64_t n) {
        #pragma omp parallel for
        for (int64_t i = 0; i < n; ++i) {
            out[i] = a[i] * 2.0;
        }
    }
    """
    nests = pm.analyze_source(source)
    assert len(nests) == 1
    assert nests[0].kind == pm.NestKind.PARALLEL


def test_a_reduction_clause_is_tagged_without_changing_the_base_kind() -> None:
    """tsvc_2_s318's argmax reduction: unconditional AND a reduction, not one or the other."""
    source = """
    extern "C" void k(const double* a, double* out, int64_t n) {
        struct pair best;
        #pragma omp parallel for reduction(best_op : best)
        for (int64_t i = 1; i < n; ++i) {
            if (a[i] > best.v) { best.v = a[i]; best.i = i; }
        }
    }
    """
    nests = pm.analyze_source(source)
    assert len(nests) == 1
    assert nests[0].kind == pm.NestKind.PARALLEL
    assert nests[0].has_reduction


def test_a_plain_for_with_no_pragma_is_sequential() -> None:
    """The baseline hand-ported C carries zero omp pragmas by construction."""
    source = """
    void k(const double* a, double* result, int64_t n) {
        double maxv = a[0];
        for (int64_t i = 1; i < n; ++i) {
            if (a[i] > maxv) { maxv = a[i]; }
        }
        result[0] = maxv;
    }
    """
    nests = pm.analyze_source(source)
    assert len(nests) == 1
    assert nests[0].kind == pm.NestKind.SEQUENTIAL


def test_omp_simd_alone_is_not_counted_as_thread_level_parallel() -> None:
    """ext_war_unit's boundary-seam loop: ``#pragma omp simd`` with no ``parallel`` or ``for``.

    DaCe's own comment above this loop says "parallel -- the iterations are independent", but
    the emitted pragma is vector-only. Folding this into the PARALLEL bucket would overstate
    thread-level parallelism the loop does not have.
    """
    source = """
    void k(double* a, const double* b, const double* seam, int64_t n) {
        // parallel -- the iterations are independent
        #pragma omp simd
        for (int64_t i = 0; i < n; i += 1) {
            a[i] = seam[0] + b[i];
        }
    }
    """
    nests = pm.analyze_source(source)
    assert len(nests) == 1
    assert nests[0].kind == pm.NestKind.SIMD_ONLY
    assert nests[0].dace_comment == "parallel -- the iterations are independent"


def test_a_contract_guarded_branch_is_neither_plain_parallel_nor_plain_sequential() -> None:
    """versioned_distance_update: if (K >= 1) takes the parallel-affine path, else it is a plain
    serial recurrence. Whether the pragma runs depends on a runtime-visible condition on K, so it
    must not silently count as unconditionally parallel.
    """
    source = """
    extern "C" void k(double* a, const double* b, const double* c, int64_t K, int64_t n) {
        if ((K >= 1)) {
            #pragma omp parallel for simd
            for (int64_t i = K; i < n; i += 1) {
                a[i] = b[i] * c[i];
            }
        } else if (((!(K >= 1)) && (K == 0))) {
            for (int64_t i = K; i < n; i = i + 1) {
                a[i] = 0.75 * a[i - K] + b[i] * c[i];
            }
        } else {
            for (int64_t i = K; i < n; i = i + 1) {
                a[i] = 0.75 * a[i - K] + b[i] * c[i];
            }
        }
    }
    """
    nests = pm.analyze_source(source)
    assert len(nests) == 3
    kinds = sorted(n.kind.value for n in nests)
    assert kinds == ["contract_guarded", "sequential", "sequential"]


def test_a_sequential_nest_inside_the_same_guarded_branch_stays_sequential() -> None:
    """versioned_distance_update's scan: the K>=1 branch itself mixes a parallel affine-delta
    loop with a genuinely sequential prefix-scan recurrence. Reclassifying every nest in a
    "mixed" branch as guarded would hide that the scan loop is sequential on every path, not
    only when the contract fails.
    """
    source = """
    extern "C" void k(double* a, const double* seed, int64_t K, int64_t n) {
        if ((K >= 1)) {
            #pragma omp parallel for simd
            for (int64_t i = K; i < n; i += 1) {
                a[i] = a[i] * 2.0;
            }
            {
                const long s = K;
                for (long r = 0; r < s; ++r) {
                    double acc = seed[r];
                    for (long k = r; k < n; k += s) {
                        acc = a[k] * acc;
                        a[k] = acc;
                    }
                }
            }
        } else {
            for (int64_t i = K; i < n; i = i + 1) {
                a[i] = a[i - K] * 2.0;
            }
        }
    }
    """
    nests = pm.analyze_source(source)
    kinds = {n.header_line: n.kind for n in nests}
    parallel_line = min(n.header_line for n in nests if n.kind == pm.NestKind.CONTRACT_GUARDED)
    scan_line = min(n.header_line for n in nests if n.depth == 2 and n.kind == pm.NestKind.SEQUENTIAL)
    assert kinds[parallel_line] == pm.NestKind.CONTRACT_GUARDED
    assert kinds[scan_line] == pm.NestKind.SEQUENTIAL


def test_a_worksharing_for_nested_in_a_sequential_outer_loop_is_a_nested_parallel_region() -> None:
    """wf_triangular's wavefront: the outer diagonal loop is a genuine RAW-carried sequential
    dimension, but it wraps a ``#pragma omp for`` over the independent tiles on each diagonal.
    The nest as a whole does real parallel work even though its own header carries no pragma.
    """
    source = """
    void k(double* a, int64_t n_tiles) {
        // wavefront tile diagonal -- sequential: the tile diagonal carries every dependence
        for (int64_t t = 0; t <= n_tiles; t = t + 1) {
            // wavefront tile column -- parallel: the tiles on one diagonal are independent
            #pragma omp for
            for (int64_t p = 0; p < t; p += 1) {
                body(a, p, t);
            }
        }
    }
    """
    nests = pm.analyze_source(source)
    assert len(nests) == 1
    nest = nests[0]
    assert nest.kind == pm.NestKind.PARALLEL
    assert nest.nested_parallel_region
    assert nest.worksharing_only


def test_an_omp_parallel_for_if_clause_is_tagged_as_a_guard_clause() -> None:
    """The clause-style guard (``omp parallel for if(cond)``) the taxonomy also has to catch,
    distinct from the branch-style contract guard above -- no separate sequential branch exists
    in the source, OpenMP itself decides at runtime whether to spawn threads.
    """
    source = """
    void k(double* a, int64_t n) {
        #pragma omp parallel for if(n > 1024)
        for (int64_t i = 0; i < n; ++i) {
            a[i] = a[i] + 1.0;
        }
    }
    """
    nests = pm.analyze_source(source)
    assert len(nests) == 1
    assert nests[0].kind == pm.NestKind.PARALLEL
    assert nests[0].has_if_clause


def test_a_for_spelled_inside_a_comment_or_string_is_not_a_loop_nest() -> None:
    """Comment- and string-aware scanning: naive regex over raw text would see two more loops."""
    source = """
    void k(double* a, int64_t n) {
        // for (int64_t ghost = 0; ghost < n; ++ghost) { a[ghost] = 0.0; }
        const char* msg = "for (int i = 0; i < 1; i++) {}";
        #pragma omp parallel for
        for (int64_t i = 0; i < n; ++i) {
            a[i] = 1.0;
        }
    }
    """
    nests = pm.analyze_source(source)
    assert len(nests) == 1
    assert nests[0].kind == pm.NestKind.PARALLEL


def test_a_brace_free_single_statement_for_body_still_parses() -> None:
    """A for-loop without a compound body must not desync the brace-depth scanner for whatever
    follows it in the same function.
    """
    source = """
    void k(double* a, double* b, int64_t n) {
        for (int64_t i = 0; i < n; ++i)
            a[i] = 0.0;
        #pragma omp parallel for
        for (int64_t i = 0; i < n; ++i) {
            b[i] = 1.0;
        }
    }
    """
    nests = pm.analyze_source(source)
    assert len(nests) == 2
    kinds = sorted(n.kind.value for n in nests)
    assert kinds == ["parallel", "sequential"]


def test_nest_depth_counts_levels_within_one_nest_not_across_helper_calls() -> None:
    """ext_war_unit: a chunk loop with an inlined nested simd loop is depth 2; a call to a
    separately-defined helper that itself loops is a SEPARATE nest at depth 1, not depth 3 --
    this module does not follow calls, and the test pins that as a stated limitation, not a bug.
    """
    source = """
    static inline void helper(double* a, int64_t n) {
        for (int64_t i = 0; i < n; ++i) {
            a[i] = a[i] + 1.0;
        }
    }

    extern "C" void k(double* a, int64_t n, int64_t chunk) {
        #pragma omp parallel for
        for (int64_t c = 1; c < n; c += chunk) {
            helper(&a[c], chunk);
        }
        #pragma omp parallel for
        for (int64_t c = 1; c < n; c += chunk) {
            #pragma omp simd
            for (int64_t i = c; i < c + 1; i += 1) {
                a[i] = a[i] * 2.0;
            }
        }
    }
    """
    nests = pm.analyze_source(source)
    assert len(nests) == 3
    depths = sorted(n.depth for n in nests)
    assert depths == [1, 1, 2]


def test_parallel_nest_fraction_counts_only_unconditional_parallel_nests() -> None:
    """The recommended metric excludes contract-guarded and simd-only nests from its numerator
    by design (Definition (a) in the report) -- they are reported as separate columns, never
    silently folded into "parallel".
    """
    nests = [
        pm.LoopNest(pm.NestKind.PARALLEL, 0, 1, False, False, False, False, False, False, ""),
        pm.LoopNest(pm.NestKind.CONTRACT_GUARDED, 1, 1, False, False, False, False, False, False, ""),
        pm.LoopNest(pm.NestKind.SIMD_ONLY, 2, 1, False, False, False, False, False, False, ""),
        pm.LoopNest(pm.NestKind.SEQUENTIAL, 3, 1, False, False, False, False, False, False, ""),
    ]
    breakdown = pm.ParallelismBreakdown.from_nests("k", "c++", nests)
    assert breakdown.total_nests == 4
    assert breakdown.parallel_nest_fraction == 0.25


def test_parallel_nest_fraction_is_zero_on_an_empty_kernel_not_a_division_error() -> None:
    breakdown = pm.ParallelismBreakdown.from_nests("k", "c", [])
    assert breakdown.total_nests == 0
    assert breakdown.parallel_nest_fraction == 0.0


def test_breakdown_for_source_carries_the_kernel_and_language_it_was_given() -> None:
    """The pure entry point the CLI and any future harness hook call: text + language in, one
    typed row out, no filesystem or dace dependency.
    """
    source = "void k(double* a) { for (int i = 0; i < 1; ++i) { a[i] = 0.0; } }"
    breakdown = pm.breakdown_for_source(source, "c", "toy_kernel")
    assert breakdown.kernel == "toy_kernel"
    assert breakdown.language == "c"
    assert breakdown.total_nests == 1


def test_multiline_block_comments_do_not_shift_line_numbers() -> None:
    """``mask_comments_and_strings`` must blank a comment's content but keep its newlines, or
    every later line number (header_line, the dace_comment lookup) is off by however many lines
    the comment spanned.
    """
    source = "void k(double* a) {\n/* a\n   multi\n   line\n   comment */\n#pragma omp parallel for\nfor (int i = 0; i < 1; ++i) { a[i] = 0.0; }\n}\n"
    nests = pm.analyze_source(source)
    assert len(nests) == 1
    assert nests[0].header_line == 6


def test_language_for_path_maps_known_cpf_suffixes() -> None:
    assert pm.language_for_path(".cpp") == "c++"
    assert pm.language_for_path(".c") == "c"
    assert pm.language_for_path(".hip") == "hip"


def test_analyzing_a_real_cpf_view_kernel_matches_the_recorded_taxonomy(tmp_path: pathlib.Path) -> None:
    """scatter_accum_dup, trimmed to its parallel scatter loop: an unconditional parallel-for
    whose write races are serialized by an atomic inside a called helper -- the atomic itself is
    invisible to this nest's own tags (documented cross-function limitation), but the loop is
    still correctly PARALLEL, not sequential because of the race.
    """
    source = """
    extern "C" void scatter_accum_dup(double* bins, const int* ip, const double* src, int64_t n) {
        // parallel -- the iterations are independent
        #pragma omp parallel for simd
        for (int64_t i = 0; i < n; i += 1) {
            loop_body(&ip[0], &src[0], &bins[0], i);
        }
    }
    """
    breakdown = pm.breakdown_for_source(source, "c++", "scatter_accum_dup")
    assert breakdown.total_nests == 1
    assert breakdown.parallel_nests == 1
    assert breakdown.atomic_nests == 0
