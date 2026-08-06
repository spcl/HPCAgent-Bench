# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The three optional diagnostics (``hpcagent_bench/perf_reports.py``): a vectorization report, a
lowered-code dump, and the auto-generated source that was compiled. Every knob defaults off, and even
switched on a report never perturbs the timed ``.so`` (the opt-report is a SEPARATE compile that leaves
it byte-identical; the disassembly and the generated-source dump only READ what a timed run made).
Native cases build a fabricated two-precision ``cpp_backend`` with a pinned source, vectorizing in one
loop and provably not in the other."""
import hashlib
import pathlib
import subprocess

import pytest

from hpcagent_bench import flags, perf_reports
from hpcagent_bench.benchmarks import cpp_runtime
from hpcagent_bench.frameworks import generate_framework
from hpcagent_bench.languages import report_flags
from tests.optional_imports import import_or_skip

#: One kernel per precision: a loop that MUST vectorize followed by one that CANNOT (a dependence),
#: so one report states both a width and a refusal. Symbol carries precision since both sources link
#: into one library (file stem == exported symbol; a shared name would collide at link).
_SRC = """
void probe_%(fp)s(%(t)s *out, const %(t)s *a, const %(t)s *b, long n) {
  for (long i = 0; i < n; i++) out[i] = a[i] * b[i] + (%(t)s)1.5;
  for (long i = 1; i < n; i++) out[i] = out[i - 1] * (%(t)s)2.0;
}
"""


@pytest.fixture
def backend(tmp_path):
    """A fabricated ``cpp_backend`` holding the two precision sources, laid out as the emitter writes them."""
    cb = tmp_path / "cpp_backend"
    cb.mkdir()
    (cb / "probe_fp64.c").write_text(_SRC % {"t": "double", "fp": "fp64"})
    (cb / "probe_fp32.c").write_text(_SRC % {"t": "float", "fp": "fp32"})
    return cb


def _md5(path: pathlib.Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# --- the knobs -------------------------------------------------------------


def test_all_knobs_default_off():
    """A default run must produce no reports at all."""
    assert perf_reports.enabled("opt_report") is False
    assert perf_reports.enabled("lowered_code") is False
    assert perf_reports.enabled("generated_source") is False


@pytest.mark.parametrize("kind", sorted(perf_reports.KINDS))
def test_knob_is_independently_enabled_by_env(kind, monkeypatch):
    """The two capabilities are independent: enabling one must not enable the other."""
    monkeypatch.setenv(f"HPCAGENT_BENCH_PERF_REPORTS_{kind.upper()}", "1")
    assert perf_reports.enabled(kind) is True
    other = next(k for k in perf_reports.KINDS if k != kind)
    assert perf_reports.enabled(other) is False


def test_unknown_kind_is_rejected():
    with pytest.raises(KeyError):
        perf_reports.enabled("no_such_report")


# --- the flag table --------------------------------------------------------


def test_report_flags_resolve_per_compiler_family():
    """One table reaches both families; each gets the channel it actually has."""
    assert report_flags("c") == flags.GCC_OPT_REPORT
    assert report_flags("fortran") == flags.GCC_OPT_REPORT
    assert report_flags("cpp", compiler="clangpp") == flags.CLANG_OPT_REPORT
    assert "-fopt-info-vec" in report_flags("c")
    assert "-Rpass" in report_flags("cpp", compiler="clangpp")


def test_report_flags_are_empty_when_no_channel_is_wired():
    """A compiler with no ``report_ref`` reports "not supported" rather than a guessed flag."""
    assert report_flags("cuda", compiler="nvcc") == ""


def test_clang_filter_never_matches_every_pass():
    """``-Rpass=.*`` floods the report with asm-printer noise; the filter must name the vectorizer passes."""
    assert "=.*" not in flags.CLANG_OPT_REPORT
    assert "loop-vectorize" in flags.CLANG_OPT_REPORT


def test_report_flags_never_name_a_missing_constant():
    """Every ``report_ref`` in the compiler table must name a real :mod:`hpcagent_bench.flags` constant."""
    compilers = cpp_runtime.FRAMEWORK_LANG
    for framework, lang in compilers.items():
        report_flags(lang, compiler=cpp_runtime.FRAMEWORK_COMPILER.get(framework))


# --- the writer ------------------------------------------------------------


def test_report_path_mirrors_the_benchmark_tree():
    p = perf_reports.report_path("scientific_computing/map_reduce/arc_distance", "arc_distance", "cc", "default",
                                 "lowered_code")
    assert p.parent == perf_reports.REPORTS / "lowered_code" / "scientific_computing/map_reduce/arc_distance"
    assert p.name == "arc_distance.cc.default.lowered_code.txt"


def test_every_kind_lands_under_its_own_subtree():
    """One root, a subdirectory per kind, and the kind spelled the SAME way in both places.

    The previous layout gave ``opt_report`` a second top-level root and wrote it as
    ``opt-report.txt`` beside ``lowered_code``'s ``asm.txt`` -- three spellings of one concept. This
    pins the invariant that replaced them: for every kind, root and suffix are the kind's own name.
    """
    opt = perf_reports.report_path("scientific_computing/map_reduce/arc_distance", "arc_distance", "cc", "default",
                                   "opt_report")
    assert opt.parent == perf_reports.REPORTS / "opt_report" / "scientific_computing/map_reduce/arc_distance"
    assert opt.name == "arc_distance.cc.default.opt_report.txt"
    roots = {kind: perf_reports.report_root(kind) for kind in perf_reports.KINDS}
    assert roots == {kind: perf_reports.REPORTS / kind for kind in perf_reports.KINDS}
    assert len(set(roots.values())) == len(perf_reports.KINDS), "two kinds share a subtree"
    for kind, suffix in perf_reports.KINDS.items():
        assert suffix == f"{kind}.txt", f"{kind} writes {suffix!r}; the suffix must be the kind's name"


def test_write_none_means_not_supported_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(perf_reports, "REPORTS", tmp_path)
    assert perf_reports.write("a/b", "k", "cc", "default", "opt_report", None) is None
    assert list(tmp_path.rglob("*")) == []


def test_write_creates_the_kernel_directory_on_demand(tmp_path, monkeypatch):
    """The kernel tree is never materialised up front -- the writer makes only the directory it needs."""
    monkeypatch.setattr(perf_reports, "REPORTS", tmp_path)
    path = perf_reports.write("scientific_computing/map_reduce/arc_distance", "arc_distance", "cc", "default",
                              "lowered_code", "TEXT")
    assert path is not None and path.read_text() == "TEXT"
    assert path.parent.is_dir()


def test_two_implementations_do_not_overwrite_each_others_report(tmp_path, monkeypatch):
    """Numba's serial and parallel tracks are separately compiled and timed; must not collapse onto one filename."""
    monkeypatch.setattr(perf_reports, "REPORTS", tmp_path)
    a = perf_reports.write("x", "k", "numba", "nopython-mode", "lowered_code", "SERIAL")
    b = perf_reports.write("x", "k", "numba", "nopython-mode-parallel", "lowered_code", "PARALLEL")
    assert a != b
    assert a.read_text() == "SERIAL" and b.read_text() == "PARALLEL"


# --- the default contract --------------------------------------------------


def test_frameworks_without_a_report_answer_not_supported():
    """``None`` is the default for every hook, so a knob can switch on across a mixed sweep."""
    numpy = generate_framework("numpy")
    assert numpy.opt_report(object(), None) is None
    assert numpy.lowered_code(object(), None) is None
    assert numpy.generated_source(object(), None) is None


# --- native: the real report -----------------------------------------------


def test_native_opt_report_names_the_vectorized_and_the_refused_loop(backend):
    """The report must carry BOTH halves: the width of what vectorized, and the reason for what did not."""
    text = cpp_runtime.opt_report_text(backend, "probe", "cc")
    assert text is not None
    assert "probe_fp64.c" in text and "probe_fp32.c" in text  # every TU, not just the last
    assert "byte vectors" in text  # the width of the loop that vectorized
    assert "missed:" in text  # the loop that could not, with its reason


def test_native_opt_report_records_the_compile_it_describes(backend):
    """A report that does not say which flags produced it cannot be read later."""
    text = cpp_runtime.opt_report_text(backend, "probe", "cc")
    assert "-O3" in text and "-march=native" in text
    assert flags.GCC_OPT_REPORT.split()[0] in text


def test_native_opt_report_is_none_when_sources_were_never_emitted(tmp_path):
    empty = tmp_path / "cpp_backend"
    empty.mkdir()
    assert cpp_runtime.opt_report_text(empty, "probe", "cc") is None


def test_opt_report_does_not_touch_the_timed_library(backend):
    """THE invariant: the report is a separate compile-only run, so the timed ``.so`` must come out
    byte-identical whether or not reports were switched on."""
    so = cpp_runtime._ensure_built(backend, "probe", "cc")
    before, before_mtime = _md5(so), so.stat().st_mtime_ns

    assert cpp_runtime.opt_report_text(backend, "probe", "cc") is not None

    assert _md5(so) == before, "the report compile rewrote the library that gets timed"
    assert so.stat().st_mtime_ns == before_mtime, "the report compile relinked the timed library"


def test_opt_report_does_not_leave_a_second_copy_of_the_library(backend):
    """The report path drops the link step; only the real build makes a ``.so``."""
    cpp_runtime.opt_report_text(backend, "probe", "cc")
    assert list(backend.rglob("*.so")) == []


# --- native: the real disassembly ------------------------------------------


def test_native_lowered_code_disassembles_the_timed_library(backend):
    so = cpp_runtime._ensure_built(backend, "probe", "cc")
    text = perf_reports.objdump(so)
    assert text is not None
    assert "Disassembly of section .text:" in text
    assert "<probe_fp64>:" in text  # the kernel symbol, not just the ELF preamble


def test_native_lowered_code_shows_the_simd_the_report_claimed(backend):
    """The two capabilities must agree: the report claims a vectorized width, checked against the
    disassembly's real instructions."""
    report = cpp_runtime.opt_report_text(backend, "probe", "cc")
    asm = perf_reports.objdump(cpp_runtime._ensure_built(backend, "probe", "cc"))
    assert "byte vectors" in report
    assert "%xmm" in asm or "%ymm" in asm or "%zmm" in asm


def test_built_so_never_builds(backend):
    """``lowered_code`` reports on an artifact a timed run made; must not compile one nobody timed."""
    assert cpp_runtime.built_so(backend, "probe", "cc") is None
    cpp_runtime._ensure_built(backend, "probe", "cc")
    assert cpp_runtime.built_so(backend, "probe", "cc") is not None


def test_objdump_of_a_missing_library_is_not_supported(tmp_path):
    """A diagnostic degrades to "no report"; it never takes down the run."""
    assert perf_reports.objdump(tmp_path / "nope.so") is None


def test_objdump_of_a_non_object_is_not_supported(tmp_path):
    junk = tmp_path / "junk.so"
    junk.write_text("not an ELF file")
    assert perf_reports.objdump(junk) is None


# --- native: the auto-generated input --------------------------------------


def test_generated_source_concatenates_both_precision_sources(backend):
    """The dump must carry every TU that was compiled, each under its own banner, so the exact input
    that was built and timed is recoverable."""
    text = cpp_runtime.generated_source_text(backend, "probe", "cc")
    assert text is not None
    assert "==== probe_fp64.c ====" in text and "==== probe_fp32.c ====" in text
    assert "void probe_fp64" in text and "void probe_fp32" in text


def test_generated_source_is_none_when_sources_were_never_emitted(tmp_path):
    empty = tmp_path / "cpp_backend"
    empty.mkdir()
    assert cpp_runtime.generated_source_text(empty, "probe", "cc") is None


def test_generated_source_only_reads_never_builds(backend):
    """Dumping the input is a pure read of files a timed run emitted; it must not compile a library."""
    cpp_runtime.generated_source_text(backend, "probe", "cc")
    assert list(backend.rglob("*.so")) == []


def test_native_framework_generated_source_hook_dumps_the_input(backend, monkeypatch):
    """The NativeFramework hook resolves its own cpp_backend + base and returns the emitted source."""
    cc = generate_framework("cc")
    monkeypatch.setattr(cc, "_cpp_backend", lambda bench: backend)
    monkeypatch.setattr(cc, "_native_base", lambda bench: "probe")
    text = cc.generated_source(object(), object())
    assert text is not None and "void probe_fp64" in text


# --- numba -----------------------------------------------------------------


def test_numba_lowered_code_dumps_real_instructions():
    """Numba never writes a ``.so``, so it answers with its own asm; compiled HERE (not cache-loaded)
    so the JIT has something to report."""
    numba = import_or_skip("numba")
    numpy = pytest.importorskip("numpy")

    @numba.njit  # NOT cache=True: this must be compiled in-process to have asm
    def scale(out, a):
        for i in range(a.shape[0]):
            out[i] = a[i] * 2.0 + 1.0

    scale(numpy.zeros(64), numpy.ones(64))
    text = generate_framework("numba").lowered_code(scale, None)
    assert text is not None
    assert "==== signature:" in text
    assert "%xmm" in text or "%ymm" in text or "%zmm" in text  # real instructions, not a stub


def test_numba_hooks_decline_a_plain_python_function():
    plain = generate_framework("numba")
    assert plain.opt_report(lambda x: x, None) is None
    assert plain.lowered_code(lambda x: x, None) is None


def test_numba_reports_nothing_for_a_cache_restored_function(tmp_path, monkeypatch):
    """A cache hit restores executable code with no compile-time by-products: ``inspect_asm`` returns
    an instruction-free stub rather than raising, so this must answer "not supported" instead."""
    numba = import_or_skip("numba")
    src = tmp_path / "cached_kernel.py"
    src.write_text("import numba as nb\n\n@nb.njit(cache=True)\ndef k(x):\n    return x * 2.0 + 1.0\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    warm = subprocess.run(
        ["python", "-c", "import cached_kernel; cached_kernel.k(3.0)"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert warm.returncode == 0, warm.stderr  # first process compiles + writes the cache

    import cached_kernel  # second process: this one LOADS the cache
    cached_kernel.k(3.0)
    sig = cached_kernel.k.signatures[0]
    assert cached_kernel.k.overloads[sig].metadata is None, "expected a cache hit"

    numba_fw = generate_framework("numba")
    assert numba_fw.lowered_code(cached_kernel.k, None) is None
    assert numba_fw.opt_report(cached_kernel.k, None) is None
