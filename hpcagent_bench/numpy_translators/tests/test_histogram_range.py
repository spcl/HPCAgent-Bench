"""np.histogram(a, bins, range=(lo, hi)) must DROP samples outside [lo, hi].

Both histogram lowerings (the AST expand_histogram in lib_nodes and the string-template
_HistogramHoister in numpy_desugar) clamped an out-of-range element into bin 0 / bin-1
instead, inflating the edge bins. numpy only keeps [lo, hi] (the last bin closed).

The AST clamp's bounds are int()-wrapped so every min/max operand is int64 -- Fortran's
min(default-int, INT(.., c_int64_t)) is a mixed-kind GNU extension that -std=f2018 rejects.

The second half of the file pins WHICH bin a sample lands in. numpy does not take the closed
form ``int((a - lo) * bins / (hi - lo))`` as final: it walks that index one step against the
``np.linspace(lo, hi, bins + 1)`` edges, because the two round apart. Both lowerings shipped the
closed form alone, and azimint_hist put 6 of 400000 fp32 samples one bin over -- it divides two
histograms, so that moved a bin ratio by 0.2% and failed the fp32 band under dace_cpu and cc
alike, while every other bin was perfect.
"""
import json
import re

import numpy as np
from _op_oracle import _bench_info, run_op

_BACKENDS = ("c", "cpp", "fortran", "numba", "pythran")


def _assert_ok(res):
    for backend, status in res.items():
        assert status == "ok" or status.startswith("skip"), f"{backend}: {status}"
    assert any(status == "ok" for status in res.values()), f"all skipped (vacuous): {res}"


def test_histogram_explicit_range_drops_out_of_range():
    # a spans [-3, 5]; with range=(-2, 2) numpy keeps only -1, 0, 1 -> counts [0,1,1,1].
    # The old clamp folded -3 into bin 0 and 3, 5 into bin 3 -> [1,1,1,3].
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           "    out[:] = np.histogram(a, 4, range=(-2.0, 2.0))[0]\n")
    a = np.array([-3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
    assert np.array_equal(np.histogram(a, 4, range=(-2.0, 2.0))[0], [0, 1, 1, 1])  # numpy anchor
    res = run_op(src, "f", {"a": a}, {"out": (4, )}, {"N": 6}, shapes={"a": "(N,)", "out": "(4,)"}, backends=_BACKENDS)
    _assert_ok(res)


def test_histogram_auto_range_unchanged():
    # No explicit range: lo/hi are a.min()/a.max(), so every element is in range and the
    # guard is a no-op -- this must still match numpy (regression guard for the fix).
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           "    out[:] = np.histogram(a, 5)[0]\n")
    a = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 2.0, 2.0, 4.0])
    res = run_op(src, "f", {"a": a}, {"out": (5, )}, {"N": 8}, shapes={"a": "(N,)", "out": "(5,)"}, backends=_BACKENDS)
    _assert_ok(res)


def _edge_probes(npt: int) -> np.ndarray:
    """Samples ON every bin edge of ``np.histogram(., npt)`` over [0, 1], and one ulp either side.

    The closed-form index and numpy's edge array round apart, so this is where they disagree:
    everywhere else the two agree and a random draw only finds it by accident (6 of 400000 at
    fp32 -- enough to fail azimint_hist, far too rare to pin a test on).
    """
    lo, hi = 0.0, 1.0
    ends = np.array([lo, hi])
    edges = np.histogram(ends, npt)[1]
    probes = [ends]
    for e in edges:
        probes.append(np.array([e, np.nextafter(e, -np.inf), np.nextafter(e, np.inf)]))
    return np.clip(np.concatenate(probes), lo, hi)


def test_histogram_bins_edge_probes_exactly():
    # Counts, not a norm: one sample in the wrong bin moves TWO bins by exactly one, and that is
    # the whole failure -- azimint_hist divides two histograms, so a single misbin shifts a ratio
    # by 1/count (0.2% at preset S, past the fp32 band) while every other bin stays perfect.
    npt = 1000
    a = _edge_probes(npt)
    ref = np.histogram(a, npt)[0]
    assert ref.sum() == a.size  # numpy anchor: every probe is in range, none is dropped
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           f"    out[:] = np.histogram(a, {npt})[0]\n")
    res = run_op(src,
                 "f", {"a": a}, {"out": (npt, )}, {"N": a.size},
                 shapes={
                     "a": "(N,)",
                     "out": f"({npt},)"
                 },
                 backends=_BACKENDS)
    _assert_ok(res)


def test_histogram_weighted_edge_probes_exactly():
    # The weighted arm bins through the SAME index and is what azimint_hist's numerator uses;
    # a misbin moves a weight rather than a count, so it needs its own consumer.
    npt = 1000
    a = _edge_probes(npt)
    w = (np.arange(a.size, dtype=np.float64) % 13.0) + 1.0
    src = ("import numpy as np\n"
           "def f(a, w, out):\n"
           f"    out[:] = np.histogram(a, {npt}, weights=w)[0]\n")
    res = run_op(src,
                 "f", {
                     "a": a,
                     "w": w
                 }, {"out": (npt, )}, {"N": a.size},
                 shapes={
                     "a": "(N,)",
                     "w": "(N,)",
                     "out": f"({npt},)"
                 },
                 backends=_BACKENDS)
    _assert_ok(res)


_HIST_SRC = ("import numpy as np\n"
             "def f(a, out):\n"
             "    out[:] = np.histogram(a, 8)[0]\n")


def _emit_sources(tmp_path):
    """``(c_source, dace_source)`` for :data:`_HIST_SRC` -- one per histogram lowering."""
    from numpyto_common.frontend import parse_kernel
    from numpyto_common.lowering import lower
    from numpyto_c.emit import emit_c
    from numpyto_c.dace_emit import emit_dace
    npy = tmp_path / "f_numpy.py"
    npy.write_text(_HIST_SRC)
    bi = tmp_path / "bench_info.json"
    bi.write_text(json.dumps(_bench_info("f", ["a"], ["out"], {"a": "(N,)", "out": "(8,)"}, {"N": 32})))
    return emit_c(lower(parse_kernel(npy, bi)), fn_name="f"), emit_dace(parse_kernel(npy, bi))


def test_both_lowerings_emit_the_edge_walk(tmp_path):
    """The walk is what makes the bin EQUAL to numpy's, so both lowerings must carry it.

    Structural, not just numeric: the numbers agree for almost every sample whether or not the
    walk is there (that is exactly why azimint_hist shipped wrong), so a count-only assertion
    passes straight through a lowering that dropped it.
    """
    c_src, dace_src = _emit_sources(tmp_path)
    c_edges = re.findall(r"__hedge_\w+", c_src)
    assert c_edges, "C: no bin-edge buffer"
    for tag, src, edges, step, lo, idx in (
        ("C", c_src, c_edges[0], c_edges[0].replace("__hedge_", "__hstep_"), "__hlo", "__bidx"),
        ("dace", dace_src, "__hist0_e", "__hist0_st", "__hist0_lo", "__hist0_b"),
    ):
        assert edges in src, f"{tag}: no bin-edge buffer"
        # The multiply and the add are SEPARATE statements writing the edge buffer: that is what
        # rounds ``j * step`` to the sample's own dtype before ``+ lo``, which is what makes the
        # edges equal to np.linspace's. One fused expression is free to evaluate wider (or to
        # contract to an fma) and shifts edges by an ulp -- ~20 misbinned samples at azimint_hist's
        # sample count.
        assert f"{edges}[0]" in src, f"{tag}: step is not parked in the edge buffer"
        assert step in src, f"{tag}: no rounded step"
        mul = [ln for ln in src.splitlines() if edges in ln and step in ln]
        add = [ln for ln in src.splitlines() if ln.count(edges) >= 2 and lo in ln]
        assert mul, f"{tag}: no `edges[j] = j * step` statement"
        assert add, f"{tag}: `+ lo` is not its own statement on the edge buffer"
        # Both walk steps, and only ever the LAST bin excluded from the step up.
        assert f"< {edges}[{idx}]" in src or f"< {edges}[({idx})]" in src, f"{tag}: no step down"
        assert f"{edges}[{idx} + 1]" in src or f"{edges}[({idx} + 1)]" in src, f"{tag}: no step up"


def test_dace_lowering_types_the_edges_from_the_sample_array(tmp_path):
    """The edge buffer takes the SAMPLE's dtype, never a hardcoded float64.

    An fp32 kernel whose edges are float64 is the bug this whole block exists for, one rounding
    smaller: the edges then sit up to half an ulp off numpy's and the walk lands on the wrong
    side for any sample in that gap.
    """
    _, dace_src = _emit_sources(tmp_path)
    line = [ln for ln in dace_src.splitlines() if "__hist0_e = " in ln]
    assert line, "no edge-buffer allocation"
    assert "a.dtype" in line[0], f"edges not typed from the samples: {line[0]}"
