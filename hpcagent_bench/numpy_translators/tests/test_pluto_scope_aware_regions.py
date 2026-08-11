"""A pluto translation unit scopes each NEST on its own merits, not the whole program at once.

pet rejects the entire scop a construct it cannot model lands in, so wrapping the kernel in one
region cost every loop nest in it whenever a single allocation or zero-fill prologue sat anywhere
inside. The emitter now splits at those statements (``numpyto_c.emit.pluto_scop_regions``) and
desugars a body-level memset into the affine loop it is (``_fill_loop_stmt``), so several scops per
translation unit is the normal output.
"""
import json
import pathlib
import re
import tempfile

from _op_oracle import _bench_info
from numpyto_c.emit import emit_pluto
from numpyto_common.frontend import parse_kernel
from numpyto_common.lowering import lower

from hpcagent_bench.pluto_affine import KNOWN_POLYCC_ISSUES, has_scop, scop_nonaffine_reason
from hpcagent_bench.pluto_transform import dedupe_scratch_declarations

#: Every construct that must never appear inside a region, per _PLUTO_UNSCOPABLE_RE.
UNSCOPABLE = ("malloc(", "calloc(", "free(", "memset(", "memcpy(", "while (")


def _lower_src(src: str, fn: str, inputs, outputs, shapes, syms):
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "k_numpy.py").write_text(src)
    (d / "bi.json").write_text(json.dumps(_bench_info(fn, inputs, outputs, shapes, syms)))
    return lower(parse_kernel(d / "k_numpy.py", d / "bi.json"))


def _regions(text: str):
    """The body of every ``#pragma scop`` region in ``text``, in order."""
    return re.findall(r"#pragma scop(.*?)#pragma endscop", text, re.S)


def _sized_zeros_kir():
    """A zero-filled local whose size is body-computed: its malloc cannot leave the body."""
    return _lower_src(
        "import numpy as np\n"
        "def sz(a, out, N):\n"
        "    m = int(a[0])\n"
        "    t = np.zeros(m)\n"
        "    for i in range(m):\n"
        "        t[i] = a[i] * 2.0\n"
        "    for i in range(N):\n"
        "        out[i] = t[0] + a[i]\n", "sz", ["a"], ["out"], {
            "a": "(N,)",
            "out": "(N,)"
        }, {"N": 64})


def _two_nests_kir():
    """Two independent nests with a body-computed allocation wedged between them."""
    return _lower_src(
        "import numpy as np\n"
        "def tn(a, out, N):\n"
        "    for i in range(N):\n"
        "        out[i] = a[i] + 1.0\n"
        "    m = int(a[0])\n"
        "    t = np.zeros(m)\n"
        "    for i in range(N):\n"
        "        out[i] = out[i] * 2.0\n", "tn", ["a"], ["out"], {
            "a": "(N,)",
            "out": "(N,)"
        }, {"N": 64})


def _clamp_kir():
    """A data-dependent ``if`` nest between two plain nests (pet_to_pluto.cpp:565 refuses it)."""
    return _lower_src(
        "import numpy as np\n"
        "def cl(a, out, N):\n"
        "    for i in range(N):\n"
        "        out[i] = a[i] + 1.0\n"
        "    for i in range(N):\n"
        "        if out[i] < 0.5:\n"
        "            out[i] = 0.5\n"
        "    for i in range(N):\n"
        "        out[i] = out[i] * 2.0\n", "cl", ["a"], ["out"], {
            "a": "(N,)",
            "out": "(N,)"
        }, {"N": 64})


def test_no_region_holds_a_construct_pet_cannot_model():
    text = emit_pluto(_sized_zeros_kir(), fn_name="sz")
    regions = _regions(text)
    assert regions, text
    for body in regions:
        for bad in UNSCOPABLE:
            assert bad not in body, f"{bad} inside a scop region:\n{body}"


def test_the_allocation_stays_in_the_translation_unit_outside_every_region():
    """Splitting must not DROP the prologue -- it moves out of the regions, not out of the kernel."""
    text = emit_pluto(_sized_zeros_kir(), fn_name="sz")
    assert "malloc(" in text, text
    assert "malloc(" not in "".join(_regions(text)), text


def test_a_body_level_zero_fill_is_a_loop_not_a_memset():
    """The desugar: a fill adjacent to compute is spelled as the affine nest it is, so it can stay in."""
    text = emit_pluto(_sized_zeros_kir(), fn_name="sz")
    fill = [ln for ln in text.split("\n") if "__zf" in ln]
    assert fill, text
    assert all("for (" in ln for ln in fill), fill


def test_a_nest_between_two_others_splits_the_kernel_into_several_regions():
    text = emit_pluto(_two_nests_kir(), fn_name="tn")
    regions = _regions(text)
    assert len(regions) == 2, f"expected one region per side of the allocation, got {len(regions)}:\n{text}"
    assert all("for (" in body for body in regions), regions


def test_an_unmodellable_nest_does_not_cost_its_scopable_neighbours():
    """Per-NEST, never per-program: the clamp is excluded, the nests around it are still scoped."""
    text = emit_pluto(_clamp_kir(), fn_name="cl")
    regions = _regions(text)
    assert len(regions) == 2, f"expected the clamp to split, not to swallow, the kernel:\n{text}"
    assert not any("if (" in body for body in regions), regions
    assert "if (" in text, text


def test_regions_never_nest():
    for kir, name in ((_sized_zeros_kir(), "sz"), (_two_nests_kir(), "tn"), (_clamp_kir(), "cl")):
        depth = 0
        for line in emit_pluto(kir, fn_name=name).split("\n"):
            depth += (line.strip() == "#pragma scop") - (line.strip() == "#pragma endscop")
            assert 0 <= depth <= 1, f"{name}: unbalanced or nested scop markers"
        assert depth == 0, f"{name}: unterminated scop"


def test_emitting_twice_gives_byte_identical_c():
    for src_fn, name in ((_sized_zeros_kir, "sz"), (_two_nests_kir, "tn"), (_clamp_kir, "cl")):
        assert emit_pluto(src_fn(), fn_name=name) == emit_pluto(src_fn(), fn_name=name), name


def test_the_affine_detector_reads_every_region_not_just_the_first():
    """A gather in the SECOND region used to go unseen, and polycc may miscompile rather than refuse."""
    text = ("#pragma scop\nfor (i = 0; i < N; i++) a[i] = 1.0;\n#pragma endscop\n"
            "#pragma scop\nfor (i = 0; i < N; i++) b[i] = a[ip[i]];\n#pragma endscop\n")
    assert scop_nonaffine_reason(text) == "indirection"


def test_a_translation_unit_with_no_region_is_not_a_scop_input():
    """A kernel pet can model no part of must decline, not be handed to polycc unchanged."""
    assert not has_scop("void f(void) { while (1) { } }")
    assert has_scop(emit_pluto(_two_nests_kir(), fn_name="tn"))


def test_polyccs_repeated_scratch_declarations_are_merged_not_dropped():
    """POLYCC-012: one declaration per counter per function, and only counters are touched."""
    src = ("void f(int N) {\n"
           "  int t1, t2;\n"
           " register int lbv, ubv;\n"
           "if (N >= 1) {\n"
           "  for (t1=0;t1<N;t1++) { }\n"
           "}\n"
           "  int t1, t2, t3;\n"
           " register int lbv, ubv;\n"
           "  double keep, me;\n"
           "}\n"
           "void g(int N) {\n"
           "  int t1;\n"
           "}\n")
    out = dedupe_scratch_declarations(src)
    assert out.count("int t1") == 2, out  # once per function, not once per region
    assert "int t3;" in out, "a counter the first declaration lacked must survive"
    assert out.count("register int lbv, ubv;") == 1, out
    assert "double keep, me;" in out, "only polycc's bare-int scratch is touched"
    assert "if (N >= 1) {" in out and out.count("for (t1=0") == 1, out


def test_a_declaration_that_went_out_of_scope_is_not_deduped_against():
    """Scope, not function: a region nested in a loop body cannot cover the block after it."""
    src = ("void f(int N) {\n"
           "for (i=0;i<N;i++) {\n"
           "  int lbp, ubp;\n"
           "  lbp = 0;\n"
           "}\n"
           "  int lbp, ubp;\n"
           "  lbp = 0;\n"
           "}\n")
    assert dedupe_scratch_declarations(src).count("int lbp, ubp;") == 2


def test_the_registry_names_this_scoping_as_what_avoids_those_bugs():
    for issue in ("POLYCC-007", "POLYCC-013"):
        assert KNOWN_POLYCC_ISSUES[issue].avoided_by == "numpyto_c.emit.pluto_scop_regions", issue
    assert KNOWN_POLYCC_ISSUES["POLYCC-011"].avoided_by == "hpcagent_bench.pluto_transform.pet_parse_env"
