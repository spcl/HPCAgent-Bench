"""``int_floor`` / ``int_ceil`` / ``python_mod`` prelude helpers (C and C++).

Neither C nor C++ has numpy's floor-division or sign-of-divisor modulo natively, so
``//`` and ``%`` always emit these helpers. Each dispatches on the OPERAND TYPE --
``_Generic`` in C, ``if constexpr`` in C++ -- rather than on a dtype the emitter infers
from the source AST. That inference is what silently mis-emitted ``int(a[i]) // 2`` as
``floor((int64)/(int64))``, where the quotient is already truncated and the floor is a
no-op (-3 instead of numpy's -4).

These compile the real prelude and check every sign combination against Python, which
defines the semantics the helpers exist to reproduce. ``int_ceil`` has no emitter caller
yet (the ceil-division idiom ``(a + b - 1) // b`` is still spelled out in lib_nodes and
holds only for a positive divisor), so it is verified here directly rather than shipped
unexercised.
"""
import math

import pytest

from _native_tu import build_run_c, have_gcc, have_gpp
from numpyto_c.emit import _C_HEADER, _CPP_HEADER, _CPP_FOOTER

# Sign matrix: same-sign, mixed-sign, exact division, and a unit divisor.
_INT_PAIRS = [(7, 2), (-7, 2), (7, -2), (-7, -2), (8, 4), (-8, 4), (8, -4), (-8, -4), (1, 3), (-1, 3), (0, 5)]
_FLT_PAIRS = [(7.5, 2.0), (-7.5, 2.0), (7.5, -2.0), (-7.5, -2.0), (1.0, 0.25), (-1.0, 0.25)]


def _driver():
    """A main() printing each helper's result, one value per line, in a fixed order."""
    lines = ["int main(void) {"]
    for a, b in _INT_PAIRS:
        lines.append(f'    printf("%lld\\n", (long long)int_floor((int64_t){a}, (int64_t){b}));')
        lines.append(f'    printf("%lld\\n", (long long)int_ceil((int64_t){a}, (int64_t){b}));')
        lines.append(f'    printf("%lld\\n", (long long)python_mod((int64_t){a}, (int64_t){b}));')
    for a, b in _FLT_PAIRS:
        lines.append(f'    printf("%.17g\\n", (double)int_floor({a!r}, {b!r}));')
        lines.append(f'    printf("%.17g\\n", (double)int_ceil({a!r}, {b!r}));')
        lines.append(f'    printf("%.17g\\n", (double)python_mod({a!r}, {b!r}));')
    lines.append("    return 0;\n}")
    return "\n".join(lines)


def _expected():
    out = []
    for a, b in _INT_PAIRS:
        out.append(float(a // b))
        out.append(float(-((-a) // b)))  # ceil-division, exact for either sign
        out.append(float(a % b))
    for a, b in _FLT_PAIRS:
        out.append(math.floor(a / b))
        out.append(math.ceil(a / b))
        out.append(a % b)
    return out


def _check(cpp):
    src = (_CPP_HEADER + _CPP_FOOTER + "#include <cstdio>\n") if cpp else (_C_HEADER + "#include <stdio.h>\n")
    run = build_run_c(src, _driver(), cpp=cpp)
    assert run.returncode == 0, run.stderr
    got = [float(line) for line in run.stdout.split()]
    exp = _expected()
    assert len(got) == len(exp), f"{len(got)} values, expected {len(exp)}"
    for g, e in zip(got, exp):
        assert g == pytest.approx(e), f"got {g}, expected {e}"


@have_gcc
def test_c_division_helpers_match_python():
    _check(cpp=False)


@have_gpp
def test_cpp_division_helpers_match_python():
    _check(cpp=True)


# --- the _Generic association list must cover every dtype the corpus can reach ------------------
# `default:` is a silent catch-all: a type absent from the list does not fail to compile, it picks
# the INTEGER helper. Two dtypes fell through -- see each test for what that cost.
_F16_DRIVER = """#include <stdio.h>
int main(void) {
    _Float16 a = 0.5f16, b = 0.25f16;
    printf("%.6f\\n", (double)int_floor(a, b));
    printf("%.6f\\n", (double)python_mod((_Float16)3.5f16, (_Float16)2.0f16));
    return 0;
}
"""

_U64_DRIVER = """#include <stdio.h>
int main(void) {
    uint64_t a = (1ULL << 63) + 5, b = 2;
    printf("%llu\\n", (unsigned long long)int_floor(a, b));
    printf("%llu\\n", (unsigned long long)python_mod(a, b));
    printf("%llu\\n", (unsigned long long)int_ceil(a, b));
    return 0;
}
"""


@have_gcc
def test_float16_operands_take_the_floating_helper():
    """GCC does not promote _Float16 in arithmetic, so `_Float16 + _Float16` is _Float16 and hit
    `default:` -- the integer helper. 0.5 // 0.25 became int_floor(0, 0) and died with SIGFPE
    (exit 136), and any non-zero pair truncated silently. C++ was unaffected (is_integral_v sends
    it to the floating branch), so the two backends disagreed on the same kernel.
    """
    run = build_run_c(_C_HEADER, _F16_DRIVER)
    # Assert the exit code first: the original defect KILLED the process (SIGFPE, rc 136), and a
    # test that only parsed stdout would report a confusing IndexError instead of the real failure.
    assert run.returncode == 0, f"exit {run.returncode} (136 = SIGFPE):\n{run.stderr}"
    out = run.stdout.splitlines()
    assert float(out[0]) == 2.0, f"0.5 // 0.25 -> {out[0]}, expected numpy's 2.0"
    assert float(out[1]) == 1.5, f"3.5 % 2.0 -> {out[1]}, expected numpy's 1.5"


@have_gcc
def test_unsigned_operands_above_int64_max_are_not_reinterpreted_as_negative():
    """uint64 is integral, so the integer helper was type-correct but SIGNED: any value above
    INT64_MAX arrived negative. (2**63 + 5) // 2 returned -4611686018427387902."""
    a = (1 << 63) + 5
    run = build_run_c(_C_HEADER, _U64_DRIVER)
    assert run.returncode == 0, f"exit {run.returncode}:\n{run.stderr}"
    out = run.stdout.splitlines()
    assert int(out[0]) == a // 2, f"floor -> {out[0]}, expected {a // 2}"
    assert int(out[1]) == a % 2, f"mod -> {out[1]}, expected {a % 2}"
    assert int(out[2]) == -(-a // 2), f"ceil -> {out[2]}, expected {-(-a // 2)}"
