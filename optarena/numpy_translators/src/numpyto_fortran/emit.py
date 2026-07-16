"""Fortran 2008 emitter built on NumpyToC's IR.

Walks the same :class:`numpyto_common.ir.KernelIR` that NumpyToC produces
and emits a self-contained Fortran subroutine + timing prelude. The
subroutine is exported with ``bind(C, name=...)`` so the harness
ctypes call links straight against it -- same convention as the C
and C++ outputs.

The body walker handles the subset NumpyToC already exercises plus
the Fortran-specific tweaks:

* 1-based indexing -- every Python loop ``for i in range(lo, hi):``
  becomes ``do i = lo, hi - 1`` (Fortran ``do`` is inclusive). Every
  subscript ``a[i]`` becomes ``a(i + 1)`` for a loop-iter index ``i``
  that was 0-based in Python.
* numpy intrinsics map to Fortran intrinsics; see :data:`NP_INTRINSICS`.
"""

import ast
import copy
import dataclasses
import math
import re
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

from numpyto_common.ir import ArrayDesc, KernelIR
from numpyto_common import dtypes, operators, parallelism
from numpyto_common.emitter import BaseEmitter
from numpyto_common.frontend import pure_int_arith

#: Whole-identifier matcher for scanning a shape-token string for the
#: names it references (so ``m`` matches in ``m + 1`` but not ``__mm``).
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")

#: Map of Fortran intrinsic functions: name as seen in the source after
#: lowering -> Fortran intrinsic to emit. Lowering already renamed
#: ``math.exp`` and ``np.exp`` to ``exp``; we just upper-case for style.
# C/libm name -> Fortran intrinsic (same elementwise math the NumpyToC
# path emits; the shared lowering produces C names so the Fortran emitter
# translates them here). Names that are ALSO Fortran intrinsics with the
# SAME spelling (tan/sinh/asin/atan2/log10/hypot/erf/floor/...) work via
# the case-insensitive fall-through and need no entry; only the genuinely
# different spellings are listed. The five with no Fortran intrinsic
# (exp2/expm1/log2/log1p/cbrt) are emitted as expressions in _emit_call.
# Fortran intrinsic / fn-expr tables live in numpyto_common.operators; aliased
# here so the existing call sites (and the public ``FORTRAN_INTRINSICS`` name)
# are unchanged.
FORTRAN_INTRINSICS = operators.FORTRAN_INTRINSICS
_FORTRAN_FN_EXPR = operators.FORTRAN_FN_EXPR

#: Integer-returning CONVERSION intrinsics (numpy name -> Fortran name). These
#: take a trailing KIND arg (``INT(x, KIND)``) and otherwise default to int32, so
#: they are emitted with the explicit int64 ABI kind (see ``_emit_call``). Note
#: ``floor`` / ``ceil`` are NOT here: numpy floor/ceil return a FLOAT and must not
#: overflow on +/-inf or |x| >= 2^63, so they lower to an AINT-based FLOAT form
#: (see ``_emit_call``) rather than the integer-returning FLOOR/CEILING intrinsic.
_INT_CONV_INTRINSIC: Dict[str, str] = {"int": "INT"}

#: Reserved name suffix for the emitted timing-buffer argument.


def _fortran_type(dtype: str) -> str:
    # Single dtype registry (numpyto_common.dtypes); ``int`` is int64 (canonical).
    # Dtypes with no Fortran kind (float16/128, complex256) fall back to double.
    try:
        return dtypes.fortran_kind(dtype)
    except KeyError:
        return "real(c_double)"


class _Fp8Fns(NamedTuple):
    """The three contained procedures for one fp8 format."""
    promote: str  # storage byte -> real(c_float)
    demote: str  # real(c_float) -> storage byte
    round: str  # real(c_float) -> real(c_float), rounded to the fp8 grid


#: Contained-procedure names per fp8 format, keyed by the CANONICAL registry
#: dtype. No leading underscores (unlike the C prelude's ``__npb_*``): a Fortran
#: identifier may not start with one.
_FP8_FNS = {
    "float8_e4m3": _Fp8Fns("npb_e4m3_to_f32", "npb_f32_to_e4m3", "npb_rn_e4m3"),
    "float8_e5m2": _Fp8Fns("npb_e5m2_to_f32", "npb_f32_to_e5m2", "npb_rn_e5m2"),
}


#: BinOp ops that are never fp8 arithmetic (bit / shift work is integer), so the
#: fp8 round-to-grid wrap skips them.
_FP8_NON_ARITH_OPS = (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)


def _fp8_fns(dtype: str):
    """:class:`_Fp8Fns` for a storage-only (fp8) dtype, else ``None`` -- gated on
    the registry, so a dtype is fp8 here iff the registry says it is."""
    if not dtype or not dtypes.is_storage_only(dtype):
        return None
    return _FP8_FNS[dtypes.canonical(dtype)]


def _fp8_dtypes_used(kir: KernelIR) -> List[str]:
    """The canonical storage-only (fp8) dtypes this kernel mentions, deduped."""
    seen: List[str] = []
    for dt in (*(a.dtype for a in kir.arrays), *(s.dtype for s in kir.scalars), *kir.local_dtypes.values(),
               kir.float_precision or ""):
        if dt and dtypes.is_storage_only(dt):
            canon = dtypes.canonical(dt)
            if canon not in seen:
                seen.append(canon)
    return seen


#: Contained procedures implementing one fp8 format, keyed by canonical dtype.
#: Fortran HAS no fp8 scalar, so -- exactly as in C -- a value is 1-byte STORAGE
#: (``integer(c_int8_t)``), promoted to real(c_float) to compute and rounded back
#: to the grid after each op. Note this needs no derived type and no
#: ``interface operator(*)``: the promote / round / demote model carries the
#: semantics, so ONE mechanism serves C, C++ and Fortran alike.
#:
#: ``transfer`` reinterprets the IEEE-754 bits (Fortran's memcpy); ``ishft`` with
#: a negative shift is a LOGICAL (zero-fill) shift, so the sign bit extracts
#: cleanly. Verified bit-exact against ml_dtypes -- see tests/test_fp8_emission.py.
_FP8_HELPER_SRC = {
    "float8_e4m3":
    """
    pure function npb_e4m3_to_f32(b) result(r)
        use, intrinsic :: iso_c_binding
        integer(c_int8_t), intent(in) :: b
        real(c_float) :: r
        integer(c_int32_t) :: bb, s, e, m, u, ex, mm
        bb = iand(int(b, c_int32_t), 255)
        s = ishft(bb, -7)
        e = iand(ishft(bb, -3), 15)
        m = iand(bb, 7)
        if (e == 15 .and. m == 7) then
            u = ior(ishft(s, 31), int(z'7fc00000', c_int32_t))
        else if (e == 0) then
            if (m == 0) then
                u = ishft(s, 31)
            else
                ex = -6
                mm = m
                do while (iand(mm, 8) == 0)
                    mm = ishft(mm, 1)
                    ex = ex - 1
                end do
                u = ior(ior(ishft(s, 31), ishft(ex + 127, 23)), ishft(iand(mm, 7), 20))
            end if
        else
            u = ior(ior(ishft(s, 31), ishft(e + 120, 23)), ishft(m, 20))
        end if
        r = transfer(u, 0.0_c_float)
    end function npb_e4m3_to_f32

    pure function npb_f32_to_e4m3(f) result(b)
        use, intrinsic :: iso_c_binding
        real(c_float), intent(in) :: f
        integer(c_int8_t) :: b
        integer(c_int32_t) :: u, s, rest, e, m, m3, sticky, half, lsb, drop, out
        u = transfer(f, 0_c_int32_t)
        s = iand(ishft(u, -31), 1)
        rest = iand(u, int(z'7fffffff', c_int32_t))
        if (rest >= int(z'7f800000', c_int32_t)) then
            out = ior(ishft(s, 7), 127)
        else
            e = ishft(rest, -23) - 127
            m = iand(rest, int(z'7fffff', c_int32_t))
            if (e >= -6) then
                drop = 20
                m3 = ishft(m, -drop)
                lsb = iand(m3, 1)
                half = ishft(1, drop - 1)
                sticky = iand(m, ishft(1, drop) - 1)
                if (sticky > half .or. (sticky == half .and. lsb == 1)) then
                    m3 = m3 + 1
                    if (m3 == 8) then
                        m3 = 0
                        e = e + 1
                    end if
                end if
                if (e > 8 .or. (e == 8 .and. m3 == 7)) then
                    out = ior(ishft(s, 7), 127)
                else
                    out = ior(ior(ishft(s, 7), ishft(e + 7, 3)), iand(m3, 7))
                end if
            else if (e < -10) then
                out = ishft(s, 7)
            else
                m = ior(m, int(z'800000', c_int32_t))
                drop = 20 + (-6 - e)
                m3 = ishft(m, -drop)
                lsb = iand(m3, 1)
                half = ishft(1, drop - 1)
                sticky = iand(m, ishft(1, drop) - 1)
                if (sticky > half .or. (sticky == half .and. lsb == 1)) m3 = m3 + 1
                out = ior(ishft(s, 7), iand(m3, 15))
            end if
        end if
        if (out > 127) out = out - 256
        b = int(out, c_int8_t)
    end function npb_f32_to_e4m3

    pure function npb_rn_e4m3(x) result(r)
        use, intrinsic :: iso_c_binding
        real(c_float), intent(in) :: x
        real(c_float) :: r
        r = npb_e4m3_to_f32(npb_f32_to_e4m3(x))
    end function npb_rn_e4m3
""",
    "float8_e5m2":
    """
    pure function npb_e5m2_to_f32(b) result(r)
        use, intrinsic :: iso_c_binding
        integer(c_int8_t), intent(in) :: b
        real(c_float) :: r
        integer(c_int32_t) :: bb, s, e, m, u, ex, mm
        bb = iand(int(b, c_int32_t), 255)
        s = ishft(bb, -7)
        e = iand(ishft(bb, -2), 31)
        m = iand(bb, 3)
        if (e == 31) then
            u = ior(ishft(s, 31), int(z'7f800000', c_int32_t))
            if (m /= 0) u = ior(u, int(z'400000', c_int32_t))
        else if (e == 0) then
            if (m == 0) then
                u = ishft(s, 31)
            else
                ex = -14
                mm = m
                do while (iand(mm, 4) == 0)
                    mm = ishft(mm, 1)
                    ex = ex - 1
                end do
                u = ior(ior(ishft(s, 31), ishft(ex + 127, 23)), ishft(iand(mm, 3), 21))
            end if
        else
            u = ior(ior(ishft(s, 31), ishft(e + 112, 23)), ishft(m, 21))
        end if
        r = transfer(u, 0.0_c_float)
    end function npb_e5m2_to_f32

    pure function npb_f32_to_e5m2(f) result(b)
        use, intrinsic :: iso_c_binding
        real(c_float), intent(in) :: f
        integer(c_int8_t) :: b
        integer(c_int32_t) :: u, s, rest, e, m, m2, sticky, half, lsb, drop, out
        u = transfer(f, 0_c_int32_t)
        s = iand(ishft(u, -31), 1)
        rest = iand(u, int(z'7fffffff', c_int32_t))
        if (rest > int(z'7f800000', c_int32_t)) then
            out = ior(ishft(s, 7), 126)
        else if (rest == int(z'7f800000', c_int32_t)) then
            out = ior(ishft(s, 7), 124)
        else
            e = ishft(rest, -23) - 127
            m = iand(rest, int(z'7fffff', c_int32_t))
            if (e >= -14) then
                drop = 21
                m2 = ishft(m, -drop)
                lsb = iand(m2, 1)
                half = ishft(1, drop - 1)
                sticky = iand(m, ishft(1, drop) - 1)
                if (sticky > half .or. (sticky == half .and. lsb == 1)) then
                    m2 = m2 + 1
                    if (m2 == 4) then
                        m2 = 0
                        e = e + 1
                    end if
                end if
                if (e > 15) then
                    out = ior(ishft(s, 7), 124)
                else
                    out = ior(ior(ishft(s, 7), ishft(e + 15, 2)), iand(m2, 3))
                end if
            else if (e < -18) then
                out = ishft(s, 7)
            else
                m = ior(m, int(z'800000', c_int32_t))
                drop = 21 + (-14 - e)
                m2 = ishft(m, -drop)
                lsb = iand(m2, 1)
                half = ishft(1, drop - 1)
                sticky = iand(m, ishft(1, drop) - 1)
                if (sticky > half .or. (sticky == half .and. lsb == 1)) m2 = m2 + 1
                out = ior(ishft(s, 7), iand(m2, 7))
            end if
        end if
        if (out > 127) out = out - 256
        b = int(out, c_int8_t)
    end function npb_f32_to_e5m2

    pure function npb_rn_e5m2(x) result(r)
        use, intrinsic :: iso_c_binding
        real(c_float), intent(in) :: x
        real(c_float) :: r
        r = npb_e5m2_to_f32(npb_f32_to_e5m2(x))
    end function npb_rn_e5m2
""",
}


def _fp8_contained(kir: KernelIR) -> str:
    """The fp8 conversion procedures this kernel needs, as CONTAINED procedures
    (siblings see each other, so ``npb_rn_*`` can call the pair; and unlike a
    module there is no .mod artefact to collide between parallel builds).
    Empty for every non-fp8 kernel."""
    return "".join(_FP8_HELPER_SRC[dt] for dt in _fp8_dtypes_used(kir))


def _round_even_helper(rk: str) -> str:
    """A contained ``pure`` half-to-even round for one real kind ``rk``.

    numpy ``round`` / ``rint`` round half-to-EVEN; Fortran ANINT rounds
    half-AWAY. Start from ANINT and, on an exact .5 tie whose ANINT is odd, step
    one ULP back toward the even neighbour by ``sign(1, x)``. Rendering it as a
    procedure -- rather than inline at each call -- keeps the (possibly large)
    argument from being duplicated six times per round."""
    return f"""\

    pure function npb_round_even(x) result(r)
        real({rk}), intent(in) :: x
        real({rk}) :: r
        r = anint(x) - merge(sign(1.0_{rk}, x), 0.0_{rk}, &
            (abs(x - aint(x)) == 0.5_{rk}) .and. (mod(anint(x), 2.0_{rk}) /= 0.0_{rk}))
    end function npb_round_even
"""


def _double_kind() -> str:
    # The ISO_C_BINDING kind token for a 64-bit (double) real, pulled FROM the
    # registry (``real(c_double)`` -> ``c_double``) so it is never hardcoded. Used
    # to force the FloorDiv divide into double precision regardless of the
    # kernel''s own float kind (a bare ``REAL()`` defaults to single and drops
    # mantissa bits on int64 / float64 operands).
    _, _, rest = _fortran_type("float64").partition("(")
    return rest.rstrip(")")


#: The dtype-registry tag for the ABI integer (the size-symbol / default int
#: kind), resolved FROM the registry so a width change (int->int64) needs no edit
#: here -- it is the int tag whose Fortran kind equals that of canonical ``int``.
#: Calls whose Fortran result is INTEGER. ``int``/``len``/``floor``/``ceil``/``round`` map to
#: INT / SIZE / FLOOR / CEILING / NINT, which return integer unconditionally; the
#: :data:`_INT_CALLS_ARGDEP` subset (MAX / MIN and the int helpers) is integer only when every
#: argument is. Shared by the min/max operand typing and :meth:`_expr_is_integer` so the two
#: cannot drift -- they disagreed before, and merge() then real-promoted an integer branch
#: sitting beside an ``int(..)`` cast (bfs).
_INT_RETURNING_CALLS = {
    "int", "len", "max", "min", "floor", "ceil", "round", "ceiling", "nint", "int_floor", "python_mod"
}
_INT_CALLS_ARGDEP = {"max", "min", "int_floor", "python_mod"}

#: Fortran intrinsics whose argument the standard requires to be REAL / COMPLEX -- an
#: INTEGER is rejected outright ("'x' argument of 'sqrt' intrinsic must be REAL or
#: COMPLEX"). numpy promotes an integer operand to float for exactly these, so mirror
#: that with an explicit REAL() instead of emitting ``SQRT(<int>)``: gpt2_block scales
#: attention by ``np.sqrt(dh)`` on the integer head dimension. ABS / MOD / MAX / MIN /
#: SIGN are excluded -- they accept an integer and must keep their integer result.
_REAL_ARG_INTRINSICS = frozenset({"exp", "sqrt", "log", "sin", "cos", "tgamma", "lgamma"})

_SYMBOL_INT_TAG = next((t for t in ("int64", "int32", "int16", "int8") if _fortran_type(t) == _fortran_type("int")),
                       "int64")


def _array_decl(arr: ArrayDesc) -> str:
    intent = "intent(inout)" if arr.is_output else "intent(in)"
    base = _fortran_type(arr.dtype)
    # Fortran rank-N array declaration: a(N), aa(N, M).
    # REVERSED shape so Fortran column-major matches the row-major
    # memory layout of the C-allocated input data: Python ``arr``
    # of shape (N, M, K) row-major has memory ``arr[i, j, k] @
    # offset i*M*K + j*K + k``; Fortran ``arr(K, M, N)`` col-major
    # has ``arr(c, b, a) @ offset (a-1)*K*M + (b-1)*K + (c-1)``,
    # so subscripts are also reversed (see _emit_subscript). Also
    # translate Python ``//`` to ``/`` (Fortran reads ``//`` as
    # string concat).
    if arr.shape:
        dims = ", ".join(_to_fortran_shape_token(s) for s in reversed(arr.shape))
        return f"{base}, {intent} :: {arr.name}({dims})"
    return f"{base}, {intent} :: {arr.name}"


def _scalar_decl(name: str, dtype: str, is_output: bool, assigned: bool = False) -> str:
    base = _fortran_type(dtype)
    # Input scalars are passed BY VALUE (the ``value`` attribute) so the C-ABI
    # matches C / C++ -- one uniform scalar convention across every
    # target (a bind(C) scalar without ``value`` would be a C pointer). An output
    # scalar (rare; kernels are C-style) stays by reference.
    if is_output:
        return f"{base}, intent(inout) :: {name}"
    # A value scalar the body REASSIGNS -- e.g. a masked-reduction nest whose
    # per-element staged read ``a_index = a[i]`` reuses the scalar dummy as a
    # loop local -- must drop ``intent(in)``: Fortran forbids an intent(in) dummy
    # on the LHS. The ``value`` attribute makes it a local copy, so dropping the
    # intent keeps the recompute legal without touching the caller or the ABI
    # (still by value). Mirrors :func:`_symbol_decl`.
    if assigned:
        return f"{base}, value :: {name}"
    return f"{base}, value, intent(in) :: {name}"


def _symbol_decl(name: str, assigned: bool = False) -> str:
    # Shape symbols are int64 passed BY VALUE (canonical ABI). Normally
    # read-only (``intent(in)``); but a kernel may RECOMPUTE a size symbol it
    # also receives -- e.g. spmv/gmres ``M = A_indptr.shape[0] - 1`` (== the
    # passed M by construction). C tolerates assigning a param; Fortran forbids
    # ``intent(in)`` on the LHS. Since the dummy has the ``value`` attribute it
    # is a local copy, so DROPPING intent(in) makes the recompute legal without
    # affecting the caller or the ABI (still by value).
    if assigned:
        return f"{_fortran_type('int')}, value :: {name}"
    return f"{_fortran_type('int')}, value, intent(in) :: {name}"


def _assigned_bool_literal(tree: ast.AST) -> Set[str]:
    """Array names assigned a bare ``True`` / ``False`` (whole-array or element).

    A numpy ``int32`` array used purely as a 0/1 flag (cloudsc's ``llfall``,
    ``llrainliq``, ``llindex3``; even a mis-typed ``float64`` ``llflag``) is a
    LOGICAL array as far as Fortran is concerned: ``arr[i] = True`` /
    ``if not arr[i]`` only type-check against ``logical``. Declaring such locals
    ``logical(c_bool)`` lets the boolean literal / ``.not.`` / ``.and.`` flow
    natively (C/C++ tolerate the int-as-bool spelling; gfortran does not)."""
    out: Set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        tgt = node.targets[0]
        if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name):
            name = tgt.value.id
        elif isinstance(tgt, ast.Name):
            name = tgt.id
        else:
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool):
            out.add(name)
    return out


def _produces_logical(rhs: ast.AST) -> bool:
    """``True`` when an RHS expression evaluates to a boolean (LOGICAL) array.

    Covers comparisons, ``and``/``or`` (``BoolOp``), ``not``, and the numpy
    elementwise spellings ``&`` / ``|`` / ``^`` (``BinOp`` with BitAnd/BitOr/
    BitXor on logical operands) and ``~mask`` (``Invert`` of a logical). This
    is what marks a local like ``in_range = (rsq < c) & (rsq > 0)`` as a
    Fortran ``logical`` array (otherwise it defaults to real and the LOGICAL
    assignment is rejected)."""
    if isinstance(rhs, (ast.Compare, ast.BoolOp)):
        return True
    if isinstance(rhs, ast.UnaryOp):
        if isinstance(rhs.op, ast.Not):
            return True
        if isinstance(rhs.op, ast.Invert):
            return _produces_logical(rhs.operand)
    if isinstance(rhs, ast.BinOp) and isinstance(rhs.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
        return _produces_logical(rhs.left) and _produces_logical(rhs.right)
    return False


def _int_literal_value(node: ast.AST) -> Optional[int]:
    """The integer value of a possibly-negated int literal (``5`` -> 5, ``-5``
    -> -5), else None. A negative literal parses as ``UnaryOp(USub,
    Constant(5))`` rather than a bare ``Constant(-5)``, so an
    ``isinstance(node, ast.Constant)`` check alone misses it -- which left a
    ``merge(ci, -1, ...)`` branch un-kinded (``-1`` defaults to int32) and
    clashed with an int64 partner."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op,
                                                     (ast.USub, ast.UAdd)) and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int) and not isinstance(node.operand.value, bool)):
        return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
    return None


# ---------------------------------------------------------------------------
# Body walker
# ---------------------------------------------------------------------------

# Operator tables live in numpyto_common.operators, keyed by target; the
# Fortran backend reads its column. Local aliases keep the existing call sites.
_BINOP = operators.BINOP["fortran"]
_CMPOP = operators.CMPOP["fortran"]
_BOOLOP = operators.BOOLOP["fortran"]


class _FortranBodyEmitter(BaseEmitter):
    """Walk the Python AST and emit Fortran statements.

    Loop iter variables are tracked so subscripts can be adjusted from
    Python's 0-based to Fortran's 1-based convention -- ``a[i]`` in a
    ``for i in range(N)`` loop emits as ``a(i + 1)`` (we always add 1
    for loop-iter names; constants and shape symbols pass through).

    ``emit_block`` / ``emit_stmt`` come from :class:`BaseEmitter`; the Fortran
    leaf hooks are a blank statement terminator, ``exit`` / ``cycle`` for
    break / continue, and an overridden ``_emit_return`` that emits a bare
    ``return`` (rather than dropping it as the C default does).
    """

    _STMT_TERM = ""
    _KW_BREAK = "exit"
    _KW_CONTINUE = "cycle"

    def emit_stmt(self, node: ast.stmt, indent: str) -> str:
        # A bare helper-subroutine call statement -- an array-returning helper's
        # out-param call ``h(args, out)`` -- emits as ``call h(args, out)`` (the
        # out-param is already the last arg; a scalar helper's ``X = h(...)`` still
        # routes through ``_emit_assign``).
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id in self._helper_out):
            args = ", ".join(self.emit_expr(a) for a in node.value.args)
            return f"{indent}call {node.value.func.id}({args})"
        return super().emit_stmt(node, indent)

    def _emit_return(self, node: ast.Return, indent: str) -> str:
        # In a HELPER subroutine the returned value is written to the out-param
        # ``return_mode`` (Fortran has no by-value return in our out-param
        # scheme -- see the user-chosen design), then a bare ``return``.
        mode = getattr(self, "return_mode", None)
        if mode is not None and node.value is not None:
            return f"{indent}{mode} = {self.emit_expr(node.value)}\n{indent}return"
        return f"{indent}return"

    def __init__(self, kir: KernelIR):
        self.kir = kir
        #: Lazy cache of the names used in an integer context (see ``_int_uses``).
        self._int_uses_cache: Optional[Set[str]] = None
        #: When this body IS a helper subroutine: the out-param name its
        #: ``return`` writes into (``None`` for the kernel).
        self.return_mode: Optional[str] = None
        #: Parallel emit variant (emit_fortran_omp): annotate each outermost
        #: independent / reduction loop with ``!$omp parallel do``. Off for the
        #: plain sequential emitter.
        self.parallel: bool = False
        #: Set while emitting the body of a loop already marked parallel, so
        #: nested loops are NOT also tagged (no nested parallel regions).
        self.parallel_active: bool = False
        #: name -> out-param name for each non-inlinable helper CALLED here, so a
        #: ``X = helper(args)`` assign lowers to ``call helper(args, X)``.
        self._helper_out: Dict[str, str] = {}
        self.array_names: Set[str] = {a.name for a in kir.arrays}
        zeros = kir.zeros_locals
        self.local_arrays: Dict[str, List[str]] = {
            name: list(shape) if shape else ["1"]
            for name, shape in zeros.items()
        }
        #: Arrays whose shape is entirely size-1 (a ``(1,)`` scalar buffer). Read bare in a value
        #: expression they must be scalarised to ``x(1)`` -- else ``a(i+1) > x`` is a rank-0 vs rank-1
        #: mismatch (mirrors the C emitter's ``x[0]`` scalarisation).
        self._size1_arrays: Set[str] = {
            a.name
            for a in kir.arrays if a.shape and all(str(s) == "1" for s in a.shape)
        }
        self._size1_arrays.update(
            name for name, shape in self.local_arrays.items() if all(str(s) == "1" for s in shape))
        self._loop_iter_names: Set[str] = set()
        # ISO_C_BINDING real kind for float literals. Fortran is strict
        # about kind mixing (a ``1.0_c_double`` literal beside a
        # ``real(c_float)`` variable is a hard "Different type kinds"
        # error, unlike C's silent promotion), so literals must match the
        # kernel's float precision set on the IR. Resolved through
        # ``compute_dtype`` so an fp8 kernel -- whose values are held in
        # real(c_float) between the promote and the demote -- suffixes its
        # literals ``_c_float``; the raw ``float8_e4m3`` would miss this map and
        # fall through to c_double, which is exactly the kind clash above.
        self._rk = {"float32": "c_float", "float16": "c_float"}.get(
            dtypes.compute_dtype(kir.float_precision or "float64"), "c_double")
        # libm functions Fortran lacks an intrinsic for, called through a bind(C)
        # interface so the result is bit-identical to the C backend (and numpy,
        # which also uses libm) -- collected here, declared in the spec part.
        self._used_libm: Set[Tuple[str, str]] = set()
        # Whether the body calls ``np.round`` / ``np.rint`` -- lowered to a CONTAINED
        # ``npb_round_even`` helper rather than an inline half-to-even expression.
        # The inline form repeats the (arbitrarily large) argument six times, so a
        # round of a big sub-expression -- histogram_equalization's CDF-normalise LUT
        # -- explodes into a statement gfortran spends minutes optimising at -O2 (past
        # the compile-timeout budget). A one-argument helper call keeps it O(1).
        self._used_round_even = False
        # Whether the body references IEEE infinity / NaN (``np.inf`` / ``np.nan``
        # lowered to the C ``INFINITY`` / ``NAN`` names), which Fortran expresses via
        # ``ieee_value`` -- gates a ``use, intrinsic :: ieee_arithmetic`` in the preamble.
        self._used_ieee = False

    def _emit_for(self, node: ast.For, indent: str) -> str:
        target = node.target
        if not isinstance(target, ast.Name):
            raise NotImplementedError("only single-name for-target supported")
        var = target.id
        if not (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"):
            raise NotImplementedError("only ``for x in range(...)`` supported")
        args = node.iter.args
        if len(args) == 1:
            lo, hi, step = "0", self.emit_expr(args[0]), "1"
        elif len(args) == 2:
            lo, hi, step = self.emit_expr(args[0]), self.emit_expr(args[1]), "1"
        elif len(args) == 3:
            lo = self.emit_expr(args[0])
            hi = self.emit_expr(args[1])
            step = self.emit_expr(args[2])
        else:
            raise NotImplementedError("range() needs 1-3 args")
        # OpenMP parallel-scope decision (parallel variant only), mirroring the C
        # emitter: tag the OUTERMOST eligible loop of a nest -- an independent map
        # -> ``!$omp parallel do``; a single-scalar reduction -> add
        # ``reduction(op:acc)``. A colliding scatter is refused up front
        # (emit_fortran_omp), so a not-parallel-safe loop here is a carried
        # dependence and stays serial. ``node`` is from the same tree the body
        # emits, so the reduction accumulator name matches the emitted code.
        omp_prefix = ""
        if self.parallel and not self.parallel_active and not parallelism.is_timestep_loop(node):
            red = parallelism.loop_reduction(node)
            if red is not None:
                op, acc = red
                omp_prefix = f"{indent}!$omp parallel do reduction({op}:{acc})\n"
            elif parallelism.loop_is_parallel_safe(node):
                omp_prefix = f"{indent}!$omp parallel do\n"
        entered_parallel = bool(omp_prefix)
        if entered_parallel:
            self.parallel_active = True
        self._loop_iter_names.add(var)
        body = self.emit_block(node.body, indent + "    ")
        self._loop_iter_names.discard(var)
        if entered_parallel:
            self.parallel_active = False
        # Fortran ``do`` is inclusive on both ends. For a positive
        # step the Python ``range(lo, hi)`` last value is ``hi - 1``;
        # for a negative step it is ``hi + 1``. (Symbolic step expressions
        # fall back to the safest form when we cannot resolve the sign.)
        negative_step = step.startswith("-") or step.startswith("(-")
        adj = "+ 1" if negative_step else "- 1"
        upper = f"({hi}) {adj}"
        if step == "1":
            return (f"{omp_prefix}{indent}do {var} = {lo}, {upper}\n"
                    f"{body}\n"
                    f"{indent}end do")
        return (f"{omp_prefix}{indent}do {var} = {lo}, {upper}, {step}\n"
                f"{body}\n"
                f"{indent}end do")

    def _emit_while(self, node: ast.While, indent: str) -> str:
        body = self.emit_block(node.body, indent + "    ")
        return (f"{indent}do while ({self.emit_expr(node.test)})\n"
                f"{body}\n"
                f"{indent}end do")

    def _emit_if(self, node: ast.If, indent: str) -> str:
        cond = self._emit_logical_test(node.test)
        then = self.emit_block(node.body, indent + "    ")
        out = [f"{indent}if ({cond}) then", then]
        # ``elif`` flattens.
        cur = node.orelse
        while cur and len(cur) == 1 and isinstance(cur[0], ast.If):
            sub = cur[0]
            cond = self._emit_logical_test(sub.test)
            sub_body = self.emit_block(sub.body, indent + "    ")
            out.append(f"{indent}else if ({cond}) then")
            out.append(sub_body)
            cur = sub.orelse
        if cur:
            else_body = self.emit_block(cur, indent + "    ")
            out.append(f"{indent}else")
            out.append(else_body)
        out.append(f"{indent}end if")
        return "\n".join(out)

    def _is_logical_operand(self, node: ast.AST) -> bool:
        """``True`` when ``node`` evaluates to a Fortran LOGICAL: a comparison,
        a boolean combination, a ``not``, or a known boolean-array local. Used
        to route ``&`` / ``|`` to ``.AND.`` / ``.OR.`` rather than IAND/IOR."""
        if isinstance(node, (ast.Compare, ast.BoolOp)):
            return True
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return True
        # ``~x`` (mask inversion) is logical iff its operand is -- so a combine
        # ``m & ~m3`` routes to ``.AND.`` of two logicals, not integer IAND.
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
            return self._is_logical_operand(node.operand)
        # A ``& | ^`` combine of logicals is itself logical, so a NESTED combine
        # ``(m1 & m2) | m3`` routes the OUTER op to ``.OR.`` of two logicals rather than
        # integer IOR (gfortran rejects IAND/IOR/IEOR of LOGICAL args). Mirrors
        # ``_is_logical_node``; without it only two-way combines lower correctly.
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
            return self._is_logical_operand(node.left) and self._is_logical_operand(node.right)
        if isinstance(node, ast.Name) and node.id in self._logical_array_locals:
            return True
        # A bool-typed scalar parameter (vexx_k config flag) is logical.
        if isinstance(node, ast.Name) and node.id in self._bool_scalar_names():
            return True
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return True
        # A subscript into a known boolean-array local is also logical.
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                and node.value.id in self._logical_array_locals:
            return True
        return False

    def _is_int_flag_scalar(self, node: ast.AST) -> bool:
        """``arr[i]`` (int PARAMETER array) OR a bare integer SCALAR parameter
        used as a 0/1 flag (ICON ``lextra_diffu`` / ``ldeepatmo``).

        Such a parameter cannot be re-typed to ``logical`` (it crosses the C
        ABI), so a boolean use must be wrapped ``/= 0`` (truthy) at the site. A
        bare flag never appears in int arithmetic, so ``_names_used_as_int``
        misses it -- recognise it from the scalar's declared integer dtype."""
        ints = vars(self).get("_int_array_names", set())
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in ints):
            return True
        int_scalars = vars(self).get("_int_scalar_names")
        if int_scalars is None:
            int_scalars = {
                s.name
                for s in self.kir.scalars
                if s.dtype in ("int", "int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8")
            }
            self._int_scalar_names = int_scalars
        return isinstance(node, ast.Name) and node.id in int_scalars

    def _is_logical_node(self, node: ast.AST) -> bool:
        """True when ``node`` emits a Fortran LOGICAL: a comparison / bool-op /
        ``not`` (``_produces_logical``), a Name / Subscript of a local the
        lowering typed boolean, or a ``~`` / ``& | ^`` combine of those (so
        ``~(m1 & m2)`` is recognised -- ``_produces_logical`` alone bottoms out
        at the bare Name locals and misses it)."""
        if _produces_logical(node):
            return True
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
            return self._is_logical_node(node.operand)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
            return self._is_logical_node(node.left) and self._is_logical_node(node.right)
        logicals = vars(self).get("_logical_array_locals", set())
        if isinstance(node, ast.Name) and node.id in logicals:
            return True
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in logicals):
            return True
        return False

    def _bool_scalar_names(self) -> Set[str]:
        """Scalar PARAMETERS the frontend typed ``bool`` (vexx_k's config flags
        ``okvan`` / ``okpaw`` / ``tqr`` / ...). They declare ``logical(c_bool)`` and
        so are ALREADY logical -- a boolean use must NOT be wrapped ``/= 0``."""
        names = vars(self).get("_bool_scalar_names_cache")
        if names is None:
            names = {s.name for s in self.kir.scalars if s.dtype in ("bool", "bool_")}
            self._bool_scalar_names_cache = names
        return names

    def _as_logical_operand(self, node: ast.AST) -> str:
        """Emit ``node`` as a Fortran LOGICAL operand for ``.and.`` / ``.or.``.

        A comparison / ``not`` / logical-local / bool literal / bool-typed scalar
        parameter already IS logical and passes through; anything else in a boolean
        context is a numeric truthiness test (cloudsc's int flag ``ldcum[jl]`` or a
        folded literal ``0``) and becomes ``(expr) /= 0`` -- the Fortran spelling of
        Python's non-zero truthiness."""
        e = self.emit_expr(node)
        if _produces_logical(node):
            return e
        # A boolean literal (a folded ``.false.`` / ``.true.``) is already logical.
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return e
        if isinstance(node, ast.Name) and node.id in self._bool_scalar_names():
            return e
        logicals = vars(self).get("_logical_array_locals", set())
        if isinstance(node, ast.Name) and node.id in logicals:
            return e
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in logicals):
            return e
        return f"({e}) /= 0"

    def _emit_logical_test(self, node: ast.AST) -> str:
        """Emit a condition expression as a Fortran scalar LOGICAL.

        Python ``if x & 1`` (or any int expr) is truthy on non-zero;
        Fortran requires an explicit comparison. When the test is an
        integer-ish expression (bitwise BinOp / Call to IAND/IOR /
        Subscript on an int array), wrap with ``/= 0``.
        """
        cond = self.emit_expr(node)
        # Already a clearly-logical expression -- no wrap needed. ``&`` / ``|``
        # / ``^`` (BitAnd/BitOr/BitXor) over LOGICAL operands (mandelbrot2's
        # ``(abs(Z) > horizon) & (N_out == 0)`` -> ``.and.`` of two
        # comparisons) yields a LOGICAL, so wrapping it in ``/= 0`` would be a
        # LOGICAL-vs-INTEGER type error.
        if _produces_logical(node):
            return cond
        # Already a clearly-logical expression -- no wrap needed.
        if isinstance(node, (ast.Compare, ast.BoolOp)):
            return cond
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return cond
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return cond
        # Heuristic: BinOp with a bitwise op, or a Call whose name is
        # one of the int intrinsics, or a Name that's in the int_uses
        # set -- wrap with ``/= 0``.
        int_uses = self._int_uses()

        def is_int_expr(n):
            if isinstance(n, ast.Constant):
                return isinstance(n.value, int) and not isinstance(n.value, bool)
            if isinstance(n, ast.BinOp):
                BITWISE = (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)
                if isinstance(n.op, BITWISE):
                    return True
                return is_int_expr(n.left) and is_int_expr(n.right)
            if isinstance(n, ast.UnaryOp):
                if isinstance(n.op, ast.Invert):
                    return True
                return is_int_expr(n.operand)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                # IAND / IOR etc. are emitted from bitwise BinOps; bare
                # ``int`` / ``len`` calls also return int.
                if n.func.id in {"len", "int", "range"}:
                    return True
            if isinstance(n, ast.Name):
                return n.id in int_uses
            return False

        # ``if arr[i]:`` where arr is an int parameter flag (cloudsc ``ldcum``).
        if is_int_expr(node) or self._is_int_flag_scalar(node):
            return f"({cond}) /= 0"
        return cond

    def _emit_assign(self, node: ast.Assign, indent: str) -> str:
        if len(node.targets) != 1:
            raise NotImplementedError("chained assignment not supported")
        target = node.targets[0]
        # ``X = helper(args)`` where helper is emitted as a subroutine with a
        # trailing out-param -> ``call helper(args, X)``.
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id in self._helper_out):
            call_args = [self.emit_expr(a) for a in node.value.args]
            call_args.append(self.emit_expr(target))
            return f"{indent}call {node.value.func.id}({', '.join(call_args)})"
        # The __optarena_zeros__ marker may have been renamed by the
        # leading-underscore-strip pass to ``x_optarena_zeros__``.
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id in {"__optarena_zeros__", "x_optarena_zeros__"}):
            # For locals whose shape uses a loop iter, emit an
            # ALLOCATE here (the loop iter is now in scope). The
            # caller pre-populates ``inline_alloc_locals`` for the
            # subset that needs this treatment. Each marker fires
            # once per kernel run; if the local is inside a deeper
            # loop the allocate is re-issued each iteration -- the
            # deallocate (in the prelude) gets a paired free at
            # function exit, so re-allocate is a leak unless we also
            # emit a deallocate before re-alloc. Keep it simple:
            # deallocate first if allocated.
            if isinstance(target, ast.Name):
                inline = getattr(self, "inline_alloc_locals", {})
                if target.id in inline:
                    rev_shape, _ftype = inline[target.id]
                    dims = ", ".join(rev_shape)
                    t = target.id
                    # Allocate only once -- and REUSE the existing buffer when
                    # the shape is unchanged. A same-shape ``__reassign__``
                    # self-assign (resnet batchnorm ``x = (x - mean)/std``) is
                    # followed by a loop that reads the OLD ``x``; an
                    # unconditional deallocate+allocate would hand it a fresh
                    # (uninitialised) array. A genuine VERSION change (a
                    # same-name local re-bound to a different shape) still
                    # reallocates -- guarded on the element count so only a real
                    # size change pays for it.
                    # Guard PER DIMENSION, not on the total element count: a
                    # reshape/transpose transient like stockham_fft's
                    # ``(R**(K-i-1), R, R**i)`` keeps a constant product (R**K)
                    # across loop iterations while each dimension's extent shifts
                    # -- a ``size(t) /= total`` test never trips, so the buffer
                    # keeps the first iteration's shape and later writes run off
                    # the end (SIG11).
                    realloc = (" .or. ".join(f"size({t}, {i + 1}) /= ({d})"
                                             for i, d in enumerate(rev_shape)) if rev_shape else f"size({t}) /= 1")
                    alloc = (f"{indent}if (.not. allocated({t})) then\n"
                             f"{indent}    allocate({t}({dims}))\n"
                             f"{indent}else if ({realloc}) then\n"
                             f"{indent}    deallocate({t})\n"
                             f"{indent}    allocate({t}({dims}))\n"
                             f"{indent}end if")
                    # An ALLOCATABLE ``np.zeros`` / ``np.ones`` must ALSO be filled
                    # after allocation -- Fortran ``allocate`` does NOT initialise
                    # the memory (the C path memsets), so without this the array is
                    # read as heap garbage (lulesh's node-normal accumulator ``pf =
                    # np.zeros((n,8,3))`` then ``pf[:,k,:] += ..``). The fixed-bound
                    # path below already fills; the inline-allocate path skipped it.
                    # The ``__reassign__`` sentinel (a self-referential reset the
                    # following loop overwrites/reads) must NOT be re-filled.
                    is_reassign = any(
                        isinstance(a, ast.Constant) and a.value == "__reassign__" for a in node.value.args)
                    if not is_reassign:
                        kind = self.kir.zeros_fills.get(target.id)
                        is_logical = target.id in vars(self).get("_logical_array_locals", set())
                        if kind in ("zeros", "zeros_like"):
                            alloc += f"\n{indent}{t} = {'.false.' if is_logical else '0'}"
                        elif kind in ("ones", "ones_like"):
                            alloc += f"\n{indent}{t} = {'.true.' if is_logical else '1'}"
                    return alloc
                # A ``__reassign__`` marker -- the lowering's no-op sentinel for a
                # whole-array reassignment ``X = f(...)`` immediately followed by a
                # per-element loop that FULLY overwrites X -- must NOT be re-zeroed:
                # the loop may READ the old X (self-referential ``qq = qq + qd*qd``,
                # bicgstab ``p = r + beta*(p - omega*v)``), and re-zeroing here
                # corrupts it. (The C emitter already treats it as a no-op; only a
                # GENUINE ``X = np.zeros(...)`` reset -- no sentinel -- zero-fills.)
                if any(isinstance(a, ast.Constant) and a.value == "__reassign__" for a in node.value.args):
                    return ""
                # A fixed-bound zeros/ones local (declared in the prelude)
                # re-constructed here must be re-filled: Fortran does NOT
                # zero arrays at declaration, and an in-loop ``X =
                # np.zeros(...)`` reset otherwise keeps the previous
                # pass's values (contour_integral's Tz accumulator). The
                # whole-array ``= 0`` / ``= 1`` converts to the local's
                # numeric kind (incl. complex).
                kind = self.kir.zeros_fills.get(target.id)
                # A LOGICAL array (cfl_clip / levmask, np.bool_) fills with
                # ``.false.`` / ``.true.`` -- Fortran rejects int 0/1 there.
                is_logical = target.id in vars(self).get("_logical_array_locals", set())
                if kind in ("zeros", "zeros_like"):
                    return f"{indent}{target.id} = {'.false.' if is_logical else '0'}"
                if kind in ("ones", "ones_like"):
                    return f"{indent}{target.id} = {'.true.' if is_logical else '1'}"
            return ""  # local declared in prelude (empty / scratch)
        # Skip tautological self-assigns ``I = I`` that the shape-
        # resolution pass leaves behind when ``utens_stage.shape[0]``
        # simplifies back to its declaring symbol. Fortran rejects
        # ``intent(in)`` parameters appearing on the LHS even when the
        # RHS is the same Name.
        if (isinstance(target, ast.Name) and isinstance(node.value, ast.Name) and target.id == node.value.id):
            return ""
        # Storing a numeric 0/1 into a LOGICAL array element -- the ``.any`` /
        # ``.all`` reduction accumulator zero-init (``levelmask[i] = 0``) -- must
        # be a logical literal in Fortran, not an integer.
        if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                and target.value.id in vars(self).get("_logical_array_locals", set())
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, bool))):
            return f"{indent}{self.emit_expr(target)} = {'.true.' if node.value.value else '.false.'}"
        rhs = self.emit_expr(node.value)
        lhs = self.emit_expr(target)
        fns = self._store_fns(target)
        if fns is not None:  # fp8 target: demote the real(c_float) RHS to the byte
            rhs = f"{fns.demote}({rhs})"
        return f"{indent}{lhs} = {rhs}"

    def _emit_augassign(self, node: ast.AugAssign, indent: str) -> str:
        lhs = self.emit_expr(node.target)
        rhs = self.emit_expr(node.value)
        fp8 = self._store_fns(node.target)
        if fp8 is not None:
            # ``y(i) += e`` on fp8 storage: the target is a 1-byte code, so the
            # READ must promote and the result demote -- the two occurrences of
            # ``lhs`` below are NOT interchangeable. Bitwise augmented ops cannot
            # reach here (an fp8 array is never an integer operand). No round()
            # wrap: the demote rounds to the same grid, so it would be a no-op.
            op = _BINOP.get(type(node.op))
            if op is None:
                raise NotImplementedError(f"augmented op {type(node.op).__name__} on fp8")
            return f"{indent}{lhs} = {fp8.demote}({fp8.promote}({lhs}) {op} ({rhs}))"
        # Bitwise / shift augmented ops -- map to the integer
        # intrinsic forms used in the BinOp emit.
        if isinstance(node.op, ast.BitAnd):
            return f"{indent}{lhs} = IAND({lhs}, {rhs})"
        if isinstance(node.op, ast.BitOr):
            return f"{indent}{lhs} = IOR({lhs}, {rhs})"
        if isinstance(node.op, ast.BitXor):
            return f"{indent}{lhs} = IEOR({lhs}, {rhs})"
        if isinstance(node.op, ast.LShift):
            return f"{indent}{lhs} = ISHFT({lhs}, {rhs})"
        if isinstance(node.op, ast.RShift):
            # numpy ``>>`` on a SIGNED integer is an ARITHMETIC (sign-preserving)
            # shift; ISHFT is logical (zero-fill). SHIFTA replicates the sign bit.
            return f"{indent}{lhs} = SHIFTA({lhs}, {rhs})"
        op = _BINOP.get(type(node.op))
        if op is None:
            raise NotImplementedError(f"augmented op {type(node.op).__name__}")
        return f"{indent}{lhs} = {lhs} {op} ({rhs})"

    def emit_expr(self, node: ast.AST) -> str:
        """Emit an expression, rounding a float BinOp result back to the fp8 grid
        when the kernel computes in fp8. Wrapping HERE rather than at each of the
        BinOp branch's exits keeps one seam for Pow / FloorDiv / MATMUL / MODULO
        and the generic arithmetic tail alike."""
        text = self._emit_expr_inner(node)
        if isinstance(node, ast.BinOp):
            return self._fp8_round(node, text)
        return text

    def _fp8_round(self, node: ast.BinOp, text: str) -> str:
        """Wrap a float BinOp result in the fp8 round-to-grid procedure.

        Load-bearing, not belt-and-braces: ml_dtypes rounds back to fp8 after
        EVERY op, so promoting on load and demoting only on store would compute
        the chain in float and round once -- measurably diverging from the numpy
        oracle (see tests/test_fp8_emission.py).
        """
        if isinstance(node.op, _FP8_NON_ARITH_OPS):
            return text
        fns = self._kernel_fp8_fns()
        if fns is None or not self._touches_fp8(node):
            return text
        return f"{fns.round}({text})"

    def _kernel_fp8_fns(self):
        """The kernel's single fp8 format's procedures, or ``None`` if it uses
        none. A kernel mixing BOTH formats has no well-defined per-op grid, so it
        is refused rather than silently rounded to the wrong one."""
        cache = vars(self).get("_fp8_fns_cache", False)
        if cache is not False:
            return cache
        used = _fp8_dtypes_used(self.kir)
        if len(used) > 1:
            raise NotImplementedError(f"kernel {self.kir.kernel_name!r} mixes fp8 formats {used}: the grid each "
                                      f"intermediate rounds to is ambiguous")
        fns = _fp8_fns(used[0]) if used else None
        self._fp8_fns_cache = fns
        return fns

    def _name_dtype(self, name: str) -> Optional[str]:
        """dtype of a Name -- an array, a local, or a by-value SCALAR param. The
        scalars are consulted too, so an fp8 ``alpha`` is promoted on read."""
        for a in self.kir.arrays:
            if a.name == name:
                return a.dtype
        dt = self.kir.local_dtypes.get(name)
        if dt is not None:
            return dt
        for sca in self.kir.scalars:
            if sca.name == name:
                return sca.dtype
        return None

    def _touches_fp8(self, node: ast.AST) -> bool:
        """True when the subtree reads an fp8 array / scalar param / local -- so
        the enclosing op yields an fp8-valued real that must be re-rounded.

        fp8-specific by design rather than reusing ``_expr_is_real``, which is
        deliberately conservative ("stays False when unsure") -- an unproven real
        would silently skip its rounding, which is a correctness gap here.
        """
        for sub in ast.walk(node):
            if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name):
                name = sub.value.id
            elif isinstance(sub, ast.Name):
                name = sub.id
            else:
                continue
            if _fp8_fns(self._name_dtype(name) or "") is not None:
                return True
        return False

    def _promote_name_read(self, node: ast.Name, access: str) -> str:
        """Promote a bare fp8 Name (a scalar param, an fp8 local, or a size-1 fp8
        array read as ``x(1)``) to real(c_float) on READ. Store ctx falls through
        to the store seam."""
        if not isinstance(node.ctx, ast.Load):
            return access
        fns = _fp8_fns(self._name_dtype(node.id) or "")
        return f"{fns.promote}({access})" if fns is not None else access

    def _store_fns(self, target: ast.AST):
        """:class:`_Fp8Fns` when an assignment TARGET is an fp8 element / name,
        else ``None`` -- the store half of the promote / demote model."""
        base = target
        while isinstance(base, ast.Subscript):
            base = base.value
        if not isinstance(base, ast.Name):
            return None
        return _fp8_fns(self._name_dtype(base.id) or "")

    def _emit_expr_inner(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool):
                return ".true." if v else ".false."
            # Parenthesise NEGATIVE literals: gfortran rejects a unary minus
            # immediately following a binary operator (``a - -0.7`` -> "Unary
            # operator following arithmetic operator"), which arises whenever a
            # negative physical constant (cloudsc ``ydthf_r4ies = -0.7``) is
            # inlined as an operand. ``a - (-0.7)`` is accepted everywhere.
            if isinstance(v, int):
                return f"({v})" if v < 0 else str(v)
            if isinstance(v, float):
                if not math.isfinite(v):
                    # inf / nan have no Fortran literal form -- express via
                    # ieee_value and flag the intrinsic use (same path the
                    # INFINITY / NAN names lowered from np.inf / np.nan take).
                    self._used_ieee = True
                    if math.isnan(v):
                        return f"ieee_value(0.0_{self._rk}, ieee_quiet_nan)"
                    sign = "ieee_positive_inf" if v > 0 else "ieee_negative_inf"
                    return f"ieee_value(0.0_{self._rk}, {sign})"
                lit = f"{v}_{self._rk}"
                return f"({lit})" if v < 0 else lit
            if isinstance(v, complex):
                # Fortran complex literal: (real, imag)
                return f"({v.real}_{self._rk}, {v.imag}_{self._rk})"
            if v is None:
                # ``None`` only appears in kwargs (e.g. ``rcond=None``);
                # the caller drops the kwarg. Emit Fortran's null
                # marker so a leaked use produces a clear runtime
                # rather than a parse error.
                return "0"
            raise NotImplementedError(f"literal {v!r}")
        if isinstance(node, ast.Name):
            # ``np.inf`` / ``np.nan`` were lowered to the C99 ``INFINITY`` / ``NAN``
            # names; Fortran has no such macro, so express them as the kind-matched
            # ``ieee_value`` (else a bare ``INFINITY`` is an undeclared real and a
            # ``merge`` beside a real(c_double) raises a kind mismatch).
            if node.id == "INFINITY":
                self._used_ieee = True
                return f"ieee_value(0.0_{self._rk}, ieee_positive_inf)"
            if node.id == "NAN":
                self._used_ieee = True
                return f"ieee_value(0.0_{self._rk}, ieee_quiet_nan)"
            # A size-1 array read bare in a value expression is its sole element ``x(1)`` (Fortran is
            # 1-based), not the whole rank-1 array -- so ``a(i+1) > x`` is a scalar comparison, not a
            # rank-0-vs-rank-1 mismatch. An explicit ``x[0]`` goes through _emit_subscript, not here.
            access = f"{node.id}(1)" if node.id in self._size1_arrays else node.id
            return self._promote_name_read(node, access)
        if isinstance(node, ast.Tuple):
            # ``(a, b, c)`` as an axis tuple / array constructor -- emit
            # the Fortran array constructor syntax. Bare elements only;
            # nested tuples are not supported.
            elts = ", ".join(self.emit_expr(e) for e in node.elts)
            return f"[{elts}]"
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Invert):
                # ``~x`` on a BOOLEAN operand is numpy logical negation (mask
                # inversion), not integer bitwise NOT -- emit ``.not.``. On an
                # integer operand it stays the Fortran ``NOT(x)`` intrinsic.
                if self._is_logical_node(node.operand):
                    return f".not. ({self.emit_expr(node.operand)})"
                return f"NOT({self.emit_expr(node.operand)})"
            if isinstance(node.op, ast.USub):
                return f"(-({self.emit_expr(node.operand)}))"
            if isinstance(node.op, ast.UAdd):
                return f"(+({self.emit_expr(node.operand)}))"
            if isinstance(node.op, ast.Not):
                # ``.not.`` needs a LOGICAL operand: an int flag (``not lvn_only``)
                # must become ``.not. ((lvn_only) /= 0)``, not ``.not. <int>``.
                return f"(.not. {self._as_logical_operand(node.operand)})"
            raise NotImplementedError(f"unary {type(node.op).__name__}")
        if isinstance(node, ast.BinOp):
            # Pow: Fortran has ``**`` for both integer and real exponents.
            if isinstance(node.op, ast.Pow):
                return (f"({self.emit_expr(node.left)} ** "
                        f"{self.emit_expr(node.right)})")
            # FloorDiv: Fortran's intrinsic ``FLOOR(a/b)`` gives the
            # numpy ``//`` semantics for mixed-sign inputs. FLOOR defaults to the
            # int32 KIND, so pin it to the int64 ABI kind -- otherwise the result
            # clashes with int64 operands (lenet ``MODULO(.., FLOOR(..))``).
            if isinstance(node.op, ast.FloorDiv):
                if self._expr_is_integer(node.left) and self._expr_is_integer(node.right):
                    # Integer ``//``: Fortran integer ``/`` TRUNCATES toward zero
                    # but numpy ``//`` FLOORS toward -inf. Both operands are cast to
                    # ONE integer kind (INT is exact for integers -- no REAL()
                    # mantissa loss above 2**53, and homogeneous kinds avoid the
                    # mixed-kind "GNU Extension" gfortran rejects). Correct the
                    # truncated quotient by -1 when the remainder is nonzero and the
                    # operand signs differ (the C ``int_floor`` rule).
                    ik = self._int_kind_selector()
                    a = f"INT({self.emit_expr(node.left)}, {ik})"
                    b = f"INT({self.emit_expr(node.right)}, {ik})"
                    return (f"({a} / {b} - MERGE(1_{ik}, 0_{ik}, MOD({a}, {b}) /= 0_{ik} "
                            f".AND. ({a} < 0_{ik}) .NEQV. ({b} < 0_{ik})))")
                # Non-integer (float) operands: ``REAL(x)`` with no KIND is DEFAULT
                # single precision, which drops mantissa bits; force the divide into
                # DOUBLE (kind from the registry) so the quotient is at full
                # precision before the floor.
                dk = _double_kind()
                return (f"FLOOR(REAL({self.emit_expr(node.left)}, {dk}) / "
                        f"REAL({self.emit_expr(node.right)}, {dk}), {self._int_kind_selector()})")
            # Bitwise ops: Fortran uses IAND / IOR / IEOR / NOT for
            # integer bit ops. Both args must share a kind, so when
            # one side resolves to a known typed Name (int64 / int32
            # / etc.) and the other is an integer literal, promote
            # the literal with the matching kind suffix.
            # ``&`` / ``|`` on LOGICAL operands (numpy uses them for elementwise
            # boolean AND/OR, e.g. force_lj's ``(rsq < c) & (rsq > 0)``) must be
            # Fortran ``.AND.`` / ``.OR.`` -- IAND/IOR are INTEGER bit ops and
            # reject a logical operand.
            if isinstance(node.op, ast.BitAnd):
                if self._is_logical_operand(node.left) and self._is_logical_operand(node.right):
                    return f"({self.emit_expr(node.left)} .AND. {self.emit_expr(node.right)})"
                left, right = self._emit_bitwise_pair(node.left, node.right)
                return f"IAND({left}, {right})"
            if isinstance(node.op, ast.BitOr):
                if self._is_logical_operand(node.left) and self._is_logical_operand(node.right):
                    return f"({self.emit_expr(node.left)} .OR. {self.emit_expr(node.right)})"
                left, right = self._emit_bitwise_pair(node.left, node.right)
                return f"IOR({left}, {right})"
            if isinstance(node.op, ast.BitXor):
                # ``^`` on LOGICAL masks is elementwise XOR -> Fortran ``.neqv.``
                # (IEOR is an INTEGER bit op and rejects a logical operand), mirroring
                # the ``&`` / ``|`` -> ``.AND.`` / ``.OR.`` handling above.
                if self._is_logical_operand(node.left) and self._is_logical_operand(node.right):
                    return f"({self.emit_expr(node.left)} .neqv. {self.emit_expr(node.right)})"
                left, right = self._emit_bitwise_pair(node.left, node.right)
                return f"IEOR({left}, {right})"
            if isinstance(node.op, ast.LShift):
                left, right = self._emit_bitwise_pair(node.left, node.right)
                return f"ISHFT({left}, {right})"
            if isinstance(node.op, ast.RShift):
                # numpy ``>>`` on a SIGNED integer is ARITHMETIC (sign-preserving);
                # ISHFT(x, -n) is a LOGICAL (zero-fill) shift. SHIFTA(x, n) shifts
                # right while replicating the sign bit -- the numpy semantics.
                left, right = self._emit_bitwise_pair(node.left, node.right)
                return f"SHIFTA({left}, {right})"
            # MatMult: ``A @ B`` should have been hoisted by an upstream
            # pass into an explicit matmul loop; if it appears here the
            # lowering missed it.
            if isinstance(node.op, ast.MatMult):
                # When BOTH operands are fully-indexed Subscripts
                # (scalar values), ``A @ B`` is just scalar*scalar.
                # MATMUL would reject rank-0 args.
                def _is_scalar_access(n):
                    if not isinstance(n, ast.Subscript):
                        return False
                    sl = n.slice
                    elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
                    # All slot elements must be non-Slice (concrete int
                    # or Name) -- if any is a Slice the result is rank>=1.
                    return all(not isinstance(e, ast.Slice) for e in elts)

                if _is_scalar_access(node.left) and _is_scalar_access(node.right):
                    return (f"({self.emit_expr(node.left)} * "
                            f"{self.emit_expr(node.right)})")
                # Two rank-1 vectors -> DOT_PRODUCT (returns scalar).
                # Detect bare 1-D Names AND
                # ``arr[gather]`` (Subscript whose slot is a Name to
                # an int array, returning a rank-1 gather result).
                def _shape_of_name(name):
                    for a in self.kir.arrays:
                        if a.name == name:
                            return a.shape
                    return None

                # Detect locals (allocatable / zeros_locals) and treat
                # known 1-D ones as rank-1.
                zl = self.kir.zeros_locals

                def _shape_of_local(name):
                    s = zl.get(name)
                    return s if s else None

                def _is_1d_operand(n):
                    if isinstance(n, ast.Name):
                        s = _shape_of_name(n.id) or _shape_of_local(n.id)
                        return s is not None and len(s) == 1
                    if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
                        s = _shape_of_name(n.value.id) or _shape_of_local(n.value.id)
                        if s is None:
                            return False
                        sl = n.slice
                        # Single-slot ``arr[expr]`` where expr is NOT a
                        # Tuple and the base array is rank-1: scalar.
                        # ``arr[gather_array]`` where gather is rank-1
                        # int array: fancy-index gather returning rank-1.
                        # ``arr[lo:hi]``: rank-1 slice.
                        if isinstance(sl, ast.Slice) and len(s) == 1:
                            return True
                        # Walk slice subtree -- if it references a 1-D
                        # int array (and the base array is rank-1),
                        # gather result is rank-1.
                        if len(s) == 1:
                            for sub in ast.walk(sl):
                                if isinstance(sub, ast.Name):
                                    slot_shape = (_shape_of_name(sub.id) or _shape_of_local(sub.id))
                                    if (slot_shape is not None and len(slot_shape) == 1):
                                        return True
                    return False

                if _is_1d_operand(node.left) and _is_1d_operand(node.right):
                    le, re = self.emit_expr(node.left), self.emit_expr(node.right)
                    # DOT_PRODUCT CONJUGATES its first arg for COMPLEX operands;
                    # numpy dot / matmul do NOT conjugate. For a complex operand
                    # emit SUM(a*b) (no conjugation); real operands keep the
                    # (BLAS-backed) DOT_PRODUCT.
                    if self._operand_is_complex(node.left) or self._operand_is_complex(node.right):
                        return f"SUM(({le}) * ({re}))"
                    return f"DOT_PRODUCT({le}, {re})"
                # Bare-Name on each side AND the name isn't a declared
                # kernel array / zeros_local -- treat as scalars.
                def _is_scalar_name(n):
                    if not isinstance(n, ast.Name):
                        return False
                    for a in self.kir.arrays:
                        if a.name == n.id:
                            return False
                    zl = self.kir.zeros_locals
                    return n.id not in zl

                if (_is_scalar_name(node.left) and _is_scalar_name(node.right)):
                    return (f"({self.emit_expr(node.left)} * "
                            f"{self.emit_expr(node.right)})")
                # Arrays are declared with REVERSED dims (col-major over the
                # row-major C buffers), so each stored array is the transpose of
                # its numpy view. numpy ``C = A @ B`` -> stored ``C^T = B^T @ A^T``
                # = ``MATMUL(B_stored, A_stored)``: emit the operands SWAPPED.
                # (Reached only for unresolved-shape matmuls the loop hoister
                # could not lower; the swap keeps it numerically correct.)
                return (f"MATMUL({self.emit_expr(node.right)}, "
                        f"{self.emit_expr(node.left)})")
            op = _BINOP.get(type(node.op))
            if op is None:
                raise NotImplementedError(f"binop {type(node.op).__name__}")
            if op == "MOD":
                # Python ``%`` (and numpy) take sign of divisor; Fortran's
                # ``MOD`` takes sign of dividend, ``MODULO`` takes sign of
                # divisor -- match Python by using MODULO. MODULO is an INTRINSIC
                # and (unlike +/-/*) requires its two args be the SAME kind, so
                # kind-match them (a literal divisor ``MODULO(i64expr, 16)`` gets
                # the int64 suffix) -- same mechanism as min/max + bitwise.
                left, right = self._emit_bitwise_pair(node.left, node.right)
                return f"MODULO({left}, {right})"
            return f"({self.emit_expr(node.left)} {op} {self.emit_expr(node.right)})"
        if isinstance(node, ast.BoolOp):
            op = _BOOLOP[type(node.op)]
            # ``and`` / ``or`` are LOGICAL operators in Fortran: an int-flag
            # operand (cloudsc ``ldcum[jl] and plude > x``) must be compared to
            # zero so the operand is logical, not the bare integer C truthiness.
            parts = [self._as_logical_operand(v) for v in node.values]
            return "(" + f" {op} ".join(parts) + ")"
        if isinstance(node, ast.Compare):
            # ``<LOGICAL> != 0`` / ``<LOGICAL> == 0`` is a truthiness test on a
            # boolean operand (from the if-guarded any/all/count_nonzero
            # reductions when applied to a mask). Fortran has no LOGICAL-vs-0
            # comparison, so emit the logical directly / negated.
            if len(node.ops) == 1 and isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
                for a, b in ((node.left, node.comparators[0]), (node.comparators[0], node.left)):
                    if self._is_logical_node(a) and isinstance(b, ast.Constant) \
                            and b.value == 0 and not isinstance(b.value, bool):
                        le = self.emit_expr(a)
                        return le if isinstance(node.ops[0], ast.NotEq) else f".not. ({le})"
            # Python chained comparison ``a < b < c`` == ``(a<b) and (b<c)``;
            # Fortran has no chaining, so emit an explicit ``.and.`` join.
            operands = [self.emit_expr(node.left)] + [self.emit_expr(c) for c in node.comparators]
            terms = [f"({operands[i]} {_CMPOP[type(op)]} {operands[i + 1]})" for i, op in enumerate(node.ops)]
            return (terms[0] if len(terms) == 1 else "(" + " .and. ".join(terms) + ")")
        if isinstance(node, ast.Subscript):
            return self._emit_subscript(node)
        if isinstance(node, ast.Call):
            return self._emit_call(node)
        if isinstance(node, ast.IfExp):
            # merge() is strict on TYPE *and* KIND: an integer-literal branch
            # defaults to int32 while a paired int64 symbol/var is int64
            # (``ending = m if negrp == 1 else 0``). Suffix a literal branch with
            # its integer partner's kind so the two branches kind-match.
            return (f"merge({self._emit_merge_branch(node.body, node.orelse)}, "
                    f"{self._emit_merge_branch(node.orelse, node.body)}, "
                    f"{self.emit_expr(node.test)})")
        # A bare ``z.real`` / ``z.imag`` Attribute never reaches emit: ``native_desugar``
        # rewrites the accessor to ``np.real(z)`` / ``np.imag(z)`` at parse time, and the
        # ``real(z, kind)`` / ``aimag(z)`` lowering lives on that canonical call form.
        raise NotImplementedError(f"expression {type(node).__name__} (line {getattr(node, 'lineno', '?')})")

    def _emit_merge_branch(self, branch: ast.AST, partner: ast.AST) -> str:
        """Emit one ``merge`` branch, suffixing an integer literal with its
        integer partner's KIND (so ``merge(m, 0, ...)`` becomes
        ``merge(m, 0_c_int64_t, ...)`` and gfortran's same-kind rule holds)."""
        litval = _int_literal_value(branch)
        if litval is not None:
            ktag = None
            if isinstance(partner, ast.Name):
                ktag = self._name_int_kind(partner.id)
            elif _int_literal_value(partner) is not None:
                ktag = "int64"
            if ktag:
                lit = f"{litval}_{self._int_kind_selector(ktag)}"
                return f"({lit})" if litval < 0 else lit
            # The partner branch is REAL: merge() is strict on TYPE, so an integer
            # literal beside a real branch must be emitted as a real of the kernel
            # float kind -- ``np.where(cond, 0, real_arr)`` -> ``merge(0.0_c_double,
            # real_arr, ..)`` (hdiff). C's ternary promotes silently; Fortran does not.
            if not self._expr_is_integer(partner):
                lit = f"{litval}.0_{self._rk}"
                return f"({lit})" if litval < 0 else lit
        # The same TYPE strictness for an integer branch the literal path cannot spell -- a
        # name / subscript / arithmetic. ``np.where(cdf > 0, cdf, npix)``
        # (histogram_equalization) would emit ``merge(cdf(..), npix, ..)``, which gfortran
        # rejects ("'fsource' argument of 'merge' must be the same type and kind as
        # 'tsource'"). numpy promotes the integer operand to the real result dtype, so
        # convert it the same way -- ``real(npix, rk)``.
        #
        # The partner must be PROVABLY real, not merely "not provably integer": the latter
        # also covers un-inferable expressions, and promoting against one breaks a pair that
        # is really int/int (bfs's ``np.where(nxt, d + 1, level)``, where the plain int local
        # ``d`` carries no registry dtype).
        if self._expr_is_integer(branch) and self._expr_is_real(partner):
            return f"real({self.emit_expr(branch)}, {self._rk})"
        # merge() is equally strict on COMPLEX. numpy's ``x.real if c else x`` yields a real (or
        # int) branch beside a complex partner -- QE vexx_k's
        # ``deexx[ikb].real if gamma_only else deexx[ikb]``. C's ternary promotes real->complex
        # silently; merge does not. Promote the non-complex branch to complex of the kernel kind,
        # mirroring the int->real promotion above (cmplx accepts a real OR integer first arg).
        if not self._operand_is_complex(branch) and self._operand_is_complex(partner):
            return f"cmplx({self.emit_expr(branch)}, 0.0_{self._rk}, {self._rk})"
        return self.emit_expr(branch)

    def _expr_is_real(self, e: ast.AST) -> bool:
        """True only when ``e`` is PROVABLY a real-typed Fortran expression.

        Deliberately NOT the complement of :meth:`_expr_is_integer`: that predicate is
        conservative, so ``not _expr_is_integer(x)`` means "real OR un-inferable" and acting
        on it promotes integer pairs by mistake. This answers the positive question and stays
        False when unsure."""
        if isinstance(e, ast.Constant):
            return isinstance(e.value, float)
        if isinstance(e, (ast.Name, ast.Subscript)):
            base = e
            while isinstance(base, ast.Subscript):
                base = base.value
            if not isinstance(base, ast.Name) or self._name_int_kind(base.id) is not None:
                return False
            for decl in (*self.kir.arrays, *self.kir.scalars):
                if decl.name == base.id:
                    # Real iff the registry says the dtype is neither integer nor logical.
                    return self._int_tag(decl.dtype) is None and decl.dtype != "bool"
            # Fresh local arrays (np.cumsum / np.where results) carry their resolved
            # element dtype in the emit-time local-dtype map, not in kir.arrays.
            dt = getattr(self, "_local_elem_dtypes", {}).get(base.id)
            if dt is not None:
                return self._int_tag(dt) is None and dt not in ("bool", "bool_")
            return False
        if isinstance(e, ast.UnaryOp):
            return self._expr_is_real(e.operand)
        if isinstance(e, ast.BinOp):
            return self._expr_is_real(e.left) or self._expr_is_real(e.right)
        return False

    def _as_real_arg(self, node: ast.AST) -> str:
        """Emit ``node`` as an argument to a REAL-only intrinsic.

        An INTEGER expression is wrapped in ``real(.., kind)`` -- numpy promotes it
        to float for these functions, and Fortran rejects it outright (see
        :data:`_REAL_ARG_INTRINSICS`). Anything not PROVABLY integer is emitted as
        is, so a real / complex operand is never needlessly retyped."""
        text = self.emit_expr(node)
        return f"real({text}, {self._rk})" if self._expr_is_integer(node) else text

    def _expr_is_integer(self, e: ast.AST) -> bool:
        """True if ``e`` is an integer-typed Fortran expression (so a merge()
        partner literal should be int-kinded, not real-promoted)."""
        if isinstance(e, ast.Constant):
            return isinstance(e.value, int) and not isinstance(e.value, bool)
        if isinstance(e, ast.Name):
            return self._name_int_kind(e.id) is not None
        if isinstance(e, ast.Subscript):
            base = e.value
            return isinstance(base, ast.Name) and self._name_int_kind(base.id) is not None
        if isinstance(e, ast.UnaryOp):
            return self._expr_is_integer(e.operand)
        if isinstance(e, ast.BinOp) and not isinstance(e.op, ast.Div):
            return self._expr_is_integer(e.left) and self._expr_is_integer(e.right)
        if isinstance(e, ast.Call):
            # An int-returning call is INTEGER -- the same rule the min/max operand typing
            # uses. Without it an ``int(..)`` cast read as non-integer, so a merge() partner
            # that IS an integer got real-promoted beside it and gfortran rejected the
            # mismatched pair (bfs: ``np.where(nxt, d + 1, level)``).
            fn = (e.func.id if isinstance(e.func, ast.Name) else (e.func.attr if isinstance(e.func, ast.Attribute)
                                                                  else ""))
            if fn in _INT_RETURNING_CALLS:
                if fn in _INT_CALLS_ARGDEP:
                    return all(self._expr_is_integer(a) for a in e.args)
                return True
        return False

    def _emit_subscript(self, node: ast.Subscript) -> str:
        # Boolean-mask indexing ``arr[mask]`` -> Fortran ``PACK(arr,
        # mask)`` -- returns elements where mask is True. Detect by
        # looking at the slice slot for a Name that resolves to a
        # known-logical local (see ``logical_array_locals``).
        if (isinstance(node.value, ast.Name) and isinstance(node.slice, ast.Name)):
            logical_locals = getattr(self, "_logical_array_locals", set())
            if node.slice.id in logical_locals:
                return f"PACK({node.value.id}, {node.slice.id})"
        # Tuple subscripted by a constant integer: resolve at emit
        # time. Comes up when ``D.shape[-2]`` lowers to
        # ``(Nqz, Nw, NA, NB, N3D, N3D)[-2]`` -- the Tuple is the
        # constant-folded shape and the index picks one element.
        if isinstance(node.value, ast.Tuple):
            elts = node.value.elts
            idx_node = node.slice
            if (isinstance(idx_node, ast.Constant) and isinstance(idx_node.value, int)):
                idx = idx_node.value
                if idx < 0:
                    idx += len(elts)
                if 0 <= idx < len(elts):
                    return self.emit_expr(elts[idx])
            if (isinstance(idx_node, ast.UnaryOp) and isinstance(idx_node.op, ast.USub)
                    and isinstance(idx_node.operand, ast.Constant) and isinstance(idx_node.operand.value, int)):
                idx = -idx_node.operand.value + len(elts)
                if 0 <= idx < len(elts):
                    return self.emit_expr(elts[idx])
        # Chained Subscripts: ``arr[a][b, c]`` is numpy slice-then-index
        # which is semantically ``arr[a, b, c]``. Walk inward, gathering
        # all axes from inner-most to outer-most, so the Fortran emit
        # produces ``arr(a+1, b+1, c+1)`` instead of the broken
        # ``arr(a+1)(b+1, c+1)``.
        inner = node.value
        prefix_axes: List[ast.expr] = []
        while isinstance(inner, ast.Subscript) and isinstance(inner.value, ast.Name):
            inner_sl = inner.slice
            inner_elts = (list(inner_sl.elts) if isinstance(inner_sl, ast.Tuple) else [inner_sl])
            prefix_axes = inner_elts + prefix_axes
            inner = inner.value
        if prefix_axes and isinstance(inner, ast.Name):
            base = inner.id
            sl = node.slice
            outer_elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
            raw_elts = prefix_axes + list(outer_elts)
            base_name = inner.id
            rank = len(raw_elts)
        else:
            # The base of a subscript is the array itself (indices added below), so use the RAW name -- NOT
            # emit_expr, which scalarises a size-1 array Name to ``x(1)`` and would double-index to ``x(1)(i)``.
            base = node.value.id if isinstance(node.value, ast.Name) else self.emit_expr(node.value)
            sl = node.slice
            raw_elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
            # Resolve the base array name so SIZE(arr, dim) works for
            # negative indices like ``arr[-1]``.
            base_name = node.value.id if isinstance(node.value, ast.Name) else base
            rank = len(raw_elts)
        adjusted: List[str] = []
        for axis, e in enumerate(raw_elts):
            # Negative integer constant: arr[-K] -> SIZE - K + 1.
            # After dim reversal axis k in Python maps to Fortran dim
            # (rank - axis). SIZE(arr, dim) returns the matching size.
            f_dim = rank - axis
            if isinstance(e, ast.Constant) and isinstance(e.value, int) and e.value < 0:
                adjusted.append(f"SIZE({base_name}, {f_dim}) + ({e.value}) + 1")
                continue
            if (isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub) and isinstance(e.operand, ast.Constant)
                    and isinstance(e.operand.value, int)):
                adjusted.append(f"SIZE({base_name}, {f_dim}) - {e.operand.value} + 1")
                continue
            # Slice axis: ``arr[lo:hi:step]`` -> Fortran ``lo+1:hi:step``.
            # Python lo/hi are half-open and 0-based; Fortran is 1-based
            # and inclusive on the upper bound, so:
            #   Python lo == None -> Fortran lo = 1 (start)
            #   Python lo == K    -> Fortran lo = K + 1
            #   Python hi == None -> Fortran hi = SIZE(arr, f_dim)
            #   Python hi == K    -> Fortran hi = K (unchanged, since
            #                        the Python half-open count equals
            #                        the Fortran inclusive count from
            #                        lo+1 to K).
            #   Python hi == -K   -> Fortran hi = SIZE - K
            if isinstance(e, ast.Slice):
                dim = f"SIZE({base_name}, {f_dim})"
                step_val = _int_literal_value(e.step) if e.step is not None else None
                # -- Negative-step RAW slice ``a[hi:lo:-1]`` / ``a[::-1]`` -------
                # numpy walks HIGH -> LOW; emit a genuine reversed Fortran section
                # ``start:end:step`` rather than the (empty) forward-direction
                # bounds. start defaults to the LAST element, end to the FIRST;
                # Python's half-open upper EXCLUDES its index going down (last
                # included is upper+1, 0-based -> upper+2 1-based).
                if step_val is not None and step_val < 0:
                    if e.lower is None:
                        lo = dim
                    else:
                        lv = _int_literal_value(e.lower)
                        lo = (f"({self.emit_expr(e.lower)}) + 1" if lv is None else
                              (str(lv + 1) if lv >= 0 else f"{dim} + ({lv}) + 1"))
                    if e.upper is None:
                        hi = "1"
                    else:
                        uv = _int_literal_value(e.upper)
                        hi = (f"({self.emit_expr(e.upper)}) + 2" if uv is None else
                              (str(uv + 2) if uv >= 0 else f"{dim} + ({uv}) + 2"))
                    adjusted.append(f"{lo}:{hi}:{self.emit_expr(e.step)}")
                    continue
                # -- Forward slice ``a[lo:hi:step]`` ----------------------------
                # Python lo/hi are half-open, 0-based; Fortran is 1-based and
                # inclusive. A negative constant LOWER wraps (dim + K + 1); a
                # positive constant UPPER is CLAMPED to the extent (numpy clamps
                # ``a[:100]`` on len-10 to 10); a negative UPPER wraps (dim + K).
                if e.lower is None:
                    lo = "1"
                else:
                    lv = _int_literal_value(e.lower)
                    lo = (f"({self.emit_expr(e.lower)}) + 1" if lv is None else
                          (str(lv + 1) if lv >= 0 else f"{dim} + ({lv}) + 1"))
                if e.upper is None:
                    hi = dim
                else:
                    uv = _int_literal_value(e.upper)
                    if uv is None:
                        hi = self.emit_expr(e.upper)
                    elif uv < 0:
                        hi = f"{dim} + ({uv})"
                    else:
                        hi = f"min({uv}, {dim})"
                if e.step is None:
                    adjusted.append(f"{lo}:{hi}")
                else:
                    adjusted.append(f"{lo}:{hi}:{self.emit_expr(e.step)}")
                continue
            adjusted.append(f"({self.emit_expr(e)}) + 1")
        # Reverse the index order so Python row-major
        # ``arr[i, j, k]`` accesses the same memory location as
        # Fortran col-major ``arr(k+1, j+1, i+1)`` against a reversed-
        # shape declaration ``arr(K, M, N)``.
        adjusted.reverse()
        access = base + "(" + ", ".join(adjusted) + ")"
        # Promote a NARROW integer array element to the int64 ABI integer on a
        # scalar READ (``INT(arr(i), c_int64_t)``) so a user-supplied int32 array
        # never forms a mixed-kind op with an int64 symbol/local (Fortran rejects
        # mixed integer kinds in operators + intrinsics). Storage keeps its width;
        # a Store (LHS) or an array-section read is left untouched.
        is_section = any(":" in a for a in adjusted)
        if is_section or not isinstance(node.ctx, ast.Load):
            return access
        # An fp8 element is a 1-byte code with no arithmetic of its own -- promote
        # it to real(c_float) on every scalar READ (the store seam demotes back).
        fns = _fp8_fns(self._name_dtype(base_name) or "")
        if fns is not None:
            return f"{fns.promote}({access})"
        # An UNSIGNED narrow int (uint8/16/32) reads back negative for a high value
        # under Fortran's signed storage -- mask to [0, 2**N) after the int64
        # promotion so the recovered value matches numpy's unsigned element.
        umask = self._unsigned_read_mask(base_name)
        sel = self._int_kind_selector()
        if umask is not None:
            return f"iand(INT({access}, {sel}), {umask}_{sel})"
        if self._is_narrow_int_array(base_name):
            return f"INT({access}, {sel})"
        return access

    def _operand_is_complex(self, node: ast.AST) -> bool:
        """True when ``node`` produces a COMPLEX value (so a rank-1 ``@`` must
        avoid DOT_PRODUCT's implicit conjugation). Resolves each Name's element
        dtype through the array + local-dtype registry, reusing the shared
        ``_walk_complex`` predicate."""
        from numpyto_common.lowering import _walk_complex
        arr_dt = {a.name: a.dtype for a in self.kir.arrays}
        arr_dt.update(self.kir.local_dtypes)
        return _walk_complex(node, arr_dt.get) is not None

    def _is_narrow_int_array(self, name: str) -> bool:
        """True when ``name`` is an integer array narrower than the int64 ABI
        integer (int8/16/32) -- its elements are promoted to int64 on read."""
        for a in self.kir.arrays:
            if a.name == name:
                return (self._int_tag(a.dtype) is not None and _fortran_type(a.dtype) != _fortran_type("int64"))
        return False

    def _unsigned_read_mask(self, name: str) -> Optional[str]:
        """The ``2**N - 1`` mask (decimal string) that recovers a uintN element's
        UNSIGNED value from Fortran's signed-integer storage, or None when the
        Name is not an unsigned narrow int. Fortran has no unsigned integer, so a
        ``uint8`` stores in ``integer(c_int8_t)``: a byte >= 128 reads back
        NEGATIVE and would index / compute wrong (histogram_equalization's
        ``lut[img]`` ran off the front of the array). Resolved through
        ``_name_dtype`` so a uint LOCAL (``flat = img.reshape(..)``) is covered
        too, not only declared array parameters. uint64 already spans int64, so it
        needs no mask."""
        dt = self._name_dtype(name)
        bits = {"uint8": 8, "uint16": 16, "uint32": 32}.get(dt or "")
        return None if bits is None else str((1 << bits) - 1)

    def _emit_sign(self, x: str) -> str:
        """numpy ``sign``: -1 / 0 / +1, and sign(NaN) == NaN. A MERGE built from
        ``x>0`` / ``x<0`` gives 0 at NaN (both comparisons false), so guard it: a
        NaN operand (``x /= x``) returns x itself (propagating the NaN)."""
        rk = self._rk
        core = (f"(merge(1.0_{rk}, 0.0_{rk}, ({x}) > 0) - "
                f"merge(1.0_{rk}, 0.0_{rk}, ({x}) < 0))")
        return f"merge({x}, {core}, ({x}) /= ({x}))"

    def _emit_call(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            fn = node.func.id
            if fn == "__optarena_zeros__":
                return ""
            # min / max require Fortran-typed-uniform args. When one
            # operand is an integer literal AND another operand is
            # a real-typed expression (e.g. relu: ``max(x, 0)``),
            # promote the int literal to real ``0.0d0``. The
            # detection uses the same int-uses analysis the implicit-
            # locals path runs.
            # ``fmax`` / ``fmin`` (relu's ``np.maximum(x, 0)`` lowers to fmax)
            # rename to the Fortran MAX / MIN intrinsic, but must STILL go
            # through arg promotion -- otherwise the int ``0`` literal stays
            # integer beside the real ``x`` and gfortran rejects the mixed-type
            # MAX. Handled here (before the FORTRAN_INTRINSICS rename at line
            # ~757, which would emit MAX without promoting).
            if fn in {"max", "min", "fmax", "fmin"}:
                all_int, arg_strs = self._minmax_arg_list(node.args)
                is_max = fn in {"max", "fmax"}
                # numpy maximum / minimum PROPAGATE NaN; Fortran MAX / MIN NaN
                # behaviour is processor-dependent. For FLOATING operands emit the
                # NaN-propagating MERGE form. Pure-integer min/max (index clamps
                # like ``min(npt-1, max(0, int(..)))``) have no NaN and keep the
                # plain intrinsic.
                if not all_int and len(arg_strs) >= 2:
                    return self._nan_minmax(is_max, arg_strs)
                out_name = "max" if is_max else "min"
                return f"{out_name}({', '.join(arg_strs)})"
            # ``pow(a, b)`` -> infix ``(a ** b)``. Fortran's ``**`` is an
            # operator, not a function, so a literal substitution would
            # emit broken ``**(a, b)``.
            if fn == "pow" and len(node.args) == 2:
                return (f"({self.emit_expr(node.args[0])} ** "
                        f"{self.emit_expr(node.args[1])})")
            # ``np.sign`` marker -> -1 / 0 / +1. Fortran has no logical
            # arithmetic, so build it from MERGE (SIGN(1,x) would give
            # +1 at x==0, not numpy's 0). The leading-underscore sanitiser
            # rewrites the marker to ``x_npb_sign`` before we see it.
            if fn in ("__npb_sign", "x_npb_sign") and len(node.args) == 1:
                return self._emit_sign(self.emit_expr(node.args[0]))
            # ``round`` / ``rint`` -> ANINT is half-AWAY; numpy round AND rint are
            # half-to-EVEN (exact .5 ties go to the even neighbour). Emit the
            # half-even form: start from ANINT and, on an exact tie whose ANINT is
            # ODD, step back toward the even value by SIGN(1, x).
            if fn in ("round", "rint") and len(node.args) == 1:
                # Call the CONTAINED half-even helper so the argument is rendered
                # ONCE. Inlining the anint/merge/sign form here repeats the argument
                # six times, and a round of a large sub-expression then blows the
                # -O2 compile past its timeout (see ``_used_round_even``).
                self._used_round_even = True
                return f"npb_round_even({self.emit_expr(node.args[0])})"
            # numpy floor / ceil return a FLOAT and never overflow on +/-inf or
            # |x| >= 2^63; the integer FLOOR / CEILING intrinsics retype + overflow.
            # AINT truncates toward zero (float result, inf-safe), then adjust by one
            # in the correct direction when a fractional part remains.
            if fn in ("floor", "ceil") and len(node.args) == 1:
                x = self.emit_expr(node.args[0])
                rk = self._rk
                t = f"aint({x})"
                if fn == "floor":
                    return f"({t} - merge(1.0_{rk}, 0.0_{rk}, ({x}) < {t}))"
                return f"({t} + merge(1.0_{rk}, 0.0_{rk}, ({x}) > {t}))"
            # libm unary funcs Fortran lacks an intrinsic for (cbrt / exp2 / log2
            # / expm1 / log1p). Emit a bind(C) call to the SAME libm the C backend
            # and numpy use, so the result is bit-identical -- not an expression
            # approximation. ``x**(1/3)`` differs from libm cbrt in ~57% of inputs;
            # ``exp(x)-1`` / ``log(1+x)`` lose all precision for small x where
            # expm1/log1p exist precisely. gfortran's own intrinsics (sin/cos/exp/
            # ...) already resolve to libm bit-for-bit, so only these five route here.
            if fn in _FORTRAN_FN_EXPR and len(node.args) == 1:
                libm = fn if self._rk == "c_double" else fn + "f"
                self._used_libm.add((libm, self._rk))
                return f"{libm}({self.emit_expr(node.args[0])})"
            # Integer-returning CONVERSIONS (int / floor / ceil -> INT / FLOOR /
            # CEILING) default to the int32 KIND in Fortran; pin them to the int64
            # ABI kind so the result does not clash with int64 operands (azimint
            # ``min(npt - 1, max(0, int(...)))``). The KIND token is derived from
            # the dtype registry, not hardcoded.
            if fn in _INT_CONV_INTRINSIC and len(node.args) == 1:
                a = self.emit_expr(node.args[0])
                return f"{_INT_CONV_INTRINSIC[fn]}({a}, {self._int_kind_selector()})"
            up = FORTRAN_INTRINSICS.get(fn)
            if up is not None:
                if fn in _REAL_ARG_INTRINSICS:
                    args = ", ".join(self._as_real_arg(a) for a in node.args)
                else:
                    args = ", ".join(self.emit_expr(a) for a in node.args)
                return f"{up}({args})"
            args = ", ".join(self.emit_expr(a) for a in node.args)
            return f"{fn}({args})"
        # ``np.X(args)`` / ``arr.X(args)`` -- map common numpy calls to
        # Fortran intrinsics so kernels whose lowering didn''t expand
        # the call (e.g. nbody's ``x.T``) still produce valid code.
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            args_e = [self.emit_expr(a) for a in node.args]
            # ``np.<dtype>(x)`` scalar constructor (``np.int64(0)`` /
            # ``np.float64(s)`` / ``np.bool_(b)`` ...) is a TYPECAST: the
            # matching Fortran conversion intrinsic (INT/REAL/CMPLX/LOGICAL)
            # with the dtype's KIND token, both resolved through the dtype
            # registry so no kind string is hardcoded. ``np.bool_`` carries a
            # trailing underscore; strip it before the lookup.
            if (isinstance(node.func.value, ast.Name) and node.func.value.id == "np" and len(node.args) == 1):
                key = attr[:-1] if attr.endswith("_") else attr
                if key in dtypes.REGISTRY or key in dtypes.SCALAR_KINDS:
                    base, _, rest = _fortran_type(key).partition("(")
                    kind = rest.rstrip(")")
                    intrinsic = {"integer": "INT", "real": "REAL", "complex": "CMPLX", "logical": "LOGICAL"}[base]
                    # A numeric cast of a LOGICAL-valued operand
                    # (``(level == d).astype(np.int64)`` -> ``np.int64(level==d)``,
                    # bfs) is invalid in Fortran: ``INT(logical)`` is rejected.
                    # numpy maps True/False to 1/0, so emit a MERGE in the target
                    # kind instead.
                    if intrinsic in ("INT", "REAL") and _produces_logical(node.args[0]):
                        one = f"1_{kind}" if intrinsic == "INT" else f"1.0_{kind}"
                        zero = f"0_{kind}" if intrinsic == "INT" else f"0.0_{kind}"
                        return f"merge({one}, {zero}, {args_e[0]})"
                    if intrinsic == "CMPLX":
                        return f"CMPLX({args_e[0]}, kind={kind})"
                    return f"{intrinsic}({args_e[0]}, {kind})"
            if attr == "transpose" and args_e:
                # Only apply when the operand is a bare Name -- on a
                # subscripted operand TRANSPOSE produces nonsense.
                if node.args and isinstance(node.args[0], ast.Name):
                    return f"TRANSPOSE({args_e[0]})"
            if attr == "mean" and args_e:
                return f"(SUM({args_e[0]}) / SIZE({args_e[0]}))"
            if attr == "sum" and args_e:
                return f"SUM({args_e[0]})"
            if attr == "prod" and args_e:
                return f"PRODUCT({args_e[0]})"
            if attr == "max" and args_e:
                return f"MAXVAL({args_e[0]})"
            if attr == "min" and args_e:
                return f"MINVAL({args_e[0]})"
            if attr == "argmax" and args_e:
                return f"MAXLOC({args_e[0]}, 1)"
            if attr == "argmin" and args_e:
                return f"MINLOC({args_e[0]}, 1)"
            if attr == "abs" and args_e:
                return f"ABS({args_e[0]})"
            if attr == "copy" and args_e:
                return args_e[0]
            # Standard Fortran array intrinsics for boolean reductions
            # over the whole array (elemental forms are also valid):
            #   np.count_nonzero(arr) -> COUNT(arr /= 0) or COUNT(arr)
            #     for a logical operand.
            #   np.any(arr) -> ANY(arr) for logical / ANY(arr /= 0).
            #   np.all(arr) -> ALL(arr) for logical / ALL(arr /= 0).
            if attr == "count_nonzero" and args_e:
                return f"COUNT({args_e[0]} /= 0)"
            if attr == "any" and args_e:
                return f"ANY({args_e[0]})"
            if attr == "all" and args_e:
                return f"ALL({args_e[0]})"
            # np.cumsum -- Fortran has no direct intrinsic; emit via
            # SUM scalar accumulator inside a Subscript context.
            # Whole-array form needs a loop.
            if attr == "fabs" and args_e:
                return f"ABS({args_e[0]})"
            # np.maximum / np.minimum -- elementwise max/min between
            # two operands. Fortran's MAX / MIN are elemental on
            # arrays, so this is direct.
            # numpy maximum / minimum PROPAGATE NaN (Fortran MAX / MIN NaN is
            # processor-dependent), so emit the NaN-propagating MERGE fold.
            if attr == "maximum" and len(args_e) >= 2:
                return self._nan_minmax(True, args_e)
            if attr == "minimum" and len(args_e) >= 2:
                return self._nan_minmax(False, args_e)
            # np.logical_not -- elemental on logicals.
            if attr == "logical_not" and args_e:
                return f"(.NOT. {args_e[0]})"
            if attr == "logical_and" and len(args_e) >= 2:
                return f"({args_e[0]} .AND. {args_e[1]})"
            if attr == "logical_or" and len(args_e) >= 2:
                return f"({args_e[0]} .OR. {args_e[1]})"
            # np.power(a, b) elemental ``a ** b``.
            if attr == "power" and len(args_e) >= 2:
                return f"({args_e[0]} ** {args_e[1]})"
            # np.true_divide / np.divide elemental ``a / b``.
            if attr == "true_divide" and len(args_e) >= 2:
                return f"({args_e[0]} / {args_e[1]})"
            # np.flip(arr) / arr.flip() -- reverse via strided slice.
            # Only emit the SIZE strided form when the operand is a bare
            # Name (so the result is a valid Fortran array section). On
            # a Subscript operand (typically ``r[i]`` -- a scalar -- as
            # in durbin where np.flip was lifted per-element), flip is
            # a no-op on a scalar; emit the operand directly.
            if attr == "flip" and args_e:
                if isinstance(node.args[0], ast.Name):
                    a = args_e[0]
                    return f"{a}(SIZE({a}):1:-1)"
                # Subscript / other operand: treat as identity. For
                # element-wise expansion this is correct (np.flip on a
                # scalar = scalar); for whole-array Subscript callers
                # this leaks but the immediate emit error is gone.
                return args_e[0]
            # np.triu(arr) -- Fortran has no direct intrinsic; emit via
            # MERGE so the upper-triangular elements pass and the rest
            # become zero. Only when operand is a Name.
            if attr == "triu" and args_e:
                if isinstance(node.args[0], ast.Name):
                    a = args_e[0]
                    return (f"MERGE({a}, 0.0_{self._rk}, "
                            f"SPREAD([(I, I=0, SIZE({a}, 2)-1)], 1, SIZE({a}, 1)) >= "
                            f"SPREAD([(I, I=0, SIZE({a}, 1)-1)], 2, SIZE({a}, 2)))")
            # np.hstack(a, b) -- ``[a, b]`` array constructor in
            # Fortran. Operands must be conformable rank-1.
            if attr == "hstack" and node.args:
                if (len(node.args) == 1 and isinstance(node.args[0], (ast.Tuple, ast.List))):
                    parts = [self.emit_expr(e) for e in node.args[0].elts]
                    return f"[{', '.join(parts)}]"
                return f"[{', '.join(args_e)}]"
            # np.linalg.X(args) -- a few of the simpler ones.
            if attr == "norm" and args_e:
                return f"SQRT(SUM({args_e[0]} ** 2))"
            # np.where(cond, a, b) -- Fortran MERGE(a, b, cond).
            # MERGE requires a and b to share type+kind, so promote
            # integer constants to real(c_double) when the other
            # branch is non-integer (typical numpy mask pattern uses
            # ``np.where(cond, expr, 0)`` to zero out).
            if attr == "where" and len(args_e) == 3:

                def _is_int_lit(a):
                    if (isinstance(a, ast.Constant) and isinstance(a.value, int) and not isinstance(a.value, bool)):
                        return a.value
                    # Negative literals parse as UnaryOp(USub, Constant) --
                    # ``np.where(cond, 2, -1)`` (smith_waterman substitution).
                    if (isinstance(a, ast.UnaryOp) and isinstance(a.op, ast.USub)
                            and isinstance(a.operand, ast.Constant) and isinstance(a.operand.value, int)
                            and not isinstance(a.operand.value, bool)):
                        return -a.operand.value
                    return None

                # An integer BRANCH is either an int literal or an explicit
                # ``np.int<N>(...)`` cast (bfs: ``np.where(nxt, np.int64(d+1),
                # level)`` where ``level`` is real). MERGE needs both sources to
                # share type+kind, and numpy promotes a mixed int/real where to
                # the real common type -- so when the OTHER branch is non-int we
                # real-promote the int branch (the literal's ``.0`` form, or the
                # cast's INNER expression cast to real). Both-int stays integer.
                def _int_cast_inner(a):
                    if (isinstance(a, ast.Call) and isinstance(a.func, ast.Attribute)
                            and isinstance(a.func.value, ast.Name) and a.func.value.id in ("np", "numpy")
                            and a.func.attr.rstrip("_").startswith("int")):
                        return a.args[0] if a.args else None
                    return None

                lit1, lit2 = _is_int_lit(node.args[1]), _is_int_lit(node.args[2])
                cast1, cast2 = _int_cast_inner(node.args[1]), _int_cast_inner(node.args[2])
                int1 = lit1 is not None or cast1 is not None
                int2 = lit2 is not None or cast2 is not None
                both_int = int1 and int2

                def _real_promote(arg_emit, lit, cast):
                    if lit is not None:
                        return f"{lit}.0_{self._rk}"
                    if cast is not None:
                        return f"real({self.emit_expr(cast)}, {self._rk})"
                    return arg_emit

                if both_int:
                    tsrc, fsrc = args_e[1], args_e[2]
                else:
                    tsrc = _real_promote(args_e[1], lit1, cast1)
                    fsrc = _real_promote(args_e[2], lit2, cast2)
                return f"MERGE({tsrc}, {fsrc}, {args_e[0]})"
            if attr == "where" and len(args_e) == 1:
                # ``np.where(cond)`` returns the indices where True;
                # PACK + index range is the Fortran analogue but
                # context-dependent. Emit MERGE-ish stub that the
                # downstream will reject; this raises so the caller
                # can fall back rather than silently mis-emit.
                pass
            # np.multiply(a, b, out=out) -> assign-multiply.
            if attr == "multiply" and len(args_e) >= 2:
                # Fortran has no in-place store; emit ``a * b``.
                # The caller statement form ``np.multiply(a, b, c)``
                # is unusual in numpy; treat as standalone product.
                return f"({args_e[0]}) * ({args_e[1]})"
            if attr == "add" and len(args_e) >= 2:
                return f"({args_e[0]}) + ({args_e[1]})"
            if attr == "subtract" and len(args_e) >= 2:
                return f"({args_e[0]}) - ({args_e[1]})"
            if attr == "divide" and len(args_e) >= 2:
                return f"({args_e[0]}) / ({args_e[1]})"
            # np.sqrt / np.exp / np.log / np.sin / np.cos / np.tanh
            if attr in {"sqrt", "exp", "log", "sin", "cos", "tanh"} and args_e:
                return f"{attr.upper()}({args_e[0]})"
            if attr in {"absolute", "fabs"} and args_e:
                return f"ABS({args_e[0]})"
            # np.conj / np.conjugate (vexx) -> Fortran CONJG intrinsic.
            if attr in {"conj", "conjugate"} and len(args_e) == 1:
                return f"CONJG({args_e[0]})"
            # ``np.real(z)`` / ``np.imag(z)`` -- the canonical function form the
            # ``.real`` / ``.imag`` accessor desugars to. ``real(z, kind)`` is the
            # real part (identity on a real operand too). ``aimag`` REQUIRES a
            # complex operand (gfortran errors on a real), so guard it: a real
            # operand's imaginary part is ``0`` -- matching numpy ``np.imag(real)``.
            if attr in {"real", "imag"} and len(args_e) == 1:
                if attr == "real":
                    return f"real({args_e[0]}, {self._rk})"
                from numpyto_common.lowering import _walk_complex
                arr_dt = {a.name: a.dtype for a in self.kir.arrays}
                arr_dt.update(self.kir.local_dtypes)
                if _walk_complex(node.args[0], arr_dt.get) is not None:
                    return f"aimag({args_e[0]})"
                return f"0.0_{self._rk}"
            # ``np.sign(x)`` in scalar context -> -1 / 0 / +1 (numpy: sign(0)==0,
            # unlike Fortran SIGN which gives +1 at 0). Built from MERGE, same as
            # the array ``__npb_sign`` marker. cloudsc scalar np.sign.
            if attr == "sign" and len(args_e) == 1:
                return self._emit_sign(args_e[0])
        raise NotImplementedError(f"call to {ast.unparse(node.func)} not supported")

    _INT_KIND_SUFFIX: Dict[str, str] = {
        "int64": "_c_int64_t",
        "int32": "_c_int32_t",
        "int16": "_c_int16_t",
        "int8": "_c_int8_t",
    }

    def _int_kind_selector(self, tag: str = "int64") -> str:
        """The Fortran KIND token for ``INT``/``FLOOR``/``CEILING``(x, KIND) and
        literal suffixes, derived from the suffix registry (``_c_int64_t`` ->
        ``c_int64_t``) -- never a freshly-hardcoded kind string."""
        return self._INT_KIND_SUFFIX[tag].lstrip("_")

    def _int_tag(self, dtype: str) -> Optional[str]:
        """Canonical int-suffix tag for ``dtype`` (``int`` -> ``int64`` via the
        registry), or None when ``dtype`` is not an integer. Resolved through
        ``_fortran_type`` so an alias like ``int`` maps to the right kind without
        a hardcoded match."""
        # An fp8 dtype is STORED as ``integer(c_int8_t)`` -- Fortran has no fp8
        # scalar -- so the _fortran_type match below would otherwise identify it
        # as int8 and treat the byte as an integer VALUE: the read promotion
        # would widen the code to int64 and the arithmetic would add bit
        # patterns. It is a float format; the fp8 promote / round / demote seams
        # own it, so it is never an int tag.
        if dtypes.is_storage_only(dtype):
            return None
        if dtype in self._INT_KIND_SUFFIX:
            return dtype
        ft = _fortran_type(dtype)
        for tag in self._INT_KIND_SUFFIX:
            if _fortran_type(tag) == ft:
                return tag
        return None

    def _name_int_kind(self, name: str) -> Optional[str]:
        """Return the int dtype tag for a Name (e.g. ``int64``) when
        the Name is a typed kernel array / scalar / symbol / known int local.
        Returns None when the Name is plain ``integer`` or not typed."""
        # Size symbols are always the int64 ABI integer (``_symbol_decl`` ->
        # ``_fortran_type('int')``); the inference must see that so a literal /
        # cast paired with one (azimint ``max(0, int(...))`` next to ``npt``)
        # resolves to int64 rather than defaulting to int32.
        for s in self.kir.symbols:
            if s.name == name:
                return _SYMBOL_INT_TAG
        for a in self.kir.arrays:
            if a.name == name and self._int_tag(a.dtype):
                return self._int_tag(a.dtype)
        for s in self.kir.scalars:
            if s.name == name and self._int_tag(s.dtype):
                return self._int_tag(s.dtype)
        # Check implicit-local types -- the Fortran emit''s own
        # bitwise-int64 propagation sets these via the emit-time
        # ``int_kinds`` map (populated by ``emit_fortran``).
        int_kinds = getattr(self, "_int_kinds", {})
        dt = int_kinds.get(name)
        if dt in self._INT_KIND_SUFFIX:
            return dt
        # Walk the body for Subscript(Name).dtype hints carried by
        # the lowering pipeline (when the local was typed via the
        # implicit-locals integer(c_int64_t) path).
        local_dtypes = self.kir.local_dtypes
        dt = local_dtypes.get(name)
        if dt in self._INT_KIND_SUFFIX:
            return dt
        # Fresh local arrays (np.zeros / np.cumsum results) carry their resolved
        # element dtype in the emit-time local-dtype map -- return its int tag so
        # an int local array is kinded in min/max / merge like a declared one.
        dt = getattr(self, "_local_elem_dtypes", {}).get(name)
        if dt is not None:
            return self._int_tag(dt)
        return None

    def _infer_int_kind(self, expr: ast.AST) -> Optional[str]:
        """Recursively look for the first typed integer Name reachable
        through Subscript / BinOp / UnaryOp / Call args. Returns the
        first concrete int dtype tag (or None)."""
        # An explicit ``int(x)`` is a TYPECAST that RESETS the kind: it emits
        # ``INT(x, c_int64_t)``, so the result is the canonical ABI integer no matter how
        # narrow ``x`` is. Descending into it reports the OPERAND's kind instead --
        # histogram_equalization's ``max(0, int(.. flat[i] ..))`` picked int32 off the uint8
        # ``flat`` and emitted ``max(0_c_int32_t, INT(.., c_int64_t))``, which gfortran
        # rejects as "Different type kinds" under -std=f2018.
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "int":
            return _SYMBOL_INT_TAG
        for sub in ast.walk(expr):
            # An ``int(..)`` cast NESTED in the expression (``max(0, int(..))``)
            # is the int64 ABI integer regardless of its operand's kind -- the same
            # reset the top-level case above applies. It is shallower than the
            # operand Names in BFS order, so reporting int64 here (rather than
            # descending to the operand's kind) makes ``min(nbins - 1, max(0,
            # int(..)))`` see the int32/int64 mix instead of a false all-int32.
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == "int":
                return _SYMBOL_INT_TAG
            if isinstance(sub, ast.Name):
                k = self._name_int_kind(sub.id)
                if k is not None:
                    return k
        return None

    def _int_uses(self) -> Set[str]:
        """Names used in an integer context anywhere in the kernel, computed
        once and cached (both the ``/= 0`` truthiness wrap and the min/max
        operand-typing consult it)."""
        if self._int_uses_cache is None:
            self._int_uses_cache = _names_used_as_int(self.kir.tree)
        return self._int_uses_cache

    def _emit_bitwise_pair(self, left: ast.AST, right: ast.AST) -> Tuple[str, str]:
        """Emit the two operands of a bitwise op with matched int
        kinds. When one side resolves to a typed integer Name and the
        other side is an integer Constant, append the kind suffix
        (``255_c_int64_t``)."""
        l_kind = self._infer_int_kind(left)
        r_kind = self._infer_int_kind(right)

        def emit_one(e, other_typed):
            base = self.emit_expr(e)
            # Add kind suffix to bare integer Constants when the OTHER
            # side resolves to a typed kind.
            if (isinstance(e, ast.Constant) and isinstance(e.value, int) and not isinstance(e.value, bool)
                    and other_typed):
                suf = self._INT_KIND_SUFFIX.get(other_typed)
                if suf and not base.endswith(suf):
                    return f"{e.value}{suf}"
            return base

        return emit_one(left, r_kind), emit_one(right, l_kind)

    def _nan_minmax(self, is_max: bool, arg_strs: List[str]) -> str:
        """Fold ``arg_strs`` into a NaN-PROPAGATING min/max. Fortran's MAX / MIN
        have processor-dependent NaN behaviour; numpy maximum / minimum propagate
        NaN (a NaN in either operand yields NaN). ``MERGE(a+b, MERGE(a, b, a>b),
        (a/=a).or.(b/=b))`` returns a+b (== NaN) when either operand is NaN, else
        the ordinary extremum -- reduced pairwise for 3+ operands."""
        cmp = ">" if is_max else "<"
        acc = arg_strs[0]
        for nxt in arg_strs[1:]:
            acc = (f"merge(({acc}) + ({nxt}), merge({acc}, {nxt}, ({acc}) {cmp} ({nxt})), "
                   f"(({acc}) /= ({acc})) .or. (({nxt}) /= ({nxt})))")
        return acc

    def _minmax_arg_list(self, args) -> Tuple[bool, List[str]]:
        """Emit args to ``min`` / ``max`` so the operand types are
        uniform. If any operand is a non-int (or Name resolving to a
        real-typed local), promote integer literals to ``0.0d0``-style
        real literals. Returns ``(all_int, emitted_args)``."""
        int_uses = self._int_uses()

        def is_int(e):
            if isinstance(e, ast.Constant):
                return isinstance(e.value, int) and not isinstance(e.value, bool)
            if isinstance(e, ast.Name):
                # Symbols / int_locals / known-int kernel scalars / loop iters.
                for s in self.kir.symbols:
                    if s.name == e.id:
                        return True
                for s in self.kir.scalars:
                    if s.name == e.id:
                        return s.dtype in ("int", "int32", "int64")
                if e.id in self.kir.int_locals:
                    return True
                if e.id in self._loop_iter_names:
                    return True
                if e.id in int_uses:
                    return True
                return False
            if isinstance(e, ast.BinOp):
                # ``a % b`` / ``a // b`` are int-returning when operands are.
                # ``a / b`` in Fortran follows the operand type (int/int=int).
                return is_int(e.left) and is_int(e.right)
            if isinstance(e, ast.UnaryOp):
                return is_int(e.operand)
            if isinstance(e, ast.Call):
                # ``int(x)`` / ``len(x)`` / nested ``max(..)`` / ``min(..)``
                # return int. Math intrinsics like ``floor`` / ``ceil`` /
                # ``round`` also map to ``FLOOR`` / ``CEILING`` / ``NINT``
                # which return integer in Fortran.
                fn = (e.func.id if isinstance(e.func, ast.Name) else
                      (e.func.attr if isinstance(e.func, ast.Attribute) else ""))
                if fn in _INT_RETURNING_CALLS:
                    # max/min/floor/ceil are int-returning iff their args are.
                    if fn in _INT_CALLS_ARGDEP:
                        return all(is_int(a) for a in e.args)
                    return True
                return False
            if isinstance(e, ast.IfExp):
                return is_int(e.body) and is_int(e.orelse)
            if isinstance(e, ast.Subscript):
                # ``A[i]`` is integer iff the base array/var is
                # integer-typed (e.g. clip on an int64 array: the
                # element read must count as int so a literal bound is
                # not real-promoted past an integer array operand).
                base = e.value
                while isinstance(base, ast.Subscript):
                    base = base.value
                if isinstance(base, ast.Name):
                    return (is_int(base) or self._name_int_kind(base.id) is not None)
                return False
            return False

        all_int = all(is_int(a) for a in args)
        # When every operand is integer-typed, Fortran MIN/MAX still
        # requires a uniform KIND: a default ``10`` beside an
        # ``integer(c_int64_t)`` array element is a kind mismatch under
        # -std=f2018. Find the concrete int kind of any typed operand
        # and suffix bare integer literals to match -- mirrors the
        # bitwise-pair kind matching (``_emit_bitwise_pair``).
        int_kind = None
        mixed_int = False
        if all_int:
            concrete = {k for k in (self._infer_int_kind(a) for a in args) if k is not None}
            if len(concrete) > 1:
                # Operands carry DIFFERENT integer kinds -- histogram_equalization's
                # ``min(nbins - 1, max(0, int(..)))`` mixes an int32 count with the
                # int64 ABI integer from the ``int()`` cast. Fortran MIN / MAX reject
                # mismatched kinds under -std=f2018 (a "GNU Extension" error), so
                # widen every operand to the int64 ABI integer.
                int_kind = "int64"
                mixed_int = True
            else:
                int_kind = next(iter(concrete), None)
        out = []
        for a in args:
            s = self.emit_expr(a)
            is_int_const = (isinstance(a, ast.Constant) and isinstance(a.value, int) and not isinstance(a.value, bool))
            if not all_int and is_int_const:
                out.append(f"{a.value}.0_{self._rk}")
            elif not all_int and is_int(a):
                # A mixed call is REAL-typed, but an integer-valued *expression*
                # (``INT(H[i,j]) - gap`` in smith_waterman / needleman_wunsch's
                # ``max(0, H + sub, H - gap, ...)``) stays integer under
                # -std=f2018 and clashes with the real operands. Wrap it in a
                # REAL conversion so every MAX/MIN operand shares the real kind.
                out.append(f"real({s}, {self._rk})")
            elif all_int and mixed_int:
                # Widen each operand to the int64 ABI integer: a literal takes the
                # int64 suffix, an operand already int64 is left alone, and a
                # narrower-kind expression is wrapped in INT(.., c_int64_t)
                # (value-preserving, and a no-op when the operand is already int64).
                if is_int_const:
                    out.append(f"{a.value}{self._INT_KIND_SUFFIX['int64']}")
                elif self._infer_int_kind(a) == "int64":
                    out.append(s)
                else:
                    out.append(f"INT({s}, {self._int_kind_selector('int64')})")
            elif all_int and int_kind and is_int_const:
                suf = self._INT_KIND_SUFFIX.get(int_kind, "")
                out.append(f"{a.value}{suf}" if suf else s)
            else:
                out.append(s)
        return all_int, out


def _fortran_safe(name: str) -> str:
    """Map a Python identifier to a gfortran-accepted name.

    gfortran rejects identifiers starting with ``_`` (single or double
    underscore). Lowering generates ``__si0`` / ``__cb1`` / ``__inl1_*``
    style temporaries; rename them by stripping leading underscores and
    prepending ``x_``. Plain underscores in the middle of the name are
    left alone (they are fine in Fortran identifiers).
    """
    if not name:
        return name
    stripped = name.lstrip("_")
    if stripped == name:
        return name
    return "x_" + stripped


_FORTRAN_TOKEN_RE = __import__("re").compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _to_fortran_shape_token(tok: str) -> str:
    """Translate a shape token from Python idioms to Fortran-valid
    syntax.

    * ``//`` -> ``/``  (Python floor-div is Fortran string-concat;
      result is the same for non-negative integer extents).
    * Python ``arr[i]`` indexing -> Fortran ``arr(i + 1)`` (1-based)
      by parsing the token into an AST and unparsing each Subscript
      as a Fortran subscript with the +1 adjustment.
    """
    if not isinstance(tok, str):
        return tok
    tok = tok.replace("//", "/")
    if "[" not in tok:
        return tok
    # Reparse and emit subscripts with +1 1-based adjustment. Falls
    # back to the original text if the token is not a valid Python
    # expression.
    try:
        tree = ast.parse(tok, mode="eval").body
    except SyntaxError:
        return tok

    def emit(n) -> str:
        if isinstance(n, ast.Subscript):
            base = emit(n.value)
            sl = n.slice
            if isinstance(sl, ast.Tuple):
                idxs = [f"({emit(e)}) + 1" for e in sl.elts]
                # Reverse for col-major.
                idxs.reverse()
                return f"{base}({', '.join(idxs)})"
            return f"{base}({emit(sl)} + 1)"
        if isinstance(n, ast.BinOp):
            ops = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "/", ast.Mod: "MOD"}
            op = ops.get(type(n.op))
            if op == "MOD":
                return f"MOD({emit(n.left)}, {emit(n.right)})"
            return f"({emit(n.left)} {op} {emit(n.right)})"
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return f"(-({emit(n.operand)}))"
        if isinstance(n, ast.Name):
            return n.id
        if isinstance(n, ast.Constant):
            return str(n.value)
        # Fallback: textual unparse (may leak Python syntax but at
        # least preserves the user-visible form).
        return ast.unparse(n)

    try:
        return emit(tree)
    except Exception:
        return tok


def _shape_token_uses_unknown(tok: str, allowed: Set[str]) -> bool:
    """Return True if ``tok`` references any identifier not in
    ``allowed``. Used to detect shape bounds that include local
    (non-dummy, non-symbol) integers -- those cannot appear in a
    Fortran declaration-time bound and force ``allocatable``."""
    if not isinstance(tok, str):
        return False
    for m in _FORTRAN_TOKEN_RE.finditer(tok):
        ident = m.group(0)
        # Skip pure-int literals and Fortran intrinsics that may
        # appear in shape expressions.
        if ident in {"min", "max", "abs"}:
            continue
        if ident in allowed:
            continue
        # ``ident`` is some other Name -- a local integer.
        return True
    return False


def _fortran_safe_token(tok: str) -> str:
    """Rewrite embedded identifiers in a shape / expression token
    string so any ``__name`` chunk maps to ``x_name``. Used for shape
    tuples in ``zeros_locals``, where the values are textual
    expressions like ``"__inl1_N * 4"`` that the rename pass over
    Name AST nodes never touches."""
    if not isinstance(tok, str):
        return tok
    return _FORTRAN_TOKEN_RE.sub(lambda m: _fortran_safe(m.group(0)), tok)


class _FortranRenameTemps(ast.NodeTransformer):
    """Rewrite every ``Name`` and ``For``-target identifier whose name
    starts with ``_`` to a Fortran-safe form. Also rewrites identifiers
    that case-insensitively collide with a reserved set (passed in via
    ``case_map``): Fortran is case-insensitive so ``K`` (symbol) and
    ``k`` (loop iter) refer to the same identifier; the offender (the
    one NOT in the reserved set) is rewritten to add a ``f_`` prefix."""

    def __init__(self, case_map: Optional[Dict[str, str]] = None):
        # ``case_map`` maps lower-cased reserved names (parameters /
        # symbols / kernel-array names that must keep their original
        # case) to a colliding rewrite. ``case_map[name.lower()] =
        # "f_" + name`` says "this identifier collides with a
        # reserved name; rewrite it to f_<name>".
        self.case_map: Dict[str, str] = case_map or {}

    def _safe(self, name: str) -> str:
        renamed = _fortran_safe(name)
        # Apply case-insensitive collision rewrite AFTER the leading-
        # underscore strip (so renamed `x_k` is also checked).
        renamed_ci = renamed.lower()
        if renamed_ci in self.case_map:
            mapped = self.case_map[renamed_ci]
            # Only rewrite if THIS occurrence is not the reserved
            # one itself (i.e. the renamed form differs from the
            # reserved one in case).
            if renamed != self.case_map.get(renamed_ci + "_reserved", renamed):
                return mapped
        return renamed

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self._safe(node.id)
        return node

    def visit_For(self, node: ast.For) -> ast.AST:
        if isinstance(node.target, ast.Name):
            node.target.id = self._safe(node.target.id)
        self.generic_visit(node)
        return node


def _require_parallelizable(kir: KernelIR) -> None:
    """Refuse a kernel the parallel variant cannot soundly emit: a colliding
    scatter (would need an atomic) or no parallelizable loop at all. The caller
    falls back to the sequential :func:`emit_fortran`, which is always valid."""
    if parallelism.has_indirect_scatter(kir.tree):
        raise parallelism.UnsupportedParallelError(
            f"{kir.kernel_name}: data-dependent scatter write needs an atomic; no parallel variant")
    if not parallelism.any_parallelizable_loop(kir.tree):
        raise parallelism.UnsupportedParallelError(
            f"{kir.kernel_name}: no iteration-independent or reduction loop to parallelize")


def emit_fortran_omp(kir: KernelIR, fn_name: Optional[str] = None) -> str:
    """Fortran with OpenMP ``!$omp parallel do`` on each outermost independent /
    reduction loop (a ``reduction(op:acc)`` clause for a single-scalar
    accumulator). Same signature and symbol as :func:`emit_fortran`; compile with
    ``-fopenmp`` -- single-core stays fair via ``OMP_NUM_THREADS=1``. Raises
    :class:`~numpyto_common.parallelism.UnsupportedParallelError` for a kernel
    with no sound parallel form (colliding scatter, or nothing to parallelize)."""
    _require_parallelizable(kir)
    return emit_fortran(kir, fn_name, parallel=True)


def emit_fortran(kir: KernelIR, fn_name: Optional[str] = None, parallel: bool = False) -> str:
    """Emit a self-contained Fortran subroutine with timing wrapper."""
    name = fn_name or f"{kir.kernel_name}_d_auto"
    # ABI parameter order (the order the binding JSON -- and thus every
    # caller -- uses). param_order() sorts names alphabetically, so it MUST
    # be captured on the ORIGINAL names, BEFORE the Fortran identifier rename
    # below: a source param with a leading underscore (e.g. polybench
    # fdtd_2d's ``_fict_``) is renamed to a valid Fortran identifier, and the
    # renamed form can sort to a different slot. Re-deriving param_order() on
    # the renamed descriptors would then desync the positional ABI from the
    # binding (the caller would feed an argument into the wrong slot). We keep
    # this order and only map the identifiers through the rename, preserving
    # each ABI slot.
    abi_param_order = kir.param_order()
    # Strip ``__`` / ``_`` prefixes from every Name in the tree
    # (Fortran identifier syntax forbids leading underscores). Also
    # handle case-insensitive collisions: Fortran treats ``K`` and
    # ``k`` as the same identifier so a kernel symbol ``K`` and a
    # loop iter ``k`` clash. Compute the case_map from declared
    # parameters / symbols / arrays and have ``_FortranRenameTemps``
    # rewrite collisions with an ``f_`` prefix.
    # Case-insensitive collision map (Fortran folds ``B`` and ``b`` to one
    # identifier). ``case_map[lc]`` is the ``f_``-prefixed rewrite for the
    # offender; ``case_map[lc + "_reserved"]`` is the ONE spelling that keeps its
    # case. Both the descriptor order and the reserved-slot claim MUST be
    # deterministic: iterating a *set* here (and letting the last writer win the
    # reserved slot) was hash-order-dependent, so a batch size symbol ``B`` that
    # collides with an input array ``b`` was kept on some runs and renamed on
    # others. Size SYMBOLS are listed FIRST and claim the reserved slot via
    # ``setdefault`` so they always win: a symbol is referenced by array shape
    # tokens (``a(K, M, B)``) that are emitted verbatim and never rewritten
    # through ``case_map``, so renaming the symbol would leave those tokens
    # dangling -- an undeclared name Fortran then implicitly types REAL, breaking
    # the integer array bound. Renaming the colliding array/scalar is safe (its
    # body uses are AST Names the rename pass rewrites), so the symbol stays put
    # and keeps its int64 declaration.
    reserved: Set[str] = set()
    reserved_ordered: List[str] = []
    for _descs in (kir.symbols, kir.arrays, kir.scalars):
        for _d in _descs:
            if _d.name not in reserved:
                reserved.add(_d.name)
                reserved_ordered.append(_d.name)
    case_map: Dict[str, str] = {}
    for r in reserved_ordered:
        case_map.setdefault(r.lower(), "f_" + r.lower())
        case_map.setdefault(r.lower() + "_reserved", r)
    # ALSO scan the AST body for local-vs-local case clashes (Fortran
    # is case-insensitive; ``i`` loop iter and ``I`` boolean-mask
    # local collapse to the same identifier and Fortran rejects the
    # duplicate declaration). For each lowercase form that has more
    # than one distinct cased member, keep the FIRST occurrence (in
    # AST order) reserved and route every other to ``f_<name>``.
    locals_by_lc: Dict[str, List[str]] = {}
    for node in ast.walk(kir.tree):
        n = None
        if isinstance(node, ast.Name):
            n = node.id
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            n = node.target.id
        if n is None:
            continue
        if n in reserved:
            continue
        # Skip leading-underscore names; they will be renamed by
        # ``_fortran_safe`` before they reach the case map.
        if n.startswith("_"):
            continue
        lc = n.lower()
        if lc in case_map:
            continue
        bucket = locals_by_lc.setdefault(lc, [])
        if n not in bucket:
            bucket.append(n)
    for lc, members in locals_by_lc.items():
        if len(members) < 2:
            continue
        # Keep the first observed cased form; rewrite the rest.
        keep = members[0]
        case_map[lc] = "f_" + lc
        case_map[lc + "_reserved"] = keep
    # Rename parameter / symbol / array / scalar descriptors (e.g.
    # polybench fdtd_2d's ``_fict_`` underscore-prefix; doitgen's
    # ``np`` lower-cased numpy-alias clashing with ``NP`` symbol).
    # Same case-insensitive collision rewrite for input_args / scalar
    # / array / symbol descriptors that the AST-Name rename applies to
    # the body. Without this, a kernel symbol ``NP`` plus a scratch
    # scalar ``np`` (lower-cased numpy-module name leaked into params)
    # would both appear in the signature and Fortran rejects the
    # duplicate.
    def _safe_with_case(n: str) -> str:
        s = _fortran_safe(n)
        if s.lower() in case_map and case_map.get(s.lower() + "_reserved") != s:
            return case_map[s.lower()]
        return s

    renamed_input_args = [_safe_with_case(n) for n in kir.input_args]
    renamed_kir_symbols = [dataclasses.replace(s, name=_safe_with_case(s.name)) for s in kir.symbols]
    renamed_kir_arrays = [dataclasses.replace(a, name=_safe_with_case(a.name)) for a in kir.arrays]
    renamed_kir_scalars = [dataclasses.replace(s, name=_safe_with_case(s.name)) for s in kir.scalars]
    kir = dataclasses.replace(kir,
                              symbols=renamed_kir_symbols,
                              arrays=renamed_kir_arrays,
                              scalars=renamed_kir_scalars,
                              input_args=renamed_input_args)
    kir_tree = copy.deepcopy(kir.tree)
    _FortranRenameTemps(case_map=case_map).visit(kir_tree)
    ast.fix_missing_locations(kir_tree)

    # Also rewrite the side-table of harvested zeros / shape locals.
    # The VALUES are shape tuples of token strings (e.g.
    # ``('__inl1_N', '__inl1_H_out')``) -- rewrite each token through
    # ``_fortran_safe_token`` so embedded ``__name`` references in the
    # shape expressions are also renamed.
    def _safe_full(name: str) -> str:
        """Apply leading-underscore strip + case-collision rewrite."""
        s = _fortran_safe(name)
        if (s.lower() in case_map and case_map.get(s.lower() + "_reserved") != s):
            return case_map[s.lower()]
        return s

    # The side-tables are typed KernelIR fields, so their Fortran-safe copies ride
    # along on the same ``dataclasses.replace`` that swaps in the renamed tree.
    # ``zeros_fills`` keys are renamed so the marker handler can re-zero a renamed
    # local (e.g. ``Tz``); ``zeros_locals`` / ``reassign_shapes`` also rename each
    # embedded ``__name`` shape token. ``int_locals`` / ``scalar_call_temps`` /
    # ``float_precision`` carry over unchanged (they held original names before).
    kir = dataclasses.replace(
        kir,
        tree=kir_tree,
        zeros_locals={
            _safe_full(k): tuple(_fortran_safe_token(tok) for tok in v) if v else v
            for k, v in kir.zeros_locals.items()
        },
        zeros_fills={
            _safe_full(k): v
            for k, v in kir.zeros_fills.items()
        },
        reassign_shapes={
            _safe_full(k): [tuple(_fortran_safe_token(t) for t in shape) for shape in v]
            for k, v in kir.reassign_shapes.items()
        },
        local_dtypes={
            _safe_full(k): v
            for k, v in kir.local_dtypes.items()
        },
    )

    sym_by_name = {s.name: s for s in kir.symbols}
    arr_by_name = {a.name: a for a in kir.arrays}
    sca_by_name = {s.name: s for s in kir.scalars}
    # Symbols the body writes to (``M = A_indptr.shape[0] - 1``) need their
    # ``intent(in)`` relaxed -- see :func:`_symbol_decl`. Collect them from the
    # (already safe-renamed) tree so the names line up with ``sym_by_name``.
    assigned_names: set = set()
    for n in ast.walk(kir.tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    assigned_names.add(t.id)
        elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
            assigned_names.add(n.target.id)
    # Signature order = the ABI order captured on the original names, with
    # each identifier mapped through the Fortran rename (positions preserved
    # so the subroutine matches the binding the caller dispatches through).
    param_names: List[str] = [_safe_with_case(n) for n in abi_param_order]
    # Fortran requires integer parameters used inside array bounds to
    # appear declared BEFORE the array declarations -- reorder the
    # declaration block (NOT the parameter list itself, which the C
    # ABI fixes via :func:`numpyto_common.frontend.parse_kernel`) so symbols
    # and integer scalars come first.
    sym_decls: List[str] = []
    sca_int_decls: List[str] = []
    arr_decls: List[str] = []
    sca_real_decls: List[str] = []
    for arg in param_names:
        if arg in sym_by_name:
            sym_decls.append(_symbol_decl(arg, assigned=arg in assigned_names))
        elif arg in sca_by_name:
            sca = sca_by_name[arg]
            d = _scalar_decl(arg, sca.dtype, sca.is_output, assigned=arg in assigned_names)
            (sca_int_decls if sca.dtype in ("int64", "int32", "int") else sca_real_decls).append(d)
        elif arg in arr_by_name:
            arr_decls.append(_array_decl(arr_by_name[arg]))
    decls = sym_decls + sca_int_decls + arr_decls + sca_real_decls

    body_emitter = _FortranBodyEmitter(kir)
    body_emitter.parallel = parallel
    # Non-inlinable helpers -> ``call helper(args, X)`` at each ``X =
    # helper(args)`` site. Keyed by the Fortran-safe helper name (the tree Names
    # were renamed above), value = return kind (unused; membership is what maps).
    body_emitter._helper_out = {_fortran_safe(h.kernel_name): h.return_kind for h in kir.helpers}
    # Pre-compute implicit-local int kinds before emit_block so the
    # body emitter can apply kind-matched bitwise literal suffixes.
    _pre_implicit = _collect_implicit_locals(kir)
    _pre_int_kinds: Dict[str, str] = {}
    for nm, ft in _pre_implicit:
        if ft == "integer(c_int64_t)":
            _pre_int_kinds[nm] = "int64"
        elif ft.startswith("integer(c_int32"):
            _pre_int_kinds[nm] = "int32"
    body_emitter._int_kinds = _pre_int_kinds
    # Pre-compute logical_array_locals so _emit_subscript can detect
    # ``arr[mask]`` boolean-indexing and emit PACK(arr, mask).
    _pre_logical_arr_locals: Set[str] = set(_assigned_bool_literal(kir.tree))
    # Array locals the lowering typed boolean (``owner = mask != 0``,
    # ``mask = cfl_clip & owner``) -- their dtype is recorded in local_dtypes
    # rather than via a bare True/False literal assignment.
    _ld_pre = kir.local_dtypes
    _pre_logical_arr_locals |= {nm for nm, dt in _ld_pre.items() if dt in ("bool", "bool_")}
    for node in ast.walk(kir.tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        tgt = node.targets[0]
        if not _produces_logical(node.value):
            continue
        # ``mask[i] = a[i] < b[i]`` (boolean ARRAY local) and, equally,
        # ``do_lj = (flags & CI_DO_LJ) != 0`` (boolean SCALAR local): both are
        # Fortran ``logical`` and the emitter must route a bare use (``.not.
        # do_lj``, ``arr[mask]``) as LOGICAL rather than wrapping it ``/= 0``.
        if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name):
            _pre_logical_arr_locals.add(tgt.value.id)
        elif isinstance(tgt, ast.Name):
            _pre_logical_arr_locals.add(tgt.id)
    body_emitter._logical_array_locals = _pre_logical_arr_locals
    # Int-typed PARAMETER arrays cannot be re-typed (C ABI), so their use as a
    # 0/1 flag in a boolean context is wrapped with ``/= 0`` at the condition
    # site instead (cloudsc's ``ldcum``). kir.arrays are the parameters.
    body_emitter._int_array_names = {a.name for a in kir.arrays if _fortran_type(a.dtype).startswith("integer")}
    # NOTE: the body is emitted further down, AFTER every body_emitter
    # side-table (especially ``inline_alloc_locals``) is populated, so
    # the ``np.zeros`` marker handler can emit an ``allocate`` for
    # loop-iter-sized locals (otherwise they stay unallocated -> SIG11).

    # Local arrays produced by ``np.zeros`` -- declare in the prelude.
    locals_block = []
    int_locals = kir.int_locals
    # Fortran is case-insensitive; collapse names that clash (in any case)
    # with parameter names already declared. The tuple-unpack rewriter
    # produces ``n = N`` -> in Fortran they refer to the same identifier
    # ``N`` so we skip the lowercase declaration entirely.
    param_names_ci = {p.lower() for p in param_names}
    seen_ci: Set[str] = set(param_names_ci)
    for name_ in int_locals:
        if name_.lower() in seen_ci:
            continue
        seen_ci.add(name_.lower())
        locals_block.append(f"    {_fortran_type('int')} :: {name_}")
    implicit = _collect_implicit_locals(kir)
    # Build a name -> int-dtype-tag map so the body emitter can use it
    # for bitwise pair-kind matching.
    int_kinds_for_body: Dict[str, str] = {}
    for name_, ftype in implicit:
        if ftype == "integer(c_int64_t)":
            int_kinds_for_body[name_] = "int64"
        elif ftype.startswith("integer(c_int32"):
            int_kinds_for_body[name_] = "int32"
    body_emitter._int_kinds = int_kinds_for_body
    for name_, ftype in implicit:
        if name_.lower() in seen_ci:
            continue
        seen_ci.add(name_.lower())
        locals_block.append(f"    {ftype} :: {name_}")
    # Identify which symbols / dummy args can appear in a Fortran
    # declaration-time array bound. Symbols and dummy-arg integers
    # are allowed; locals (anything else used as int) are NOT.
    allowed_bound_names: Set[str] = set()
    for s in kir.symbols:
        allowed_bound_names.add(s.name)
    for s in kir.scalars:
        if s.dtype in ("int", "int32", "int64"):
            allowed_bound_names.add(s.name)
    # Detect array locals whose body uses Compare/BoolOp on per-
    # element assignments -- those must be Fortran ``logical`` arrays.
    # Walks every Subscript(Name) = Compare/BoolOp/Not assignment.
    logical_array_locals: Set[str] = set(_assigned_bool_literal(kir.tree))
    # Array locals the lowering typed boolean via local_dtypes (whole-array
    # Compare / BoolOp / bitwise-of-bools) are logical too -- declare them so.
    _ld_decl = kir.local_dtypes
    logical_array_locals |= {nm for nm, dt in _ld_decl.items() if dt in ("bool", "bool_")}
    # SCALAR locals whose RHS is boolean-valued are Fortran ``logical`` too
    # (GROMACS ``do_lj = (flags & CI_DO_LJ) != 0``, ``half_lj = ... and
    # do_coul``). They are already DECLARED logical (the implicit-locals
    # ``logical_uses`` pass), but the emitter's operand routing keys off this
    # set: without them here a bare use (``.not. do_lj``, ``... .and.
    # do_coul``) is treated as an integer flag and wrapped ``/= 0``, which
    # gfortran rejects against the LOGICAL declaration.
    for node in ast.walk(kir.tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and _produces_logical(node.value)):
            logical_array_locals.add(node.targets[0].id)
    # Track inferred dtype for array locals via per-element assigns.
    # ``cols[si0] = A_col[expr]`` -> cols inherits A_col''s dtype.
    inferred_local_dtypes: Dict[str, str] = {}
    array_dtype_map: Dict[str, str] = {}
    for a in kir.arrays:
        array_dtype_map[a.name] = a.dtype
    for node in ast.walk(kir.tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].value, ast.Name)):
            rhs = node.value
            if _produces_logical(rhs):
                logical_array_locals.add(node.targets[0].value.id)
            # ``X[i] = Y[j]`` where Y is a typed array -- propagate
            # Y''s dtype to X.
            if isinstance(rhs, ast.Subscript) and isinstance(rhs.value, ast.Name):
                src = rhs.value.id
                tgt = node.targets[0].value.id
                if src in array_dtype_map and tgt not in inferred_local_dtypes:
                    inferred_local_dtypes[tgt] = array_dtype_map[src]
    # Collect for-loop iter names; locals whose shape uses any of
    # them must be allocated INSIDE the loop body, not at function
    # top (the iter is not yet in scope at function start).
    loop_iter_names: Set[str] = set()
    for node in ast.walk(kir.tree):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            loop_iter_names.add(node.target.id)

    def _shape_uses_loop_iter(rev_shape):
        for tok in rev_shape:
            t = str(tok)
            for it in loop_iter_names:
                idx = t.find(it)
                while idx >= 0:
                    lo_ok = (idx == 0 or not (t[idx - 1].isalnum() or t[idx - 1] == "_"))
                    hi_ok = (idx + len(it) >= len(t) or not (t[idx + len(it)].isalnum() or t[idx + len(it)] == "_"))
                    if lo_ok and hi_ok:
                        return True
                    idx = t.find(it, idx + 1)
        return False

    # Scalar locals COMPUTED in the body (``m = min(max_iter, n)`` in
    # gmres, ``x_inl3_H_out = ...`` from conv2d inlining): every Name
    # assigned in the body that is not itself a local array or a loop
    # iter. An array whose ALLOCATE bound references one of these cannot
    # be allocated at function-top -- the scalar is still undefined there,
    # so the bound is garbage (-> SIG11 on first write). Such arrays defer
    # their allocate to the marker site, which follows the scalar's
    # assignment in straight-line order (mirrors the NumpyToC fix).
    _array_local_names = set(body_emitter.local_arrays.keys())
    computed_scalars: Set[str] = set()
    for node in ast.walk(kir.tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            nm = node.targets[0].id
            if nm not in _array_local_names and nm not in loop_iter_names:
                computed_scalars.add(nm)

    def _shape_uses_computed_scalar(rev_shape):
        for tok in rev_shape:
            for m in _IDENT_RE.findall(str(tok)):
                if m in computed_scalars:
                    return True
        return False

    # Track allocatable locals so we can emit allocate/deallocate.
    # ``inline_alloc_locals`` is the subset whose allocate must go
    # at the marker site (inside the for-loop scope).
    allocatable_locals: List[Tuple[str, List[str], str]] = []
    inline_alloc_locals: Dict[str, Tuple[List[str], str]] = {}
    # Element registry-dtype of each local array (``cdf`` -> ``float64``). Most of
    # these carry no recorded dtype and fall to the kernel float default, so the
    # array + scalar decls the type predicates consult never learn they are real /
    # int. Record the RESOLVED dtype here so ``_expr_is_real`` / ``_name_int_kind``
    # can classify a fresh local -- e.g. ``merge(cdf, npix, ..)`` then promotes the
    # int ``npix`` beside the real ``cdf`` (histogram_equalization's CDF clamp).
    local_elem_dtypes: Dict[str, str] = {}
    for name_, shape in body_emitter.local_arrays.items():
        if name_.lower() in seen_ci:
            # A param-array re-harvested into ``local_arrays`` is already declared
            # (same entity) -- skipping its local declaration is correct. But an
            # array whose lowercased name clashes with an already-declared SCALAR
            # / int-local / other-case array is a genuine conflict: Fortran is
            # case-insensitive, so both would bind to one symbol and indexing the
            # "array" hits a scalar -> the body is unclassifiable. Fail loudly
            # (rename the local at its source) instead of silently dropping the
            # declaration and miscompiling.
            if name_.lower() not in param_names_ci:
                raise NotImplementedError(f"local array {name_!r} clashes case-insensitively with an "
                                          "already-declared name; rename it (Fortran is case-insensitive)")
            continue
        seen_ci.add(name_.lower())
        # REVERSED shape for col-major / row-major interop -- see
        # ``_array_decl`` for the kernel-arg case. Also translate
        # Python ``//`` (floor-div) to ``/`` since Fortran reads
        # ``//`` as the string-concat operator.
        rev_shape = ([_to_fortran_shape_token(s) for s in reversed(shape)] if shape else ["1"])
        local_dtypes = kir.local_dtypes
        # A float temp with no recorded dtype defaults to the KERNEL's
        # float precision, not a hard-coded ``float64`` -- in fp32 mode the
        # locals must be ``real(c_float)`` so a relu ``max(x + bias,
        # 0.0_c_float)`` does not mix kinds with the c_float inputs (lenet).
        _default_float = kir.float_precision or "float64"
        dt = local_dtypes.get(name_, inferred_local_dtypes.get(name_, _default_float))
        # Bool-typed locals declare as Fortran ``logical(c_bool)`` -- the 1-byte
        # C-ABI logical (from the dtype registry), so a comparison RHS
        # (``mask = (a < b) .AND. (b < c)``) lands on a typed-compatible LHS that
        # also matches C's 1-byte ``_Bool`` across the bind(C) seam (a bare
        # ``logical`` is the 4-byte default kind -- an ABI mismatch).
        if dt in ("bool", "bool_") or name_ in logical_array_locals:
            ftype = _fortran_type("bool")
            local_elem_dtypes[name_] = local_elem_dtypes[_fortran_safe(name_)] = "bool"
        else:
            ftype = _fortran_type(dt)
            local_elem_dtypes[name_] = local_elem_dtypes[_fortran_safe(name_)] = dt
        # If any shape token references a Name that is NOT a symbol
        # or dummy int arg, the declaration-time bound is illegal --
        # fall back to ``allocatable`` and emit ALLOCATE / DEALLOCATE
        # around the body. Required for lenet/resnet/conv2d/mlp/etc.
        # where shapes use locals like ``x_inl1_K``.
        needs_alloc = any(_shape_token_uses_unknown(tok, allowed_bound_names) for tok in rev_shape)
        if needs_alloc:
            colons = ", ".join(":" for _ in rev_shape)
            locals_block.append(f"    {ftype}, allocatable :: {name_}({colons})")
            if _shape_uses_loop_iter(rev_shape) \
                    or _shape_uses_computed_scalar(rev_shape):
                inline_alloc_locals[name_] = (rev_shape, ftype)
            else:
                allocatable_locals.append((name_, rev_shape, ftype))
        else:
            dims = ", ".join(rev_shape)
            locals_block.append(f"    {ftype} :: {name_}({dims})")
    # Tell the body emitter which locals to inline-allocate at their
    # ``Name = __optarena_zeros__()`` marker site (instead of the
    # function-top allocate list above).
    body_emitter.inline_alloc_locals = inline_alloc_locals
    body_emitter._local_elem_dtypes = local_elem_dtypes

    # Now that every side-table is set, emit the body. The ``np.zeros``
    # / slice-copy markers can now emit ``allocate(<local>(...))`` for
    # loop-iter-sized locals at their in-scope marker site.
    body = body_emitter.emit_block(kir.tree.body, indent="    ")

    # Loop iter var declarations (Fortran needs explicit declaration).
    iter_vars = _collect_for_targets(kir.tree.body)
    # Loop iterators at the int64 ABI width (same as the size symbols they range
    # over): a default-kind ``integer`` iterator clashes with an int64 symbol the
    # moment they meet in an intrinsic / comparison (``MODULO(i * 7, LEN_1D)``,
    # ``min(jj + W, N - 1)``) under -std=f2018. Subscripts accept any kind, so the
    # ``a(i + 1)`` offsets were fine; the arithmetic/intrinsic uses were not.
    iter_decls = [f"    {_fortran_type('int')} :: " + ", ".join(sorted(iter_vars))] if iter_vars else []

    # ALLOCATE / DEALLOCATE around the body for any allocatable locals.
    if allocatable_locals:
        alloc_lines = [f"    allocate({n}({', '.join(s)}))" for n, s, _ in allocatable_locals]
        dealloc_lines = [f"    deallocate({n})" for n, _, _ in allocatable_locals]
        body = "\n".join(alloc_lines) + "\n" + body + "\n" + "\n".join(dealloc_lines)

    # bind(C) interface block for any libm functions Fortran lacks (cbrt/exp2/
    # log2/expm1/log1p) -- declared so the body's ``cbrt(x)`` calls resolve to
    # the C library (linked via -lm), bit-identical to the C/numpy result.
    libm_iface = ""
    if body_emitter._used_libm:
        lines = ["    interface"]
        for libm, rk in sorted(body_emitter._used_libm):
            lines.append(f'        pure real({rk}) function {libm}(x) bind(C, name="{libm}")')
            lines.append(f"            import :: {rk}")
            lines.append(f"            real({rk}), value :: x")
            lines.append(f"        end function {libm}")
        lines.append("    end interface")
        libm_iface = "\n".join(lines)

    contained = _fp8_contained(kir) + "".join(_emit_fortran_helper(h) for h in kir.helpers)
    # numpy round / rint are half-to-EVEN; Fortran ANINT is half-AWAY. The
    # correction (step back toward the even neighbour on an exact .5 tie whose
    # ANINT is odd) is emitted ONCE as a contained pure function -- callers pass
    # the argument once instead of inlining it six times (see _used_round_even).
    if body_emitter._used_round_even:
        contained += _round_even_helper(body_emitter._rk)
    return _format_subroutine(
        name=name,
        params=param_names,
        decls=decls,
        iter_decls=iter_decls,
        locals_block=locals_block,
        body=body,
        interface_block=libm_iface,
        use_ieee=body_emitter._used_ieee,
        contained=contained,
    )


def _collect_implicit_locals(kir: KernelIR) -> List[Tuple[str, str]]:
    """Return ``(name, fortran_type)`` for scalar locals needing a decl.

    Names used as array subscripts or as ``range()`` arguments promote
    to ``integer``; everything else falls back to ``real(c_double)``.
    The detection walks subscript / call arithmetic so that
    ``b(LEN_1D - k + 1)`` correctly promotes ``k`` even though it
    appears under a BinOp.
    """
    # Float locals follow the kernel's precision (real(c_float) at fp32),
    # else a double local clashes with float32 arrays/values.
    rk = {"float32": "c_float", "float16": "c_float"}.get(
            dtypes.compute_dtype(kir.float_precision or "float64"), "c_double")
    real_t = f"real({rk})"
    ck = {
        "float32": "c_float_complex",
        "float16": "c_float_complex"
    }.get(kir.float_precision or "float64", "c_double_complex")
    complex_t = f"complex({ck})"
    # The lowering records each local's dtype in ``local_dtypes``. Walk it ONCE
    # and derive every consumer from that single pass instead of re-iterating the
    # dict (and re-running ``_fortran_type``) per consumer:
    #   * ``recorded_ftype`` -- the authoritative name -> Fortran-type map (PRIMARY
    #     dtype source; outranks the usage-role heuristics below, which only guess a
    #     class from a subscript / assignment shape and else fall to real(c_double)).
    #     Complex / real map to the kernel-precision types (``complex_t`` /
    #     ``real_t``, matching the rest of the emitter); integer / logical keep their
    #     exact registry kind -- int32 vs int64 matters for ``-std=f2018`` kind
    #     matching, so it comes from the registry (``_fortran_type``), never a
    #     literal. A recorded integer still widens to int64 in ``_classify`` when a
    #     bitwise / kind source demands it.
    #   * ``complex_names`` -- complex locals, for the float-assign exclusion and the
    #     classify fallback (contour_integral's ``zz`` / linalg-solve ``__sol_tmp``).
    #     A real decl would silently drop the imaginary part. Kept ``startswith``-
    #     based so a complex dtype outside the registry is still caught.
    #   * ``recorded_int64_local`` / ``recorded_real_local`` -- local int64 / real
    #     names that seed the int64 propagation and the real-assignment detection
    #     further down (nqueens' int64 stack, azimint's real bounds).
    # ``local_dtypes`` is keyed by the pre-rename name; every derived key is stored
    # under both the raw and the fortran-safe name so lookups work before and after
    # sanitising.
    _ldt = kir.local_dtypes
    int64_kind = _fortran_type("int64")
    complex_names: Set[str] = set()
    recorded_ftype: Dict[str, str] = {}
    recorded_int64_local: Set[str] = set()
    recorded_real_local: Set[str] = set()
    for _k, _v in _ldt.items():
        if not isinstance(_v, str):
            continue
        _safe = _fortran_safe(_k)
        if _v.startswith("complex"):
            complex_names.add(_k)
            complex_names.add(_safe)
        if _v not in dtypes.REGISTRY:
            continue
        _ft = _fortran_type(_v)
        if _ft.startswith("complex"):
            _rt = complex_t
        elif _ft.startswith("real"):
            _rt = real_t
            recorded_real_local.add(_k)
        else:  # integer / logical: keep the exact registry kind
            _rt = _ft
            if _ft == int64_kind:
                recorded_int64_local.add(_k)
        recorded_ftype[_k] = _rt
        recorded_ftype[_safe] = _rt
    declared: Set[str] = set()
    declared.update(kir.input_args)
    declared.update(kir.int_locals)
    declared.update(kir.zeros_locals.keys())
    # Loop iter vars are declared via _collect_for_targets.
    for s in ast.walk(kir.tree):
        if isinstance(s, ast.For) and isinstance(s.target, ast.Name):
            declared.add(s.target.id)
    int_uses = _names_used_as_int(kir.tree)
    BITWISE_OPS = (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)

    def _produces_bool(node) -> bool:
        # A boolean-valued RHS: a comparison / boolean op / logical-not, OR a
        # numpy mask combine where ``&`` / ``|`` / ``^`` / ``~`` is applied to
        # boolean operands (``(a < b) & (c > 0)``). Such a target is ``logical``.
        if isinstance(node, (ast.Compare, ast.BoolOp)):
            return True
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, BITWISE_OPS):
            return _produces_bool(node.left) or _produces_bool(node.right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
            return _produces_bool(node.operand)
        return False

    # Names whose RHS is boolean-valued -- type as ``logical`` so a comparison /
    # mask-combine result can be assigned without a type error.
    logical_uses: Set[str] = set()
    for node in ast.walk(kir.tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and _produces_bool(node.value)):
            logical_uses.add(node.targets[0].id)
    # Add bitwise-op targets and bitwise-op operands to int_uses --
    # all of ``IAND`` / ``IOR`` / ``IEOR`` / ``ISHFT`` / ``NOT``
    # arguments must be INTEGER in Fortran, so the local that ends up
    # in either position needs an int decl. A ``&`` / ``|`` / ``^`` / ``~``
    # whose operand ``_produces_bool`` is a numpy MASK COMBINE, not integer
    # bitwise arithmetic -- those operands stay logical / real and are skipped.

    def _walk_bitwise_operands(rhs):
        # Walk a RHS expression and add every Name reachable through
        # bitwise BinOps / Invert UnaryOps to int_uses.
        if isinstance(rhs, ast.BinOp) and isinstance(rhs.op, BITWISE_OPS):
            for sub in (rhs.left, rhs.right):
                if _produces_bool(sub):  # logical mask combine, not int
                    continue
                for n in ast.walk(sub):
                    if isinstance(n, ast.Name):
                        int_uses.add(n.id)
                _walk_bitwise_operands(sub)
        elif (isinstance(rhs, ast.UnaryOp) and isinstance(rhs.op, ast.Invert)):
            if not _produces_bool(rhs.operand):  # ``~mask`` stays logical
                for n in ast.walk(rhs.operand):
                    if isinstance(n, ast.Name):
                        int_uses.add(n.id)
                _walk_bitwise_operands(rhs.operand)
        elif isinstance(rhs, ast.BinOp):
            _walk_bitwise_operands(rhs.left)
            _walk_bitwise_operands(rhs.right)
        elif isinstance(rhs, ast.UnaryOp):
            _walk_bitwise_operands(rhs.operand)
        elif isinstance(rhs, ast.Compare):
            # A bitwise op can hide inside a comparison (``(flags & CI_DO_LJ)
            # != 0``): the ``flags`` operand of IAND must still be INTEGER, so
            # descend into the compared expressions.
            _walk_bitwise_operands(rhs.left)
            for c in rhs.comparators:
                _walk_bitwise_operands(c)
        elif isinstance(rhs, ast.BoolOp):
            # ...and inside ``and`` / ``or`` (``(flags & 4) != 0 or ...``).
            for v in rhs.values:
                _walk_bitwise_operands(v)

    for node in ast.walk(kir.tree):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            rhs = node.value
            tgt = (node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else
                   node.target if isinstance(node, ast.AugAssign) else None)
            # If the RHS itself is a top-level bitwise op, the LHS is
            # also int -- UNLESS it is a logical mask combine, whose result
            # is a boolean array (``logical``), handled by ``logical_uses``.
            if isinstance(rhs, ast.BinOp) and isinstance(rhs.op, BITWISE_OPS) and not _produces_bool(rhs):
                if isinstance(tgt, ast.Name):
                    int_uses.add(tgt.id)
            if (isinstance(rhs, ast.UnaryOp) and isinstance(rhs.op, ast.Invert) and not _produces_bool(rhs)):
                if isinstance(tgt, ast.Name):
                    int_uses.add(tgt.id)
            _walk_bitwise_operands(rhs)
    seen: Set[str] = set(declared)
    out: List[Tuple[str, str]] = []
    # Collect names that interact with int64 arrays / scalars through
    # bitwise / shift ops -- those need ``integer(c_int64_t)`` kind so
    # IEOR / IAND etc. don't reject mismatched kinds.
    int64_uses: Set[str] = set()
    int64_names: Set[str] = set()
    # Seed from every name whose Fortran kind IS the int64 integer kind --
    # determined programmatically from the dtype (not a literal dtype string):
    # the size SYMBOLS (always the c_int64_t ABI integer) plus any int scalar /
    # array whose dtype maps to that kind (``int`` and ``int64`` both do; an
    # ``int32`` index array does not). Propagation below then carries int64-ness
    # to any local that meets one of these in an assignment / intrinsic, so e.g.
    # gmres ``f_n = N`` then ``m = min(max_iter, f_n)`` resolve to int64 and do
    # not clash with the int64 dummies under ``-std=f2018``.
    # A name is int64 iff its dtype maps to the int64 Fortran kind (floats map
    # to ``real(...)`` and int32 to ``integer(c_int32_t)``, so neither matches).
    # ``int64_kind`` is computed with the single local_dtypes pass near the top.
    for s in kir.symbols:
        int64_names.add(s.name)
    for a in kir.arrays:
        if _fortran_type(a.dtype) == int64_kind:
            int64_names.add(a.name)
    for s in kir.scalars:
        if _fortran_type(s.dtype) == int64_kind:
            int64_names.add(s.name)
    # LOCAL int64 arrays/scalars (``np.zeros(N + 1, dtype=np.int64)`` -- the
    # nqueens backtracking stack) carry their dtype in ``local_dtypes``, not in
    # ``kir.arrays``; seed them so their elements/derived scalars stay int64 and
    # don't clash under -std=f2018 (collected in the single local_dtypes pass above).
    int64_names |= recorded_int64_local
    # A ``x = np.<dtype>(...)`` scalar cast (``total = np.int64(0)``) types ``x``
    # by that dtype -- mark integer casts as int (and int64 when the kind
    # matches) so the local is declared integer, not the real(c_double) default.
    for node in ast.walk(kir.tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == "np"):
            key = node.value.func.attr
            key = key[:-1] if key.endswith("_") else key
            if (key in dtypes.REGISTRY or key in dtypes.SCALAR_KINDS) \
                    and _fortran_type(key).startswith("integer"):
                int_uses.add(node.targets[0].id)
                if _fortran_type(key) == int64_kind:
                    int64_names.add(node.targets[0].id)
    # FOR-loop iterators are the int64 ABI integer (declared via
    # ``_fortran_type('int')``); seed them as int64 sources so a bitwise op
    # mixing an iterator with another local (bitonic ``partner = i ^ j`` /
    # ``i & k``) propagates int64 to that local instead of leaving it int32 and
    # clashing under -std=f2018.
    for node in ast.walk(kir.tree):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            int64_names.add(node.target.id)
    # Fixed-point propagate: any Name that shares a bitwise BinOp with
    # an int64 Name is itself int64. ``crc = IEOR(crc, poly)`` -- crc
    # is int64 because poly is. A boolean mask combine (``(a < b) & ...``)
    # is excluded -- it is logical, not integer bitwise arithmetic.
    changed = True
    int64_uses |= int64_names
    while changed:
        changed = False
        for node in ast.walk(kir.tree):
            if (isinstance(node, ast.BinOp) and isinstance(node.op, BITWISE_OPS) and not _produces_bool(node)):
                names = [n.id for n in ast.walk(node) if isinstance(n, ast.Name)]
                if any(n in int64_uses for n in names):
                    for n in names:
                        if n not in int64_uses:
                            int64_uses.add(n)
                            changed = True
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in {"IAND", "IOR", "IEOR", "ISHFT", "NOT"}):
                names = [n.id for n in ast.walk(node) if isinstance(n, ast.Name)]
                if any(n in int64_uses for n in names):
                    for n in names:
                        if n not in int64_uses:
                            int64_uses.add(n)
                            changed = True
            # Through assignments: ``x = bitwise_expr_with_int64`` makes
            # x int64.
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                rhs = node.value
                rhs_names = [n.id for n in ast.walk(rhs) if isinstance(n, ast.Name)]
                if any(n in int64_uses for n in rhs_names):
                    tgts = (node.targets if isinstance(node, ast.Assign) else [node.target])
                    for tgt in tgts:
                        if isinstance(tgt, ast.Name) and tgt.id not in int64_uses:
                            int64_uses.add(tgt.id)
                            changed = True
    # A scalar local assigned from a REAL array element holds a real value, so
    # it must be declared REAL even when it also flows into an integer-
    # truncating expression. azimint_hist's histogram bounds ``__hlo =
    # radius[0]`` feed the real bin formula ``... // (__hhi - __hlo)`` that is
    # INT()-truncated to the bin index; the int-use fixed-point propagation
    # walks into that truncated expression and would mis-mark the bounds as
    # int. Assignment-from-real wins over the int-use heuristic, matching the C
    # backend (whose dtype inference is assignment-based).
    real_array_names = {a.name for a in kir.arrays if _fortran_type(a.dtype).startswith("real")}
    real_array_names |= recorded_real_local  # local reals from the single local_dtypes pass above
    float_assigned: Set[str] = set()
    for node in ast.walk(kir.tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Name)
                and node.value.value.id in real_array_names):
            float_assigned.add(node.targets[0].id)
    # Pick a local's Fortran type from its inferred role. Integer kinds come
    # from the dtype registry (``_fortran_type``), never a literal kind string:
    # an int that meets an int64 source is the int64 kind, a plain int is the
    # int32 kind (== Fortran default-integer kind), so the two only differ when
    # the inference says so.
    def _classify(name: str) -> str:
        # 1. The lowering-recorded dtype is authoritative: it names the local's
        #    class outright, so it outranks every usage-based guess below. A
        #    recorded integer still widens to int64 when a bitwise / kind source
        #    demands it (f2018 kind matching); real / complex / logical are final.
        rec = recorded_ftype.get(name)
        if rec is not None:
            if rec.startswith("integer") and name in int64_uses:
                return int64_kind
            return rec
        # 2. Value-based inference for untagged locals: a boolean-valued RHS is
        #    logical; a scalar read from a real array element is real (this wins
        #    over the int-use heuristic -- azimint's ``__hlo = radius[0]`` feeds an
        #    INT()-truncated bin index but is itself real).
        if name in logical_uses:
            return _fortran_type("bool")  # logical(c_bool): 1-byte, matches C _Bool
        if name in float_assigned and name not in complex_names:
            return real_t
        # 3. Usage-role inference (weakest): a name used as a subscript / range arg
        #    or bitwise operand is integer, int64 when it meets an int64 source.
        if name in int64_uses and name in int_uses:
            return _fortran_type("int64")
        if name in int_uses:
            return _fortran_type("int32")
        if name in complex_names:
            return complex_t
        return real_t

    for node in ast.walk(kir.tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id not in seen:
                    out.append((tgt.id, _classify(tgt.id)))
                    seen.add(tgt.id)
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name) and node.target.id not in seen:
                out.append((node.target.id, _classify(node.target.id)))
                seen.add(node.target.id)
    return out


def _names_used_as_int(tree: ast.AST) -> Set[str]:
    """Names that flow into an integer-only position (subscript / range arg).

    Walks BinOps, UnaryOps, and Call args so ``b(LEN_1D - k + 1)``
    promotes ``k`` and ``LEN_1D``, not just the literal Name in slot 0.
    """
    int_uses: Set[str] = set()

    def collect(node):
        if node is None:
            return
        if isinstance(node, ast.Name):
            int_uses.add(node.id)
        elif isinstance(node, ast.BinOp):
            collect(node.left)
            collect(node.right)
        elif isinstance(node, ast.UnaryOp):
            collect(node.operand)
        elif isinstance(node, ast.Call):
            for arg in node.args:
                collect(arg)
        elif isinstance(node, ast.Subscript):
            collect(node.value)
            sl = node.slice
            elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
            for e in elts:
                collect(e)

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            sl = node.slice
            elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
            for e in elts:
                collect(e)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range"):
            for arg in node.args:
                collect(arg)
    # Fixed-point propagation: when X is int-typed and ``X = expr``
    # assigns from another Name Y, Y is also int-typed (the result
    # flows into an int target). Required for chains like
    # ``m_lbound = min(...)`` then ``m_start = max(i - m_lbound, 0)``
    # where m_start is used as a range arg. Bounded by
    # :func:`pure_int_arith` so the closure never propagates BACKWARD
    # across a float divide / sqrt or an ``int(...)`` truncation: without
    # it, GROMACS ``ri = int(rs)`` (ri indexes the Coulomb table) walked
    # into ``rs = rsq * rinv * tab_coul_scale`` and mistyped the whole
    # distance chain (``rsq`` / ``rinv`` / ``dx``) as integer, so every
    # coordinate difference truncated to 0 and SQRT rejected the int arg.
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in int_uses and pure_int_arith(node.value)):
                before = len(int_uses)
                collect(node.value)
                if len(int_uses) > before:
                    changed = True
    return int_uses


def _collect_for_targets(stmts: List[ast.stmt]) -> Set[str]:
    found: Set[str] = set()
    for s in ast.walk(ast.Module(body=stmts, type_ignores=[])):
        if isinstance(s, ast.For) and isinstance(s.target, ast.Name):
            found.add(s.target.id)
    return found


def _helper_returns_int(hkir: KernelIR) -> bool:
    """True when every ``return`` value of a scalar helper is an integer literal
    (so its out-param is integer-typed rather than the default real)."""
    rets = [n.value for n in ast.walk(hkir.tree) if isinstance(n, ast.Return) and n.value is not None]
    return bool(rets) and all(
        isinstance(v, ast.Constant) and isinstance(v.value, int) and not isinstance(v.value, bool) for v in rets)


def _rename_helper_to_fortran_safe(hkir: KernelIR) -> KernelIR:
    """Fortran-safe rename of a captured helper KIR -- its body tree, the harvested
    side-tables, and every descriptor -- mirroring the kernel-level rename so the
    helper's lowering temps (``__cb1``, ``__w0``) and out-param (``__hret_0``) in
    the body match the names its declarations use."""
    htree = copy.deepcopy(hkir.tree)
    _FortranRenameTemps().visit(htree)
    ast.fix_missing_locations(htree)
    r_arrays = [
        dataclasses.replace(a, name=_fortran_safe(a.name), shape=tuple(_fortran_safe_token(t) for t in a.shape))
        for a in hkir.arrays
    ]
    r_scalars = [dataclasses.replace(s, name=_fortran_safe(s.name)) for s in hkir.scalars]
    r_symbols = [dataclasses.replace(s, name=_fortran_safe(s.name)) for s in hkir.symbols]
    r_return = (_fortran_safe(hkir.return_kind) if hkir.return_kind not in (None, "scalar") else hkir.return_kind)
    # The harvested side-tables are typed KernelIR fields; rename their keys and
    # embedded ``__name`` shape tokens the same way and carry them on the replace.
    return dataclasses.replace(hkir,
                               tree=htree,
                               arrays=r_arrays,
                               scalars=r_scalars,
                               symbols=r_symbols,
                               input_args=[_fortran_safe(p) for p in hkir.input_args],
                               return_kind=r_return,
                               zeros_locals={
                                   _fortran_safe(k): tuple(_fortran_safe_token(t) for t in v) if v else v
                                   for k, v in hkir.zeros_locals.items()
                               },
                               zeros_fills={
                                   _fortran_safe(k): v
                                   for k, v in hkir.zeros_fills.items()
                               },
                               reassign_shapes={
                                   _fortran_safe(k): [tuple(_fortran_safe_token(t) for t in sh) for sh in v]
                                   for k, v in hkir.reassign_shapes.items()
                               },
                               local_dtypes={
                                   _fortran_safe(k): v
                                   for k, v in hkir.local_dtypes.items()
                               })


def _emit_fortran_helper(hkir: KernelIR) -> str:
    """Emit a non-inlinable helper as a CONTAINED Fortran subroutine whose return
    value comes back through a trailing out-param (Fortran functions are avoided
    -- the out-param mirrors the C scheme, per the chosen design). A scalar
    return synthesises a scalar ``intent(out)`` param; an array return reuses the
    out-param already present in ``input_args``."""
    hkir = _rename_helper_to_fortran_safe(hkir)
    name = _fortran_safe(hkir.kernel_name)
    sym_by = {s.name: s for s in hkir.symbols}
    arr_by = {a.name: a for a in hkir.arrays}
    sca_by = {s.name: s for s in hkir.scalars}
    if hkir.return_kind == "scalar":
        ret_name = "hret_"
        ret_dtype = "int64" if _helper_returns_int(hkir) else "float64"
        ret_decl = f"{_fortran_type(ret_dtype)}, intent(out) :: {ret_name}"
        param_names = [_fortran_safe(p) for p in hkir.input_args] + [ret_name]
    else:
        ret_name = _fortran_safe(hkir.return_kind)
        ret_decl = None
        param_names = [_fortran_safe(p) for p in hkir.input_args]
    # Names the helper body reassigns need their ``intent(in)`` relaxed (a value
    # dummy on the LHS) -- same rule as the top-level kernel. Collect from the
    # already-safe-renamed helper tree so the names line up with ``sca_by``.
    hassigned: set = set()
    for n in ast.walk(hkir.tree):
        if isinstance(n, ast.Assign):
            hassigned.update(t.id for t in n.targets if isinstance(t, ast.Name))
        elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
            hassigned.add(n.target.id)
    # Symbols / int scalars declared before the arrays that use them as bounds.
    sym_decls: List[str] = []
    sca_int_decls: List[str] = []
    arr_decls: List[str] = []
    sca_real_decls: List[str] = []
    for orig in hkir.input_args:
        safe = _fortran_safe(orig)
        if orig in sym_by:
            sym_decls.append(_symbol_decl(safe, assigned=safe in hassigned))
        elif orig in sca_by:
            sca = sca_by[orig]
            d = _scalar_decl(safe, sca.dtype, sca.is_output, assigned=safe in hassigned)
            (sca_int_decls if sca.dtype in ("int64", "int32", "int") else sca_real_decls).append(d)
        elif orig in arr_by:
            a = arr_by[orig]
            arr_decls.append(
                _array_decl(ArrayDesc(name=_fortran_safe(orig), dtype=a.dtype, shape=a.shape, is_output=a.is_output)))
    decls = sym_decls + sca_int_decls + arr_decls + sca_real_decls
    if ret_decl:
        decls.append(ret_decl)
    # Local arrays the lowering harvested inside the helper (np.where / np.maximum
    # temps like ``x_cb1``, the ``gf`` / ``nonsing`` work arrays) need explicit
    # fixed-shape declarations -- a contained subroutine has no enclosing scope to
    # inherit them from. Shapes are reversed for the C-interop (col-major) layout,
    # matching :func:`_array_decl`; the dtype comes from the lowering's
    # ``local_dtypes`` (a bool mask -> ``logical``, a complex temp -> ``complex``).
    rk = {"float32": "c_float", "float16": "c_float"}.get(
        dtypes.compute_dtype(hkir.float_precision or "float64"), "c_double")
    default_real = f"real({rk})"
    ldt = hkir.local_dtypes
    param_set = set(hkir.input_args)
    local_arr_decls: List[str] = []
    for lname, lshape in hkir.zeros_locals.items():
        if lname in param_set:
            continue
        rev = [_to_fortran_shape_token(s) for s in reversed(lshape)] if lshape else ["1"]
        dt = ldt.get(lname)
        ftype = _fortran_type(dt) if dt else default_real
        local_arr_decls.append(f"{ftype} :: {lname}({', '.join(rev)})")
    implicit = _collect_implicit_locals(hkir)
    local_decls = local_arr_decls + [f"{ft} :: {nm}" for nm, ft in implicit]
    iter_vars = _collect_for_targets(hkir.tree.body)
    iter_decls = [f"{_fortran_type('int')} :: " + ", ".join(sorted(iter_vars))] if iter_vars else []
    be = _FortranBodyEmitter(hkir)
    be.return_mode = ret_name
    body = be.emit_block(hkir.tree.body, indent="            ")
    decl_lines = "\n".join(f"        {d}" for d in decls + iter_decls + local_decls)
    # A contained helper has its own specification part: when its body emits a
    # non-finite constant (``ieee_value``) it must import ieee_arithmetic itself.
    # Host association is not relied on -- the host imports it only when ITS OWN
    # body uses a non-finite value, so a helper-only inf/nan would otherwise
    # reference ieee_value with no import anywhere.
    ieee_use = "        use, intrinsic :: ieee_arithmetic\n" if be._used_ieee else ""
    return (f"    subroutine {name}({', '.join(param_names)})\n"
            f"        use, intrinsic :: iso_c_binding\n"
            f"{ieee_use}"
            f"{decl_lines}\n{body}\n"
            f"    end subroutine {name}\n")


#: Physical-line budget for emitted Fortran. gfortran free-form rejects any
#: CODE line past column 132 under ``-Werror=line-truncation``; we wrap well
#: before that so the trailing `` &`` continuation marker also fits.
_MAX_LINE_COLS = 120
#: Highest column at which a break (the space before `` &``) may land, leaving
#: room for the appended `` &`` within :data:`_MAX_LINE_COLS`.
_WRAP_COL = 118


def _wrap_fortran_line(line: str) -> str:
    """Continue one physical Fortran free-form CODE line so no piece exceeds
    :data:`_MAX_LINE_COLS` columns.

    Free-form continues a line by ending it with `` &`` and (optionally) opening
    the continuation with a leading ``&``; the whole physical line (including the
    ``&``) must stay within 132 columns. We break at the last whitespace at or
    before :data:`_WRAP_COL` so no token / name / number is split, append `` &``
    (which re-supplies the separating space), and resume on the next line with the
    original indentation + ``&`` (so the indent is ignored and the tokens stay
    exactly one space apart). Comment lines (first non-blank char ``!``) are left
    untouched -- gfortran does not truncate a comment as code. Re-wraps the tail
    until every physical line is within budget."""
    if len(line) <= _MAX_LINE_COLS:
        return line
    stripped = line.lstrip()
    if not stripped or stripped.startswith("!"):
        return line
    indent = line[:len(line) - len(stripped)]
    cont_indent = indent + "&"
    out: List[str] = []
    rest = line
    while len(rest) > _MAX_LINE_COLS:
        # Last space at/before the wrap column, but past the indent so a break
        # never lands inside the leading whitespace. ``rfind`` end is exclusive.
        hi = min(_WRAP_COL, len(rest) - 1)
        brk = rest.rfind(" ", len(indent) + 1, hi + 1)
        if brk <= len(indent):
            # No breakable space within budget -- a single over-long token that
            # cannot be split without corrupting it; emit it whole.
            break
        out.append(rest[:brk] + " &")
        rest = cont_indent + rest[brk + 1:]
    out.append(rest)
    return "\n".join(out)


def _wrap_fortran_text(text: str) -> str:
    """Apply :func:`_wrap_fortran_line` to every physical line of ``text`` so no
    emitted CODE line exceeds the column budget (comments pass through)."""
    return "\n".join(_wrap_fortran_line(ln) for ln in text.split("\n"))


def _format_subroutine(name: str,
                       params: List[str],
                       decls: List[str],
                       iter_decls: List[str],
                       locals_block: List[str],
                       body: str,
                       interface_block: str = "",
                       use_ieee: bool = False,
                       contained: str = "") -> str:
    param_list = ", ".join(params)
    decl_block = "\n".join(f"    {d}" for d in decls)
    iter_block = "\n".join(iter_decls)
    locals_block_text = "\n".join(locals_block)
    iface = (interface_block + "\n") if interface_block else ""
    ieee_use = "    use, intrinsic :: ieee_arithmetic\n" if use_ieee else ""
    # Non-inlinable helpers are CONTAINED procedures (automatic explicit
    # interface, no bind(C) needed -- they are called only from Fortran).
    contains_block = f"contains\n{contained}" if contained else ""
    text = f"""\
subroutine {name}({param_list}) bind(C, name="{name}")
    use, intrinsic :: iso_c_binding
{ieee_use}{iface}{decl_block}
{iter_block}
{locals_block_text}
{body}
{contains_block}
end subroutine {name}
"""
    # Wrap any over-long physical line with `` &`` continuations so gfortran's
    # free-form 132-column limit (``-Werror=line-truncation``) is never hit --
    # purely physical formatting, no semantic change (see ``_wrap_fortran_line``).
    return _wrap_fortran_text(text)
