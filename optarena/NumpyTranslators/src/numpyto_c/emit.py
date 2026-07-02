"""C99 / C++ / Pluto-input emitters via a Python AST -> C walker.

Two design rules drive this emitter:

1. **1D pointers, always.** Every array parameter is emitted as a
   ``T *restrict name`` regardless of rank; multi-dim subscripts
   ``A[i, j]`` are lowered to ``A[i*N + j]`` using the array's
   declared shape symbols. Same applies in the C++ output (kept
   close to C) and the Pluto input (polycc's ``--pet`` mode handles
   1D arithmetic well enough; if we ever want multi-dim VLAs for
   Pluto specifically we can add a separate ``emit_pluto_vla``).
2. **Hand-rolled Python -> C walker, no ast.unparse.** Python-syntax
   loops and bare assignments would otherwise leak through.

Slicing (``A[1:N-1, 1:N-1] = expr``) lowers to a loop nest -- per
the design discussion: framework-side map fusion is out of scope
for this tool.
"""

import ast
import re
from typing import Dict, List, Optional, Set, Tuple

from numpyto_c.ir import ArrayDesc, KernelIR
from numpyto_common import dtypes, operators
from numpyto_common.emitter import BaseEmitter
from numpyto_common.frontend import _names_used_as_int

#: Whole-identifier matcher for scanning a shape-token string for the
#: names it references (so ``m`` matches in ``m + 1`` but not ``__mm``).
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")


def _c_type(dtype: str) -> str:
    # The dtype -> C type mapping is the single registry in numpyto_common.dtypes
    # (note: the canonical ``int`` is int64_t so index arithmetic is 64-bit).
    try:
        return dtypes.c_type(dtype)
    except KeyError:
        return "double"


#: ``u?int{8,16,32}_t`` -- the integer C types NARROWER than the int64 ABI
#: integer (matched on the registry's C-type string, not a dtype-name list).
_NARROW_INT_CT = re.compile(r"u?int(8|16|32)_t")


def _is_narrow_int(dtype: str) -> bool:
    """True for an integer dtype narrower than the int64 ABI integer -- its
    array elements are promoted to int64 on read (see ``_promote_read``)."""
    try:
        return bool(_NARROW_INT_CT.fullmatch(dtypes.c_type(dtype)))
    except KeyError:
        return False


def _array_signature(arr: ArrayDesc) -> str:
    """Every array is a 1D pointer; rank is encoded in subscript arithmetic.

    Reads ``arr.dtype`` directly -- precision is set on the IR upstream by
    ``ir.apply_precision`` (no per-emit override).
    """
    base = _c_type(arr.dtype)
    qual = "" if arr.is_output else "const "
    return f"{qual}{base} *restrict {arr.name}"


def _emit_signature(kir: KernelIR, fn_name: str) -> str:
    """Emit the C signature in ``kir.input_args`` order.

    The original OptArena JSON's ``input_args`` list controls the
    positional argument order the harness uses to call the kernel.
    Emitting in any other order would make every ctypes call swap
    array pointers with int sizes -- which is precisely the bug
    that surfaced as `llvm_auto` validation failures on s111.
    """
    parts: List[str] = []
    sym_by_name = {s.name: s for s in kir.symbols}
    arr_by_name = {a.name: a for a in kir.arrays}
    sca_by_name = {s.name: s for s in kir.scalars}
    for name in kir.param_order():
        if name in sym_by_name:
            parts.append(f"{dtypes.c_type('int')} {name}")  # int64_t (canonical)
        elif name in arr_by_name:
            parts.append(_array_signature(arr_by_name[name]))
        elif name in sca_by_name:
            sca = sca_by_name[name]
            c_ty = _c_type(sca.dtype)
            parts.append(f"{c_ty} {name}")
        else:
            raise ValueError(f"unknown parameter {name!r} in kernel {kir.kernel_name}")
    parts.append("int64_t *restrict time_ns")
    return f"void {fn_name}({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Body walker
# ---------------------------------------------------------------------------

# Operator tables live in numpyto_common.operators, keyed by target; the C
# backend reads its column. Local aliases keep the existing call sites.
_BINOP = operators.BINOP["c"]
_CMPOP = operators.CMPOP["c"]
_BOOLOP = operators.BOOLOP["c"]


class _CBodyEmitter(BaseEmitter):
    """Walk a Python AST function body and emit C99 statements.

    Tracks the per-array shape symbols so multi-D subscripts can be
    flattened to 1D arithmetic. For a 2-D array ``A[N, M]``,
    ``A[i, j]`` lowers to ``A[(i)*M + (j)]``; for 3-D ``A[N, M, K]``,
    ``A[i, j, k]`` becomes ``A[((i)*M + (j))*K + (k)]``.

    ``emit_block`` / ``emit_stmt`` come from :class:`BaseEmitter`; the C leaf
    hooks are the statement terminator and break / continue keywords. A
    ``return`` is dropped (BaseEmitter default) since OptArena kernels are void.
    """

    _STMT_TERM = ";"
    _KW_BREAK = "break;"
    _KW_CONTINUE = "continue;"

    def __init__(self, kir: KernelIR, multidim_arrays: Optional[Set[str]] = None):
        self.kir = kir
        #: Pluto path: names that have a multidimensional ``double (*A)[M][K]``
        #: VIEW declared in the prelude -- their subscripts stay multidimensional
        #: (``A[i][j][k]``) instead of flattened 1-D arithmetic, so the
        #: polyhedral analyzer sees affine array references.
        self.multidim_arrays: Set[str] = multidim_arrays or set()
        #: Pluto only: emit local arrays as multidimensional pointer-to-array.
        self.pluto: bool = False
        #: Pluto only: ``name -> "[d1][d2]"`` trailing-dim string for a local
        #: array declared as a pointer-to-array (so its deferred-malloc marker
        #: casts to the matching multidimensional pointer type).
        self.md_trailing: Dict[str, str] = {}
        self.array_shapes: Dict[str, List[str]] = {a.name: list(a.shape) for a in kir.arrays}
        zeros = getattr(kir.tree, "zeros_locals", {}) or {}
        for name, shape in zeros.items():
            self.array_shapes[name] = list(shape) if shape else ["1"]
        self._loop_iter_names: Set[str] = set()
        # Per-statement shape tracking for reassigned locals. The
        # lowering pipeline stashes a FIFO of shapes (one per
        # reassignment) on the tree; we pop the next entry every
        # time we cross a ``Name = __optarena_zeros__()`` marker in
        # source order. Lets lenet/resnet ``x = relu(...);
        # x = maxpool2d(x); ...`` emit each ``x[w0, w1, w2, w3]`` use
        # against the THEN-current shape of ``x`` instead of
        # falling through to chained ``[][]`` because
        # ``array_shapes[x]`` carries the FINAL rank-2 FC shape.
        self._reassign_shapes: Dict[str, List[Tuple[str, ...]]] = {
            k: list(v)
            for k, v in (getattr(kir.tree, "reassign_shapes", {}) or {}).items()
        }

    # ----- statement-level ------------------------------------------------

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
            lo, hi, step = (self.emit_expr(args[0]), self.emit_expr(args[1]), self.emit_expr(args[2]))
        else:
            raise NotImplementedError("range() needs 1-3 args")

        self._loop_iter_names.add(var)
        body = self.emit_block(node.body, indent + "  ")
        self._loop_iter_names.discard(var)
        # Negative step -> reverse loop, condition must be ``i > hi``.
        # Detect the sign from the AST (the emitted text may be ``(-1)``
        # from a UnaryOp, which ``startswith("-")`` would miss).
        step_node = args[2] if len(args) == 3 else None
        neg = False
        if step_node is not None:
            if isinstance(step_node, ast.UnaryOp) and isinstance(step_node.op, ast.USub):
                neg = True
            elif (isinstance(step_node, ast.Constant) and isinstance(step_node.value, (int, float))):
                neg = step_node.value < 0
            else:
                neg = step.startswith("-")
        cmp = ">" if neg else "<"
        if step == "1":
            inc = f"++{var}"
        elif step == "-1":
            inc = f"--{var}"
        else:
            inc = f"{var} += {step}"
        # Loop iterators are the int64 ABI integer (canonical ``int``), matching
        # the size symbols they range over -- a 32-bit iterator would overflow a
        # large extent and mix widths with int64 bounds.
        return (f"{indent}for ({_c_type('int')} {var} = {lo}; {var} {cmp} {hi}; {inc}) {{\n"
                f"{body}\n"
                f"{indent}}}")

    def _emit_while(self, node: ast.While, indent: str) -> str:
        body = self.emit_block(node.body, indent + "  ")
        return (f"{indent}while ({self.emit_expr(node.test)}) {{\n"
                f"{body}\n"
                f"{indent}}}")

    def _emit_if(self, node: ast.If, indent: str) -> str:
        then = self.emit_block(node.body, indent + "  ")
        chained = bool(node.orelse) and len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If)
        else_str = ""
        if node.orelse:
            else_str = self._emit_if(node.orelse[0], indent) if chained else self.emit_block(node.orelse, indent + "  ")
        # A guard whose branches are both empty (``if bad: raise ...`` after the
        # raise is dropped -- minife's ``if not np.issubdtype(...): raise``,
        # lavamd's ``if np.any(...): raise``) has no effect; drop the whole ``if``
        # so its condition (a pure validation predicate the backend may not be able
        # to emit) is never emitted.
        if not then.strip() and not else_str.strip():
            return ""
        cond = self.emit_expr(node.test)
        out = [f"{indent}if ({cond}) {{", then, f"{indent}}}"]
        if node.orelse:
            if chained:
                out.append(f"{indent}else " + else_str.lstrip())
            else:
                out.append(f"{indent}else {{")
                out.append(else_str)
                out.append(f"{indent}}}")
        return "\n".join(out)

    def _emit_assign(self, node: ast.Assign, indent: str) -> str:
        if len(node.targets) != 1:
            raise NotImplementedError("chained assignment not supported")
        target = node.targets[0]
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "__optarena_zeros__"):
            # Per-statement shape update: each marker for a reassigned
            # local advances the FIFO so subsequent multi-D subscripts
            # against this name flatten against the THEN-current shape.
            is_reassign = bool(node.value.args) and (isinstance(node.value.args[0], ast.Constant)
                                                     and node.value.args[0].value == "__reassign__")
            if isinstance(target, ast.Name):
                t = target.id
                fifo = self._reassign_shapes.get(t)
                if fifo:
                    self.array_shapes[t] = list(fifo.pop(0))
                # Deferred-malloc local: shape depends on a body-computed
                # scalar (``Q = np.empty((n, m + 1))`` after
                # ``m = min(max_iter, n)``). The NULL pointer was declared
                # at fn-top; allocate (and fill, if zeros/ones) HERE, now
                # that the scalar is in scope. Re-occurrence (a reset
                # inside a loop) re-mallocs after freeing the old buffer.
                deferred = vars(self).get("deferred_malloc_decls", {})
                if t in deferred:
                    size, c_type, fill = deferred[t]
                    # A deferred-malloc local (shape known only once a
                    # body-computed scalar is in scope) is declared NULL at
                    # fn-top. (Re)allocate ONLY when the buffer does not yet
                    # exist or its SIZE actually changes; otherwise reuse it in
                    # place. Emitting ``free()``+``malloc()`` on every marker is
                    # both wasteful and WRONG for a same-size ``__reassign__``
                    # self-assign (``x = (x - mean)/std``): the following loop
                    # reads the OLD ``x``, but a fresh ``malloc`` hands it an
                    # uninitialised buffer whose leading bytes the allocator
                    # clobbers with bookkeeping -- the resnet batchnorm
                    # corruption at element 0. ``free`` is emitted only for a
                    # genuine reallocation (a prior buffer of a different size).
                    sizes = vars(self).setdefault("_deferred_alloc_size", {})
                    prev = sizes.get(t)
                    if prev == size:
                        # Reuse in place. A reassign reads its own old values
                        # (no refill); a genuine zeros/ones reset still refills.
                        if is_reassign or fill is None:
                            return ""
                        return _zero_fill_stmt(t, size, c_type, fill, indent)
                    realloc = prev is not None
                    sizes[t] = size
                    lines = []
                    if realloc:
                        lines.append(f"{indent}free({t});")
                    # Pluto: cast to the multidimensional pointer-to-array type
                    # matching the declaration (``T (*X)[d1]``); else flat ``T*``.
                    cast = (f"({c_type} (*){self.md_trailing[t]})"
                            if t in self.md_trailing else f"({c_type} *)")
                    lines.append(f"{indent}{t} = {cast}malloc(({size}) "
                                 f"* sizeof({c_type}));")
                    if fill is not None:
                        lines.append(_zero_fill_stmt(t, size, c_type, fill, indent))
                    return "\n".join(lines)
                # Inline-declare this local here if its shape depends
                # on a loop variable that is only in scope inside this
                # block. C99 VLAs are valid in any block scope.
                inline_locals = vars(self).get("inline_local_decls", {})
                if t in inline_locals:
                    shape = inline_locals.pop(t)  # only emit decl once
                    local_dtypes = vars(self).get("local_dtypes_for_inline", {})
                    size_tokens = [f"({_c_shape_token(s)})" for s in shape] if shape else []
                    size = " * ".join(size_tokens) if size_tokens else "1"
                    default_float = vars(self.kir.tree).get("float_precision", "float64")
                    dtype_tag = local_dtypes.get(t, default_float)
                    c_type = _c_type(dtype_tag)
                    return f"{indent}{c_type} {t}[{size}];"
                # A fn-top zeros/ones local re-constructed here (a
                # ``X = np.zeros(...)`` reset, typically inside a loop)
                # must be re-filled -- it was allocated once at the top,
                # so without this it keeps the previous pass''s values.
                # A ``__reassign__`` marker, though, is followed by a loop
                # that FULLY overwrites the buffer, so re-zeroing it here
                # would corrupt a self-referential ``p = f(p)`` step
                # (bicgstab's ``p = r + beta * (p - omega * v)`` reads the
                # old p). Skip the refill for reassignments; genuine
                # ``np.zeros(...)`` resets (no sentinel) still refill.
                refill = vars(self).get("zeros_refill", {})
                if t in refill and not is_reassign:
                    size, c_type, kind = refill[t]
                    return _zero_fill_stmt(t, size, c_type, kind, indent)
            return ""  # local already declared at top of function
        # Name = Name alias: inherit the source's current shape so
        # downstream subscripts on the LHS flatten correctly.
        if (isinstance(target, ast.Name) and isinstance(node.value, ast.Name) and node.value.id in self.array_shapes):
            self.array_shapes[target.id] = list(self.array_shapes[node.value.id])
        rhs = self.emit_expr(node.value)
        lhs = self.emit_expr(target)
        return f"{indent}{lhs} = {rhs};"

    def _emit_augassign(self, node: ast.AugAssign, indent: str) -> str:
        op = _BINOP.get(type(node.op))
        if op is None:
            raise NotImplementedError(f"augmented op {type(node.op).__name__}")
        lhs = self.emit_expr(node.target)
        rhs = self.emit_expr(node.value)
        return f"{indent}{lhs} {op}= {rhs};"

    # ----- expression-level -----------------------------------------------

    def emit_expr(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool):
                return "1" if v else "0"
            if isinstance(v, int):
                return str(v)
            if isinstance(v, float):
                return repr(v)
            if isinstance(v, complex):
                # C99 ``_Complex`` literal via ``_Complex_I`` (C-only;
                # avoids the bare ``I`` macro which collides with
                # user variable names like mandelbrot''s boolean mask
                # ``I``). C++ complex requires ``std::complex`` and
                # is not currently supported by NumpyToC -- complex
                # kernels emit C-compatible code only.
                return f"({v.real!r} + {v.imag!r} * _Complex_I)"
            raise NotImplementedError(f"literal {v!r}")
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.UnaryOp):
            op = {ast.USub: "-", ast.UAdd: "+", ast.Not: "!", ast.Invert: "~"}.get(type(node.op))
            if op is None:
                raise NotImplementedError(f"unary {type(node.op).__name__}")
            return f"({op}{self.emit_expr(node.operand)})"
        if isinstance(node, ast.BinOp):
            # ``a ** b`` -> ``pow(a, b)`` (always, even for integer
            # exponents -- the compiler folds small constants and we
            # never have to second-guess promotion rules).
            #
            # Complex special cases:
            #   * a ** 2 (integer 2 on complex base) -> ``a * a`` --
            #     cheaper than any ``cpow`` invocation and avoids
            #     branch-cut precision artefacts at the principal
            #     branch.
            #   * a ** k for any other shape on a complex base ->
            #     ``cpow(a, k)`` (the macro defined in the prelude
            #     wraps ``exp(w * log(z))``).
            if isinstance(node.op, ast.Pow):
                if self._is_complex_operand(node.left):
                    if (isinstance(node.right, ast.Constant) and node.right.value == 2):
                        z = self.emit_expr(node.left)
                        return f"(({z})*({z}))"
                    return (f"cpow({self.emit_expr(node.left)}, "
                            f"{self.emit_expr(node.right)})")
                # Integer-typed operands -> ``__npb_int_pow`` (int64
                # binary-exponentiation helper). Detect via known-int
                # locals / symbols / range args. Falls back to
                # double-precision ``pow`` otherwise.
                if (self._is_int_operand(node.left) and self._is_int_operand(node.right)):
                    return (f"__npb_int_pow({self.emit_expr(node.left)}, "
                            f"{self.emit_expr(node.right)})")
                return (f"pow({self.emit_expr(node.left)}, "
                        f"{self.emit_expr(node.right)})")
            # ``a // b`` -> ``int_floor(a, b)``: C's ``/`` truncates
            # toward zero, Python's ``//`` floors toward -inf; the macro
            # in the header bridges the gap so mixed-sign operands match
            # numpy.
            if isinstance(node.op, ast.FloorDiv):
                return (f"int_floor({self.emit_expr(node.left)}, "
                        f"{self.emit_expr(node.right)})")
            # ``a % b`` -> ``python_mod(a, b)`` because Python (and numpy)
            # take the sign of the divisor; C/C++ take the sign of the
            # dividend. The macro converts.
            if isinstance(node.op, ast.Mod):
                return (f"python_mod({self.emit_expr(node.left)}, "
                        f"{self.emit_expr(node.right)})")
            # ``scalar @ scalar`` (numpy treats ``@`` between two
            # 0-D values as ordinary multiplication). Reaches emit
            # only when the matmul hoister rejected it because both
            # operand iter-extents are None -- safely lower to ``*``.
            if isinstance(node.op, ast.MatMult):
                return (f"({self.emit_expr(node.left)} * "
                        f"{self.emit_expr(node.right)})")
            op = _BINOP.get(type(node.op))
            if op is None:
                raise NotImplementedError(f"binop {type(node.op).__name__}")
            return f"({self.emit_expr(node.left)} {op} {self.emit_expr(node.right)})"
        if isinstance(node, ast.BoolOp):
            op = _BOOLOP[type(node.op)]
            parts = [self.emit_expr(v) for v in node.values]
            return "(" + f" {op} ".join(parts) + ")"
        if isinstance(node, ast.Compare):
            # Python chained comparison ``a < b < c`` means
            # ``(a < b) and (b < c)`` -- each adjacent pair compared, the
            # middle operand reused. C has no chaining, so emit an explicit
            # conjunction. (A single comparison is just the one term.)
            operands = [self.emit_expr(node.left)] + [self.emit_expr(c) for c in node.comparators]
            terms = [f"({operands[i]} {_CMPOP[type(op)]} {operands[i + 1]})" for i, op in enumerate(node.ops)]
            return terms[0] if len(terms) == 1 else "(" + " && ".join(terms) + ")"
        if isinstance(node, ast.Subscript):
            return self._emit_subscript(node)
        if isinstance(node, ast.Call):
            return self._emit_call(node)
        if isinstance(node, ast.IfExp):
            return (f"({self.emit_expr(node.test)} ? "
                    f"{self.emit_expr(node.body)} : "
                    f"{self.emit_expr(node.orelse)})")
        # ``z.real`` / ``z.imag`` accessor on a complex scalar -> ``creal``/
        # ``cimag`` (C99 <complex.h>); on a real operand ``.real`` is the value and
        # ``.imag`` is 0. Used by the complex-Hermitian eigh Jacobi.
        if isinstance(node, ast.Attribute) and node.attr in ("real", "imag"):
            x = self.emit_expr(node.value)
            if self._is_complex_operand(node.value):
                return f"creal({x})" if node.attr == "real" else f"cimag({x})"
            return f"({x})" if node.attr == "real" else "0.0"
        raise NotImplementedError(
            f"expression {type(node).__name__} "
            f"(line {getattr(node, 'lineno', '?')}): {ast.unparse(node)[:120]}")

    def _unchain_subscript(self, node: ast.Subscript) -> Tuple[ast.AST, List[str]]:
        """Collapse a subscript CHAIN ``a[i][j]...`` that bottoms out at a base
        expression into ``(base_node, [i, j, ...])`` (outermost-first index
        order). Slice fusion of a column gather (``log_emit[:, c]`` reduced to
        ``log_emit[w][c]``) produces such a chain; flattening it with the
        array's shape gives the same row-major access as a tuple index
        ``a[i, j]`` instead of an invalid ``a[i][j]`` on a flat pointer."""
        chain: List[str] = []
        cur: ast.AST = node
        while isinstance(cur, ast.Subscript):
            sl = cur.slice
            if isinstance(sl, ast.Tuple):
                chain = [self.emit_expr(e) for e in sl.elts] + chain
            else:
                chain = [self.emit_expr(sl)] + chain
            cur = cur.value
        return cur, chain

    def _emit_subscript(self, node: ast.Subscript) -> str:
        # Fold a constant-index subscript of a tuple literal: ``(n,)[0]`` -> ``n``.
        # This arises when a 1-D ``x.shape`` is substituted to its tuple form and
        # then indexed (``int(x.shape[0])`` in xsbench's ``n = x.shape[0]``).
        if isinstance(node.value, ast.Tuple) and isinstance(node.slice, ast.Constant) and isinstance(
                node.slice.value, int):
            elts = node.value.elts
            if -len(elts) <= node.slice.value < len(elts):
                return self.emit_expr(elts[node.slice.value])
        base_node, indices = self._unchain_subscript(node)
        base = self.emit_expr(base_node)
        # Flatten multi-D indexing using the array's shape symbols.
        # Row-major: index = ((i_0)*d_1 + i_1)*d_2 + i_2 + ...
        if len(indices) == 1 or not isinstance(base_node, ast.Name):
            access = base + "".join(f"[{i}]" for i in indices)
            return self._promote_read(node, access)
        shape = self.array_shapes.get(base_node.id)
        if shape is None or len(shape) != len(indices):
            # Fall back to chained [][]... if we have no shape info.
            return self._promote_read(node, base + "".join(f"[{i}]" for i in indices))
        # Pluto path: keep the access multidimensional so polycc sees an affine
        # array reference (against a ``double (*A)[d1][d2]`` view declared in the
        # prelude), rather than the flattened ``A[(i*d1+j)*d2+k]`` arithmetic.
        # Only names with a declared view qualify; flat locals stay flattened.
        if base_node.id in self.multidim_arrays:
            return self._promote_read(node, base + "".join(f"[{i}]" for i in indices))
        flat = indices[0]
        for k in range(1, len(indices)):
            # ``//`` and ``**`` in shape tokens come from Python source
            # idioms; map to C-valid form via ``_c_shape_token``. The
            # floor-div vs trunc-div distinction doesn''t matter for
            # non-negative integer extents.
            # Parenthesise the stride: a compound extent like ``J+3-1``
            # used bare would mis-associate (``(flat)*J + 3 - 1`` instead
            # of ``(flat)*(J+3-1)``) -- the hdiff 3-D-stencil OOB.
            dim = f"({_c_shape_token(shape[k])})"
            flat = f"({flat})*{dim} + ({indices[k]})"
        return self._promote_read(node, f"{base}[{flat}]")

    def _promote_read(self, node: ast.Subscript, access: str) -> str:
        """Promote a NARROW integer array element to the int64 ABI integer on
        READ (``(int64_t)(arr[i])``), so a user-supplied int32 index/value array
        never forms a mixed-width op with an int64 symbol/local. Array STORAGE
        keeps its declared width; a write (Store ctx) is untouched and an
        assignment back into the narrow array narrows implicitly."""
        base = node.value
        while isinstance(base, ast.Subscript):   # chained ``a[i][j]`` -> Name a
            base = base.value
        if (isinstance(node.ctx, ast.Load) and isinstance(base, ast.Name)
                and _is_narrow_int(self._dtype_for_name(base.id) or "")):
            return f"(({_c_type('int')})({access}))"
        return access

    def _emit_call(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            fn = node.func.id
            if fn == "__optarena_zeros__":
                return ""
            # Math intrinsic complex specialisations. ``abs / sqrt /
            # exp / log / sin / cos`` resolve to the float / double
            # overload by default which mishandles a complex operand;
            # route through the ``c*`` macros defined in the prelude.
            _COMPLEX_INTRINSIC = {
                "abs": "cabs",
                "fabs": "cabs",
                "sqrt": "csqrt",
                "exp": "cexp",
                "log": "clog",
            }
            if (fn in _COMPLEX_INTRINSIC and len(node.args) == 1 and self._is_complex_operand(node.args[0])):
                args = self.emit_expr(node.args[0])
                return f"{_COMPLEX_INTRINSIC[fn]}({args})"
            # Python ``abs(x)`` on a floating operand must be C ``fabs`` --
            # plain ``abs`` is <stdlib.h> INTEGER abs and would truncate
            # the double (s3113 / s318: ``abs(a[i])``). Integer operands
            # keep ``abs`` (``fabs`` would break an int array subscript).
            if (fn == "abs" and len(node.args) == 1 and self._is_float_operand(node.args[0])):
                return f"fabs({self.emit_expr(node.args[0])})"
            # ``pow(complex_value, K)`` -> integer-2 fast path or
            # ``cpow``. ``pow`` in C++ has no complex overload.
            if (fn == "pow" and len(node.args) == 2 and self._is_complex_operand(node.args[0])):
                if (isinstance(node.args[1], ast.Constant) and node.args[1].value == 2):
                    z = self.emit_expr(node.args[0])
                    return f"(({z})*({z}))"
                z, w = (self.emit_expr(node.args[0]), self.emit_expr(node.args[1]))
                return f"cpow({z}, {w})"
            # Python ``int(x)`` is a TYPECAST in both C and C++. C
            # syntax ``(int)expr`` works in both; ``int(expr)`` only
            # in C++. Emit the portable cast form so the C compile
            # accepts it too.
            if fn == "int" and len(node.args) == 1:
                return f"((int)({self.emit_expr(node.args[0])}))"
            # ``np.sign`` marker -> -1 / 0 / +1. C bool subtraction gives
            # the integer result, promoted to the target's type.
            if fn == "__npb_sign" and len(node.args) == 1:
                x = self.emit_expr(node.args[0])
                return f"((({x}) > 0) - (({x}) < 0))"
            # Variadic builtin ``max(a, b, c, ...)`` / ``min(...)``: the C and
            # C++ ``max``/``min`` are 2-arg macros (prelude), so fold a 3+-arg
            # call into a left-nested chain ``max(max(a, b), c)`` (needleman_
            # wunsch's 3-way recurrence). numpy reductions np.max/np.min go
            # through the lib-node path, not here.
            if fn in ("max", "min") and len(node.args) > 2:
                acc = self.emit_expr(node.args[0])
                for a in node.args[1:]:
                    acc = f"{fn}({acc}, {self.emit_expr(a)})"
                return acc
            args = ", ".join(self.emit_expr(a) for a in node.args)
            return f"{fn}({args})"
        # ``np.X(arg)`` / ``arr.X(...)`` -- handle a small set of
        # passthrough / identity intrinsics that survived lowering.
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            # ``np.<dtype>(x)`` scalar constructor (``np.int64(0)`` /
            # ``np.float64(s)`` / ``np.bool_(b)`` ...) is a TYPECAST. Emit the
            # C cast to that dtype's C type, resolved through the dtype registry
            # so no width string is hardcoded. ``np.bool_`` carries a trailing
            # underscore; strip it before the registry lookup.
            if (isinstance(node.func.value, ast.Name) and node.func.value.id == "np"
                    and len(node.args) == 1):
                key = attr[:-1] if attr.endswith("_") else attr
                if key in dtypes.REGISTRY or key in dtypes.SCALAR_KINDS:
                    return f"(({dtypes.c_type(key)})({self.emit_expr(node.args[0])}))"
            # ``np.flip(scalar)`` / ``np.copy(scalar)`` / ``np.transpose(scalar)``
            # on a scalar Subscript is a no-op; this happens when the
            # slice-fusion lifts e.g. ``np.flip(r[:k])`` into a per-
            # element loop where the operand becomes ``r[i]``.
            if attr in {"flip", "copy", "transpose"} and len(node.args) == 1:
                return self.emit_expr(node.args[0])
            # ``z.conjugate()`` / ``z.conj()`` -- complex conjugate on a
            # scalar value. Routes to ``conj(z)`` in both C and C++ via
            # the ``__npb_make_complex(creal(z), -cimag(z))`` macro
            # provided in the prelude.
            if attr in {"conjugate", "conj"} and not node.args:
                z = self.emit_expr(node.func.value)
                return f"__npb_conj({z})"
            # ``np.conj(z)`` / ``np.conjugate(z)`` -- function form (vexx
            # ``np.conj(exxbuff)``, scalarised to a per-element operand).
            if (isinstance(node.func.value, ast.Name)
                    and node.func.value.id in ("np", "numpy")
                    and attr in {"conj", "conjugate"} and len(node.args) == 1):
                return f"__npb_conj({self.emit_expr(node.args[0])})"
            # ``z.real`` / ``z.imag`` -- accessor on a complex scalar.
            # (Not reached for an Attribute Load, only Call here, but
            # kept symmetric for an Attribute-call form.)
            # ``np.where(cond, a, b)`` in scalar context (inside an
            # already-scalarised per-element body) -- numpy semantics
            # are ``(a if cond else b)`` per element. Lower to the C
            # ternary so hdiff's ``out[i,j,k] = np.where(...)`` body
            # compiles.
            if attr == "where" and len(node.args) == 3:
                c = self.emit_expr(node.args[0])
                a = self.emit_expr(node.args[1])
                b = self.emit_expr(node.args[2])
                return f"({c} ? {a} : {b})"
            # ``np.sign(x)`` in scalar context -> -1 / 0 / +1 (numpy's
            # convention: sign(0) == 0). Same inline as the array ``__npb_sign``
            # marker. cloudsc's ``max(0.0, 1.0 * np.sign(ztp1[..] - rtt))``.
            if (isinstance(node.func.value, ast.Name)
                    and node.func.value.id in ("np", "numpy")
                    and attr == "sign" and len(node.args) == 1):
                x = self.emit_expr(node.args[0])
                return f"((({x}) > 0) - (({x}) < 0))"
            # ``np.abs(x)`` in scalar context (whole-array ``np.abs(arr)`` is
            # scalarised to per-element form by lowering): complex -> ``cabs``,
            # float -> ``fabs`` (plain int ``abs`` would truncate a double),
            # integer -> ``abs``. Mirrors the builtin ``abs`` handling above.
            if (isinstance(node.func.value, ast.Name)
                    and node.func.value.id in ("np", "numpy")
                    and attr in ("abs", "absolute", "fabs") and len(node.args) == 1):
                x = node.args[0]
                if self._is_complex_operand(x):
                    return f"cabs({self.emit_expr(x)})"
                if attr == "fabs" or self._is_float_operand(x):
                    return f"fabs({self.emit_expr(x)})"
                return f"abs({self.emit_expr(x)})"
            # ``np.hypot(a, b)`` -> C99 ``hypot`` (both operands real -- the
            # eigh Jacobi's ``np.hypot(z.real, z.imag)`` = |z|).
            if (isinstance(node.func.value, ast.Name) and node.func.value.id in ("np", "numpy")
                    and attr == "hypot" and len(node.args) == 2):
                return f"hypot({self.emit_expr(node.args[0])}, {self.emit_expr(node.args[1])})"
        raise NotImplementedError(f"call to {ast.unparse(node.func)} not supported")

    def _is_int_operand(self, node: ast.AST) -> bool:
        """Conservative int-typed operand detection.

        Returns True for: integer ``Constant``, Names that resolve to
        a symbol / int local / declared ``int`` parameter, and
        BinOp / UnaryOp whose subtree contains only int operands.
        """
        if isinstance(node, ast.Constant):
            return isinstance(node.value, int) and not isinstance(node.value, bool)
        if isinstance(node, ast.Name):
            n = node.id
            # Kernel symbols are always int.
            for s in self.kir.symbols:
                if s.name == n:
                    return True
            # ``int_locals`` are tuple-unpack int locals.
            int_locals = getattr(self.kir.tree, "int_locals", []) or []
            if n in int_locals:
                return True
            # For-loop iter names are always declared ``int`` in the
            # emitted C; recognise them so ``R ** i`` for a loop iter
            # routes through ``__npb_int_pow`` instead of
            # double-precision ``pow``.
            if n in self._loop_iter_names:
                return True
            # ``M_PI`` / ``M_E`` / ``INFINITY`` / ``NAN`` are math
            # macros, NOT int (we don''t want ``__npb_int_pow(M_PI, K)``).
            if n in {"M_PI", "M_E", "INFINITY", "NAN"}:
                return False
            # Implicit int scalar locals (those flagged via the
            # ``needs_int`` promotion path in _collect_implicit_locals).
            if n in self._all_int_locals():
                return True
            return False
        if isinstance(node, ast.BinOp):
            return (self._is_int_operand(node.left) and self._is_int_operand(node.right))
        if isinstance(node, ast.UnaryOp):
            return self._is_int_operand(node.operand)
        return False

    def _all_int_locals(self) -> Set[str]:
        """Cached set of all locals known to be ``int`` -- ``int``
        kernel scalars + tuple-unpack ``int_locals`` + the
        implicit-locals path's ``needs_int`` promotions."""
        cached = getattr(self, "_int_locals_cache", None)
        if cached is not None:
            return cached
        out: Set[str] = set()
        for s in self.kir.scalars:
            if s.dtype in {"int", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}:
                out.add(s.name)
        # ``needs_int`` collected by _names_used_as_int -- any Name
        # used as an array subscript / range arg / bitwise op operand.
        out.update(_names_used_as_int(self.kir.tree))
        self._int_locals_cache = out
        return out

    def _is_complex_operand(self, node: ast.AST) -> bool:
        """Return True when ``node``''s element dtype is a complex form.

        Recognises:
        * Direct ``Subscript(Name(...))`` against a known-complex array.
        * BinOp / UnaryOp / Call whose AST contains ANY complex literal
          (``Constant(complex)``) or Name reference resolving to a
          complex-dtype array. Walks the whole subtree so a deeper
          ``exp(BinOp_with_complex_literal)`` is recognised.
        """
        # Walk every Subscript / Constant in the node and short-circuit
        # on the first match.
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, complex):
                return True
            if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name):
                dt = self._dtype_for_name(sub.value.id)
                if dt and dt.startswith("complex"):
                    return True
            if isinstance(sub, ast.Name):
                dt = self._dtype_for_name(sub.id)
                if dt and dt.startswith("complex"):
                    return True
        return False

    def _is_float_operand(self, node: ast.AST, _scalars=None) -> bool:
        """Return True when ``node`` is provably floating-point: a float
        Constant, or a Subscript/Name resolving to a float-dtype array or
        local. Used to route ``abs`` -> ``fabs``. Unknown -> False (keep
        ``abs``) so an integer index ``abs(k)`` is never turned into a
        ``fabs`` that would make ``arr[fabs(k)]`` a non-integer subscript.

        ``_scalars`` is the in-progress float-scalar-local set during fixpoint
        computation; callers leave it ``None`` to use the cached result."""
        scalars = self._float_scalar_names() if _scalars is None else _scalars
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, float):
                return True
            if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name):
                dt = self._dtype_for_name(sub.value.id)
                if dt and dt.startswith("float"):
                    return True
            if isinstance(sub, ast.Name):
                dt = self._dtype_for_name(sub.id)
                if dt and dt.startswith("float"):
                    return True
                if sub.id in scalars:
                    return True
        return False

    def _float_scalar_names(self) -> set:
        """Body-computed scalar locals that hold a float value (``zc =
        z_w_con_c[...]``; ``vcfl = zc * dtime / h``). These are absent from the
        array / local_dtypes tables, so ``abs()`` on them would otherwise stay
        C's INTEGER abs and truncate the double. Inferred to a fixpoint: a bare
        ``name = <expr>`` makes ``name`` float when the RHS is float-provable
        (using the floats found so far)."""
        cache = vars(self).get("_fsn_cache")
        if cache is not None:
            return cache
        floats: set = set()
        for _ in range(8):                          # small fixpoint
            changed = False
            for node in ast.walk(self.kir.tree):
                if (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id not in floats
                        and self._is_float_operand(node.value, floats)):
                    floats.add(node.targets[0].id)
                    changed = True
            if not changed:
                break
        self._fsn_cache = floats
        return floats

    def _dtype_for_name(self, name: str):
        local_dtypes = getattr(self.kir.tree, "local_dtypes", {}) or {}
        dt = local_dtypes.get(name)
        if dt is None:
            for a in self.kir.arrays:
                if a.name == name:
                    return a.dtype
        return dt

    def _is_complex_subscript_legacy(self, node: ast.AST) -> bool:
        """Retained for the original Subscript(Name) shortcut."""
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            name = node.value.id
            local_dtypes = getattr(self.kir.tree, "local_dtypes", {}) or {}
            dt = local_dtypes.get(name)
            if dt is None:
                for a in self.kir.arrays:
                    if a.name == name:
                        dt = a.dtype
                        break
            if dt is not None and dt.startswith("complex"):
                return True
        return False


# ---------------------------------------------------------------------------
# Top-level emitters
# ---------------------------------------------------------------------------


def _c_shape_token(tok: str) -> str:
    """Translate a Python shape token to a C-valid integer expression.

    * ``//`` -> ``/`` -- floor-div / trunc-div agree on non-negative
      integer extents.
    * ``a ** b`` -> ``__npb_int_pow(a, b)`` -- integer power. The
      helper is defined in the C / C++ preludes.
    """
    out = str(tok).replace("//", "/")
    # ``**`` -> ``__npb_int_pow(a, b)`` -- match the textual form
    # left-to-right; nested forms ``a ** b ** c`` are right-assoc in
    # Python and the helper preserves that via the textual recursion.
    while "**" in out:
        idx = out.index("**")
        # Find the base token: walk left over an identifier-or-paren
        # chain.
        i = idx - 1
        while i >= 0 and out[i] == " ":
            i -= 1
        if i < 0:
            break
        # Identifier / digit run on the left.
        if out[i] == ")":
            depth = 1
            j = i - 1
            while j >= 0 and depth > 0:
                if out[j] == ")":
                    depth += 1
                elif out[j] == "(":
                    depth -= 1
                j -= 1
            base_start = j + 1
        else:
            j = i
            while j >= 0 and (out[j].isalnum() or out[j] == "_"):
                j -= 1
            base_start = j + 1
        base = out[base_start:i + 1]
        # Right side: number / identifier / paren run.
        k = idx + 2
        while k < len(out) and out[k] == " ":
            k += 1
        if k < len(out) and out[k] == "(":
            depth = 1
            m = k + 1
            while m < len(out) and depth > 0:
                if out[m] == "(":
                    depth += 1
                elif out[m] == ")":
                    depth -= 1
                m += 1
            exp = out[k:m]
            exp_end = m
        else:
            m = k
            while m < len(out) and (out[m].isalnum() or out[m] == "_"):
                m += 1
            exp = out[k:m]
            exp_end = m
        out = out[:base_start] + f"__npb_int_pow({base}, {exp})" + out[exp_end:]
    return out


def _collect_implicit_locals(kir: KernelIR) -> List[Tuple[str, str]]:
    """Return ``(name, c_type)`` pairs for scalar locals needing a C decl.

    Any ``Name = expr`` in the body that is not already a parameter,
    not an ``int_local`` produced by tuple-unpack lowering, and not a
    local array (``np.zeros``) is implicit. The type is inferred in
    priority order:

    1. ``tree.local_dtypes`` (populated by the lowering pipeline for
       loop-vars that inherit the source array's element dtype, e.g.
       ``for b in data:`` where ``data`` is ``uint8``).
    2. Used-as-int promotion (subscript / range / bitwise operand).
    3. Default to ``double``.
    """
    declared: Set[str] = set()
    declared.update(kir.input_args)
    declared.update(getattr(kir.tree, "int_locals", []) or [])
    declared.update((getattr(kir.tree, "zeros_locals", {}) or {}).keys())
    local_dtypes = getattr(kir.tree, "local_dtypes", {}) or {}
    out: List[Tuple[str, str]] = []
    needs_int = _names_used_as_int(kir.tree)
    seen: Set[str] = set(declared)
    # Per-array element-dtype map for ``Name = Subscript(arr, scalar)``
    # inheritance (``x = data[i]`` where data is uint8 => x is uint8).
    array_dtypes = {a.name: a.dtype for a in kir.arrays}

    def _ctype_for(name: str, value: Optional[ast.AST] = None) -> str:
        # Highest priority: explicit dtype from the lowering pipeline.
        if name in local_dtypes:
            return _c_type(local_dtypes[name])
        # ``needs_int`` -- used as a subscript or range arg -- takes
        # precedence over the source-array-dtype inheritance below
        # because using a float as an array subscript is a hard C
        # error (``s4114``: ``k = ip[i]; c[LEN_1D - k - 1]``).
        if name in needs_int:
            return "int"
        # ``x = arr[i]`` (scalar Subscript on a Name with known dtype).
        if value is not None and isinstance(value, ast.Subscript) \
                and isinstance(value.value, ast.Name):
            src_dt = array_dtypes.get(value.value.id) or local_dtypes.get(value.value.id)
            if src_dt is not None:
                return _c_type(src_dt)
        return "double"

    for node in ast.walk(kir.tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id not in seen:
                    out.append((tgt.id, _ctype_for(tgt.id, node.value)))
                    seen.add(tgt.id)
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name) and node.target.id not in seen:
                out.append((node.target.id, _ctype_for(node.target.id)))
                seen.add(node.target.id)
    return out


def _zero_fill_stmt(name: str, size: str, c_type: str, kind: str, indent: str) -> str:
    """C statement that fills ``name[0:size]`` per the numpy constructor
    ``kind``: ``ones``/``ones_like`` -> 1, anything else (zeros/
    zeros_like) -> ``memset`` to 0. Used both for the initial fill of a
    fresh fn-top local and to RE-fill it at a ``np.zeros`` marker that
    re-occurs inside a loop (an in-loop accumulator reset)."""
    if kind in ("ones", "ones_like"):
        return (f"{indent}for (int64_t __zf = 0; __zf < ({size}); ++__zf) "
                f"{name}[__zf] = 1;")
    return f"{indent}memset({name}, 0, ({size}) * sizeof({c_type}));"


def _md_trailing(shape) -> str:
    """``[d1][d2]...`` -- the trailing dimensions of a pointer-to-array view
    (the leading dimension is implicit in the pointer). Empty for rank<2."""
    return "".join(f"[{_c_shape_token(d)}]" for d in shape[1:])


def _emit_body(kir: KernelIR, indent: str = "  ",
               multidim_arrays: Optional[Set[str]] = None,
               pluto: bool = False, return_parts: bool = False):
    emitter = _CBodyEmitter(kir, multidim_arrays=multidim_arrays)
    emitter.pluto = pluto
    zeros = getattr(kir.tree, "zeros_locals", {}) or {}
    zeros_fills = vars(kir.tree).get("zeros_fills", {}) or {}
    int_locals = getattr(kir.tree, "int_locals", []) or []
    implicit = _collect_implicit_locals(kir)
    local_dtypes = getattr(kir.tree, "local_dtypes", {}) or {}
    # Output parameters that a ``np.zeros``/``np.empty`` in the kernel
    # body aliases (e.g. ``table = np.zeros((N, N)); return table``). These
    # must NOT get a fresh local declaration -- that would shadow the
    # caller's buffer, so every write lands in scratch that is freed on
    # return and the caller sees an uninitialised result. Instead we
    # initialise the parameter in place (see ``param_inits`` below).
    params = set(kir.param_order())
    arr_by_name = {a.name: a for a in kir.arrays}
    # Collect the set of for-loop iter names. Any zeros_local whose
    # shape contains one of these as a free token must be allocated
    # INLINE at its marker site (inside the loop scope where the iter
    # is bound) rather than hoisted to function-top -- C99 VLAs are
    # allowed in any block scope.
    loop_iters: Set[str] = set()
    for node in ast.walk(kir.tree):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            loop_iters.add(node.target.id)

    def _shape_uses_loop_iter(shape) -> bool:
        for tok in shape:
            for it in loop_iters:
                # Free-token match: ``it`` must appear as a whole word.
                t = str(tok)
                idx = t.find(it)
                while idx >= 0:
                    lo = idx == 0 or not (t[idx - 1].isalnum() or t[idx - 1] == "_")
                    hi = (idx + len(it) >= len(t) or not (t[idx + len(it)].isalnum() or t[idx + len(it)] == "_"))
                    if lo and hi:
                        return True
                    idx = t.find(it, idx + 1)
        return False

    # Scalar locals COMPUTED in the body (``m = min(max_iter, n)`` in
    # gmres, ``__inl3_H_out = ...`` from conv2d inlining). A local array
    # whose malloc size references one of these CANNOT be allocated at
    # function-top -- the scalar is still uninitialised there, so the size
    # evaluates to garbage (malloc(0) -> 1-byte region -> heap overflow on
    # first write). Such arrays defer their malloc to the body marker site,
    # which always follows the scalar's assignment in straight-line order.
    computed_scalars: Set[str] = {n for n, _ in implicit} | set(int_locals)

    def _shape_uses_computed_scalar(shape) -> bool:
        for tok in shape:
            t = str(tok)
            for m in _IDENT_RE.findall(t):
                if m in computed_scalars:
                    return True
        return False

    inline_locals: Dict[str, Tuple[str, ...]] = {}
    deferred_malloc_locals: Dict[str, Tuple[str, ...]] = {}
    fn_top_locals: Dict[str, Tuple[str, ...]] = {}
    param_inits: Dict[str, Tuple[str, ...]] = {}
    for name, shape in zeros.items():
        if name in params:
            param_inits[name] = shape  # alias of an output buffer
        elif _shape_uses_loop_iter(shape):
            inline_locals[name] = shape
        elif _shape_uses_computed_scalar(shape):
            deferred_malloc_locals[name] = shape
        else:
            fn_top_locals[name] = shape
    emitter.inline_local_decls = inline_locals
    emitter.local_dtypes_for_inline = local_dtypes
    # Pluto only: local rank>=2 arrays are declared as pointer-to-array so the
    # scop body indexes them ``X[i][j]`` (affine) instead of ``X[i*M+j]``. Mark
    # them multidim (drives _emit_subscript) and record their trailing dims (for
    # the deferred-malloc marker's cast). pluto=False leaves this empty -- the
    # C / C++ backends are byte-for-byte unchanged.
    md_locals: Set[str] = set()
    if pluto:
        for _nm, _shp in (list(fn_top_locals.items())
                          + list(deferred_malloc_locals.items())
                          + list(inline_locals.items())):
            if len(_shp) >= 2:
                md_locals.add(_nm)
                emitter.md_trailing[_nm] = _md_trailing(_shp)
        emitter.multidim_arrays = set(emitter.multidim_arrays) | md_locals
    # Default dtype for a float temp not listed in local_dtypes (e.g. a
    # matmul scratch) follows the kernel's float precision set on the IR.
    default_float = vars(kir.tree).get("float_precision", "float64")
    # Register each local ARRAY's resolved dtype. Source-level float
    # locals (gmres ``Q`` / ``H`` / ``e1`` from ``np.zeros``) are declared
    # as the default float but never tagged in ``local_dtypes``; without a
    # tag ``_is_float_operand`` cannot prove ``H[...]`` is float and
    # ``abs(H[...])`` stays C's INTEGER ``abs`` -- truncating the double to
    # 0 and triggering a spurious early break. ``setdefault`` never
    # overrides an explicit int / complex tag.
    for name in (*fn_top_locals, *deferred_malloc_locals, *inline_locals):
        local_dtypes.setdefault(name, default_float)
    kir.tree.local_dtypes = local_dtypes
    decls: List[str] = []
    frees: List[str] = []
    for name in int_locals:
        decls.append(f"{indent}int {name};")
    for name, ctype in implicit:
        decls.append(f"{indent}{ctype} {name};")
    # Fresh ``np.zeros``/``np.ones`` locals need an initial fill (a bare
    # malloc leaves garbage). Output-param aliases are handled separately
    # in ``param_inits``; here we fill the non-param fn-top locals. The
    # ``zeros_refill`` table is also handed to the emitter so a
    # ``X = np.zeros(...)`` marker that re-occurs INSIDE a loop re-zeros
    # the (top-allocated) buffer each iteration -- without it an in-loop
    # accumulator like contour_integral''s ``Tz`` accrues across passes.
    zeros_refill: Dict[str, Tuple[str, str, str]] = {}
    for name, shape in fn_top_locals.items():
        size_tokens = [f"({_c_shape_token(s)})" for s in shape] if shape else []
        size = " * ".join(size_tokens) if size_tokens else "1"
        dtype_tag = local_dtypes.get(name, default_float)
        c_type = _c_type(dtype_tag)
        # A symbolic-sized local (e.g. matmul temp ``__mm1[NI * NJ]``) is
        # heap-allocated: at polybench sizes (NI=1000) a stack VLA would
        # be ~megabytes and overflow the stack. The array is already
        # flattened row-major so pointer indexing ``name[i*NJ + j]`` is
        # unchanged. A literal-sized local stays a stack array.
        if name in md_locals:
            # Pluto: pointer-to-array (heap) so ``name[i][j]`` is affine.
            tr = emitter.md_trailing[name]
            decls.append(f"{indent}{c_type} (*{name}){tr} = "
                         f"({c_type} (*){tr})malloc(({size}) * sizeof({c_type}));")
            frees.append(f"{indent}free({name});")
        elif any(c.isalpha() for c in size):
            decls.append(f"{indent}{c_type} *{name} = "
                         f"({c_type} *)malloc(({size}) * sizeof({c_type}));")
            frees.append(f"{indent}free({name});")
        else:
            decls.append(f"{indent}{c_type} {name}[{size}];")
        # Only fill locals EXPLICITLY built by a zeros/ones constructor;
        # ``empty``-kind and matmul/scratch temps (no recorded kind) are
        # fully written by the kernel, so skip them.
        kind = zeros_fills.get(name)
        if kind is None or kind in ("empty", "empty_like", "ndarray"):
            continue
        zeros_refill[name] = (size, c_type, kind)
        decls.append(_zero_fill_stmt(name, size, c_type, kind, indent))
    # Deferred-malloc locals: shape depends on a body-computed scalar, so
    # only the NULL pointer is declared at fn-top; the malloc itself is
    # emitted at the marker (after the scalar is assigned) by the body
    # emitter, and the buffer is freed at function end. Keeps pointer
    # semantics (unlike a VLA, so aliases / later reads work).
    deferred_specs: Dict[str, Tuple[str, str, Optional[str]]] = {}
    for name, shape in deferred_malloc_locals.items():
        size_tokens = [f"({_c_shape_token(s)})" for s in shape] if shape else []
        size = " * ".join(size_tokens) if size_tokens else "1"
        dtype_tag = local_dtypes.get(name, default_float)
        c_type = _c_type(dtype_tag)
        if name in md_locals:
            decls.append(f"{indent}{c_type} (*{name}){emitter.md_trailing[name]} = NULL;")
        else:
            decls.append(f"{indent}{c_type} *{name} = NULL;")
        frees.append(f"{indent}free({name});")
        kind = zeros_fills.get(name)
        fill = None if (kind is None or kind in ("empty", "empty_like", "ndarray")) else kind
        deferred_specs[name] = (size, c_type, fill)
    emitter.deferred_malloc_decls = deferred_specs
    emitter.zeros_refill = zeros_refill
    # Initialise output-parameter aliases in place (no shadowing local).
    # The element type MUST match the parameter's signature type (so the
    # memset byte-count is right), not the numpy local dtype.
    for name, shape in param_inits.items():
        size_tokens = [f"({_c_shape_token(s)})" for s in shape] if shape else []
        size = " * ".join(size_tokens) if size_tokens else "1"
        arr = arr_by_name.get(name)
        c_type = _c_type(arr.dtype if arr else default_float)
        kind = zeros_fills.get(name, "zeros")
        if kind in ("empty", "empty_like", "ndarray"):
            continue  # caller buffer, kernel writes all
        if kind in ("ones", "ones_like"):
            decls.append(f"{indent}for (int64_t __i = 0; __i < ({size}); ++__i) "
                         f"{name}[__i] = 1;")
        else:  # zeros / zeros_like / default
            decls.append(f"{indent}memset({name}, 0, ({size}) * sizeof({c_type}));")
    body = emitter.emit_block(kir.tree.body, indent)
    if return_parts:
        # Pluto: keep allocations / frees OUT of the loop body so the caller can
        # place them outside ``#pragma scop`` (malloc/free are non-affine and
        # break the polyhedral SCoP).
        return ("\n".join(d for d in decls if d), body,
                "\n".join(f for f in frees if f))
    return "\n".join(d for d in (*decls, body, *frees) if d)


_C_HEADER = ("#define _POSIX_C_SOURCE 199309L\n"
             "#define _USE_MATH_DEFINES\n"
             "#include <time.h>\n"
             "#include <stdint.h>\n"
             "#include <stdlib.h>\n"
             "#include <stdbool.h>\n"
             "#include <string.h>\n"
             "#include <math.h>\n"
             "#include <complex.h>\n"
             "/* ``z.conjugate()`` -- portable complex-conjugate scalar\n"
             " * helper. Inline static so callers see the same signature\n"
             " * in C and C++. */\n"
             "static inline double _Complex __npb_conj(double _Complex z) {\n"
             "    return __builtin_complex(__real__ z, -__imag__ z);\n"
             "}\n"
             "/* M_PI / M_E etc. are POSIX/GNU extensions -- ensure they\n"
             " * are defined even on strict-C builds (glibc 2.27+ /\n"
             " * BSDs / MSVC). */\n"
             "#ifndef M_PI\n#define M_PI 3.14159265358979323846\n#endif\n"
             "#ifndef M_E\n#define M_E 2.71828182845904523536\n#endif\n"
             "/* ``<complex.h>`` defines ``I`` as the imaginary unit;\n"
             " * undef it so user variable names like ``I`` (mandelbrot\n"
             " * boolean mask) don''t collide. Complex literals continue\n"
             " * to use the portable ``_Complex_I`` form. */\n"
             "#ifdef I\n#undef I\n#endif\n"
             "/* Operand order matches Python's builtin ``max``/``min``: the\n"
             " * result is the SECOND arg only when it strictly wins, else the\n"
             " * FIRST -- so a NaN first operand propagates (``max(nan, x) ==\n"
             " * nan``) exactly as Python/numpy do, rather than the\n"
             " * NaN-suppressing ``fmax`` semantics a naive ``a > b`` gives. */\n"
             "#ifndef min\n"
             "#define min(a, b) (((b) < (a)) ? (b) : (a))\n"
             "#endif\n"
             "#ifndef max\n"
             "#define max(a, b) (((b) > (a)) ? (b) : (a))\n"
             "#endif\n"
             "/* Python ``//`` floor-toward-neg-inf vs C trunc-toward-zero;\n"
             " * matches numpy ``//`` for both same- and mixed-sign inputs. */\n"
             "#ifndef int_floor\n"
             "#define int_floor(a, b) ((a)/(b) - (((a)%(b)!=0) && (((a)<0)^((b)<0))))\n"
             "#endif\n"
             "/* Python ``%`` returns sign of divisor; C returns sign of dividend. */\n"
             "#ifndef python_mod\n"
             "#define python_mod(a, b) (((a) % (b) + (b)) % (b))\n"
             "#endif\n"
             "/* Integer power for VLA shape bounds like ``R ** K``. */\n"
             "static inline int64_t __npb_int_pow(int64_t base, int64_t exp) {\n"
             "    int64_t result = 1;\n"
             "    while (exp > 0) {\n"
             "        if (exp & 1) result *= base;\n"
             "        base *= base;\n"
             "        exp >>= 1;\n"
             "    }\n"
             "    return result;\n"
             "}\n")

# C++ prelude: ZERO preprocessor macros -- every constant is a ``constexpr``
# value and every helper a ``constexpr`` (compile-and-runtime) or ``inline``
# (libm-backed) function. ``consteval`` is deliberately NOT used for max / min /
# int_floor / python_mod: those are called with RUNTIME arguments, and a
# consteval function may only be invoked in a constant-expression context, so it
# would fail to compile every runtime call. ``constexpr`` gives the same
# compile-time evaluability where the args happen to be constant while still
# permitting runtime use. ``<complex.h>`` is dropped (its C99 ``creal`` / ``cabs``
# / ... declarations would clash with our own definitions); ``double _Complex``
# is a GCC/Clang extension available without it.
_CPP_HEADER = ('#include <chrono>\n#include <cstdint>\n#include <cmath>\n'
               '#include <cstring>\n'
               '// Math constants as typed constexpr values. ``<cmath>`` may\n'
               '// predefine M_PI / M_E as macros (glibc __USE_MISC); undefine\n'
               '// them so the names rebind to our constexpr values -- we emit no\n'
               '// macro DEFINITION, only remove the platform ones.\n'
               '#ifdef M_PI\n#undef M_PI\n#endif\n'
               '#ifdef M_E\n#undef M_E\n#endif\n'
               'constexpr double M_PI = 3.14159265358979323846;\n'
               'constexpr double M_E  = 2.71828182845904523536;\n'
               '// Complex support via the GCC/Clang ``double _Complex`` extension\n'
               '// (no <complex.h>, so no name clashes). The imaginary unit and\n'
               '// the C99-named helpers are constexpr/inline FUNCTIONS, not macros.\n'
               'constexpr double creal(double _Complex z) { return __real__ z; }\n'
               'constexpr double cimag(double _Complex z) { return __imag__ z; }\n'
               'inline double _Complex __npb_make_complex(double re, double im) {\n'
               '    double _Complex z; __real__ z = re; __imag__ z = im; return z;\n'
               '}\n'
               'static const double _Complex _Complex_I = __npb_make_complex(0.0, 1.0);\n'
               'inline double cabs(double _Complex z) {\n'
               '    return sqrt(creal(z)*creal(z) + cimag(z)*cimag(z));\n'
               '}\n'
               'inline double carg(double _Complex z) { return atan2(cimag(z), creal(z)); }\n'
               '/* ``cexp(z) = exp(re) * (cos(im) + i*sin(im))``. */\n'
               'inline double _Complex cexp(double _Complex z) {\n'
               '    return __npb_make_complex(exp(creal(z))*cos(cimag(z)),\n'
               '                             exp(creal(z))*sin(cimag(z)));\n'
               '}\n'
               '/* ``clog(z) = log(|z|) + i*arg(z)``. */\n'
               'inline double _Complex clog(double _Complex z) {\n'
               '    return __npb_make_complex(log(cabs(z)), carg(z));\n'
               '}\n'
               '/* ``csqrt(z) = exp((1/2) * log(z))`` -- principal branch. */\n'
               'inline double _Complex csqrt(double _Complex z) {\n'
               '    double _Complex l = clog(z);\n'
               '    return cexp(__npb_make_complex(0.5*creal(l), 0.5*cimag(l)));\n'
               '}\n'
               '/* ``cpow(z, w) = exp(w * log(z))`` -- general complex pow. */\n'
               'inline double _Complex cpow(double _Complex z, double _Complex w) {\n'
               '    double _Complex l = clog(z);\n'
               '    return cexp(__npb_make_complex(\n'
               '        creal(w)*creal(l) - cimag(w)*cimag(l),\n'
               '        creal(w)*cimag(l) + cimag(w)*creal(l)));\n'
               '}\n'
               '/* ``z.conjugate()`` -- complex-conjugate scalar helper. */\n'
               'inline double _Complex __npb_conj(double _Complex z) {\n'
               '    return __npb_make_complex(creal(z), -cimag(z));\n'
               '}\n'
               '/* Integer power for VLA shape bounds. */\n'
               'constexpr int64_t __npb_int_pow(int64_t base, int64_t exp) {\n'
               '    int64_t result = 1;\n'
               '    while (exp > 0) {\n'
               '        if (exp & 1) result *= base;\n'
               '        base *= base;\n'
               '        exp >>= 1;\n'
               '    }\n'
               '    return result;\n'
               '}\n'
               '/* Ternary-form ``max`` / ``min`` as constexpr function templates\n'
               ' * so a mixed call like ``max(double, int)`` promotes the int\n'
               ' * operand via the usual arithmetic conversions (``std::max``\n'
               ' * would require both args to share a type). Operand order picks\n'
               ' * the SECOND arg only when it strictly wins, else the FIRST --\n'
               ' * matching Python''s builtin max/min so a NaN first operand\n'
               ' * propagates (``max(nan, x) == nan``), not the NaN-suppressing\n'
               ' * ``fmax`` behaviour a plain ``a > b`` would give. */\n'
               'template <class A, class B>\n'
               'constexpr auto max(A a, B b) { return b > a ? b : a; }\n'
               'template <class A, class B>\n'
               'constexpr auto min(A a, B b) { return b < a ? b : a; }\n'
               '/* Python ``//`` floor-toward-neg-inf (C/C++ ``/`` truncates\n'
               ' * toward zero); matches numpy ``//`` for mixed-sign inputs. */\n'
               'template <class A, class B>\n'
               'constexpr auto int_floor(A a, B b) {\n'
               '    return a / b - ((a % b != 0) && ((a < 0) ^ (b < 0)));\n'
               '}\n'
               '/* Python ``%`` returns the sign of the divisor; C/C++ the\n'
               ' * dividend. ``python_mod`` bridges the gap. */\n'
               'template <class A, class B>\n'
               'constexpr auto python_mod(A a, B b) { return (a % b + b) % b; }\n\n'
               'extern "C" {\n')
_CPP_FOOTER = '} // extern "C"\n'

_C_PRELUDE = ("    struct timespec __t1, __t2;\n"
              "    int64_t __dsec, __dnsec;\n"
              "    clock_gettime(CLOCK_MONOTONIC, &__t1);\n"
              "    {\n")

_C_EPILOGUE = ("    }\n"
               "    clock_gettime(CLOCK_MONOTONIC, &__t2);\n"
               "    __dsec  = __t2.tv_sec  - __t1.tv_sec;\n"
               "    __dnsec = __t2.tv_nsec - __t1.tv_nsec;\n"
               "    time_ns[0] = __dsec * 1000000000LL + __dnsec;\n")

_CPP_PRELUDE = ("    auto __t1 = std::chrono::high_resolution_clock::now();\n"
                "    {\n")

_CPP_EPILOGUE = ("    }\n"
                 "    auto __t2 = std::chrono::high_resolution_clock::now();\n"
                 "    time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(\n"
                 "                       __t2 - __t1).count();\n")


def emit_c(kir: KernelIR, fn_name: Optional[str] = None) -> str:
    name = fn_name or f"{kir.kernel_name}_d_c"
    signature = _emit_signature(kir, name)
    body = _emit_body(kir, indent="        ")
    return f"{_C_HEADER}\n{signature} {{\n{_C_PRELUDE}{body}\n{_C_EPILOGUE}}}\n"


def emit_cpp(kir: KernelIR, fn_name: Optional[str] = None) -> str:
    name = fn_name or f"{kir.kernel_name}_d"
    signature = _emit_signature(kir, name)
    # ``restrict`` is a C99 keyword; C++ accepts it under the
    # ``__restrict__`` GCC / Clang extension. Rewrite for the C++ output
    # so the same body string serves both targets.
    signature = signature.replace("*restrict ", "*__restrict__ ")
    body = _emit_body(kir, indent="        ")
    return (f"{_CPP_HEADER}\n{signature} {{\n{_CPP_PRELUDE}{body}\n"
            f"{_CPP_EPILOGUE}}}\n{_CPP_FOOTER}")


def _pluto_multidim_array_signature(arr: ArrayDesc) -> str:
    """Pluto signature for a rank>=2 array: a DIRECT VLA parameter
    ``T name[restrict d0][d1]...``. pet extracts an affine scop from these
    (``A[i][j]``); the flat-pointer + local cast-view form yields ZERO statements
    -- pet drops every scop that reaches an array through a cast pointer, silently
    miscompiling the kernel to a no-op (the output write vanishes). The VLA dims
    must be in scope, so :func:`_emit_pluto_signature` emits the size symbols BEFORE
    the array params and :func:`emit_pluto_binding` matches that order."""
    base = _c_type(arr.dtype)
    qual = "" if arr.is_output else "const "
    dims = (f"[restrict {_c_shape_token(arr.shape[0])}]"
            + "".join(f"[{_c_shape_token(d)}]" for d in arr.shape[1:]))
    return f"{qual}{base} {arr.name}{dims}"


def _emit_pluto_signature(kir: KernelIR, fn_name: str, multidim: Set[str]) -> str:
    """Pluto signature with rank>=2 arrays as direct VLA params (see
    :func:`_pluto_multidim_array_signature`). Because a VLA dim must be lexically
    in scope, the order is regrouped from the canonical C-ABI: SIZE SYMBOLS first,
    then array params, then scalars, then ``time_ns``. :func:`emit_pluto_binding`
    emits the matching arg order so the harness marshals correctly."""
    sym_by_name = {s.name: s for s in kir.symbols}
    arr_by_name = {a.name: a for a in kir.arrays}
    sca_by_name = {s.name: s for s in kir.scalars}
    order = kir.param_order()
    for nm in order:
        if nm not in sym_by_name and nm not in arr_by_name and nm not in sca_by_name:
            raise ValueError(f"unknown parameter {nm!r} in kernel {kir.kernel_name}")
    parts: List[str] = [f"{dtypes.c_type('int')} {nm}" for nm in order if nm in sym_by_name]
    for nm in order:
        if nm in arr_by_name:
            arr = arr_by_name[nm]
            parts.append(_pluto_multidim_array_signature(arr) if nm in multidim else _array_signature(arr))
    parts += [f"{_c_type(sca_by_name[nm].dtype)} {nm}" for nm in order if nm in sca_by_name]
    parts.append("int64_t *restrict time_ns")
    return f"void {fn_name}({', '.join(parts)})"


def emit_pluto(kir: KernelIR, fn_name: Optional[str] = None) -> str:
    name = fn_name or f"{kir.kernel_name}_d_pluto"
    # Rank>=2 array PARAMS are direct VLA parameters so polycc/pet see affine
    # ``A[i][j]`` references (see _pluto_multidim_array_signature). Rank-1 params
    # and malloc'd local scratch stay flat / cast-view (a 1-D ``a[i]`` is already
    # affine, and pet accepts malloc'd cast-views -- only PARAM cast-views break it).
    multidim = {a.name for a in kir.arrays if len(a.shape) >= 2}
    signature = _emit_pluto_signature(kir, name, multidim)
    decls, body, frees = _emit_body(kir, indent="        ", multidim_arrays=multidim,
                                    pluto=True, return_parts=True)
    # Local allocations / frees live OUTSIDE ``#pragma scop`` -- malloc/free are
    # non-affine and would break the polyhedral region; only the affine loop
    # nests stay inside. (Deferred-malloc locals, whose shape needs a body
    # scalar, still allocate at their marker inside the scop -- rare.)
    decl_block = (decls + "\n") if decls else ""
    free_block = (frees + "\n") if frees else ""
    return (f"{_C_HEADER}\n{signature} {{\n{_C_PRELUDE}"
            f"{decl_block}    #pragma scop\n{body}\n    #pragma endscop\n"
            f"{free_block}{_C_EPILOGUE}}}\n")
