"""Call normalisation: astype, reshape methods, numpy aliases, matmul spellings, enumerate/zip, transpose."""

import ast
import copy
from collections.abc import Callable

from hpcagent_bench.translators.numpyto_common.statement_desugar import bind, element_read, indexed_loop, pair_names


class AstypeRewriter(ast.NodeTransformer):
    """Lower ``<expr>.astype(<dtype>)`` on ANY receiver (not just a bare
    Name, which ``MethodCallRewriter`` is limited to) into the cast form the
    emitters already handle.

    * ``x.astype(np.int32)`` / ``x.astype(np.float64)`` / ``x.astype("int64")``
      -> ``np.int32(x)`` -- a per-element cast that the emitters render as a
      C/Fortran cast (integer targets truncate, matching numpy).
    * ``x.astype(other.dtype)`` -> ``x`` -- a cast to *another array's* dtype is
      realised by the destination's declared dtype on store, so it drops to the
      receiver (the frontend already read the ``.astype`` for dtype inference).
      This is what makes ``(labels[:, None] == ids).astype(X.dtype)`` (kmeans)
      and ``(level == d).astype(np.int64)`` (bfs) lowerable.
    """

    def __init__(self, array_dtypes: dict[str, str] | None = None, default_float: str = "") -> None:
        #: ``{array_name: dtype}`` so ``(cmp).astype(X.dtype)`` can resolve
        #: ``X.dtype`` to a concrete cast when the receiver is logical.
        self.array_dtypes = array_dtypes or {}
        #: What an UNTYPED array is: every other pass reads one as the kernel's float, so a
        #: ``.astype(tmp.dtype)`` off an intermediate resolves to the same thing rather than
        #: dropping the cast (fv3_dycore's y stage casts off ``q_advected_x``, which carries no
        #: recorded dtype).
        self.default_float = default_float

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "astype" and node.args):
            return node
        recv, dt = f.value, node.args[0]
        name = None
        if isinstance(dt, ast.Attribute) and isinstance(dt.value, ast.Name) and dt.value.id in ("np", "numpy"):
            name = dt.attr  # np.<dtype>
        elif isinstance(dt, ast.Name) and dt.id in ("int", "float", "bool"):
            name = {"int": "int64", "float": "float64", "bool": "bool_"}[dt.id]
        elif isinstance(dt, ast.Constant) and isinstance(dt.value, str):
            name = dt.value  # "float64" etc.
        if name is None:
            # ``other.dtype``: normally the destination's declared dtype
            # realises the cast, so we drop to the receiver. BUT a comparison /
            # boolean receiver leaves a LOGICAL value -- Fortran cannot store it
            # into / sum it as the REAL destination (kmeans' one-hot
            # ``(labels == ids).astype(X.dtype)``). Resolve the source array's
            # dtype and emit the concrete cast so the merge(1, 0, cond) path
            # fires and the destination is declared REAL.
            bitwise = (isinstance(recv, ast.BinOp) and isinstance(recv.op, (ast.BitAnd, ast.BitOr, ast.BitXor))) or (
                isinstance(recv, ast.UnaryOp) and isinstance(recv.op, (ast.Not, ast.Invert))
            )
            # A bitwise combination of masks (fv3_xppm's ``(smt5 | smt5_m1).astype(q.dtype)``) is as
            # LOGICAL as the comparisons it joins; its operands are locals, so their dtype is not in
            # this table yet and the receiver's own shape is the only evidence. Dropping the cast
            # left Fortran multiplying REAL(8) by LOGICAL(1). ``& | ^ ~`` reject floats in numpy, so
            # the operand is boolean or integer either way and the concrete cast is right for both.
            if (
                (isinstance(recv, (ast.Compare, ast.BoolOp)) or bitwise)
                and isinstance(dt, ast.Attribute)
                and dt.attr == "dtype"
                and isinstance(dt.value, ast.Name)
            ):
                name = self.array_dtypes.get(dt.value.id) or self.default_float or None
            if name is None:
                return recv
        return ast.copy_location(
            ast.Call(
                func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr=name, ctx=ast.Load()),
                args=[recv],
                keywords=[],
            ),
            node,
        )


def match_reshape(node: ast.AST):
    """If ``node`` is a reshape call (method ``X.reshape(shape...)`` OR func
    ``np.reshape(X, shape)``), return ``(base_expr, shape_elts)`` -- the array
    being reshaped and the list of shape AST elements. Else ``None``.

    Both the single-tuple (``X.reshape((a, b))``) and the varargs
    (``X.reshape(a, b)``) method spellings are accepted."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "reshape"):
        return None
    recv = node.func.value
    if isinstance(recv, ast.Name) and recv.id in ("np", "numpy"):
        if len(node.args) < 2:
            return None
        shape = node.args[1]
        elts = list(shape.elts) if isinstance(shape, (ast.Tuple, ast.List)) else [shape]
        return node.args[0], elts
    # method form: receiver is the array
    if len(node.args) == 1 and isinstance(node.args[0], (ast.Tuple, ast.List)):
        return recv, list(node.args[0].elts)
    return recv, list(node.args)


class ReshapeMethodRewriter(ast.NodeTransformer):
    """Normalize the method form ``X.reshape(a, b)`` / ``X.reshape((a, b))`` to
    the function form ``np.reshape(X, (a, b))`` so the single ``expand_reshape``
    path handles every spelling (lulesh uses the varargs method form)."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "reshape"):
            return node
        recv = node.func.value
        if isinstance(recv, ast.Name) and recv.id in ("np", "numpy"):
            return node  # already the function form
        matched = match_reshape(node)
        if matched is None:
            return node
        base, elts = matched
        # Preserve ``order=`` (C/F) so expand_reshape can honour a column-major
        # reshape; every other kwarg is dropped (the method form has none else).
        keep = [kw for kw in node.keywords if kw.arg == "order"]
        return ast.copy_location(
            ast.Call(
                func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="reshape", ctx=ast.Load()),
                args=[base, ast.Tuple(elts=elts, ctx=ast.Load())],
                keywords=keep,
            ),
            node,
        )


FFT_FNS = {"fftn": (False, True), "ifftn": (True, True), "fft": (False, False), "ifft": (True, False)}


def match_fft(node: ast.AST):
    """If ``node`` is ``np.fft.{fftn,ifftn,fft,ifft}(arg, ...)``, return
    ``(fn_name, arg, keywords)``; else ``None``."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in FFT_FNS):
        return None
    f = node.func
    if (
        isinstance(f.value, ast.Attribute)
        and f.value.attr == "fft"
        and isinstance(f.value.value, ast.Name)
        and f.value.value.id in ("np", "numpy")
        and node.args
    ):
        return f.attr, node.args[0], node.keywords
    return None


#: numpy free-function aliases that are exact synonyms of a name the
#: translator already lowers. Normalising them at the AST level means the whole
#: downstream machinery (expander, shape derivation, hoister) is reused with no
#: duplication. ``permute_dims`` is the array-API spelling of ``transpose``
#: (both take ``(a, axes)``); ``amax``/``amin`` are the long names of max/min.
NP_FUNC_ALIASES: dict[str, str] = {
    "permute_dims": "transpose",
    "permute": "transpose",
    "amax": "max",
    "amin": "min",
}


class NpAliasRewriter(ast.NodeTransformer):
    """Rename ``np.<alias>(...)`` to its canonical ``np.<name>(...)`` form so a
    single lowering path serves every spelling (e.g. ``np.permute_dims`` ->
    ``np.transpose``)."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        f = node.func
        if (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Name)
            and f.value.id in ("np", "numpy")
            and f.attr in NP_FUNC_ALIASES
        ):
            f.attr = NP_FUNC_ALIASES[f.attr]
        return node


class ConditionalNoneAllocRewriter(ast.NodeTransformer):
    """``X = <expr> if cond else None`` (and the mirror ``X = None if cond else <expr>``)
    -> ``X = <expr>``.

    An array that is a value in one branch and ``None`` in the other is a
    CONDITIONALLY-ALLOCATED buffer (QE vexx_k's ``deexx``; an ML optional
    residual/bias accumulator). The backends have no ``None``, and a *valid* kernel
    only reads ``X`` where it was allocated -- reading it on the ``None`` branch would be a
    ``None``-index error -- so unconditionally taking the allocated branch is sound: the
    extra buffer is written/read only under the same guard, and is otherwise never
    observed. Left untouched when ``X`` is later tested with ``is None`` / ``is not None``
    (there its None-ness is observable, so allocating unconditionally would flip the
    guard); that case is the separate is-None allocation-check handling."""

    def __init__(self) -> None:
        self._none_checked: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Names whose None-ness is observed (``x is None`` / ``x is not None``): a
        # conditional alloc into one of these must NOT be forced, so record them first.
        checked = set()
        for cmp in ast.walk(node):
            if (
                isinstance(cmp, ast.Compare)
                and isinstance(cmp.left, ast.Name)
                and any(isinstance(op, (ast.Is, ast.IsNot)) for op in cmp.ops)
                and any(isinstance(c, ast.Constant) and c.value is None for c in cmp.comparators)
            ):
                checked.add(cmp.left.id)
        self._none_checked = checked
        self.generic_visit(node)
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if not (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.IfExp)
            and node.targets[0].id not in self._none_checked
        ):
            return node
        ifexp = node.value
        body_none = isinstance(ifexp.body, ast.Constant) and ifexp.body.value is None
        orelse_none = isinstance(ifexp.orelse, ast.Constant) and ifexp.orelse.value is None
        if orelse_none and not body_none:
            node.value = ifexp.body  # X = A if cond else None -> X = A
        elif body_none and not orelse_none:
            node.value = ifexp.orelse  # X = None if cond else A -> X = A
        return node


class MatmulCallRewriter(ast.NodeTransformer):
    """Normalize ``np.matmul(a, b)`` to the ``a @ b`` BinOp so the call reuses
    the existing matmul machinery (the ``MatmulHoister`` loop lowering and the
    Fortran ``MATMUL`` / ``DOT_PRODUCT`` emit path) -- no parallel detector."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        f = node.func
        if (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Name)
            and f.value.id in ("np", "numpy")
            and f.attr == "matmul"
            and len(node.args) == 2
            and not node.keywords
        ):
            return ast.copy_location(ast.BinOp(left=node.args[0], op=ast.MatMult(), right=node.args[1]), node)
        return node


class ScalarTimesMatmulRewriter(ast.NodeTransformer):
    """Restructure ``alpha * A @ B`` so the matmul hoister can fire.

    Python parses ``alpha * A @ B`` left-to-right as ``(alpha * A) @ B``;
    the matmul's left operand is a BinOp not a Name, so the hoister
    rejects it. We rewrite the assignment to lift ``alpha * A`` into a
    per-element scaled-copy temp first; the next pass then matmuls the
    temp with B.

    Triggered only when the assignment LHS is a slice / Subscript --
    the typical gemm / k2mm shape. The IR's shape table tells us
    A's shape so we can declare the temp.
    """

    def __init__(self, shape_table: dict[str, list[str]], temps: dict[str, tuple[str, ...]], counter) -> None:
        self.shape_table = shape_table
        self.temps = temps
        self.counter = counter
        self.pre_stmts: list[ast.stmt] = []

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        # Pattern: ``(scalar * Name) @ B`` -- left is BinOp(Mult), right is anything.
        if isinstance(node.op, ast.MatMult) and isinstance(node.left, ast.BinOp) and isinstance(node.left.op, ast.Mult):
            inner = node.left
            scaled_name = None
            scalar = None
            if isinstance(inner.left, ast.Name) and inner.left.id in self.shape_table:
                scaled_name = inner.left
                scalar = inner.right
            elif isinstance(inner.right, ast.Name) and inner.right.id in self.shape_table:
                scaled_name = inner.right
                scalar = inner.left
            if scaled_name is not None and scalar is not None:
                shape = self.shape_table.get(scaled_name.id)
                if shape:
                    self.counter[0] += 1
                    temp = f"__sm{self.counter[0]}"
                    self.temps[temp] = tuple(shape)
                    self.shape_table[temp] = shape
                    iters = [f"__si{i}" for i in range(len(shape))]
                    idx = (
                        ast.Name(id=iters[0], ctx=ast.Load())
                        if len(iters) == 1
                        else ast.Tuple(elts=[ast.Name(id=i, ctx=ast.Load()) for i in iters], ctx=ast.Load())
                    )
                    body = [
                        ast.Assign(
                            targets=[
                                ast.Subscript(value=ast.Name(id=temp, ctx=ast.Load()), slice=idx, ctx=ast.Store())
                            ],
                            value=ast.BinOp(
                                left=scalar,
                                op=ast.Mult(),
                                right=ast.Subscript(
                                    value=ast.Name(id=scaled_name.id, ctx=ast.Load()), slice=idx, ctx=ast.Load()
                                ),
                            ),
                        )
                    ]
                    out = body
                    for v, b in zip(reversed(iters), reversed(shape)):
                        out = [
                            ast.For(
                                target=ast.Name(id=v, ctx=ast.Store()),
                                iter=ast.Call(
                                    func=ast.Name(id="range", ctx=ast.Load()),
                                    args=[
                                        ast.Name(id=b, ctx=ast.Load())
                                        if not b.isdigit()
                                        else ast.Constant(value=int(b))
                                    ],
                                    keywords=[],
                                ),
                                body=out,
                                orelse=[],
                            )
                        ]
                    self.pre_stmts.extend(out)
                    # Replace ``alpha * A`` in this MatMult with the temp.
                    node.left = ast.Name(id=temp, ctx=ast.Load())
        return node


class EnumerateZipRewriter(ast.NodeTransformer):
    """Desugar ``for x in enumerate(arr):`` and ``for x in zip(a, b):``
    to plain ``for __i in range(N):`` with the per-iteration assignments
    inlined as the first statement of the loop body.
    """

    def __init__(self, extent_of: Callable[[str], ast.expr | None]) -> None:
        self.extent_of = extent_of

    @staticmethod
    def enumerate_start(call: ast.Call) -> ast.expr:
        """The ``start=`` of ``enumerate(seq, start=s)`` (positional or kw), else 0."""
        for kw in call.keywords:
            if kw.arg == "start":
                return kw.value
        if len(call.args) >= 2:
            return call.args[1]
        return ast.Constant(value=0)

    def visit_For(self, node: ast.For) -> ast.AST:
        self.generic_visit(node)
        it = node.iter
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name):
            # ``for m, w in enumerate((a, b, c), start=s):`` over a LITERAL/const
            # sequence (the finite-difference-stencil idiom ``enumerate(_CW)``):
            # unroll to straight-line ``m = s+i; w = <elt i>; <body>`` blocks so the
            # element values are compile-time constants (an axis/shift a roll needs).
            if (
                it.func.id == "enumerate"
                and it.args
                and isinstance(it.args[0], (ast.Tuple, ast.List))
                and isinstance(node.target, ast.Tuple)
                and len(node.target.elts) == 2
            ):
                idx_name, val_name = node.target.elts[0], node.target.elts[1]
                start = self.enumerate_start(it)
                out: list[ast.stmt] = []
                for i, elt in enumerate(it.args[0].elts):
                    out.append(
                        ast.Assign(
                            targets=[ast.Name(id=idx_name.id, ctx=ast.Store())],
                            value=ast.BinOp(left=copy.deepcopy(start), op=ast.Add(), right=ast.Constant(value=i)),
                        )
                    )
                    out.append(
                        ast.Assign(targets=[ast.Name(id=val_name.id, ctx=ast.Store())], value=copy.deepcopy(elt))
                    )
                    out.extend(copy.deepcopy(stmt) for stmt in node.body)
                return out
            pair = pair_names(node.target)
            if pair is not None and it.func.id == "enumerate" and it.args and isinstance(it.args[0], ast.Name):
                sequence = it.args[0].id
                extent = self.extent_of(sequence)
                if extent is not None:
                    # idx = start + __ei ; val = arr[__ei]
                    position = ast.BinOp(
                        left=copy.deepcopy(self.enumerate_start(it)),
                        op=ast.Add(),
                        right=ast.Name(id="__ei", ctx=ast.Load()),
                    )
                    binds = [bind(pair[0], position), bind(pair[1], element_read(sequence, "__ei"))]
                    return indexed_loop(node, "__ei", extent, binds)
            if (
                pair is not None
                and it.func.id == "zip"
                and len(it.args) == 2
                and all(isinstance(a, ast.Name) for a in it.args)
            ):
                left, right = it.args[0].id, it.args[1].id
                extent = self.extent_of(left)
                if extent is not None:
                    binds = [bind(pair[0], element_read(left, "__zi")), bind(pair[1], element_read(right, "__zi"))]
                    return indexed_loop(node, "__zi", extent, binds)
        return node


class TransposeRewriter(ast.NodeTransformer):
    """Normalize both transpose spellings to the ``np.transpose(A[, axes])``
    function form so the single ``expand_transpose`` path serves every spelling:

    * the property ``A.T`` -> ``np.transpose(A)``;
    * the method ``A.transpose()`` / ``A.transpose(axes)`` / ``A.transpose(1, 0)``
      -> ``np.transpose(A[, (axes)])`` (the varargs ints are packed into a tuple).
    """

    def __init__(self, sparse_names=None) -> None:
        #: Logical sparse matrices whose ``A.T`` / ``A.transpose()`` must stay a
        #: transpose ATTRIBUTE/method -- the sparse matmul hoister turns ``A.T @
        #: x`` into a transpose SpMV on A's own buffers (CSR<->CSC). Densifying it
        #: via np.transpose would index the sparse buffers as a dense 2-D matrix
        #: (wrong + uncompilable).
        self.sparse_names = set(sparse_names or ())

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if node.attr != "T":
            return node
        base = node.value
        # Bare Name -- the original path (a sparse buffer keeps its ``.T`` for the
        # sparse matmul hoister, which lowers ``A.T @ x`` on A's own CSR/CSC buffers).
        if isinstance(base, ast.Name):
            if base.id in self.sparse_names:
                return node
        elif isinstance(base, ast.Subscript):
            # A subscript of a sparse buffer likewise keeps its transpose.
            if isinstance(base.value, ast.Name) and base.value.id in self.sparse_names:
                return node
        elif not isinstance(base, (ast.BinOp, ast.Call)):
            # ``.T`` on a matmul result (``(Yf.T @ Wf).T`` -- the LS3DF generalized
            # Rayleigh-Ritz symmetriser) or another array-valued call. Anything else
            # (a scalar attribute chain) is left intact.
            return node
        # ``<array-expr>.T`` -> ``np.transpose(<array-expr>)``. The call hoister
        # materialises a non-Name argument (the matmul) into a temp before
        # ``expand_transpose`` lowers it, so the transpose never survives as an
        # attribute the per-element scalarizer would misapply.
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="transpose", ctx=ast.Load()),
            args=[base],
            keywords=[],
        )

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "transpose"):
            return node
        if isinstance(f.value, ast.Name) and f.value.id in ("np", "numpy"):
            return node  # already the ``np.transpose(...)`` function form
        if isinstance(f.value, ast.Name) and f.value.id in self.sparse_names:
            return node  # sparse transpose stays a method on its own buffers
        base = f.value
        if len(node.args) == 1 and isinstance(node.args[0], (ast.Tuple, ast.List)):
            args = [base, node.args[0]]  # x.transpose((1, 0))
        elif node.args:
            args = [base, ast.Tuple(elts=list(node.args), ctx=ast.Load())]  # x.transpose(1, 0)
        else:
            args = [base]  # x.transpose() -- full reverse
        return ast.copy_location(
            ast.Call(
                func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="transpose", ctx=ast.Load()),
                args=args,
                keywords=[],
            ),
            node,
        )
