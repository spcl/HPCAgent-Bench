"""Emit a DaCe @dc.program from the canonical numpy reference, sharing IR/classification with the C/Fortran emitters."""

import ast
import copy
import dataclasses
import functools
import itertools
import logging
import re
from typing import cast
from collections.abc import Callable, Iterable, Sequence

from hpcagent_bench.translators.numpyto_common import dtypes
from hpcagent_bench.translators.numpyto_common import frontend as common_frontend
from hpcagent_bench.translators.numpyto_common.frontend import PinnedValue, field_nodes, fold_shape_expr
from hpcagent_bench.translators.numpyto_common.emit_helpers.numpy_names import is_numpy_module
from hpcagent_bench.translators.numpyto_common.emit_helpers.tokens import IDENT_RE
from hpcagent_bench.translators.numpyto_common.ir import ArrayDesc, KernelIR, shape_dimension_symbols
from hpcagent_bench.translators.numpyto_common.lib_nodes import shape_exprs_equal, sympify_shape
from hpcagent_bench.translators.numpyto_common.lowering import lower
from hpcagent_bench.translators.numpyto_common.numpy_desugar import (
    AUG_OP_SRC,
    dtype_kind,
    dtype_table_,
    kind_of_dtype_str,
    promote_kind,
    desugar_for_python_backend,
    expr_rank,
    name_binding_index,
    rank_table,
)
from hpcagent_bench.translators.numpyto_common.ordered import OrderedSet
from hpcagent_bench.translators.numpyto_common.statement_desugar import (
    DesugarArrayIteration,
    SplitChainedAssign,
    SplitTupleUnpack,
    Spelled,
    is_scalar_literal,
)

#: A decimal point or an exponent -- what makes a shape token a float rather than an extent.
FLOAT_LITERAL_RE = re.compile(r"\d*\.\d|\d[eE][-+]?\d")


class ShapeToSymbol(ast.NodeTransformer):
    """Replace each <array>.shape[<const k>] with the array's k-th declared symbolic shape token."""

    def __init__(self, arr_shapes: dict[str, list[str]]) -> None:
        self.arr_shapes = arr_shapes

    def visit_Subscript(self, node: ast.Subscript):
        self.generic_visit(node)
        v = node.value
        if (
            isinstance(v, ast.Attribute)
            and v.attr == "shape"
            and isinstance(v.value, ast.Name)
            and v.value.id in self.arr_shapes
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)
        ):
            toks = self.arr_shapes[v.value.id]
            if 0 <= node.slice.value < len(toks):
                return ast.copy_location(ast.parse(toks[node.slice.value], mode="eval").body, node)
        return node


class SplitTupleAssign(SplitTupleUnpack):
    """Lower a tuple assignment into one statement per name.

    ``n, c, h, w = x.shape`` is what the helper inliner emits, and it is the single biggest reason a
    generated program is refused: each unpacked name reaches the frontend as an ordinary local, so
    it mints a fresh opaque symbol per use and the buffer sized from them cannot be written from
    ``x`` -- ``[batch_size, 3, 224, 224]`` against ``[__sym___inl6_n_0, ...]``. Split into
    ``n = x.shape[0]`` etc., the existing shape passes resolve each one: declared arrays through
    :class:`ShapeToSymbol`, transients through :func:`inline_transient_shape_scalars`.

    A SWAP (``a, b = b, a``) goes through ``__hpcagent_bench_tuple<k>`` temporaries: emitting the
    statements in order would overwrite ``a`` before ``b`` reads it, which is a silent wrong answer
    rather than a refusal.
    """

    def values(self, targets: list[ast.expr], value: ast.expr) -> Spelled | None:
        if not (isinstance(value, ast.Attribute) and value.attr == "shape"):
            return super().values(targets, value)
        # Re-reading ``.shape`` per name is free: it is resolved to declared extents below, and
        # never survives as a runtime read.
        base = value.value
        prelude: list[ast.stmt] = []
        if not isinstance(base, ast.Name):
            # ``n, c, h, w = np.maximum(t, 0.0).shape``: name the operand first, or each of the
            # four reads would carry its own copy of the call. The temporary is elementwise, so
            # the shape resolver can follow it to the operand's own extents.
            temporary = f"__hpcagent_bench_shaped{self.temps}"
            self.temps += 1
            prelude.append(ast.Assign(targets=[ast.Name(id=temporary, ctx=ast.Store())], value=base))
            base = ast.Name(id=temporary, ctx=ast.Load())
        reads: list[ast.expr] = [
            ast.Subscript(
                value=ast.Attribute(value=copy.deepcopy(base), attr="shape", ctx=ast.Load()),
                slice=ast.Constant(value=index),
                ctx=ast.Load(),
            )
            for index in range(len(targets))
        ]
        return prelude, reads

    def temp_name(self, position: int) -> str:
        return f"__hpcagent_bench_tuple{self.temps}"


class DropSymbolAssign(ast.NodeTransformer):
    """Drop <sym> = ... where <sym> is a declared size symbol (dace symbols are immutable)."""

    def __init__(self, symbols: Iterable[str]) -> None:
        self.symbols = set(symbols)

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in self.symbols:
            return None
        return node


class ResolveZeros(ast.NodeTransformer):
    """Resolve a lowered kir's __hpcagent_bench_zeros__() allocation marker to an explicit np.zeros/np.ones() call."""

    def __init__(
        self,
        zeros_locals: dict[str, tuple[str, ...]],
        zeros_fills: dict[str, str],
        local_dtypes: dict[str, str],
        default_dtype: str,
    ) -> None:
        self.zeros_locals = zeros_locals
        self.zeros_fills = zeros_fills
        self.local_dtypes = local_dtypes
        self.default_dtype = default_dtype
        self.allocated: dict[str, tuple[str, ...]] = {}  # name -> last-allocated shape

    def visit_Assign(self, node: ast.Assign):
        if not (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "__hpcagent_bench_zeros__"
        ):
            return node
        name = node.targets[0].id
        if name not in self.zeros_locals:
            return None  # a reassigned param (spmm's output C): update in place, never allocate
        # Detect the self-referential sentinel the same way the C/Fortran emitters do.
        is_reassign = any(isinstance(a, ast.Constant) and a.value == "__reassign__" for a in node.value.args)
        shape = self.zeros_locals[name] or ("1",)
        prev_shape = self.allocated.get(name)
        # An in-place reuse whose loop reads OLD values -> drop the re-zero; a shape change still allocates.
        if is_reassign and prev_shape == shape:
            return None
        self.allocated[name] = shape
        ctor = "np.ones" if self.zeros_fills.get(name) in ("ones", "ones_like") else "np.zeros"
        dtype = dace_dtype(self.local_dtypes.get(name) or self.default_dtype)
        elts = ", ".join(str(s) for s in shape) + ("," if len(shape) == 1 else "")
        return ast.copy_location(ast.parse(f"{name} = {ctor}(({elts}), dtype={dtype})").body[0], node)


def declare_unallocated_zeros_locals(
    fn: ast.FunctionDef,
    zeros_locals: dict[str, tuple[str, ...]],
    allocated: dict[str, tuple[str, ...]],
    zeros_fills: dict[str, str],
    local_dtypes: dict[str, str],
    default_dtype: str,
    params: set[str],
) -> None:
    """Allocate a lowered local that carries no ``__hpcagent_bench_zeros__`` marker.

    :class:`ResolveZeros` rewrites markers, but not every expander leaves one: ``np.stack``
    writes its temp straight into a per-operand copy nest and leaves the allocation to the
    emitter's locals table. C and Fortran declare EVERY ``zeros_locals`` entry from that table, so
    both were correct; dace only ever saw the marker rewrite, so the program reached the frontend
    reading a name nothing defines. That is a PARSE-time "Use of undefined variable", raised long
    after emit reported success -- the emit is what has to change.

    Declared at the top of the body, which is where the C and Fortran locals are declared too, so a
    shape expressed in the kernel's symbols is in scope wherever the first write happens.
    """
    used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    missing = [nm for nm in zeros_locals if nm in used and nm not in allocated and nm not in params]
    if not missing:
        return
    decls: list[ast.stmt] = []
    for nm in missing:
        shape = zeros_locals[nm] or ("1",)
        ctor = "np.ones" if zeros_fills.get(nm) in ("ones", "ones_like") else "np.zeros"
        dtype = dace_dtype(local_dtypes.get(nm) or default_dtype)
        elts = ", ".join(str(s) for s in shape) + ("," if len(shape) == 1 else "")
        decls.append(ast.parse(f"{nm} = {ctor}(({elts}), dtype={dtype})").body[0])
    fn.body[:0] = decls
    ast.fix_missing_locations(fn)


class AnnotateEmptyDtype(ast.NodeTransformer):
    """Give a bare ``np.empty(shape)`` the dtype dace's replacement requires but never defaults.

    ``array_creation_dace.py``'s ``_numpy_empty(pv, sdfg, state, shape, dtype)`` has no default,
    unlike its ``zeros``/``ones``/``full`` siblings (which fall back to float64, matching real
    numpy) -- an asymmetry in dace, not something this generator should keep relying on. A source
    call with no dtype IS real numpy's own float64 default, so filling in the kernel's
    precision-driven float global reproduces that default rather than guessing one.
    """

    def __init__(self, dtype_expr: str) -> None:
        self.dtype_expr = dtype_expr

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if not (
            isinstance(node.func, ast.Attribute) and node.func.attr == "empty" and is_numpy_module(node.func.value)
        ):
            return node
        if len(node.args) != 1 or any(kw.arg == "dtype" for kw in node.keywords):
            return node  # dtype already positional/keyword, or not a plain empty(shape) call
        node.keywords.append(ast.keyword(arg="dtype", value=ast.parse(self.dtype_expr, mode="eval").body))
        return node


#: Value each numpy allocator fills with, for a re-allocation that has to become an in-place fill.
#: ``full`` carries its own; ``empty`` has none, so its statement is dropped rather than rewritten.
REALLOC_FILL = {"zeros": "0", "ones": "1", "empty": None, "full": None}


def alloc_shape_tokens(call: ast.Call) -> list[str] | None:
    """The shape a numpy allocator call names, whitespace-normalized; ``None`` if it names none."""
    if not call.args:
        return None
    first = call.args[0]
    elts = first.elts if isinstance(first, (ast.Tuple, ast.List)) else [first]
    return [ast.unparse(e).replace(" ", "") for e in elts]


class FillOutputParamRealloc(ast.NodeTransformer):
    """Rewrite a re-allocation of an OUTPUT PARAMETER into an in-place fill of it.

    A numpy reference RETURNS what it allocates -- nbody's ``KE = np.zeros(Nt + 1)`` ends in
    ``return KE, PE`` -- but the emitted program takes the same names as parameters, because that
    is how the native backends hand a promoted return back. The allocation then rebinds the name
    to a fresh transient and the caller's array is never written: dace answered
    ``Missing program argument "KE"`` and, once passed, would have graded untouched zeros.

    Only a SHAPE-MATCHING allocation is rewritten. A local that merely shares the name and has a
    different extent is a different container, and filling the parameter with it would be a
    miscompile rather than the missed write it replaces.
    """

    def __init__(self, shapes: dict[str, list[str]]) -> None:
        self.shapes = shapes

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return node
        want = self.shapes.get(node.targets[0].id)
        if want is None or not isinstance(node.value, ast.Call):
            return node
        func = node.value.func
        if not (isinstance(func, ast.Attribute) and func.attr in REALLOC_FILL and is_numpy_module(func.value)):
            return node
        if alloc_shape_tokens(node.value) != want:
            return node
        fill = REALLOC_FILL[func.attr]
        if func.attr == "full":
            if len(node.value.args) < 2:
                return node
            fill = ast.unparse(node.value.args[1])
        if fill is None:
            return None  # np.empty promises nothing: the parameter the caller passed already is it
        return ast.parse(f"{node.targets[0].id}[:] = {fill}").body[0]


#: numpy dtype tag -> dace type expression (floats route through the precision-driven globals).
DTYPE_TO_DACE = {
    "float64": "dc_float",
    "float32": "dc_float",
    "complex128": "dc_complex_float",
    "complex64": "dc_complex_float",
    "int64": "dc.int64",
    "int32": "dc.int32",
    "int16": "dc.int16",
    "int8": "dc.int8",
    "uint64": "dc.uint64",
    "uint32": "dc.uint32",
    "uint16": "dc.uint16",
    "uint8": "dc.uint8",
    "int": "dc.int64",
    "bool": "dc.bool",
}


def dace_dtype(tag: str) -> str:
    """Map a numpy dtype tag to its dace spelling, FAILING LOUDLY on an unknown integer tag
    rather than silently declaring a float. A declared-int array typed as ``dc_float`` reaches
    the frontend as a double, and the first bitwise op on it dies inside dace with an operand-type
    error that names nothing in this file (``BitAnd: 'double' and 'int64_t'`` -- int4 shipped that
    way). Sub-byte dtypes route through their STORAGE dtype: an int4 array IS an int8 buffer."""
    mapped = DTYPE_TO_DACE.get(tag) or DTYPE_TO_DACE.get(dtypes.storage_dtype(tag))
    if mapped is not None:
        return mapped
    # float16/float128 and the fp8 pair have no dace spelling of their own here: like float64 and
    # float32 they compute through the precision-driven float global.
    if tag.startswith("complex"):
        return "dc_complex_float"
    if tag.startswith("float"):
        return "dc_float"
    raise ValueError(
        f"dace emit: cannot map dtype {tag!r} (not in _DTYPE_TO_DACE and not a float "
        f"family tag); refusing to default to a float declaration"
    )


class FloorDivToIntFloor(ast.NodeTransformer):
    """``a // b`` -> ``dc.symbolic.int_floor(a, b)``, recursively."""

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        if not isinstance(node.op, ast.FloorDiv):
            return node
        symbolic = ast.Attribute(value=ast.Name(id="dc", ctx=ast.Load()), attr="symbolic", ctx=ast.Load())
        call = ast.Call(
            func=ast.Attribute(value=symbolic, attr="int_floor", ctx=ast.Load()),
            args=[node.left, node.right],
            keywords=[],
        )
        return ast.copy_location(call, node)


@functools.lru_cache(maxsize=4096, typed=True)
def extent_without_dead_symbols(text: str) -> str:
    """``text`` with any symbol that CANCELS out of it gone, or ``text`` unchanged.

    A slice extent arrives as ``hi - lo``: ``x[ci * 4:(ci + 1) * 4]`` declares
    ``(ci + 1) * 4 - ci * 4``, which is 4 whatever ``ci`` is. dace reads the declaration literally,
    cannot solve a symbol the shape does not really contain, and asks the caller for it --
    "Argument number mismatch ... Missing arguments: {'ci'}".

    Only a strictly SMALLER symbol set is taken, and only when the folded form needs no
    floor/ceiling: an extent this cannot simplify keeps the exact spelling the rest of the emitter
    matches on, and ``//`` never round-trips through sympy's ``floor``.
    """
    named = {i for i in IDENT_RE.findall(text)}
    if not named:
        return text
    folded = sympify_shape(text)
    if folded is None or len(folded.free_symbols) >= len(named):
        return text
    rendered = str(folded)
    if any(head in rendered for head in ("floor", "ceiling", "Piecewise")):
        return text
    return rendered


def declared_extent(dim: str) -> str:
    """One declared extent, with ``//`` spelled the way the frontend spells it inside the body.

    A signature annotation is evaluated by PYTHON, where ``//`` on a dace symbol is sympy's
    ``floor(x/2 + 1/2)``; the SAME text inside the program is parsed by dace, which maps
    ``ast.FloorDiv`` to ``int_floor(x, 2)``. One extent then reaches the write under two spellings
    and the frontend, unable to prove them equal, refuses the broadcast -- stride-2 dilated conv
    was the first corpus case where the divisor is not 1 and the two stop folding together.
    """
    text = extent_without_dead_symbols(str(dim))
    if "//" not in text:
        return text
    tree = FloorDivToIntFloor().visit(ast.parse(text, mode="eval"))
    return ast.unparse(ast.fix_missing_locations(tree))


def array_annotation(arr: ArrayDesc) -> str:
    """``a`` of shape ``(LEN_1D,)`` float64 -> ``dc_float[LEN_1D]``; a 0-d array -> a dace scalar.

    A shapeless manifest entry is a SCALAR the harness happens to file under ``arrays``. Declaring
    it ``[1]`` gave the program a rank the initializer does not build and the other backends do not
    pass -- the same declared-rank-vs-initialize disagreement that runs and still lies.
    """
    if not arr.shape:
        return dace_dtype(arr.dtype)
    return f"{dace_dtype(arr.dtype)}[{', '.join(declared_extent(s) for s in arr.shape)}]"


#: Map framework precision globals (np_float/np_complex) to the dace globals the module imports.
FRAMEWORK_DTYPE_TO_DACE = {"np_float": "dc_float", "np_complex": "dc_complex_float"}


class RewriteFrameworkDtype(ast.NodeTransformer):
    """Rewrite leaked np_float/np_complex tokens to the dace precision global; tracks complex usage for the import."""

    def __init__(self) -> None:
        self.used_complex = False

    def visit_Name(self, node: ast.Name):
        mapped = FRAMEWORK_DTYPE_TO_DACE.get(node.id)
        if mapped is None:
            return node
        if mapped == "dc_complex_float":
            self.used_complex = True
        return ast.copy_location(ast.Name(id=mapped, ctx=node.ctx), node)

    def visit_Assign(self, node: ast.Assign):
        """Drop a reference's call-time rebinding of the precision globals.

        A reference reads them off the framework module (``np_float = framework.np_float``) rather
        than importing the names, because a ``from ... import np_float`` snapshots the value at
        first import and a process that runs two precisions keeps the first one. Renaming that
        statement's target would emit ``dc_float = framework.np_float`` into a module that has no
        ``framework`` and already imports ``dc_float`` -- so the whole assignment goes."""
        targets: list[ast.expr] = []
        for t in node.targets:
            targets.extend(t.elts if isinstance(t, ast.Tuple) else [t])
        named = [t for t in targets if isinstance(t, ast.Name)]
        if len(named) != len(targets) or not named or not all(t.id in FRAMEWORK_DTYPE_TO_DACE for t in named):
            return self.generic_visit(node)
        values = node.value.elts if isinstance(node.value, ast.Tuple) else [node.value]
        if not all(isinstance(v, ast.Attribute) and v.attr in FRAMEWORK_DTYPE_TO_DACE for v in values):
            return self.generic_visit(node)
        if any(FRAMEWORK_DTYPE_TO_DACE[t.id] == "dc_complex_float" for t in named):
            self.used_complex = True
        return None


#: Python builtin used as a numpy dtype -> the spelling dace accepts. numpy reads the builtin as its
#: default of that kind, so these are the same dtype written a way dace's property setter takes.
BUILTIN_DTYPE = {"bool": "np.bool_", "int": "np.int64"}


class RewriteBuiltinDtype(ast.NodeTransformer):
    """Spell a ``dtype=bool`` / ``dtype=int`` / ``dtype=float`` argument the way dace accepts.

    numpy takes the builtin as its default dtype of that kind. dace hands it to the descriptor's
    dtype property as a plain ``str`` and the property rejects it -- ``Received str for property
    dtype of type dace.dtypes.typeclass``, from inside ``data.Array.__init__``, naming no allocation
    and no kernel. ``float`` routes through the precision-driven global, like every other float the
    emitter declares, so the fp32 leg does not allocate an fp64 workspace.
    """

    def __init__(self, float_dtype: str) -> None:
        self.float_dtype = float_dtype

    def visit_keyword(self, node: ast.keyword):
        self.generic_visit(node)
        if node.arg != "dtype" or not isinstance(node.value, ast.Name):
            return node
        spelled = BUILTIN_DTYPE.get(node.value.id) or (self.float_dtype if node.value.id == "float" else None)
        if spelled is None:
            return node
        node.value = ast.copy_location(ast.parse(spelled, mode="eval").body, node.value)
        return node


class TernaryValueHoister(ast.NodeTransformer):
    """Hoist each ternary-used-as-value to a scalar temp assigned by a guarding if/else appended to prelude."""

    def __init__(self, owner: "DesugarTernary", prelude: list[ast.stmt]) -> None:
        self.owner = owner
        self.prelude = prelude

    def visit_IfExp(self, node: ast.IfExp):
        self.generic_visit(node)  # hoist any nested ternary first
        tmp = f"__hpcagent_bench_ternary{self.owner.ctr}"
        self.owner.ctr += 1
        body, orelse = self.owner.joined(node.body, node.orelse)
        self.prelude.append(
            ast.If(
                test=node.test,
                body=[ast.Assign(targets=[ast.Name(id=tmp, ctx=ast.Store())], value=body)],
                orelse=[ast.Assign(targets=[ast.Name(id=tmp, ctx=ast.Store())], value=orelse)],
            )
        )
        return ast.copy_location(ast.Name(id=tmp, ctx=ast.Load()), node)


#: A dtype kind -> the dace dtype an array of that kind is declared with.
KIND_DACE_DTYPE = {"bool": "np.bool_", "int": "np.int64", "float": "dc_float", "complex": "dc_complex_float"}


class DesugarTernary(ast.NodeTransformer):
    """Lower a ternary (assignment RHS or nested value) to the if/else statement dace's frontend accepts."""

    def __init__(self, dtypes: dict[str, str] | None = None, ranks: dict[str, int] | None = None) -> None:
        self.ctr = 0
        self.dtypes = dtypes or {}
        self.ranks = ranks or {}

    def joined(self, body: ast.expr, orelse: ast.expr) -> tuple[ast.expr, ast.expr]:
        """Both branch values at numpy's join dtype, when they are arrays of different known kinds.

        The if/else binds one name on both branches, and dace refuses to rebind an array to another
        dtype, while numpy lets each branch keep its own. vexx_k's ``d.real if gamma_only else d``
        is float on one side and complex on the other; the join holds every value either produces.
        The cast names the wider branch's own ``.dtype`` when that branch is a bare name, so an fp32
        leg joins to the width it declared.
        """
        kinds = (dtype_kind(body, self.dtypes), dtype_kind(orelse, self.dtypes))
        join = promote_kind(kinds[0], kinds[1])
        ranks = (expr_rank(body, self.ranks), expr_rank(orelse, self.ranks))
        if join is None or kinds[0] == kinds[1] or 0 in ranks or not any(ranks):
            return body, orelse
        wide = body if kinds[0] == join else orelse
        dtype = f"{wide.id}.dtype" if isinstance(wide, ast.Name) else KIND_DACE_DTYPE[join]
        cast = [
            value
            if kind == join
            else ast.Call(
                func=ast.Attribute(value=value, attr="astype", ctx=ast.Load()), args=[parse_expr(dtype)], keywords=[]
            )
            for value, kind in zip((body, orelse), kinds)
        ]
        return cast[0], cast[1]

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node.body = self.process_body_(node.body)
        return node

    def visit_For(self, node: ast.For):
        node.body = self.process_body_(node.body)
        node.orelse = self.process_body_(node.orelse)
        return node

    def visit_While(self, node: ast.While):
        node.body = self.process_body_(node.body)
        node.orelse = self.process_body_(node.orelse)
        return node

    def visit_If(self, node: ast.If):
        node.body = self.process_body_(node.body)
        node.orelse = self.process_body_(node.orelse)
        return node

    def process_body_(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for stmt in stmts:
            if isinstance(stmt, (ast.For, ast.While, ast.If)):
                out.append(self.visit(stmt))  # recurse: ternaries in nested bodies hoist there
                continue
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.IfExp) and len(stmt.targets) == 1:
                tgt = stmt.targets[0]
                body, orelse = self.joined(stmt.value.body, stmt.value.orelse)
                new_if = ast.If(
                    test=stmt.value.test,
                    body=self.process_body_([ast.Assign(targets=[copy.deepcopy(tgt)], value=body)]),
                    orelse=self.process_body_([ast.Assign(targets=[copy.deepcopy(tgt)], value=orelse)]),
                )
                out.append(ast.copy_location(new_if, stmt))
                continue
            prelude: list[ast.stmt] = []
            new_stmt = TernaryValueHoister(self, prelude).visit(stmt)
            out.extend(prelude)
            out.append(new_stmt)
        return out


class MethodReceiverHoister(ast.NodeTransformer):
    """Bind a method call's non-Name receiver to a temp, appended to ``prelude``."""

    def __init__(self, owner: "BindMethodReceiver", prelude: list[ast.stmt]) -> None:
        self.owner = owner
        self.prelude = prelude

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)  # innermost receiver first, so a chain unwinds bottom-up
        if not isinstance(node.func, ast.Attribute):
            return node
        base = node.func.value
        while isinstance(base, ast.Attribute):
            base = base.value
        if isinstance(base, ast.Name):
            return node
        tmp = f"__hpcagent_bench_recv{self.owner.ctr}"
        self.owner.ctr += 1
        self.prelude.append(
            ast.copy_location(ast.Assign(targets=[ast.Name(id=tmp, ctx=ast.Store())], value=node.func.value), node)
        )
        node.func.value = ast.copy_location(ast.Name(id=tmp, ctx=ast.Load()), node)
        return node


class BindMethodReceiver(ast.NodeTransformer):
    """Give every method call a receiver dace can NAME.

    ``dace.frontend.python.astutils.rname`` resolves a method call by walking the attribute chain
    down to a Name; anything else (a call, a subscript) raises "Unsupported AST <node> nested inside
    AST call node" before the frontend looks at what the call means. ``np.asarray(npw).reshape(-1)``
    is refused for the receiver, not the reshape, so binding the receiver to a temporary is the
    desugaring -- the value computed is identical.

    A ``while`` test is left alone: its receiver is re-evaluated per iteration, and hoisting it
    before the loop would freeze the first value. That construct stays refused, which is honest.
    """

    def __init__(self) -> None:
        self.ctr = 0

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node.body = self.process_body(node.body)
        return node

    def visit_For(self, node: ast.For):
        node.body = self.process_body(node.body)
        node.orelse = self.process_body(node.orelse)
        return node

    def visit_While(self, node: ast.While):
        node.body = self.process_body(node.body)
        node.orelse = self.process_body(node.orelse)
        return node

    def visit_If(self, node: ast.If):
        node.body = self.process_body(node.body)
        node.orelse = self.process_body(node.orelse)
        return node

    def process_body(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for stmt in stmts:
            prelude: list[ast.stmt] = []
            if isinstance(stmt, ast.For):
                stmt.iter = MethodReceiverHoister(self, prelude).visit(stmt.iter)
                out.extend(prelude)  # the iterable is evaluated once, at loop entry
                out.append(self.visit(stmt))
                continue
            if isinstance(stmt, ast.If):
                stmt.test = MethodReceiverHoister(self, prelude).visit(stmt.test)
                out.extend(prelude)
                out.append(self.visit(stmt))
                continue
            if isinstance(stmt, ast.While):
                out.append(self.visit(stmt))
                continue
            out.append(MethodReceiverHoister(self, prelude).visit(stmt))
            out[-1:] = prelude + out[-1:]
        return out


#: numpy constructors that are the IDENTITY on an argument that is already an ndarray.
ASARRAY_IDENTITY = ("asarray", "ascontiguousarray", "asanyarray")


class DropIdentityAsarray(ast.NodeTransformer):
    """Drop ``np.asarray(x)`` when ``x`` is already an array.

    dace registers no ``asarray`` replacement at all, so the call survives into the frontend as an
    opaque object and the next method on it reports a type nobody wrote (``Method "reshape" is not
    registered for object type "Scalar"``). On an ndarray the call is numpy's own identity, so
    dropping it emits the same numbers with a receiver dace can trace. An argument of unknown rank
    keeps its call: there the constructor is doing real work.
    """

    def __init__(self, ranks: dict[str, int]) -> None:
        self.ranks = ranks

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in ASARRAY_IDENTITY
            and is_numpy_module(node.func.value)
        ):
            return node
        if len(node.args) != 1 or node.keywords:
            return node  # a dtype/order argument makes it a CONVERSION, not an identity
        arg = node.args[0]
        if isinstance(arg, ast.Name) and (self.ranks.get(arg.id) or 0) >= 1:
            return ast.copy_location(arg, node)
        return node


class DesugarChainedCompare(ast.NodeTransformer):
    """Split ``a < b < c`` into ``a < b and b < c`` -- dace's frontend takes one comparator only.

    Python evaluates the middle operand once; the split evaluates it twice, so this rewrites only
    when every repeated operand is a Name or a Constant. Anything else (a call, a subscript) keeps
    its chain and is refused by dace, which is the honest outcome: a duplicated side effect would
    be a miscompile, and a duplicated array read would be a second memlet.
    """

    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        if len(node.ops) < 2:
            return node
        operands = [node.left, *node.comparators]
        if not all(isinstance(x, (ast.Name, ast.Constant)) for x in operands[1:-1]):
            return node
        links: list[ast.expr] = [
            ast.Compare(left=copy.deepcopy(left), ops=[op], comparators=[copy.deepcopy(right)])
            for left, op, right in zip(operands, node.ops, operands[1:])
        ]
        return ast.copy_location(ast.BoolOp(op=ast.And(), values=links), node)


def is_negative_one(node: ast.expr) -> bool:
    """``-1`` reaches the AST as a USub over a Constant, never as a negative literal."""
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and node.operand.value == 1
    )


def reshape_target(node: ast.Call) -> tuple[str | None, list[ast.expr]]:
    """``(name, shape_args)`` for a reshape call on a plain name, else ``(None, [])``."""
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "reshape":
        return None, []
    if is_numpy_module(node.func.value):
        return (node.args[0].id, node.args[1:]) if node.args and isinstance(node.args[0], ast.Name) else (None, [])
    return (node.func.value.id, node.args) if isinstance(node.func.value, ast.Name) else (None, [])


class ResolveInferredReshape(ast.NodeTransformer):
    """Replace the ``-1`` in ``x.reshape(1, -1, 1, 1)`` with the extent numpy would infer.

    numpy reads ``-1`` as "work it out from the size"; dace takes the shape literally and rejects
    a negative dimension. The inferred extent is the operand's size over the product of the dims
    that were spelled out, so it is only computable here when the operand's shape is known and
    every other dim is a literal -- otherwise the chain is left for dace to refuse rather than
    guessed at.
    """

    def __init__(self, arr_shapes: dict[str, list[str]]) -> None:
        self.arr_shapes = arr_shapes

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        base, args = reshape_target(node)
        if base not in self.arr_shapes or not args:
            return node
        dims = args[0].elts if len(args) == 1 and isinstance(args[0], (ast.Tuple, ast.List)) else args
        inferred = [i for i, d in enumerate(dims) if is_negative_one(d)]
        spelled = [d for i, d in enumerate(dims) if i not in inferred]
        extents = [d.value for d in spelled if isinstance(d, ast.Constant) and isinstance(d.value, int)]
        if len(inferred) != 1 or len(extents) != len(spelled):
            return node
        divisor = 1
        for extent_value in extents:
            divisor *= extent_value
        size = " * ".join(f"({tok})" for tok in self.arr_shapes[base])
        extent = size if divisor == 1 else f"({size}) // {divisor}"
        dims[inferred[0]] = ast.parse(extent, mode="eval").body
        return ast.fix_missing_locations(node)


class NormalizeReshape(ast.NodeTransformer):
    """Spell a reshape the two ways dace can follow: a tuple shape, and ``ravel`` for a bare ``-1``.

    numpy takes ``a.reshape(6)`` and ``a.reshape((6,))`` as the same call. dace's
    ``_ndarray_reshape`` unconditionally unwraps its varargs to the first element and then iterates
    it, so a single scalar extent reaches ``reshape`` as a bare symbol and dies with ``'symbol'
    object is not iterable`` -- a message that names no reshape and no kernel.

    A lone ``-1`` cannot become a tuple: dace takes the shape literally and allocates a negative
    extent. It is numpy's flatten-to-1-D, which is exactly ``ravel``, and dace does register that.
    The ``order=`` keyword goes with it: a plain ``ravel()`` reads C order, so an F-order flatten
    would come back permuted.
    """

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        base, args = reshape_target(node)
        # A resolved base already proves the callee is ``<receiver>.reshape``.
        if base is None or len(args) == 0 or not isinstance(node.func, ast.Attribute):
            return node
        numpy_form = is_numpy_module(node.func.value)
        dims = args[0].elts if len(args) == 1 and isinstance(args[0], (ast.Tuple, ast.List)) else list(args)
        if len(dims) == 1 and is_negative_one(dims[0]):
            receiver = node.args[0] if numpy_form else node.func.value
            order = [keyword for keyword in node.keywords if keyword.arg == "order"]
            ravel = ast.Call(func=ast.Attribute(value=receiver, attr="ravel", ctx=ast.Load()), args=[], keywords=order)
            return ast.fix_missing_locations(ast.copy_location(ravel, node))
        if len(args) == 1 and isinstance(args[0], (ast.Tuple, ast.List)):
            return node
        shape = ast.Tuple(elts=list(args), ctx=ast.Load())
        node.args = [node.args[0], shape] if numpy_form else [shape]
        return ast.fix_missing_locations(node)


#: numpy calls that return a new array no other name refers to. The ``np.fft`` transforms always
#: allocate their result (vexx_k's ``fwfft`` reshapes an ``fftn`` result and writes through it).
FRESH_ARRAY_CALLS = frozenset(
    {"zeros", "empty", "ones", "full", "zeros_like", "empty_like", "ones_like", "full_like", "copy"}
    | {f"fft.{kind}{axes}" for kind in ("fft", "ifft", "rfft", "irfft") for axes in ("", "2", "n")}
)


def ordered_reshape_source(value: ast.expr) -> str | None:
    """The source name of ``src.reshape(...)`` or ``np.reshape(src, ...)`` with a non-C ``order=``, else None."""
    if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "reshape"):
        return None
    order = next((keyword.value for keyword in value.keywords if keyword.arg == "order"), None)
    if not (isinstance(order, ast.Constant) and order.value != "C"):
        return None
    numpy_form = is_numpy_module(value.func.value)
    source = (value.args[0] if value.args else None) if numpy_form else value.func.value
    return source.id if isinstance(source, ast.Name) else None


def stored_through(root: ast.AST, name: str) -> bool:
    """Whether some statement under ``root`` writes into ``name``'s elements, as numpy does in place."""
    for node in ast.walk(root):
        if isinstance(node, ast.AugAssign):
            targets: list[ast.expr] = [node.target]
        elif isinstance(node, ast.Assign):
            targets = [target for target in node.targets if isinstance(target, ast.Subscript)]
        else:
            continue
        for target in targets:
            while isinstance(target, ast.Subscript):
                target = target.value
            if isinstance(target, ast.Name) and target.id == name:
                return True
    return False


def name_uses(nodes: Iterable[ast.AST], name: str) -> list[ast.Name]:
    """Every occurrence of ``name`` under ``nodes``, read or bound."""
    return [found for node in nodes for found in ast.walk(node) if isinstance(found, ast.Name) and found.id == name]


def fresh_binding_index(block: list[ast.stmt], at: int, name: str) -> int | None:
    """The index of the last statement before ``at`` binding ``name``, when it binds a new array."""
    for index in range(at - 1, -1, -1):
        stmt = block[index]
        if not name_uses([stmt], name) or not isinstance(stmt, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in stmt.targets):
            continue
        fresh = isinstance(stmt.value, ast.Call) and np_call_name(stmt.value) in FRESH_ARRAY_CALLS
        return index if fresh and len(stmt.targets) == 1 else None
    return None


def statement_blocks(root: ast.AST) -> Iterable[list[ast.stmt]]:
    """Every statement list under ``root``: bodies, else branches and finally blocks."""
    for node in ast.walk(root):
        for field in ("body", "orelse", "finalbody"):
            block = vars(node).get(field)
            if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                yield block


class MaterializeWrittenReshape(ast.NodeTransformer):
    """Copy a non-C-order reshape that is later written through, when its source is private to it.

    numpy returns a fresh array for such a reshape. dace materializes it as well, then refuses a store
    through the result because the write would not reach the source (vexx_k's ``rhocg[nl0] += ...``
    on what ``fwfft`` returned). An explicit ``np.copy`` is the same program without that view.

    Copied only when the source is bound to a new array in the same block, above the reshape, and is
    neither rebound in between nor named anywhere else in the function. Then nothing can observe
    whether the write reached the source, whatever layout numpy gave it.
    """

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        for block in statement_blocks(node):
            for at, stmt in enumerate(block):
                self.materialize(node, block, at, stmt)
        return node

    def materialize(self, fn: ast.FunctionDef, block: list[ast.stmt], at: int, stmt: ast.stmt) -> None:
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
            return
        source = ordered_reshape_source(stmt.value)
        if source is None or not stored_through(fn, stmt.targets[0].id):
            return
        start = fresh_binding_index(block, at, source)
        if start is None:
            return
        rebound = any(isinstance(use.ctx, ast.Store) for use in name_uses(block[start + 1 : at + 1], source))
        if rebound or len(name_uses(block[start : at + 1], source)) != len(name_uses([fn], source)):
            return
        stmt.value = ast.copy_location(ast.Call(func=parse_expr("np.copy"), args=[stmt.value], keywords=[]), stmt.value)


class DesugarUnreplacedCalls(ast.NodeTransformer):
    """Rewrite a numpy call dace has no replacement for; unrewritten it becomes an untyped callback
    ("KeyError: pyobject"). outer -> broadcast product, ascontiguousarray -> copy (also contiguous)."""

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if not (isinstance(node.func, ast.Attribute) and is_numpy_module(node.func.value) and not node.keywords):
            return node
        if node.func.attr == "outer" and len(node.args) == 2:
            a, b = ast.unparse(node.args[0]), ast.unparse(node.args[1])
            return ast.copy_location(ast.parse(f"({a})[:, None] * ({b})[None, :]", mode="eval").body, node)
        if node.func.attr == "ascontiguousarray" and len(node.args) == 1:
            return ast.copy_location(ast.parse(f"({ast.unparse(node.args[0])}).copy()", mode="eval").body, node)
        return node


class DesugarContractionFreeEinsum(ast.NodeTransformer):
    """Rewrite a two-operand einsum that sums NOTHING into the broadcast product it already is.

    ``np.einsum('ei,ek->eik', a, b)`` is ``np.outer`` batched over ``e``: every index survives into
    the output, so no index is contracted. dace routes every two-operand einsum through its GEMM
    path anyway, and with an empty sum that mints a batched MatMul of K=1. ``simplify`` then
    collapses the ``[E, 4, 1]`` and ``[E, 1, 8]`` operand views back to rank 2, which MatMul's
    dispatch has no case for, so expansion dies in ``NotImplementedError: Matrix multiplication not
    implemented for shapes: [E, 4] and [E, 8]`` -- shapes that never conform as a matrix product
    because they were never meant to be one.

    Value-preserving by construction: with nothing summed, each output element is a single product
    of one element from each operand, so there is no accumulation to reorder. Same rewrite
    ``np.outer`` already gets above, generalised to leading batch axes.
    """

    @staticmethod
    def broadcast_index(operand: str, out: str) -> str | None:
        """``operand``'s subscripts as an index into ``out``'s axes, or None if it needs a transpose."""
        positions = [out.index(ch) for ch in operand]
        # An operand whose axes reach the output out of order needs a real transpose; leave it for
        # dace rather than guess a permutation this kernel has never asked for.
        if positions != sorted(positions):
            return None
        return ", ".join(":" if ch in operand else "None" for ch in out)

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "einsum"
            and is_numpy_module(node.func.value)
            and len(node.args) == 3
        ):
            return node
        # ``optimize`` picks a contraction ORDER, and there is no contraction here; any other
        # keyword (``out``, ``dtype``) changes what the call means, so it keeps its einsum.
        if any(kw.arg != "optimize" for kw in node.keywords):
            return node
        spec = node.args[0]
        if not (isinstance(spec, ast.Constant) and isinstance(spec.value, str) and "->" in spec.value):
            return node
        lhs, unused, out = spec.value.replace(" ", "").partition("->")
        operands = lhs.split(",")
        if len(operands) != 2:
            return node
        a, b = operands
        # A repeated subscript is a diagonal and a dropped one is a reduction -- neither is a plain
        # product. Requiring the union to BE the output rules both out, along with any summed index.
        if len({*a}) != len(a) or len({*b}) != len(b) or len({*out}) != len(out):
            return node
        if {*a} | {*b} != {*out}:
            return node
        left, right = self.broadcast_index(a, out), self.broadcast_index(b, out)
        if left is None or right is None:
            return node
        src = f"({ast.unparse(node.args[1])})[{left}] * ({ast.unparse(node.args[2])})[{right}]"
        return ast.copy_location(ast.parse(src, mode="eval").body, node)


class DesugarReverseSlice(ast.NodeTransformer):
    """Rewrite x[::-1] to np.flip(x) -- dace rejects negative-stride subscripts."""

    @staticmethod
    def is_neg_one(node: ast.AST | None) -> bool:
        # ``-1`` parses to ``UnaryOp(USub, Constant(1))``, not ``Constant(-1)``.
        if isinstance(node, ast.Constant):
            return node.value == -1
        return (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and node.operand.value == 1
        )

    def visit_Subscript(self, node: ast.Subscript):
        self.generic_visit(node)
        sl = node.slice
        if isinstance(sl, ast.Slice) and sl.lower is None and sl.upper is None and self.is_neg_one(sl.step):
            # ``axis=0`` is not decoration: ``x[::-1]`` reverses the FIRST axis only, while a bare
            # ``np.flip`` reverses every one of them. The two agree at rank 1 and diverge above it.
            flip = ast.Call(
                func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="flip", ctx=ast.Load()),
                args=[node.value],
                keywords=[ast.keyword(arg="axis", value=ast.Constant(value=0))],
            )
            return ast.copy_location(flip, node)
        return node


class DivisibleStridedSpan(ast.NodeTransformer):
    """Respell a strided slice's stop so its length is DIVISIBLE by the step.

    The tap-loop idiom every pooling and convolution port uses takes one wide strided slice per
    kernel tap: ``padded[..., ky:ky + span:stride]`` with ``span = (out_len - 1) * stride + 1``. That
    span is the tight one -- it stops exactly on the last element -- and it is what makes dace refuse
    the write. dace sizes the slice as ``ceiling(span / stride)``, which for ``A * stride + 1``
    simplifies to ``A + ceiling(1/stride)`` and no further: ``stride`` is a runtime scalar, so sympy
    cannot rule out 0 and will not fold ``ceiling(1/stride)`` to 1. The accumulator it is added into
    is ``A + 1`` long, spelled directly, and the two shapes then fail to broadcast even though they
    are the same number.

    Spelling the span ``(A + 1) * stride`` instead makes ``ceiling((A + 1) * stride / stride)`` fold
    to ``A + 1`` exactly, matching the target with no assumption about ``stride`` at all. The slice
    selects the SAME elements either way -- both yield ``A + 1`` of them for any ``stride >= 1``; only
    the stop moves, from the last element to one full step past it, which a slice clamps.
    """

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        for element in elements:
            if isinstance(element, ast.Slice):
                self.respell(element)
        return node

    def respell(self, element: ast.Slice) -> None:
        """Rewrite ``lower : lower + (A * step + 1) : step`` in place to ``lower : lower + (A + 1) * step``."""
        if element.step is None or element.upper is None or negative_step(element.step):
            return
        step = ast.unparse(element.step)
        span = self.span_expr(element)
        if span is None:
            return
        # ``A * step + 1``: the multiplier must be the step ITSELF, or the rewrite changes which
        # elements the slice picks rather than only where it stops.
        if not (isinstance(span, ast.BinOp) and isinstance(span.op, ast.Add) and is_literal_one(span.right)):
            return
        product = span.left
        if not (isinstance(product, ast.BinOp) and isinstance(product.op, ast.Mult)):
            return
        if ast.unparse(product.right) == step:
            count = product.left
        elif ast.unparse(product.left) == step:
            count = product.right
        else:
            return
        wider = ast.BinOp(
            op=ast.Mult(),
            left=ast.BinOp(left=count, op=ast.Add(), right=ast.Constant(value=1)),
            right=copy.deepcopy(element.step),
        )
        element.upper = (
            wider if element.lower is None else ast.BinOp(left=copy.deepcopy(element.lower), op=ast.Add(), right=wider)
        )
        ast.fix_missing_locations(element)

    @staticmethod
    def span_expr(element: ast.Slice) -> ast.expr | None:
        """The span out of ``upper``: ``upper`` itself when the slice starts at 0, else what is left
        once ``lower`` is taken out of it. A stop that does not contain ``lower`` is not this idiom.

        Over the whole ``+`` chain, not just its top node: ``oy0 + (h - 1) * stride + 1`` parses
        left-associatively, so ``upper.left`` is ``oy0 + (h - 1) * stride`` and ``upper.right`` is
        ``1``, and a match against ``lower`` on either one fails for a slice that carries the idiom.
        """
        upper = element.upper
        if element.lower is None:
            return upper
        lower = ast.unparse(element.lower)
        terms: list[ast.expr] = []
        pending = [upper]
        while pending:
            term = pending.pop()
            if isinstance(term, ast.BinOp) and isinstance(term.op, ast.Add):
                pending.extend([term.right, term.left])
            else:
                terms.append(term)
        spelled = [ast.unparse(t) for t in terms]
        if lower not in spelled:
            return None
        rest = [t for t, text in zip(terms, spelled) if text != lower]
        if len(rest) != len(terms) - 1 or not rest:
            return None  # ``lower`` appearing twice is not this idiom either
        return functools.reduce(lambda left, right: ast.BinOp(left=left, op=ast.Add(), right=right), rest)


def is_literal_one(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == 1 and not isinstance(node.value, bool)


def negative_step(step: ast.expr) -> bool:
    """True when ``step`` is negative on its face.

    The respelling is only an identity for a POSITIVE step: ``ceiling((A*s + 1)/s)`` and
    ``ceiling((A + 1)*s/s)`` are both ``A + 1`` for ``s >= 1``, and disagree for every ``s <= -1``
    (measured: 635 disagreements over lengths 1..24, steps -1..-3). A reverse slice never carries
    this idiom -- ``A*s + 1`` is at most 1 with a negative ``s``, so the tight slice is empty -- and
    ``DesugarReverseSlice`` has already rewritten the literal ones by the time this pass runs. The
    guard is here so the rewrite does not depend on that ordering.

    A symbolic step is treated as positive: it is a stride, positive by construction, and it is the
    same assumption dace's own nonnegative symbol canonicalization makes. A step that is zero is
    rejected by the slice itself, under either spelling.
    """
    if isinstance(step, ast.UnaryOp) and isinstance(step.op, ast.USub):
        return True
    return isinstance(step.value, (int, float)) and step.value < 0 if isinstance(step, ast.Constant) else False


class FlipReplacer(ast.NodeTransformer):
    """Replace a materialisable np.flip(base[lo:hi]) with a reversing-copy workspace slice, via the owner."""

    def __init__(self, owner: "MaterializeDynamicFlip", prelude: list[ast.stmt]) -> None:
        self.owner = owner
        self.prelude = prelude

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)  # innermost flips first (their copy loop precedes the outer's)
        spec = self.owner.match_dynamic_flip(node)
        if spec is None:
            return node
        return self.owner.materialize(spec, self.prelude)


class MaterializeDynamicFlip(ast.NodeTransformer):
    """Materialise a dynamic-length np.flip into a fixed-extent reversing-copy workspace -- dace rejects a View there."""

    def __init__(self, arr_shapes: dict[str, list[str]], arr_dtypes: dict[str, str], symbols: set[str]) -> None:
        self.arr_shapes = arr_shapes
        self.arr_dtypes = arr_dtypes
        self.symbols = set(symbols)
        self.ctr = 0
        self.workspaces: dict[str, tuple[str, str]] = {}  # ws name -> (extent token, dtype expr)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node.body = self.process_body_(node.body)
        if not self.workspaces:
            return node
        decls = [
            ast.parse(f"{ws} = np.zeros(({ext},), dtype={dt})").body[0] for ws, (ext, dt) in self.workspaces.items()
        ]
        at = (
            1
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            )
            else 0
        )
        node.body[at:at] = decls
        ast.fix_missing_locations(node)
        return node

    def visit_For(self, node: ast.For):
        node.body = self.process_body_(node.body)
        node.orelse = self.process_body_(node.orelse)
        return node

    def visit_While(self, node: ast.While):
        node.body = self.process_body_(node.body)
        node.orelse = self.process_body_(node.orelse)
        return node

    def visit_If(self, node: ast.If):
        node.body = self.process_body_(node.body)
        node.orelse = self.process_body_(node.orelse)
        return node

    def process_body_(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for stmt in stmts:
            if isinstance(stmt, (ast.For, ast.While, ast.If)):
                out.append(self.visit(stmt))  # recurse: flips inside nested bodies hoist there
                continue
            prelude: list[ast.stmt] = []
            new_stmt = FlipReplacer(self, prelude).visit(stmt)
            out.extend(prelude)
            out.append(new_stmt)
        return out

    def match_dynamic_flip(self, node: ast.Call) -> tuple[str, ast.expr | None, ast.expr] | None:
        """Return ``(base, lo, hi)`` for a materialisable dynamic-length ``np.flip``, else None."""
        if not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "flip"
            and is_numpy_module(node.func.value)
            and len(node.args) == 1
        ):
            return None
        for kw in node.keywords:  # only a bare / axis=0 flip is an unambiguous axis-0 reverse
            if not (kw.arg == "axis" and isinstance(kw.value, ast.Constant) and kw.value.value == 0):
                return None
        arg = node.args[0]
        if not (
            isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name) and isinstance(arg.slice, ast.Slice)
        ):
            return None
        base = arg.value.id
        if base not in self.arr_shapes or len(self.arr_shapes[base]) != 1 or arg.slice.step is not None:
            return None
        hi = arg.slice.upper
        # A whole-array or static-length reverse lowers on its own; only a runtime-length reverse needs materialising.
        if hi is None or is_symbol_expr(hi, self.symbols):
            return None
        return base, arg.slice.lower, hi

    def materialize(self, spec: tuple[str, ast.expr | None, ast.expr], prelude: list[ast.stmt]) -> ast.AST:
        base, lo, hi = spec
        ws, fi = f"__hpcagent_bench_flip{self.ctr}", f"__hpcagent_bench_fi{self.ctr}"
        self.ctr += 1
        self.workspaces[ws] = (self.arr_shapes[base][0], self.arr_dtypes.get(base, "dc_float"))
        hi_src = ast.unparse(hi)
        length = hi_src if lo is None else f"({hi_src}) - ({ast.unparse(lo)})"
        loop = f"for {fi} in range({length}):\n    {ws}[{fi}] = {base}[({hi_src}) - 1 - {fi}]"
        prelude.append(ast.parse(loop).body[0])
        return ast.parse(f"{ws}[0:{length}]", mode="eval").body


def for_target_names(node: ast.For) -> list[str]:
    """The names a ``for`` binds -- one, or each element of a tuple target."""
    targets = node.target.elts if isinstance(node.target, ast.Tuple) else [node.target]
    return [t.id for t in targets if isinstance(t, ast.Name)]


def uniquify_nested_loop_targets(fn_ast: ast.FunctionDef) -> None:
    """Rename a ``for`` target that shadows an ENCLOSING one -- dace gives both the same variable.

    Python scopes nothing here but keeps an iterator per loop, so two nested ``for _ in range(...)``
    are independent. dace's frontend mints ONE symbol per name and codegen declares it once at
    function scope, so the inner loop's ``_ = 0`` resets the outer loop's counter: distribution_search
    nests ``range(60)`` inside ``range(200)`` on ``_``, the inner one breaks on its first trial, and
    the emitted C loops forever. It is a hang, not a slow kernel -- 1200 s of the numeric gate's cap
    with the answer never arriving.

    Only where the shadowed name is read NOWHERE outside the inner loop's own body. A read after the
    inner loop sees the INNER value in Python, and renaming would quietly change which value that is.
    """
    taken = {n.id for n in ast.walk(fn_ast) if isinstance(n, ast.Name)}

    def rename(node: ast.For, old: str, new: str) -> None:
        for stmt in [node.target] + node.body + node.orelse:
            for name in ast.walk(stmt):
                if isinstance(name, ast.Name) and name.id == old:
                    name.id = new

    def visit(node: ast.AST, enclosing: set[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, ast.For):
                visit(child, enclosing)
                continue
            inner = set(enclosing)
            for old in for_target_names(child):
                if old in enclosing and not read_outside(fn_ast, child, old):
                    stem = old.lstrip("_") or "it"
                    new = next(f"{stem}_nested{k}" for k in itertools.count(1) if f"{stem}_nested{k}" not in taken)
                    taken.add(new)
                    rename(child, old, new)
                    old = new
                inner.add(old)
            visit(child, inner)

    visit(fn_ast, set())
    ast.fix_missing_locations(fn_ast)


def read_outside(fn_ast: ast.FunctionDef, loop: ast.For, name: str) -> bool:
    """Is ``name`` LOADED anywhere in ``fn_ast`` other than inside ``loop``'s body?"""
    inside = {id(n) for stmt in loop.body + loop.orelse for n in ast.walk(stmt)}
    return any(
        isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load) and id(n) not in inside
        for n in ast.walk(fn_ast)
    )


def loop_target_ranks(fn_ast: ast.FunctionDef) -> dict[str, int]:
    """Rank 0 for every ``for`` target name.

    :func:`numpyto_common.numpy_desugar.rank_table` only walks assignments, so a name the loop
    binds has no rank at all -- and a consumer that reads "unknown" as "array" indexes a scalar.
    Iterating a rank-1 value (a range, an index vector, a tuple of coefficients) yields rank-0
    elements; iteration over an array VALUE is already rewritten to an indexed range by
    :class:`DesugarArrayIteration` before this is read.
    """
    ranks: dict[str, int] = {}
    for node in ast.walk(fn_ast):
        if not isinstance(node, ast.For):
            continue
        targets = node.target.elts if isinstance(node.target, ast.Tuple) else [node.target]
        for t in targets:
            if isinstance(t, ast.Name):
                ranks[t.id] = 0
    return ranks


class PointwiseScatterToLoop(ast.NodeTransformer):
    """``A[i, j] = / += rhs`` with INDEX ARRAYS -> the explicit point-wise loop.

    numpy zips the index vectors: element ``p`` of the selection is ``A[i[p], j[p]]``. dace does not
    lower that write at all -- it produced a uniform garbage value across the whole array for
    chebyshev's ``lap[idx, (idx + m) % N] += w``, a SILENT wrong answer rather than a refusal, which
    is why this lowers here instead of waiting for dace to grow the write.

    Only the point-wise WRITE is lowered, and only when every index is a scalar or a rank-1 array:
    a slice or an Ellipsis among the indices is a mixed basic/advanced selection whose result axes
    are not the zip, and a rank>=2 index selects a grid. A repeated value inside one index vector
    ACCUMULATES here where numpy's gather-add-scatter applies the update once -- the same caveat
    :class:`numpyto_common.numpy_desugar.IxWriteToLoop` carries, and undetectable statically.
    """

    def __init__(self, ranks: dict[str, int]) -> None:
        self.ranks = ranks
        self.ctr = 0

    def lower(self, node: ast.Assign | ast.AugAssign, target: ast.expr, op: str) -> ast.stmt | list[ast.stmt]:
        if not (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and isinstance(target.slice, ast.Tuple)
        ):
            return node
        elts = target.slice.elts
        if any(isinstance(e, (ast.Slice, ast.Starred)) for e in elts):
            return node
        if any(isinstance(e, ast.Constant) and e.value is Ellipsis for e in elts):
            return node
        ranks = [expr_rank(e, self.ranks) for e in elts]
        if any(r is None or r > 1 for r in ranks) or 1 not in ranks:
            return node
        prefix = f"__hpcagent_bench_scatter{self.ctr}"
        self.ctr += 1
        lines: list[str] = []

        def bind(expr: ast.expr, tmp: str) -> str:
            """Name the operand once, before the nest: numpy evaluates the whole right-hand side
            before the scattered store, and an in-loop array expression would rebuild it per point."""
            if isinstance(expr, ast.Name):
                return expr.id
            lines.append(f"{tmp} = {ast.unparse(expr)}")
            return tmp

        names = [ast.unparse(e) if r == 0 else bind(e, f"{prefix}_x{k}") for k, (e, r) in enumerate(zip(elts, ranks))]
        value_rank = expr_rank(node.value, self.ranks)
        if value_rank is None or value_rank > 1:
            return node  # an unknown or grid-shaped rhs: guessing how it lines up would be a miscompile
        value = ast.unparse(node.value) if value_rank == 0 else bind(node.value, f"{prefix}_v")
        driver = names[ranks.index(1)]
        it = f"{prefix}_i"
        index = ", ".join(nm if r == 0 else f"{nm}[{it}]" for nm, r in zip(names, ranks))
        rhs = value if value_rank == 0 else f"{value}[{it}]"
        lines.append(f"for {it} in range({driver}.shape[0]):")
        lines.append(f"    {target.value.id}[{index}] {op} {rhs}")
        return [ast.copy_location(stmt, node) for stmt in ast.parse("\n".join(lines)).body]

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        if len(node.targets) != 1:
            return node
        return self.lower(node, node.targets[0], "=")

    def visit_AugAssign(self, node: ast.AugAssign):
        self.generic_visit(node)
        op = AUG_OP_SRC.get(type(node.op))
        return node if op is None else self.lower(node, node.target, op)


class DesugarAugAssign(ast.NodeTransformer):
    """``t op= v`` -> ``t = t op v``: dace turns every augmented store into a WCR edge.

    Canonicalization privatizes those edges into copies CPF refuses, and a broadcasting one on a
    name is an invalid SDFG outright. The read-modify-write is also numpy's meaning: a fancy index
    with repeats updates each element once, where a WCR accumulates the repeats. An array name is
    written back through ``[:]``, in place and in its own dtype; a rank-0 name rebinds, like a numpy
    scalar.

    Index parts that call anything are bound to a temp above the store, so they run once. A target
    that cannot be spelled twice -- a call in its base, a name of unknown rank -- is left alone.
    """

    def __init__(self, ranks: dict[str, int]) -> None:
        self.ranks = ranks
        self.counter = 0

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.stmt | list[ast.stmt]:
        prelude: list[ast.stmt] = []
        store = self.store(node.target, prelude)
        if store is None:
            return node
        load = (
            ast.Name(id=node.target.id, ctx=ast.Load()) if isinstance(node.target, ast.Name) else copy.deepcopy(store)
        )
        load.ctx = ast.Load()
        if isinstance(store, ast.Subscript) and self.index_arrays(store.slice) >= 2:
            loop = self.element_loop(store, node, prelude)
            if loop is None:
                return node
            return [*prelude, ast.fix_missing_locations(ast.copy_location(loop, node))]
        assign = ast.Assign(targets=[store], value=ast.BinOp(left=load, op=node.op, right=node.value))
        return [*prelude, ast.copy_location(assign, node)]

    def index_arrays(self, index: ast.expr) -> int:
        """How many parts of a store's index are arrays."""
        parts = index.elts if isinstance(index, ast.Tuple) else [index]
        return sum(1 for part in parts if not isinstance(part, ast.Slice) and (expr_rank(part, self.ranks) or 0) >= 1)

    def element_loop(self, store: ast.Subscript, node: ast.AugAssign, prelude: list[ast.stmt]) -> ast.For | None:
        """``t[r, c] op= v`` as a loop of element read-modify-writes, index arrays paired by position.

        dace refuses a plain store of an array through more than one index array (cp2k_density_matrix_trs4's
        ``x_blocks[rows, cols, cols] -= s``). A repeated index tuple is updated once per repeat, where numpy's
        buffered store updates it once; an accumulation is spelled ``np.add.at``. Only 1-D index arrays beside
        scalars and a scalar or 1-D value: a slice or a multi-dimensional index array moves axes, so such a
        store keeps its ``op=``.
        """
        parts = store.slice.elts if isinstance(store.slice, ast.Tuple) else [store.slice]
        ranks = [None if isinstance(part, ast.Slice) else expr_rank(part, self.ranks) for part in parts]
        value_rank = expr_rank(node.value, self.ranks)
        if any(rank is None or rank > 1 for rank in ranks) or value_rank is None or value_rank > 1:
            return None
        at = f"__hpcagent_bench_aug{self.counter}"
        self.counter += 1
        arrays = [self.bound(part, prelude) if rank == 1 else part for part, rank in zip(parts, ranks)]
        element = [
            ast.Subscript(value=part, slice=ast.Name(id=at, ctx=ast.Load()), ctx=ast.Load()) if rank == 1 else part
            for part, rank in zip(arrays, ranks)
        ]
        write = ast.Subscript(
            value=copy.deepcopy(store.value), slice=ast.Tuple(elts=element, ctx=ast.Load()), ctx=ast.Store()
        )
        read = copy.deepcopy(write)
        read.ctx = ast.Load()
        value = self.bound(node.value, prelude) if value_rank == 1 else self.once(node.value, prelude)
        if value_rank == 1:
            value = ast.Subscript(value=value, slice=ast.Name(id=at, ctx=ast.Load()), ctx=ast.Load())
        first = next(part for part, rank in zip(arrays, ranks) if rank == 1)
        return ast.For(
            target=ast.Name(id=at, ctx=ast.Store()),
            iter=parse_expr(f"range({ast.unparse(first)}.shape[0])"),
            body=[ast.Assign(targets=[write], value=ast.BinOp(left=read, op=node.op, right=value))],
            orelse=[],
        )

    def bound(self, expr: ast.expr, prelude: list[ast.stmt]) -> ast.Name:
        """``expr`` as a name: a bare name as is, anything else bound to a temp above the loop, evaluated once."""
        if isinstance(expr, ast.Name):
            return ast.Name(id=expr.id, ctx=ast.Load())
        name = f"__hpcagent_bench_aug{self.counter}"
        self.counter += 1
        prelude.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=copy.deepcopy(expr)))
        return ast.Name(id=name, ctx=ast.Load())

    def store(self, target: ast.expr, prelude: list[ast.stmt]) -> ast.Name | ast.Subscript | None:
        if isinstance(target, ast.Name):
            rank = self.ranks.get(target.id)
            if rank is None:
                return None
            if rank == 0:
                return ast.Name(id=target.id, ctx=ast.Store())
            return ast.Subscript(value=ast.Name(id=target.id, ctx=ast.Load()), slice=ast.Slice(), ctx=ast.Store())
        if not isinstance(target, ast.Subscript):
            return None
        base = self.base(target.value, prelude)
        return (
            None if base is None else ast.Subscript(value=base, slice=self.once(target.slice, prelude), ctx=ast.Store())
        )

    def base(self, expr: ast.expr, prelude: list[ast.stmt]) -> ast.expr | None:
        """The array a store lands on, its own indices bound once; None if reaching it calls anything."""
        if isinstance(expr, ast.Name):
            return ast.Name(id=expr.id, ctx=ast.Load())
        if isinstance(expr, ast.Attribute):
            inner = self.base(expr.value, prelude)
            return None if inner is None else ast.Attribute(value=inner, attr=expr.attr, ctx=ast.Load())
        if isinstance(expr, ast.Subscript):
            inner = self.base(expr.value, prelude)
            return (
                None
                if inner is None
                else ast.Subscript(value=inner, slice=self.once(expr.slice, prelude), ctx=ast.Load())
            )
        return None

    def once(self, expr: ast.expr, prelude: list[ast.stmt]) -> ast.expr:
        """``expr`` safe to spell twice: each tuple element or slice bound that calls anything becomes a temp."""
        if isinstance(expr, ast.Tuple):
            return ast.Tuple(elts=[self.once(element, prelude) for element in expr.elts], ctx=ast.Load())
        if isinstance(expr, ast.Slice):
            lower, upper, step = (
                None if part is None else self.once(part, prelude) for part in (expr.lower, expr.upper, expr.step)
            )
            return ast.Slice(lower=lower, upper=upper, step=step)
        if not any(isinstance(sub, ast.Call) for sub in ast.walk(expr)):
            return copy.deepcopy(expr)
        name = f"__hpcagent_bench_aug{self.counter}"
        self.counter += 1
        prelude.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=expr))
        return ast.Name(id=name, ctx=ast.Load())


#: Builtins that return a Python scalar whatever they read.
SCALAR_BUILTINS = frozenset({"bool", "complex", "float", "int"})


def binding_rank(value: ast.expr, ranks: dict[str, int]) -> int | None:
    """:func:`expr_rank`, plus the scalar builtins it leaves unranked: cp2k's ``span0 = int(..)``."""
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in SCALAR_BUILTINS:
        return 0
    return expr_rank(value, ranks)


def settled_ranks(fn: ast.AST, ranks: dict[str, int]) -> dict[str, int]:
    """``ranks`` plus each unranked name whose bindings all agree on scalar-or-array.

    ``conv = np.zeros(<5-d>)`` then ``conv = conv.reshape(<4-d>)`` has no one rank, but it is an
    array at every point, and that is all :class:`DesugarAugAssign` asks of a name.
    """
    seen: dict[str, set[int | None]] = {}
    for name, value in name_binding_index(fn)[0]:
        seen.setdefault(name, set()).add(binding_rank(value, ranks))
    settled = dict(ranks)
    for name, bound in seen.items():
        known = {rank for rank in bound if rank is not None}
        if name not in settled and known == bound and (min(known) >= 1 or max(known) == 0):
            settled[name] = min(known)
    return settled


def is_full_slice(node: ast.AST) -> bool:
    """True iff a subscript index selects everything -- ``[:]``, or a tuple of ``:``."""
    if isinstance(node, ast.Slice):
        return node.lower is None and node.upper is None and node.step is None
    return isinstance(node, ast.Tuple) and bool(node.elts) and all(is_full_slice(e) for e in node.elts)


class DropRedundantSliceStore(ast.NodeTransformer):
    """``cn[l][:] = v`` -> ``cn[l] = v``: dace mis-sizes the chained store. Base must ALREADY be a
    subscript -- on a bare name ``y[:] = v`` writes in place where ``y = v`` would rebind."""

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        node.targets = [self.trim(target) for target in node.targets]
        return node

    def trim(self, target: ast.expr) -> ast.expr:
        while (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Subscript)
            and is_full_slice(target.slice)
        ):
            target = target.value
            target.ctx = ast.Store()
        return target


def dace_chained_assign_split(seed_ranks: dict[str, int] | None = None) -> SplitChainedAssign:
    """dace cannot codegen ``a = b = rhs``: split it, a literal repeated (dace issue 05), a temp ``__hpcagent_bench_chain<k>``."""
    return SplitChainedAssign(lambda ordinal: f"__hpcagent_bench_chain{ordinal}", True, seed_ranks)


class SubstituteNames_(ast.NodeTransformer):
    """Replace every load of a name in ``mapping`` with a copy of its expression."""

    def __init__(self, mapping: dict[str, ast.AST]) -> None:
        self.mapping = mapping

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id in self.mapping:
            return ast.copy_location(copy.deepcopy(self.mapping[node.id]), node)
        return node


class DropAliasAssign(ast.NodeTransformer):
    """Drop ``<name> = ...`` for each inlined alias name (its uses are substituted)."""

    def __init__(self, names: Iterable[str]) -> None:
        self.names = set(names)

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in self.names:
            return None
        return node


def view_slice_binding(node: ast.stmt) -> str | None:
    """The bound name of ``name = arr[...]`` when the subscript keeps a dimension, else ``None``.

    That is the spelling numpy answers with a VIEW rather than a copy or a scalar, and the one dace
    turns into a View node.
    """
    if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
        return None
    target, value = node.targets[0], node.value
    if not (isinstance(target, ast.Name) and isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name)):
        return None
    index = value.slice
    elements = index.elts if isinstance(index, ast.Tuple) else [index]
    if not any(isinstance(element, (ast.Slice, ast.Starred)) for element in elements):
        return None
    return target.id


def bare_alias_binding(node: ast.stmt, symbols: frozenset[str] = frozenset()) -> str | None:
    """The bound name of ``name = other`` -- the whole-array spelling numpy answers with a view.

    ``arr[...]`` is not the only way to reach a View node: dace makes one for a bare rebinding too,
    and refuses the next one exactly the same way. esirkepov's ``idx`` is bound to ``j`` in two arms
    of a five-way branch and to ``j - 1`` in the rest, which comes back as ``Variable __inl11_idx
    has been already defined`` (or ``Cannot reassign View`` when both arms are bare).
    """
    if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
        return None
    target, value = node.targets[0], node.value
    if not (isinstance(target, ast.Name) and isinstance(value, ast.Name)):
        return None
    # A dc.symbol is not storage: ``m_iter = m`` reads a scalar the caller bound, and copying it
    # asks dace for an array of a symbol. gmres seeds its runtime count exactly this way.
    return None if value.id in symbols else target.id


def view_binding(node: ast.stmt, symbols: frozenset[str] = frozenset()) -> str | None:
    """Either spelling that leaves dace holding a View: a kept-dimension slice or a bare alias."""
    return view_slice_binding(node) or bare_alias_binding(node, symbols)


def as_stmt_block(raw: object) -> list[ast.stmt]:
    """One non-empty statement list off an ast node's field dict; anything else reads as empty.

    ``isinstance(raw, list)`` proves a sequence and nothing about its members, so the first member
    is the evidence that decides -- an ast field holding a list holds one node type throughout."""
    return cast("list[ast.stmt]", raw) if isinstance(raw, list) and raw and isinstance(raw[0], ast.stmt) else []


def statement_lists(root: ast.AST) -> list[list[ast.stmt]]:
    """Every statement list in the subtree -- the blocks a name's live range can be confined to."""
    blocks: list[list[ast.stmt]] = []
    for parent in ast.walk(root):
        # An ast node keeps its fields in ``__dict__``, and most node types carry none of these.
        for field in ("body", "orelse", "finalbody"):
            block = as_stmt_block(vars(parent).get(field))
            if block:
                blocks.append(block)
    return blocks


#: numpy calls that build a FRESH buffer, so the name they bind is a new array rather than a rebind
#: of the old one. dace sizes one descriptor per name and refuses a second of a different shape.
ALLOCATION_CALLS = frozenset({"empty", "zeros", "ones", "full", "empty_like", "zeros_like", "ones_like", "full_like"})


def allocation_binding(node: ast.stmt) -> str | None:
    """The bound name of ``name = np.empty(..)`` and friends, else ``None``."""
    if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
        return None
    call = node.value
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr in ALLOCATION_CALLS):
        return None
    return node.targets[0].id


def value_binding(node: ast.stmt) -> str | None:
    """The bound name of any ``name = <expr>``, else ``None``.

    The widest of the binding predicates, and the one that catches what the others miss: a name
    bound to a COMPUTED value in two arms of a branch. dace gives it one descriptor and refuses the
    second binding (``Cannot reassign value to variable``), whether or not the two are spelled the
    same -- esirkepov's ``cum_x = np.cumsum(...)`` appears verbatim in three arms of a five-way
    branch, and conv_pointwise_2d's ``padded`` is an allocation in one arm and a plain alias in the
    other. Safe to be this wide only because :func:`version_rebound_names` versions nothing whose
    bindings do not already have disjoint live ranges: anything needing a phi is declined, not
    renamed.
    """
    if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
        return None
    return node.targets[0].id


def binding_regions(
    blocks: list[list[ast.stmt]], name: str, binding_of: Callable[[ast.stmt], str | None]
) -> list[tuple[ast.Assign, list[ast.AST]]]:
    """``(binding, statements the binding owns)`` per binding of ``name``, in source order.

    A binding's region runs from the statement AFTER it to the next binding in the same block. The
    next binding's own right-hand side belongs to this region, not to itself: ``e = e[1:]`` reads
    the value the PREVIOUS binding holds.
    """
    regions: list[tuple[ast.Assign, list[ast.AST]]] = []
    for block in blocks:
        indices = [i for i, stmt in enumerate(block) if binding_of(stmt) == name]
        for position, index in enumerate(indices):
            binding = block[index]
            if not isinstance(binding, ast.Assign):
                continue
            stop = indices[position + 1] if position + 1 < len(indices) else len(block)
            owned: list[ast.AST] = list(block[index + 1 : stop])
            following = block[stop] if stop < len(block) else None
            if isinstance(following, ast.Assign):
                owned.append(following.value)
            regions.append((binding, owned))
    regions.sort(key=lambda region: region[0].lineno)
    return regions


def written_through(fn: ast.AST) -> set[str]:
    """Names an element or slice store lands on. A copy of one of those is not the same array."""
    names: set[str] = set()
    for node in ast.walk(fn):
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else ([node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else [])
        )
        # A tuple target is a list of stores, not one: ``a[i], b[j] = ...`` lands on both. Missing
        # that read daubechies_dwt2d's four quadrant writes as no store at all.
        flat: list[ast.expr] = []
        for target in targets:
            flat.extend(target.elts if isinstance(target, ast.Tuple) else [target])
        for target in flat:
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                names.add(target.value.id)
    return names


def copy_view_bindings(fn: ast.FunctionDef, names: set[str], symbols: frozenset[str] = frozenset()) -> None:
    """Rewrite each view binding of ``names`` to ``np.copy(..)``, in place.

    The name stops being a View and becomes a plain array, which dace rebinds freely as long as the
    shape holds. A name written THROUGH is left alone: a copy no longer reaches the base array, and
    a wrong port is worse than an unported kernel.
    """
    names = set(names) - written_through(fn)
    if not names:
        return
    for block in statement_lists(fn):
        for stmt in block:
            if isinstance(stmt, ast.Assign) and view_binding(stmt, symbols) in names:
                copied = ast.Call(
                    func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="copy", ctx=ast.Load()),
                    args=[stmt.value],
                    keywords=[],
                )
                stmt.value = ast.copy_location(copied, stmt.value)


def views_of_written_bases(fn: ast.FunctionDef) -> set[str]:
    """View names whose BASE array is stored into later in the same block.

    dace's simplify fuses a straight-line block into ONE dataflow state, and inside one state there
    is no ordering edge between a read through a View and a write to the array that view reads:
    codegen may serialize the write first. daubechies_dwt2d binds ``block = out[:s, :s]``, reads it
    for both column bands, then writes four quadrants of ``out``; one term of the high band was
    emitted after the first quadrant write and read back what that write had just replaced.

    Only bindings whose every read precedes the first such store are named. A kernel that reads the
    view AFTER writing the base is relying on the aliasing, and a copy would answer the wrong array.
    """
    names: set[str] = set()
    for block in statement_lists(fn):
        # Per statement, once per block: rescanning the tail for every binding was quadratic.
        stores: list[set[str]] = []
        reads: list[set[str]] = []
        for index, stmt in enumerate(block):
            name = view_slice_binding(stmt)
            # A view slice binding is ``<name> = <array>[<slice>]``, so both bases are named.
            if name is None or not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Subscript):
                continue
            if not isinstance(stmt.value.value, ast.Name):
                continue
            if not stores:
                stores = [written_through(later) for later in block]
                reads = [loaded_names(later) for later in block]
            base = stmt.value.value.id
            tail = range(index + 1, len(block))
            store = next((i for i in tail if base in stores[i]), None)
            if store is None:
                continue
            read = max((i for i in tail if name in reads[i]), default=-1)
            if read < store:
                names.add(name)
    return names


def loaded_names(node: ast.AST) -> set[str]:
    """Every name READ in the subtree."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def mixed_view_names(fn: ast.FunctionDef, symbols: frozenset[str] = frozenset()) -> set[str]:
    """Names bound BOTH to a view and to a computed value.

    dace makes a View node for ``horiz = padded[:, 0:W]`` and then refuses the ``horiz =
    np.maximum(horiz, ..)`` that follows (``Cannot reassign View``; the loop-carried spelling says
    ``Variable .. has been already defined``). Versioning cannot separate them -- the value binding
    reads the name it rebinds -- so the view becomes the array that binding materializes anyway.
    """
    views: set[str] = set()
    valued: set[str] = set()
    for block in statement_lists(fn):
        for stmt in block:
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if not isinstance(target, ast.Name):
                    continue
                (views if view_binding(stmt, symbols) == target.id else valued).add(target.id)
    for node in ast.walk(fn):
        target = node.target if isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For)) else None
        if isinstance(target, ast.Name):
            valued.add(target.id)
    return views & valued


def inplace_update_targets(fn: ast.FunctionDef) -> set[int]:
    """``id()`` of every bare-name ``x += ..`` target -- a store that UPDATES rather than binds.

    numpy and dace agree on what one of these means: the buffer the name already holds is read and
    written, its shape untouched. So it is not a rebinding, and the name's value still comes from
    the bindings alone -- which is what :func:`version_rebound_names` has to know before it may
    split a name whose accumulate sits between two of them.
    """
    return {
        id(node.target)
        for node in ast.walk(fn)
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name)
    }


def version_rebound_views(fn: ast.FunctionDef, symbols: frozenset[str] = frozenset()) -> list[str]:
    """Give each rebinding of a view name its own name. Returns the names it DECLINED.

    Both View spellings count. ls3df_scf's inlined CheFSI binds ``X = <reshaped block>`` and swaps
    ``X, Y = Y, Ynew`` in the loop: every binding of ``X`` is a bare alias, and dace refused the
    loop's one (``Cannot reassign View``) because only a slice binding was considered here.
    """
    return version_rebound_names(fn, lambda stmt: view_binding(stmt, symbols))


def version_reallocations(fn: ast.FunctionDef) -> None:
    """Give each re-ALLOCATION of a name its own name, where the allocations differ.

    ``padded = np.empty((H, W + 2 * r))`` then ``padded = np.empty((H + 2 * r, W))`` is one dace
    descriptor asked to hold two shapes: ``Cannot reassign value to variable "padded"``. Two names
    are two descriptors. Allocations spelled identically are left alone -- dace accepts those, and a
    second name would cost a second buffer for nothing.
    """
    spellings: dict[str, set[str]] = {}
    for block in statement_lists(fn):
        for stmt in block:
            name = allocation_binding(stmt)
            if name is not None and isinstance(stmt, ast.Assign):
                spellings.setdefault(name, set()).add(ast.unparse(stmt.value))
    version_rebound_names(fn, allocation_binding, {n for n, texts in spellings.items() if len(texts) > 1})


def binds_a_view(node: ast.stmt) -> bool:
    """``name = <expr>[...]`` whose subscript keeps a dimension, over any base: the View dace refuses to
    rebind. A chained ``a[f][..., 0]`` counts, which :func:`view_slice_binding` does not name; a bare
    ``name = other`` does not, since without the symbol table ``edge = N`` may read a dc.symbol."""
    if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Subscript)):
        return False
    index = node.value.slice
    elements = index.elts if isinstance(index, ast.Tuple) else [index]
    return any(
        isinstance(element, (ast.Slice, ast.Starred))
        or (isinstance(element, ast.Constant) and element.value is Ellipsis)
        for element in elements
    )


#: The binding a read sees when no binding of the name precedes it: a parameter, a global, or nothing.
UNBOUND = 0


def statements_touching(fn: ast.FunctionDef, name: str) -> set[int]:
    """``id()`` of every statement whose subtree names ``name`` or leaves its block early.

    Every other statement neither binds, reads nor redirects the name, so a dataflow walk may skip it.
    """
    touching: set[int] = set()

    def visit(node: ast.AST) -> bool:
        hit = (isinstance(node, ast.Name) and node.id == name) or isinstance(
            node, (ast.Break, ast.Continue, ast.Return, ast.Raise)
        )
        for child in ast.iter_child_nodes(node):
            hit = visit(child) or hit
        if hit and isinstance(node, ast.stmt):
            touching.add(id(node))
        return hit

    visit(fn)
    return touching


class ReachingBindings:
    """Which bindings of one name reach each ``Name`` node of it, over a structured function body.

    Reaching definitions on the ast: an ``if`` joins its arms, a loop iterates to a fixed point and
    joins its ``break`` states, ``return`` and ``raise`` reach nothing. A statement it does not model
    (``try``, ``with``, ``match``, a nested def) that touches the name clears ``sound``.
    """

    __slots__ = ("name", "bindings", "touching", "reached", "breaks", "continues", "sound")

    def __init__(self, fn: ast.FunctionDef, name: str, bindings: set[int]) -> None:
        self.name = name
        self.bindings = bindings
        self.touching = statements_touching(fn, name)
        self.reached: dict[int, frozenset[int]] = {}
        self.breaks: list[frozenset[int]] = []
        self.continues: list[frozenset[int]] = []
        self.sound = True
        self.block(fn.body, frozenset({UNBOUND}))

    def record(self, node: ast.AST, state: frozenset[int]) -> None:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id == self.name:
                self.reached[id(sub)] = self.reached.get(id(sub), frozenset()) | state

    def block(self, stmts: list[ast.stmt], state: frozenset[int]) -> frozenset[int]:
        for stmt in stmts:
            if id(stmt) in self.touching:
                state = self.statement(stmt, state)
        return state

    def statement(self, stmt: ast.stmt, state: frozenset[int]) -> frozenset[int]:
        if isinstance(stmt, ast.Assign) and id(stmt) in self.bindings:
            self.record(stmt.value, state)
            return frozenset({id(stmt)})
        if isinstance(stmt, ast.If):
            self.record(stmt.test, state)
            return self.block(stmt.body, state) | self.block(stmt.orelse, state)
        if isinstance(stmt, (ast.For, ast.While)):
            return self.loop(stmt, state)
        if isinstance(stmt, (ast.Break, ast.Continue)):
            (self.breaks if isinstance(stmt, ast.Break) else self.continues).append(state)
            return frozenset()
        if "body" in vars(stmt) or isinstance(stmt, ast.Match):
            self.sound = False
            return state
        self.record(stmt, state)
        return frozenset() if isinstance(stmt, (ast.Return, ast.Raise)) else state

    def loop(self, stmt: ast.For | ast.While, state: frozenset[int]) -> frozenset[int]:
        if isinstance(stmt, ast.For):
            self.record(stmt.iter, state)
        outer = (self.breaks, self.continues)
        head = state
        while True:
            self.breaks, self.continues = [], []
            if isinstance(stmt, ast.While):
                self.record(stmt.test, head)
            widened = head.union(self.block(stmt.body, head), *self.continues)
            if widened == head:
                break
            head = widened
        breaks = self.breaks
        self.breaks, self.continues = outer
        return self.block(stmt.orelse, head).union(*breaks)


def sole_reaching_bindings(
    fn: ast.FunctionDef, name: str, bindings: set[int], touches: list[ast.Name]
) -> dict[int, int] | None:
    """``id(touch) -> id(binding)`` when exactly one binding of ``name`` reaches every touch, else ``None``.

    A touch two bindings reach needs a phi; one no binding reaches reads a value from outside them.
    """
    reaching = ReachingBindings(fn, name, bindings)
    if not reaching.sound:
        return None
    owners: dict[int, int] = {}
    for touch in touches:
        sources = reaching.reached.get(id(touch), frozenset())
        if len(sources) != 1 or UNBOUND in sources:
            return None
        owners[id(touch)] = next(iter(sources))
    return owners


def fresh_version(name: str, version: int, taken: set[str]) -> str:
    """``<name>__v<version>``, counting past every spelling in ``taken``; the result is reserved."""
    renamed = f"{name}__v{version}"
    while renamed in taken:
        version += 1
        renamed = f"{name}__v{version}"
    taken.add(renamed)
    return renamed


def version_rebound_names(
    fn: ast.FunctionDef,
    binding_of: Callable[[ast.stmt], str | None],
    candidates: set[str] | None = None,
) -> list[str]:
    """Give each rebinding of a name its own name, in place. Returns the names it DECLINED.

    ``col = a[k]`` twice is a numpy REFERENCE rebind, but dace makes a View node per binding and the
    second has nowhere to go (``Cannot reassign View``). Distinct names say the same thing, and cost
    nothing -- a view is a descriptor, not a buffer.

    Only names whose bindings already have disjoint live ranges are versioned. A name bound in one
    branch of a conditional and read after the merge needs a phi, and so does one rebound inside a
    loop and read after it; both show up as a read reachable from two regions, or from none.
    Renaming those would bind the read to whichever binding the parser saw last, so they are
    declined here for :func:`copy_view_bindings`, which pays for a buffer to say the same thing.

    Regions are built per statement list, so a binding NESTED inside another's extent needs its own
    decline: the outer region's owned statements include the whole loop or branch, and every read
    the inner binding feeds is counted against the outer region alone. The read-ownership check
    then passes while the inner region owns nothing, and versioning it produces a dead store --
    gmres' ``m_iter``, seeded at top level and advanced by ``m_iter = k + 1`` two blocks down,
    stopped advancing. Bindings in SIBLING blocks are unaffected, which is the common case this
    function exists for: esirkepov binds ``cum_x`` in three arms of one branch, none inside another.
    A nested name is still versioned when reaching definitions show exactly one binding reaches each
    touch (:func:`sole_reaching_bindings`). cegterg binds ``psi_k = psi[:kdim, :nbase]`` before its
    loop and in three branch arms inside it, each read right after; the copies that declining made
    were sized by different versions of the reassigned ``nbase`` symbol, which dace refuses to rebind.

    ``acc += tap`` UPDATES the binding in scope rather than making a new one, so it is read like a
    read and renamed like one -- the accumulate belongs to whichever region reaches it. Counting it
    as a foreign store instead declined every accumulator that is later reshaped, which is the shape
    conv_depthwise_2d_square_input_asymmetric_kernel's ``out = out.reshape(..)`` asks dace to give
    one descriptor.

    A nested value binding is not declined when an OUTER binding of the name is a view and there are
    two or more outer bindings: it is part of the live range of the outer region enclosing it, and
    renames with that region. ls3df_scf binds ``v`` to a view, rebinds it to ``v / norm``, then
    updates it in a loop; declining the whole name for the loop left the View and the value under
    one name, which dace refuses. A name bound only to values keeps its name: dace rebinds those.
    """
    declined: list[str] = []
    blocks = statement_lists(fn)
    stores: dict[str, list[ast.Name]] = {}
    loads: dict[str, list[ast.Name]] = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Name):
            (stores if isinstance(node.ctx, ast.Store) else loads).setdefault(node.id, []).append(node)
    updates = inplace_update_targets(fn)
    taken = set(loads) | set(stores) | {arg.arg for arg in fn.args.args}
    # Node ids under each statement, walked once per call: renaming rewrites Name.id, never the tree.
    subtree: dict[int, set[int]] = {}

    def node_ids(stmt: ast.AST) -> set[int]:
        found = subtree.get(id(stmt))
        if found is None:
            found = subtree[id(stmt)] = {id(node) for node in ast.walk(stmt)}
        return found

    for name in sorted({n for block in blocks for stmt in block if (n := binding_of(stmt))}):
        if candidates is not None and name not in candidates:
            continue
        regions = binding_regions(blocks, name, binding_of)
        if len(regions) < 2:
            continue
        bound_here = {id(binding.targets[0]) for binding, unused in regions}
        if any(id(store) not in bound_here | updates for store in stores.get(name, [])):
            declined.append(name)
            continue  # something else writes the name; its value is no longer just these bindings
        reached = [set().union(*map(node_ids, owned)) for unused, owned in regions]
        nested = {
            id(binding) for binding, owned_statements in regions if any(id(binding) in nodes for nodes in reached)
        }
        touches = [node for node in loads.get(name, []) + stores.get(name, []) if id(node) not in bound_here]
        if nested:
            # A nested value binding belongs to the one outer region enclosing it and renames with it.
            # Only done to separate a View from the values bound after it: dace rebinds a value name.
            outer = [index for index, (binding, owned_statements) in enumerate(regions) if id(binding) not in nested]
            if (
                len(outer) < 2
                or not any(binds_a_view(regions[index][0]) for index in outer)
                or any(view_binding(binding) for binding, owned_statements in regions if id(binding) in nested)
                or any(sum(key in reached[index] for index in outer) != 1 for key in nested)
            ):
                owners = sole_reaching_bindings(fn, name, {id(binding) for binding, unused in regions}, touches)
                if owners is None:
                    declined.append(name)
                    continue  # a binding NESTED in another's extent: the reads after it belong to both
                # Every touch sees one binding, so each binding names its own touches wherever they sit.
                spelled = {id(regions[0][0]): name}
                for version, (binding, owned) in enumerate(regions[1:], start=2):
                    spelled[id(binding)] = fresh_version(name, version, taken)
                    bound = binding.targets[0]
                    if isinstance(bound, ast.Name):
                        bound.id = spelled[id(binding)]
                for touch in touches:
                    touch.id = spelled[owners[id(touch)]]
                continue
            regions = [regions[index] for index in outer]
            reached = [reached[index] for index in outer]
        if any(sum(id(touch) in nodes for nodes in reached) != 1 for touch in touches):
            declined.append(name)
            continue  # a read or update no region owns, or one two regions reach: neither is a rename
        for version, (binding, owned) in enumerate(regions[1:], start=2):
            renamed = fresh_version(name, version, taken)
            bound = binding.targets[0]
            if isinstance(bound, ast.Name):
                bound.id = renamed
            for stmt in owned:
                for node in ast.walk(stmt):
                    if isinstance(node, ast.Name) and node.id == name:
                        node.id = renamed
    return declined


#: numpy allocators whose first arg is a shape tuple (dims dace requires to be symbolic).
#: Calls whose result has the same shape as their first shaped argument -- elementwise, so a read of
#: ``.shape`` on the result is a read of that argument's shape.
ELEMENTWISE_CALLS = frozenset(
    {
        "maximum",
        "minimum",
        "add",
        "subtract",
        "multiply",
        "divide",
        "power",
        "exp",
        "log",
        "sqrt",
        "tanh",
        "sin",
        "cos",
        "abs",
        "absolute",
        "where",
        "clip",
        "sign",
        "floor",
        "ceil",
        "round",
        "square",
        "reciprocal",
        "negative",
    }
)


def inserts_axis(element: ast.AST) -> bool:
    """True iff a subscript element INSERTS a length-1 axis -- ``None`` or ``np.newaxis``."""
    if isinstance(element, ast.Constant) and element.value is None:
        return True
    return (isinstance(element, ast.Name) and element.id == "newaxis") or (
        isinstance(element, ast.Attribute) and element.attr == "newaxis"
    )


class ResolveShapeReads(ast.NodeTransformer):
    """Rewrite every ``<name>.shape[k]`` to the symbolic extent in effect at that point.

    DaCe has no runtime ``.shape``: an array's extents ARE symbols, so a shape read has to be
    resolved before the frontend sees it. ``ShapeToSymbol`` did this for the declared arguments
    only, and a read on a TRANSIENT survived -- ``(h.shape[3] + 2 - kw) // 1 + 1``. That is not
    merely unresolved: it makes the enclosing size expression non-symbolic, and because
    :func:`plan_size_promotion` is all-or-nothing, ONE such read stops every size scalar in the
    kernel from becoming a symbol. The whole conv family refuses on that.

    The table is flow-sensitive -- ``h`` is rebound per layer and its extents change with it -- so
    the target's shape is learned only AFTER its right-hand side is rewritten, and statements are
    visited in order.

    Inference is deliberately conservative: an extent guessed wrong is a miscompile, not a refusal.
    Only an alias, an allocation, a reshape, a transpose, a rank-2 matmul (exact, not a guess) and a
    broadcast whose every operand is known are inferred; ONE unknown operand poisons the whole
    expression, leaving the name unknown and its ``.shape`` read intact. Taking the known side of an
    elementwise pair -- what this did before -- is what miscompiled ``flat @ clusters - bn_mean``:
    the rank-2 matmul is unknown, so the result adopted ``bn_mean``'s RANK-1 shape and axis 1's
    extent was read as axis 0's.
    """

    def __init__(self, shapes: dict[str, list[str]]) -> None:
        self.shapes: dict[str, list[str]] = {k: [fold_shape_expr(t) for t in v] for k, v in shapes.items()}
        self.aliases: dict[str, ast.AST] = {}
        self.alias_seen: set[str] = set()

    def canon(self, token: str) -> str:
        """An extent token with its size-scalar aliases substituted away, then folded -- resnet's
        residual reaches one extent as both ``__inl12_oh`` and ``__inl3_oh``."""
        if not self.aliases:
            return fold_shape_expr(token)
        try:
            tree = ast.parse(token, mode="eval")
        except SyntaxError:
            return token
        return fold_shape_expr(ast.unparse(SubstituteNames_(self.aliases).visit(tree).body))

    def note_alias(self, name: str, value: ast.AST) -> None:
        """Record ``name = <integer expression>`` so :meth:`canon` can substitute it away."""
        rank0 = {nm for nm, shape in self.shapes.items() if shape == []}
        reads_self = any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(value))
        # A rebind and a self-update (``n = n + 1``, once per loop trip) are both dropped: an alias
        # standing for the wrong trip would equate two extents that differ.
        if name in self.alias_seen or reads_self or not is_symbol_expr(value, rank0 | set(self.aliases)):
            self.aliases.pop(name, None)
            self.alias_seen.add(name)
            return
        self.alias_seen.add(name)
        # Folded on the way in: without it alias N carries alias N-1's whole expansion, so the AST
        # deepens once per layer and resnet101's 101 layers overflow the deepcopy in _SubstituteNames.
        self.aliases[name] = fold_expr(SubstituteNames_(self.aliases).visit(copy.deepcopy(value)))

    def cumulative_axis(self, node: ast.Call) -> tuple[ast.expr, int] | None:
        """``(operand, axis)`` of an ``np.cumsum``/``np.cumprod`` written with a literal axis, else
        ``None``. A ``dtype=``/``out=`` spelling is left alone: the rewrite below moves the axis, and
        carrying the rest of the call across it would be a guess about what they mean here."""
        if len(node.args) == 2 and not node.keywords:
            axis = node.args[1]
        elif len(node.args) == 1 and len(node.keywords) == 1 and node.keywords[0].arg == "axis":
            axis = node.keywords[0].value
        else:
            return None
        if not (isinstance(axis, ast.Constant) and isinstance(axis.value, int)):
            return None
        return node.args[0], axis.value

    def visit_Call(self, node: ast.Call):
        """``np.swapaxes(x, i, j)`` -> ``np.transpose(x, perm)``, and an INNER-axis cumulative scan
        -> the same scan on the last axis between two transposes. Both need the operand RANK, and
        this table is the emitter's only flow-SENSITIVE one -- netvlad rebinds a name across ranks."""
        self.generic_visit(node)
        if not (isinstance(node.func, ast.Attribute) and is_numpy_module(node.func.value)):
            return node
        if node.func.attr == "swapaxes" and len(node.args) == 3 and not node.keywords:
            shape = self.infer(node.args[0])
            axes = [a.value for a in node.args[1:] if isinstance(a, ast.Constant) and isinstance(a.value, int)]
            if not shape or len(axes) != 2:
                return node
            perm = list(range(len(shape)))
            i, j = axes[0] % len(shape), axes[1] % len(shape)
            perm[i], perm[j] = perm[j], perm[i]
            order = ", ".join(str(p) for p in perm)
            return ast.copy_location(
                ast.parse(f"np.transpose({ast.unparse(node.args[0])}, ({order}))", mode="eval").body, node
            )
        # dace lowers a prefix scan along the LAST axis only -- an inner axis is a strided chain per
        # outer index, which its Scan libnode's single ``stride`` cannot express. The scan axis is
        # swapped to the end, scanned there, and swapped back; the permutation is its own inverse,
        # so one order string spells both transposes.
        if node.func.attr in ("cumsum", "cumprod"):
            spec = self.cumulative_axis(node)
            shape = self.infer(spec[0]) if spec else None
            if spec is None or not shape or len(shape) < 2:
                return node
            operand, axis = spec
            rank = len(shape)
            axis %= rank
            if axis == rank - 1:
                return node
            perm = list(range(rank))
            perm[axis], perm[rank - 1] = perm[rank - 1], perm[axis]
            order = ", ".join(str(p) for p in perm)
            scan = f"np.{node.func.attr}(np.transpose({ast.unparse(operand)}, ({order})), axis={rank - 1})"
            return ast.copy_location(ast.parse(f"np.transpose({scan}, ({order}))", mode="eval").body, node)
        return node

    def visit_Subscript(self, node: ast.Subscript):
        self.generic_visit(node)
        value = node.value
        # Inferred, not looked up by name: densenet reads a dimension off a SLICE (y[:, 0:64]).
        if (
            isinstance(value, ast.Attribute)
            and value.attr == "shape"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)
        ):
            tokens = self.infer(value.value)
            if tokens is not None and 0 <= node.slice.value < len(tokens):
                token = fold_shape_expr(tokens[node.slice.value])
                return ast.copy_location(ast.parse(token, mode="eval").body, node)
        return node

    def visit_Assign(self, node: ast.Assign):
        node.value = self.visit(node.value)  # resolve reads against the shapes in effect BEFORE this
        inferred = self.infer(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                if inferred is not None:
                    self.shapes[target.id] = [fold_shape_expr(t) for t in inferred]
                elif not self.accumulates(target.id, node.value):
                    self.shapes.pop(target.id, None)  # rebound to something unknown: forget the old
                if not inferred:  # rank 0 or unknown: the only forms that can be a size alias
                    self.note_alias(target.id, node.value)
        return node

    def visit_For(self, node: ast.For):
        """A ``range`` loop target is a rank-0 integer: lstm indexes ``w_hh[l - 1]`` with one."""
        if (
            isinstance(node.target, ast.Name)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ):
            self.shapes[node.target.id] = []
        self.generic_visit(node)
        return node

    def accumulates(self, name: str, value: ast.AST) -> bool:
        """True iff ``value`` is an elementwise UPDATE of ``name`` -- ``out = np.maximum(out, ...)``,
        alexnet's max-pool workspace. Not a guess: one transient keeps one descriptor, so a write
        back into it has the shape dace already has. ``@`` is excluded -- it changes the extents."""
        if name not in self.shapes:
            return False
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.MatMult):
            return False
        elementwise = isinstance(value, (ast.BinOp, ast.UnaryOp)) or (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr in ELEMENTWISE_CALLS
        )
        return elementwise and any(
            isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load) for n in ast.walk(value)
        )

    def tuple_tokens(self, node: ast.AST) -> list[str] | None:
        """The extent tokens of a shape ARGUMENT, or None when its RANK is not established.

        A tuple spells its own extents. A ``.shape`` read carries the array's WHOLE rank:
        ``N = np.zeros(C.shape)`` reading as the rank-1 ``['C.shape']`` is what rewrote
        ``N.shape[0]`` to a bare ``C.shape`` -- a tuple where mandelbrot's loop wanted an extent --
        while ``N.shape[1]`` fell out of range and survived, so one nest disagreed with itself.
        Anything else is one extent only if it is PROVABLY rank 0; an expression whose rank is
        unknown refuses rather than donating a rank of 1.
        """
        if isinstance(node, ast.Tuple):
            return [ast.unparse(e) for e in node.elts] if node.elts else None
        if isinstance(node, ast.Attribute) and node.attr == "shape" and isinstance(node.value, ast.Name):
            tokens = self.shapes.get(node.value.id)
            return list(tokens) if tokens else None
        return [ast.unparse(node)] if self.infer(node) == [] else None

    def infer(self, node: ast.AST) -> list[str] | None:
        if is_scalar_literal(node):
            return []  # rank 0: a literal broadcasts against anything and decides no extent
        if isinstance(node, ast.Name):
            return self.shapes.get(node.id)
        if isinstance(node, ast.UnaryOp):
            return self.infer(node.operand)
        if isinstance(node, ast.Compare):
            return self.broadcast([node.left, *node.comparators])
        if isinstance(node, ast.BinOp):
            return self.matmul(node) if isinstance(node.op, ast.MatMult) else self.broadcast([node.left, node.right])
        if isinstance(node, ast.Attribute) and node.attr == "T":
            base = self.infer(node.value)
            return None if base is None else list(reversed(base))
        if isinstance(node, ast.Subscript):
            return self.sliced(node)
        if not isinstance(node, ast.Call):
            return None
        if isinstance(node.func, ast.Name):
            # A scalar builtin over rank-0 arguments is rank 0. cp2k_grid_integrate reads its
            # angular momenta as ``lamax = int(la_max[task])``, and leaving that rankless poisoned
            # every extent downstream of it -- lp, nlp, and the index grids built from nlp.
            if node.func.id not in SCALAR_BUILTINS_ or not all(self.infer(a) == [] for a in node.args):
                return None
            return []
        if not isinstance(node.func, ast.Attribute):
            return None
        name, args = node.func.attr, node.args
        if name in ALLOC_FUNCS and args:
            return self.tuple_tokens(args[0])
        if name == "reshape" and len(args) > 1:
            return self.tuple_tokens(args[1])
        if name == "transpose" and args:
            return self.transposed(args)
        if name in ELEMENTWISE_CALLS:
            return self.broadcast(args)
        if name == "dot" and len(args) == 2:
            return self.dotted(args)
        if name == "arange":
            return self.aranged(args)
        return None

    def aranged(self, args: list[ast.expr]) -> list[str] | None:
        """``np.arange`` is rank 1 and its extent is EXACT: the stop for one argument, the span for
        two. A step makes it a ceiling division this declines to spell rather than guess.

        Without this an index grid poisons every expression it reaches. cp2k_grid_integrate builds
        its degree masks as ``np.arange(nlp)[:, None, None]``; the ``arange`` was rankless, so the
        whole ``(zi <= si) & (yi + xi <= lp - si)`` condition was, and the fill that
        :class:`BroadcastScalarWhere` exists to apply never fired.
        """
        if len(args) == 1:
            return [ast.unparse(args[0])]
        if len(args) == 2:
            return [f"({ast.unparse(args[1])}) - ({ast.unparse(args[0])})"]
        return None

    def dotted(self, args: list[ast.expr]) -> list[str] | None:
        """``np.dot`` of two rank-1 operands is rank 0 -- numpy's inner product.

        Every other rank combination is declined rather than guessed: rank-2 ``dot`` is a matmul and
        a rank-0 operand is a broadcast, and an invented rank is a miscompile. Rank 0 is what
        :class:`CopyScalarAlias` needs to see: minife's ``rtrans = float(np.dot(r, r))`` left the
        name rankless, so ``oldrtrans = rtrans`` was not recognised as a scalar alias, dace issue
        05 aliased the container, and ``beta = rtrans / oldrtrans`` was 1.0 on every CG trip.
        """
        ranks = [self.infer(a) for a in args]
        return [] if all(r is not None and len(r) == 1 for r in ranks) else None

    def sliced(self, node: ast.Subscript) -> list[str] | None:
        """A subscript's extents by numpy's rank rules: a slice KEEPS an axis, an integer index
        DROPS it, ``None`` INSERTS a length-1 one. Any other form is declined -- an invented extent
        is a miscompile, not a refusal."""
        base = self.infer(node.value)
        if base is None:
            return None
        elements = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        if sum(0 if inserts_axis(e) else 1 for e in elements) > len(base):
            return None
        rank0 = {name for name, shape in self.shapes.items() if shape == []}
        tokens: list[str] = []
        axis = 0
        for element in elements:
            if inserts_axis(element):
                tokens.append("1")
                continue
            extent = base[axis]
            axis += 1
            if isinstance(element, ast.Slice):
                span = self.span(extent, element)
                if span is None:
                    return None
                tokens.append(span)
            elif not is_symbol_expr(element, rank0):
                return None  # a mask, an index array or an ellipsis: not this one's rank to guess
        return tokens + base[axis:]

    def span(self, extent: str, element: ast.expr) -> str | None:
        """The extent one slice leaves behind, or None when the form is not one this can spell."""
        if not isinstance(element, ast.Slice) or element.step is not None:
            return None  # a strided slice's length is a ceiling division, not a difference
        rank0 = {name for name, shape in self.shapes.items() if shape == []}
        bounds = [b for b in (element.lower, element.upper) if b is not None]
        if any(not is_symbol_expr(b, rank0) for b in bounds):
            return None
        if element.upper is None:
            return extent if element.lower is None else f"{extent} - ({ast.unparse(element.lower)})"
        upper = ast.unparse(element.upper)
        if element.lower is None or (isinstance(element.lower, ast.Constant) and element.lower.value == 0):
            return upper
        return f"{upper} - ({ast.unparse(element.lower)})"

    def transposed(self, args: list[ast.expr]) -> list[str] | None:
        base = self.infer(args[0])
        if base is None:
            return None
        if len(args) == 1:
            return list(reversed(base))
        order = args[1].elts if isinstance(args[1], ast.Tuple) else []
        axes = [a.value for a in order if isinstance(a, ast.Constant) and isinstance(a.value, int)]
        if len(axes) != len(base) or sorted(axes) != list(range(len(base))):
            return None
        return [base[axis] for axis in axes]

    def matmul(self, node: ast.BinOp) -> list[str] | None:
        """``[m, k] @ [k, n]`` is exact; any other rank pair stays unknown."""
        left, right = self.infer(node.left), self.infer(node.right)
        if left is None or right is None or len(left) != 2 or len(right) != 2:
            return None
        return [left[0], right[1]]

    def broadcast(self, operands: list[ast.expr]) -> list[str] | None:
        """The broadcast shape of an elementwise operand list, or None when it is not certain.

        Numpy's own rule, applied exactly: align right, and each axis takes whichever operand
        carries a non-1 extent there. Two operands disagreeing on a non-1 axis is refused -- numpy
        raises on it too, so there is no answer to give. One unknown operand poisons the result:
        taking the known side instead would adopt its RANK, and a rank-1 shape read as a rank-2
        value's is a miscompile rather than a refusal.

        Reading the widest operand alone -- what this did before -- is not the same thing and cost
        cp2k_grid_integrate its rank: ``zi <= si`` pairs a ``[nlp, 1, 1]`` grid with a
        ``[nlp, 1, 1, 1]`` one, so axis 1's extent lives only on the SHORTER side and the whole
        condition came back unknown.
        """
        shapes: list[list[str]] = []
        for operand in operands:
            shape = self.infer(operand)
            if shape is None:
                return None
            shapes.append(shape)
        no_operand: list[str] = []
        result: list[str] = list(max(shapes, key=len, default=no_operand))
        for shape in shapes:
            offset = len(result) - len(shape)
            for axis, extent in enumerate(shape):
                standing = result[offset + axis]
                if standing == "1":
                    result[offset + axis] = extent
                # Canonically: an extent reached two ways is spelled two ways.
                elif extent != "1" and self.canon(extent) != self.canon(standing):
                    return None
        return result


ALLOC_FUNCS = frozenset({"zeros", "empty", "ones", "full"})


class BroadcastScalarWhere(ResolveShapeReads):
    """Fill a scalar branch of ``np.where`` to the condition's shape where the branches under-size it.

    DaCe sizes a ``where`` from its BRANCHES only. Two scalar branches leave the result shapeless and
    it refuses outright ("Both x and y cannot be scalars in numpy.where"); one scalar branch beside a
    NARROWER array branch is worse, because it answers -- with a result a rank short of numpy's, and
    the refusal lands later, on whatever reads the missing axis. Filling the scalar branch to the
    condition's own extents keeps numpy's answer exactly: numpy broadcasts all three operands, and
    the fill contributes the extents the condition already had.

    Inference is the base class's, which poisons on an unknown operand: ``x @ w + bias`` taking
    ``bias``'s rank-1 shape is a miscompile for a fill and for a ``.shape`` read alike.
    """

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)  # innermost first: a nested where is filled before it is measured
        if not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "where"
            and is_numpy_module(node.func.value)
            and len(node.args) == 3
            and not node.keywords
        ):
            return node
        # A scalar PARAMETER (match_score in needleman_wunsch) is rank 0 exactly like a literal --
        # infer() already resolves it via value_shapes -- but is_scalar_literal alone missed it,
        # so a where(cond, scalar_param, scalar_param) passed through unfilled and dace refused.
        scalars = [k for k in (1, 2) if is_scalar_literal(node.args[k]) or self.infer(node.args[k]) == []]
        if not scalars:
            return node
        shape = self.infer(node.args[0])
        if not shape:
            return node  # unknown, or a scalar condition: an invented extent would be a miscompile
        if len(scalars) == 1:
            # One array branch: dace sizes the result from it alone, so a WIDER condition comes out
            # a rank short of what numpy gives. cp2k_grid_integrate's
            # ``np.where(<4-D mask>, cxyz[:nlp, :nlp, :nlp], 0.0)`` became rank 3, and the
            # ``np.tensordot(gated, ..., axes=([3], [2]))`` that reads it said only "Axes for left
            # tensor are out-of-bounds" -- the rank was already lost two statements earlier. Filled
            # only where the condition is WIDER: at equal rank the branches already carry it, and
            # materialising a fill there costs a temp the size of the result for nothing.
            other = self.infer(node.args[3 - scalars[0]])
            if not other or len(shape) <= len(other):
                return node
        extents = ", ".join(shape) + ("," if len(shape) == 1 else "")
        fill = scalars[0]
        node.args[fill] = ast.parse(f"np.full(({extents}), {ast.unparse(node.args[fill])})", mode="eval").body
        return ast.fix_missing_locations(node)


#: Bare-name calls whose result is rank 0 when every argument is, and whose kind is decided.
SCALAR_BUILTINS_ = frozenset({"int", "float", "abs", "min", "max", "round"})
FLOAT_CALLS = frozenset(
    {
        "float",
        "float32",
        "float64",
        "exp",
        "log",
        "log2",
        "log10",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "tanh",
        "arctan2",
        "atan2",
        "fabs",
        "hypot",
        "erf",
        "mean",
        "std",
        "var",
        "linalg",
    }
)
INT_CALLS = frozenset({"int", "int32", "int64", "len", "argmax", "argmin", "floor_divide"})


def is_float_expr(node: ast.AST, floats: set[str]) -> bool:
    """Conservative: True only where the value is CERTAINLY floating point (or complex).

    One-directional on purpose. Both consumers fall back to the integer spelling when this says no,
    and an integer spelling is right for a float too (``x + 0`` keeps float64) while the reverse
    would widen an index scalar.
    """
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (float, complex))
    if isinstance(node, ast.Name):
        return node.id in floats
    if isinstance(node, ast.UnaryOp):
        return is_float_expr(node.operand, floats)
    if isinstance(node, ast.BinOp):
        return isinstance(node.op, ast.Div) or is_float_expr(node.left, floats) or is_float_expr(node.right, floats)
    if isinstance(node, ast.Subscript):
        return is_float_expr(node.value, floats)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            name = node.func.id
        else:
            name = ""
        if name in INT_CALLS:
            return False
        return name in FLOAT_CALLS or any(is_float_expr(a, floats) for a in node.args)
    return False


def float_names(fn_ast: ast.FunctionDef, declared: set[str]) -> set[str]:
    """Every name that certainly holds a floating-point value, to a least fixed point."""
    floats = set(declared)
    while True:
        grown: set[str] = set()
        for node in ast.walk(fn_ast):
            if not isinstance(node, (ast.Assign, ast.AugAssign)) or not is_float_expr(node.value, floats):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            grown |= {t.id for t in targets if isinstance(t, ast.Name) and t.id not in floats}
        if not grown:
            return floats
        floats |= grown


class CopyScalarAlias(ResolveShapeReads):
    """``x = y`` on a scalar makes ``x`` a second NAME for ``y``'s container (dace issue 05), so a
    later write through either one lands in the other: spell it ``x = y + 0`` to force a copy.

    Only a bare rank-0 Name is rewritten. An array alias is numpy's own semantics, a declared
    parameter is copied by the frontend already, and an operand whose rank the base class cannot
    infer is left alone -- an invented copy on a rank it guessed wrong is a miscompile.
    """

    def __init__(self, shapes: dict[str, list[str]], floats: set[str], skip: set[str]) -> None:
        super().__init__(shapes)
        self.floats = floats
        self.skip = skip

    def infer(self, node: ast.AST) -> list[str] | None:
        """The base class declines a bare-name call; ``center = int(center0_value)`` is rank 0."""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in SCALAR_BUILTINS_:
            ranks = [super(CopyScalarAlias, self).infer(a) for a in node.args]
            return [] if ranks and all(r == [] for r in ranks) else None
        return super().infer(node)

    def visit_Assign(self, node: ast.Assign):
        node = super().visit_Assign(node)
        value = node.value
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(value, ast.Name)
            and value.id not in self.skip
            and value.id != node.targets[0].id
            and self.infer(value) == []
        ):
            zero = ast.Constant(value=0.0 if is_float_expr(value, self.floats) else 0)
            node.value = ast.copy_location(ast.BinOp(left=value, op=ast.Add(), right=zero), node)
        return node


def widen_int_seeds(fn_ast: ast.FunctionDef, floats: set[str], skip: set[str]) -> None:
    """``udiff = 1`` fixes an int64 container that silently TRUNCATES a later float store into it
    (dace issue 06), so the convergence loop it opens exits after two trips: seed it as a float."""
    seeds: dict[str, list[ast.Assign]] = {}
    widen: set[str] = set()
    for node in ast.walk(fn_ast):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if name in skip:
            continue
        if (
            isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
            and not isinstance(node.value.value, bool)
        ):
            seeds.setdefault(name, []).append(node)
        elif is_float_expr(node.value, floats):
            widen.add(name)
    for name in widen & set(seeds):
        for node in seeds[name]:
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
                widened = ast.Constant(value=float(node.value.value))
                node.value = ast.copy_location(widened, node.value)


def is_symbol_expr(node: ast.AST, allowed: set[str]) -> bool:
    """True iff node is a shape expression dace can evaluate as a symbol (names, int consts, + - * // %, min/max).

    A ``.shape[k]`` read is included whatever its receiver: dace's own array descriptor already
    carries a symbolic shape, so reading one axis of it is exactly as "symbol" as a name already in
    ``allowed`` -- max_filter's ``nblocks = -(-length // w)`` feeds a reshape, and ``length`` itself
    is one array's ``shape[0]`` plus two symbols, which used to make the WHOLE chain look
    data-dependent and left ``nblocks`` a plain scalar reshape then auto-promoted and collided with.
    """
    if is_shape_subscript(node):
        return True
    if isinstance(node, ast.Name):
        return node.id in allowed
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
        return is_symbol_expr(node.left, allowed) and is_symbol_expr(node.right, allowed)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return is_symbol_expr(node.operand, allowed)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("min", "max"):
        return bool(node.args) and all(is_symbol_expr(a, allowed) for a in node.args)
    return False


#: Where each ALLOCATION call keeps the shape the caller asked for. ``reshape`` is handled
#: separately below: DaCe NAMES the container it builds after the shape EXPRESSION --
#: ``batch_size * oh * ow`` becomes ``batch_size_oh_times_ow`` -- and then wants a symbol of that
#: same name, which is the "Cannot create symbol X, the name is used by a data descriptor"
#: refusal. A shape that is one plain name gives it nothing to mint.
SHAPE_ARG_INDEX = {"zeros": 0, "empty": 0, "ones": 0, "full": 0}


def reshape_argument(node: ast.AST):
    """The shape argument of a ``reshape`` call only -- the one place hoisting is needed.

    An ALLOCATION takes a compound extent happily (``np.zeros((N, m + 1))`` always worked). It is
    ``reshape`` that makes DaCe name the container after the expression and then collide with it, so
    hoisting anywhere else would mint symbols that buy nothing.

    The shape is always the LAST argument, one tuple/list: :class:`NormalizeReshape` runs before
    this and leaves every reshape call in exactly that form, whether it started as the method
    (``x.reshape(a, b)``, one arg after normalizing) or the function (``np.reshape(x, a, b)``, two
    -- the receiver stays first). Indexing a fixed position instead read the receiver as the shape
    for the method form and refused every kernel that reshapes by method rather than by function.
    """
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "reshape"
        and node.args
        and isinstance(node.args[-1], (ast.Tuple, ast.List))
    ):
        return node.args[-1]
    return None


def shape_argument(node: ast.AST):
    """The shape argument of an allocation or reshape call, or None."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return None
    if node.func.attr == "reshape":
        return reshape_argument(node)
    index = SHAPE_ARG_INDEX.get(node.func.attr)
    if index is None or len(node.args) <= index:
        return None
    return node.args[index]


class HoistCompoundExtents(ast.NodeTransformer):
    """Give every compound shape expression a NAME, so promotion can turn it into one symbol.

    Hoisting alone is not enough and was measured not to be: the hoisted name must also be
    PROMOTED, which needs every ``.shape`` read already resolved (see :class:`ResolveShapeReads`)
    because :func:`plan_size_promotion` is all-or-nothing.

    The definition goes at TOP LEVEL, before the first statement that uses it: a use can sit inside
    a loop while another sits after it, so defining at the point of first use would leave the second
    undefined. Only expressions over names already defined before that statement are hoisted --
    anything else would move a read above its write.
    """

    def __init__(self, known: set[str]) -> None:
        self.known = known
        self.names: dict[str, str] = {}
        self.plan: list[tuple[int, str, ast.expr]] = []  # statement index to define before, name, expression

    def collect(self, fn_ast: ast.FunctionDef) -> None:
        defined = set(self.known)
        for index, stmt in enumerate(fn_ast.body):
            for node in ast.walk(stmt):
                shape = reshape_argument(node)
                if shape is None:
                    continue
                for element in shape.elts if isinstance(shape, ast.Tuple) else [shape]:
                    if not isinstance(element, ast.BinOp) or not is_symbol_expr(element, defined):
                        continue
                    text = ast.unparse(element)
                    if text not in self.names:
                        self.names[text] = f"__hpcagent_bench_extent{len(self.names)}"
                        self.plan.append((index, self.names[text], element))
            for node in ast.walk(stmt):
                if isinstance(node, ast.Assign):
                    defined.update(t.id for t in node.targets if isinstance(t, ast.Name))

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        # Collected from reshape, but substituted in EVERY shape: the allocation and the reshape
        # must name the same symbol or DaCe cannot see they are the same extent -- measured, it
        # reports "[__extent0, 96] into [oh*ow*batch_size, 96]" and refuses the write.
        shape = shape_argument(node)
        if shape is None:
            return node
        elements = shape.elts if isinstance(shape, ast.Tuple) else [shape]
        for position, element in enumerate(elements):
            name = self.names.get(ast.unparse(element)) if isinstance(element, ast.BinOp) else None
            if name is not None:
                elements[position] = ast.copy_location(ast.Name(id=name, ctx=ast.Load()), element)
        return node


def hoist_compound_extents(fn_ast: ast.FunctionDef, known: set[str]) -> ast.FunctionDef:
    """Name every compound shape expression, defining each above the first statement that uses it."""
    hoister = HoistCompoundExtents(known)
    hoister.collect(fn_ast)
    if not hoister.plan:
        return fn_ast
    fn_ast = hoister.visit(fn_ast)
    for index, name, element in reversed(hoister.plan):
        definition = ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=copy.deepcopy(element))
        fn_ast.body.insert(index, ast.copy_location(definition, fn_ast.body[index]))
    ast.fix_missing_locations(fn_ast)
    return fn_ast


def shape_base_ids(node: ast.AST) -> set[int]:
    """``id()`` of every Name read as ``<x>.shape`` -- x is a DIMENSION source, never an integer value."""
    return {
        id(a.value)
        for a in ast.walk(node)
        if isinstance(a, ast.Attribute) and a.attr == "shape" and isinstance(a.value, ast.Name)
    }


def shape_reaching_names(body: ast.AST, direct: set[str]) -> set[str]:
    """Names whose VALUE reaches a shape, following assignments -- not only the names written in one.

    conv2d_instance_norm_divide reads its stride, padding and dilation out of manifest scalars,
    derives the convolution's output extents from them (``oh = (height + 2*ph - dh*(ks - 1) - 1) //
    sh + 1``), and reshapes the im2col patch to ``(batch * oh * ow, in_per_group)``. Every name in
    that shape is a local, so a syntactic scan of the shape finds nothing, the scalars stay runtime
    data, and DaCe refuses the extent -- a data descriptor cannot be a shape.

    The seed must NOT be filtered by the rebound names. A rebound name cannot become a dc.symbol,
    which is a fact about what may be PROMOTED; as a HOP from a scalar to an extent it is perfectly
    good, and dropping it cuts every chain at its first local.

    A ``.shape`` receiver is skipped. It names a DIMENSION SOURCE rather than an integer value, so
    following it drags whole arrays in and makes an array alias read as a size expression.
    """
    assigns: dict[str, list[ast.expr]] = {}
    for node in ast.walk(body):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assigns.setdefault(node.targets[0].id, []).append(node.value)
    reaching = set(direct)
    frontier = list(direct)
    while frontier:
        for rhs in assigns.get(frontier.pop(), ()):
            bases = shape_base_ids(rhs)
            for sub in ast.walk(rhs):
                if isinstance(sub, ast.Name) and id(sub) not in bases and sub.id not in reaching:
                    reaching.add(sub.id)
                    frontier.append(sub.id)
    return reaching


class SubstituteScalarValues(ast.NodeTransformer):
    """Replace every READ of a named scalar with its literal value."""

    def __init__(self, values: dict[str, int]) -> None:
        self.values = values

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id in self.values:
            return ast.copy_location(ast.Constant(value=self.values[node.id]), node)
        return node


def freeze_pinned_extent_scalars(kir: KernelIR) -> KernelIR:
    """Substitute the value of every manifest-pinned integer scalar that reaches an EXTENT.

    A benchmark pins each scalar to ONE value across S/M/L/XL, so a scalar an extent depends on is a
    compile-time constant however the reference spells it. Promoting it to a dc.symbol instead was
    measured and does not work: the manifest declares conv2d_instance_norm_divide's output as
    ``(batch, out_channels, height - kernel_size + 1, width - kernel_size + 1)``, which is the
    convolution's general extent formula ALREADY EVALUATED at stride 1, padding 0, dilation 1. Left
    symbolic, the body computes ``int_floor(2*conv_padding - conv_dilation*(kernel_size - 1) +
    height - 1, conv_stride) + 1`` and nothing can prove the two equal -- the parse refuses the
    write to ``out``. A symbol only agrees when the declared shapes name that same symbol, which is
    the case the direct scan already handles.

    The scalar stays a PARAMETER of the emitted program, so the binding and the ABI are unchanged;
    it is simply no longer read. Floats are excluded: an extent is an integer, and a float scalar
    reached through the chain is a tolerance or a scale, not a size.

    Only a scalar that reaches an extent INDIRECTLY, through a chain of local assignments, is
    frozen. One the body spells DIRECTLY in a shape or a slice bound is promoted to a dc.symbol by
    the scan in :func:`emit_dace` instead, and the manifest value is test DATA there, not a
    compile-time constant: cloudsc's ``za_col = za[jk - 1, kidia - 1:kfdia]`` became
    ``za[jk - 1, 1 - 1:0]``, an empty range that builds, runs and computes nothing. The measured
    conv case below needs the fold precisely because it is indirect -- no shape argument names the
    scalar, only the ``//``-derived locals do, and nothing can fold those back to the declaration.

    Two further names are never substituted, and both were caught as regressions rather than
    predicted:

    * one a DECLARED ARRAY SHAPE mentions. nbody declares ``KE: (Nt + 1,)`` and also lists ``Nt``
      under scalars, so ``Nt`` has to be a dc.symbol -- the existing direct scan promotes it. Give
      the body the literal instead and the declaration still says ``Nt + 1`` while the body says
      ``1``; the frontend answers "Cannot reassign value to variable KE".
    * one that IS a declared array. cfd lists ``neigh`` under scalars AND under arrays with shape
      ``(ncells, 4)``. It is an array; substituting a scalar's value for it replaces the array with
      an integer, and a local derived from it becomes an undefined name.
    """
    scalars = {s.name: s for s in kir.scalars}
    if not scalars:
        return kir
    declared = {a.name for a in kir.arrays}
    for array in kir.arrays:
        for token in array.shape:
            declared.update(IDENT_RE.findall(str(token)))
    probe = NormalizeReshape().visit(copy.deepcopy(kir.tree))
    seeds: set[str] = set()
    for node in ast.walk(probe):
        shape_arg = shape_argument(node)
        if shape_arg is None:
            continue
        elements = shape_arg.elts if isinstance(shape_arg, (ast.Tuple, ast.List)) else [shape_arg]
        for element in elements:
            seeds.update(n.id for n in ast.walk(element) if isinstance(n, ast.Name))
    frozen: dict[str, int] = {}
    for name in shape_reaching_names(probe, seeds) - seeds:
        if name not in scalars or name in declared:
            continue
        desc = scalars[name]
        if desc.dtype.startswith(("int", "uint")) and type(desc.value) is int:
            frozen[name] = desc.value
    if not frozen:
        return kir
    tree = SubstituteScalarValues(frozen).visit(copy.deepcopy(kir.tree))
    ast.fix_missing_locations(tree)
    return dataclasses.replace(kir, tree=tree)


def freeze_shape_only_parameters(kir: KernelIR) -> KernelIR:
    """Spell every :attr:`KernelIR.shape_only_consts` name as its literal in the declared shapes.

    Such a name reaches the emitted program through one declared extent and nowhere else, so the
    scan in :func:`emit_dace` mints a free dc.symbol for it -- a symbol the body can never mention,
    and therefore one no write to that array can ever be proved against. conv_depthwise_separable_2d
    declares ``out`` through ``dilation`` and computes it through the pinned scalar
    ``depthwise_dilation``, which :func:`freeze_pinned_extent_scalars` has already turned into ``1``:
    the frontend is then asked to broadcast ``height - kernel_size + 1`` into
    ``height - dilation * (kernel_size - 1)`` and refuses. Freezing the one spelling and not the
    other is what makes the two extents unprovable, so both are frozen.

    Shapes only. The body never names one of these, the signature never takes one, and the
    manifest binds it to the same value for every preset -- so the ABI and the numbers are the
    same either way, and only the proof obligation changes.
    """
    if not kir.shape_only_consts:
        return kir
    values = {name: int(value) for name, value in kir.shape_only_consts.items()}
    arrays = [dataclasses.replace(a, shape=tuple(frozen_extent(s, values) for s in a.shape)) for a in kir.arrays]
    # The body too: once helpers are kept, the buffer the kernel allocates for a helper argument is
    # spelled off the same declared extent, and freezing only the declaration left mlp's ``w1`` at
    # ``[C_in, 30000]`` against a ``[N, S0]`` argument buffer dace could not relate to it.
    tree = ast.fix_missing_locations(SubstituteScalarValues(values).visit(copy.deepcopy(kir.tree)))
    return dataclasses.replace(kir, arrays=arrays, tree=tree)


def frozen_extent(dim: str, values: dict[str, int]) -> str:
    """One declared extent with every ``values`` name replaced by its literal; unchanged if unparsable."""
    text = str(dim)
    if not any(ident in values for ident in IDENT_RE.findall(text)):
        return text
    try:
        tree = SubstituteScalarValues(values).visit(ast.parse(text, mode="eval"))
    except SyntaxError:
        return text
    return ast.unparse(ast.fix_missing_locations(tree))


def shape_ident_candidates(fn_ast: ast.FunctionDef, known: set[str]) -> set[str]:
    """Identifiers in an np.zeros/empty/ones shape arg not already array/scalar/symbol -- promotion candidates."""
    names: set[str] = set()
    for node in ast.walk(fn_ast):
        shape_arg = shape_argument(node)
        if shape_arg is not None:
            shape_bases = shape_base_ids(shape_arg)
            for sub in ast.walk(shape_arg):
                if isinstance(sub, ast.Name) and id(sub) not in shape_bases and sub.id not in known:
                    names.add(sub.id)
    return names


def scan_size_assigns(
    fn_ast: ast.FunctionDef, targets: set[str]
) -> tuple[dict[str, ast.expr], list[str], OrderedSet[str]]:
    """For each name in targets: its first (defining) RHS, def order, and which names are reassigned.

    Only a name whose every store is a plain ``name = ...`` has a definition. One also stored as a
    counter, a loop target or a tuple target (``n += 1``) holds a path-dependent value and is left out:
    inlined as its first value, or promoted to a symbol bound to it, every read after the update is wrong.
    """
    first_rhs: dict[str, ast.expr] = {}
    order: list[str] = []
    counts: dict[str, int] = {}
    plain: set[int] = set()
    mutated: set[str] = set()
    for node in ast.walk(fn_ast):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            nm = node.targets[0].id
            plain.add(id(node.targets[0]))
            if nm in targets:
                counts[nm] = counts.get(nm, 0) + 1
                if nm not in first_rhs:
                    first_rhs[nm] = node.value
                    order.append(nm)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in targets:
            if id(node) not in plain:
                mutated.add(node.id)
    order = [nm for nm in order if nm not in mutated]
    first_rhs = {nm: first_rhs[nm] for nm in order}
    # Ordered: the caller PREPENDS one ``<nm>_iter = <nm>`` statement per reassigned name to the
    # emitted body, so this order is statement order in the generated program.
    reassigned = OrderedSet(nm for nm, c in counts.items() if c > 1 and nm not in mutated)
    return first_rhs, order, reassigned


def once_bound_locals(fn_ast: ast.FunctionDef, known: set[str]) -> set[str]:
    """Body locals bound EXACTLY once and never written through a subscript.

    One binding is what makes a name substitutable at all: a second one, or an ``x[...] = ...``,
    makes the name data whose value depends on where in the body it is read.
    """
    bindings: dict[str, int] = {}
    stored: set[str] = set()
    for node in ast.walk(fn_ast):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bindings[node.id] = bindings.get(node.id, 0) + 1  # every rebinding: assign, augassign, for, walrus
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store) and isinstance(node.value, ast.Name):
            stored.add(node.value.id)  # ``x[...] = ...`` is data, and a name cannot be both data and symbol
    return {nm for nm, count in bindings.items() if count == 1 and nm not in stored and nm not in known}


def constant_int_locals(fn_ast: ast.FunctionDef, known: set[str]) -> set[str]:
    """Body locals bound ONCE to a CONSTANT integer expression -- inlinable as that constant.

    dace folds such a local inside arithmetic, but NOT in a slice bound: a local ``IC = 2``
    reaches ``strx[IC:N_I - 2]`` as data and the frontend mints ``__sym_IC`` for it, which it then
    cannot prove equal to the ``__sym_IM2`` minted for the ``IM2 = 0`` two lines above -- two
    windows of one array that differ by a constant come out as unrelated extents. Substituting the
    literal is what the numpy reference already means, and leaves the frontend one expression.

    Distinct from :func:`mintable_int_locals`, which requires the definition to READ a symbol:
    minting a dc.symbol for a constant only adds one the caller then has to bind, while inlining it
    adds nothing at all.
    """
    once = once_bound_locals(fn_ast, known)
    first_rhs, order, unused = scan_size_assigns(fn_ast, once)
    return {
        nm
        for nm in order
        if is_symbol_expr(first_rhs[nm], set())
        and not any(
            isinstance(sub, ast.Name) or (isinstance(sub, ast.Constant) and isinstance(sub.value, bool))
            for sub in ast.walk(first_rhs[nm])
        )
    }


def mintable_int_locals(fn_ast: ast.FunctionDef, symbols: set[str], known: set[str]) -> set[str]:
    """Body locals bound ONCE to an integer expression over declared symbols -- mintable as dc.symbols.

    Seeding promotion from shape arguments alone leaves ``k = K`` a scalar transient, and dace's
    frontend then mints a FRESH symbol for it that it never unifies with the one it came from --
    ``[__sym_k_0]`` into ``[K]``, the largest refusal class in the generated corpus. A name minted
    here keeps the one spelling both sides agree on.

    Atoms are the DECLARED SYMBOLS, never the wider ``known``: an array (``B = A``) and a float
    scalar (``c = 2 * alpha``) both read as integer symbol expressions against ``known``, and either
    one minted as an int64 symbol is a wrong answer rather than a refusal.
    """
    first_rhs, order, unused = scan_size_assigns(fn_ast, once_bound_locals(fn_ast, known))
    cand: set[str] = set()
    while True:  # least fixed point: a name qualifies once every name its definition reads does
        atoms = symbols | cand
        # Reading a symbol is required, not just being integer: dace folds a literal-valued local
        # (``vl = 64``) already, so minting one only adds a symbol the caller then has to bind.
        grown = {
            nm
            for nm in order
            if nm not in cand
            and is_symbol_expr(first_rhs[nm], atoms)
            and any(isinstance(sub, ast.Name) and sub.id in atoms for sub in ast.walk(first_rhs[nm]))
        }
        if not grown:
            return cand
        cand |= grown


class StripIdentityIntCasts(ast.NodeTransformer):
    """Drop ``int(...)`` where the operand is already an integer symbol expression.

    Every dc.symbol is minted int64 and :func:`is_symbol_expr` admits only integer-valued forms,
    so the cast computes nothing -- but it hides an alias from the inliner, which is what costs the
    kernel. warpx_field_gather's ``o = int(depos_order)`` stayed a body local; ``__inl1_o + 1`` then
    reached one allocation as an expression over the minted ``__sym___inl1_o`` and another as the
    whole-expression symbol ``__sym___inl1_o_plus_1``, and the frontend cannot prove one equals the
    other. With the cast gone ``o`` folds to ``depos_order`` and both spellings become the same one.

    Only that case: ``int()`` on a float is a truncation and its operand fails ``is_symbol_expr``,
    so it is left alone.
    """

    def __init__(self, symbols: set[str]) -> None:
        self.symbols = symbols

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "int"
            and len(node.args) == 1
            and not node.keywords
            and is_symbol_expr(node.args[0], self.symbols)
        ):
            return node.args[0]
        return node


def loop_induction_symbols(fn_ast: ast.FunctionDef) -> OrderedSet[str]:
    """Names bound by ``for <name> in range(...)`` -- symbols to dace, not data.

    An induction variable is an atom the alias inliner must count as symbolic, or a scalar derived
    from one stays a body local and lands in a slice bound as DATA. dace then mints a fresh symbol
    for the whole bound and has nothing left to relate it to the start: conv3d's
    ``padded_g[:, icg, iz0:iz0 + span_d]`` came out as an extent
    ``-__sym___inl1_iz0 + __sym___inl1_iz0_plus_depth_1_kernel_size_1_1_1_1_1``, which the frontend
    cannot prove equal to the accumulator's ``depth - kernel_size + 1``. With ``iz0 = kz * 1``
    folded to its induction variable the extent is the span expression itself, spelled once.
    """
    names: OrderedSet[str] = OrderedSet()
    for node in ast.walk(fn_ast):
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ):
            names.add(node.target.id)
    return names


#: Bound on alias-inliner rounds; each exposes names one definition deeper.
INLINE_ALIAS_ROUNDS = 25


def fold_expr(node: ast.AST) -> ast.AST:
    """Fold an integer expression AST through :func:`fold_shape_expr`; unchanged if it will not parse."""
    try:
        return ast.parse(fold_shape_expr(ast.unparse(node)), mode="eval").body
    except SyntaxError:
        return node


#: Calls the DaCe frontend has no replacement for. Reaching one makes it a CALLBACK -- an opaque
#: Python call whose return type it cannot infer ("Trying to operate on a callback return value with
#: an undefined type"), so the parse fails and, where it does not, the kernel is no longer a kernel.
#: Every entry here is lowered by :class:`LowerCallsDaceCannotReplace` into forms dace does have:
#: its ufuncs, its BLAS ``Dot`` node, plain subscripts, or an explicit loop.
CALLS_WITHOUT_A_DACE_REPLACEMENT = ("take", "round", "searchsorted", "linalg.norm", "fft.fftfreq", "ufunc.at")


def np_call_name(node: ast.AST) -> str | None:
    """``np.take`` -> ``"take"``, ``np.linalg.norm`` -> ``"linalg.norm"``, else None."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return None
    base = node.func.value
    if is_numpy_module(base):
        return node.func.attr
    if isinstance(base, ast.Attribute) and is_numpy_module(base.value):
        return f"{base.attr}.{node.func.attr}"
    return None


def kwarg_value(node: ast.Call, name: str, position: int) -> ast.expr | None:
    """The argument bound to ``name``, whether it was passed by keyword or at ``position``."""
    for kw in node.keywords:
        if kw.arg == name:
            return kw.value
    return node.args[position] if len(node.args) > position else None


def parse_expr(text: str) -> ast.expr:
    """One expression, parsed. The lowerings below are clearer written out than built node by node."""
    return ast.parse(text, mode="eval").body


class LowerCallsDaceCannotReplace(ast.NodeTransformer):
    """Rewrite every call in :data:`CALLS_WITHOUT_A_DACE_REPLACEMENT` into something dace replaces.

    dace covers 90 numpy ufuncs and ~150 named functions; what it does not cover it turns into a
    callback, and a callback is not a kernel -- it is a Python call the code generator cannot see
    into, schedule, or type. Six spellings in the corpus land there, and each has an exact
    equivalent dace does implement:

    ``np.take``          a plain subscript -- dace already lowers an index-array gather
    ``np.add.at``        the scatter loop it is defined as; sequential, because the indices repeat
    ``np.searchsorted``  the binary search, NOT a scan: xsbench looks up tens of thousands of edges
    ``np.linalg.norm``   ``sqrt(dot(v, v))``, which reaches the BLAS ``Dot`` library node
    ``np.fft.fftfreq``   its closed form; it is a frequency ladder, not a transform
    ``np.round``         floor-and-correct, keeping numpy's round-HALF-TO-EVEN

    A form outside what each handler proves is left alone rather than guessed at: the parse then
    fails on the callback, which is the honest outcome. Runs before the ``.shape`` passes so the
    extents these emit are resolved with every other one.
    """

    def __init__(self, ranks: dict[str, int], complex_arrays: set[str] | None = None) -> None:
        self.ranks = ranks
        self.complex_arrays = complex_arrays or set()
        self.counter = 0
        self.prelude: list[ast.stmt] = []
        self.changed = False

    def temp(self, stem: str) -> str:
        self.counter += 1
        return f"__{stem}{self.counter - 1}"

    def bind(self, stem: str, value: ast.expr) -> ast.Name:
        """Evaluate ``value`` once into a fresh name -- these lowerings read their operand twice."""
        name = self.temp(stem)
        self.prelude.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=value))
        return ast.Name(id=name, ctx=ast.Load())

    def block(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        """Visit one statement list, splicing each statement's prelude in above it."""
        out: list[ast.stmt] = []
        for stmt in stmts:
            outer, self.prelude = self.prelude, []
            scattered = self.scatter_loop(stmt)
            lowered: ast.stmt | list[ast.stmt] = self.visit(stmt) if scattered is None else scattered
            out.extend(self.prelude)
            out.extend(lowered if isinstance(lowered, list) else [lowered])
            self.prelude = outer
        for stmt in out:
            ast.fix_missing_locations(stmt)
        return out

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.body = self.block(node.body)
        return node

    def visit_For(self, node: ast.For) -> ast.AST:
        node.iter = self.visit(node.iter)
        node.body, node.orelse = self.block(node.body), self.block(node.orelse)
        return node

    def visit_While(self, node: ast.While) -> ast.AST:
        node.test = self.visit(node.test)
        node.body, node.orelse = self.block(node.body), self.block(node.orelse)
        return node

    def visit_If(self, node: ast.If) -> ast.AST:
        node.test = self.visit(node.test)
        node.body, node.orelse = self.block(node.body), self.block(node.orelse)
        return node

    def scatter_loop(self, stmt: ast.stmt) -> list[ast.stmt] | None:
        """``np.add.at(a, idx, v)`` -> the loop nest it is defined as, or None if not one.

        A statement, never an expression: ``add.at`` returns nothing and exists precisely because
        ``a[idx] += v`` drops every repeat. The indices DO repeat here -- lulesh scatters element
        forces onto shared nodes -- so the loop stays sequential and accumulates each one.

        numpy evaluates the index and the value once, before the first write. A non-Name operand
        (vexx_k's ``ikb.ravel()``) is bound to a local above the loop, so it is neither re-run per
        element nor parsed as ``expr[k]`` with the subscript binding tighter than the expression.
        """
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
            return None
        call = stmt.value
        func = call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "at" and isinstance(func.value, ast.Attribute)):
            return None
        base = func.value
        if not (is_numpy_module(base.value) and base.attr == "add"):
            return None
        if len(call.args) != 3:
            return None
        target, index, value = call.args
        rank = expr_rank(index, self.ranks)
        if rank is None or rank < 1:
            return None
        stem = self.temp("scatter")
        index_name = self.operand_name(f"{stem}_idx", index)
        value_name = self.operand_name(f"{stem}_val", value)
        iters = [f"{stem}_{k}" for k in range(rank)]
        at = ", ".join(iters)
        body = f"{ast.unparse(target)}[{index_name}[{at}]] += {value_name}[{at}]"
        for depth in reversed(range(rank)):
            body = f"for {iters[depth]} in range({index_name}.shape[{depth}]):\n" + indent_block(body)
        self.changed = True
        return [ast.copy_location(new, stmt) for new in ast.parse(body).body]

    def operand_name(self, name: str, value: ast.expr) -> str:
        """``value``'s own name when it is a bare Name, else ``name`` bound to it once in the prelude."""
        if isinstance(value, ast.Name):
            return value.id
        self.prelude.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=value))
        return name

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        name = np_call_name(node)
        handlers: dict[str, Callable[[ast.Call], ast.expr | None]] = {
            "take": self.take,
            "round": self.round_half_to_even,
            "searchsorted": self.searchsorted,
            "linalg.norm": self.norm,
            "fft.fftfreq": self.fftfreq,
        }
        handler = None if name is None else handlers.get(name)
        if handler is None:
            return node
        lowered = handler(node)
        if lowered is None:
            return node
        self.changed = True
        return ast.copy_location(lowered, node)

    def take(self, node: ast.Call) -> ast.expr | None:
        """``np.take(a, i, axis=k)`` -> ``a[:, .., i]``. dace lowers an index-array gather already.

        An absent axis flattens ``a`` first, so it is only the same subscript when ``a`` is already
        rank 1 -- otherwise this leaves the call alone rather than silently gathering a wrong axis.
        """
        if len(node.args) < 2:
            return None
        source, index = node.args[0], node.args[1]
        axis = kwarg_value(node, "axis", 2)
        if axis is None:
            if not (isinstance(source, ast.Name) and self.ranks.get(source.id) == 1):
                return None
            leading = 0
        elif isinstance(axis, ast.Constant) and isinstance(axis.value, int) and axis.value >= 0:
            leading = axis.value
        else:
            return None
        return parse_expr(f"{ast.unparse(source)}[{', '.join([':'] * leading + [ast.unparse(index)])}]")

    def round_half_to_even(self, node: ast.Call) -> ast.expr | None:
        """``np.round(x)`` -> floor-and-correct. numpy rounds a HALF to the EVEN neighbour.

        ``floor(x + 0.5)`` alone disagrees on every exact half -- 2.5 goes to 3 where numpy gives 2 --
        and histogram_equalization feeds the result to a lookup table the oracle compares elementwise.
        """
        if len(node.args) != 1 or node.keywords:
            return None
        value = self.bind("round_x", node.args[0])
        up = self.bind("round_up", parse_expr(f"np.floor({ast.unparse(value)} + 0.5)"))
        return parse_expr(
            f"np.where(({ast.unparse(up)} - {ast.unparse(value)} == 0.5) & "
            f"(np.mod({ast.unparse(up)}, 2.0) != 0.0), {ast.unparse(up)} - 1.0, {ast.unparse(up)})"
        )

    def norm(self, node: ast.Call) -> ast.expr | None:
        """``np.linalg.norm(v)`` -> the 2-norm of the flattened operand, which is what it is defined as.

        With neither ``ord`` nor ``axis`` numpy returns the 2-norm of ``v.ravel()`` at ANY rank, so
        the lowering holds for the rank-3 fragment ls3df passes as much as for a vector. A rank-1
        operand goes through ``np.dot``, which dace expands to its BLAS ``Dot`` library node -- but only
        when it is PROVABLY real. Everything else takes ``sum(abs(v) ** 2)``, numpy's own definition
        and the only form that also holds for a complex operand: ``v * v`` and ``dot(v, v)`` both drop
        the conjugate there and return a different number.

        Anything with an ``ord`` or an ``axis`` asks for a different quantity and is left alone.
        """
        if len(node.args) != 1 or node.keywords or not isinstance(node.args[0], ast.Name):
            return None
        name = node.args[0].id
        if self.ranks.get(name) == 1 and name not in self.complex_arrays:
            return parse_expr(f"np.sqrt(np.dot({name}, {name}))")
        return parse_expr(f"np.sqrt(np.sum(np.abs({name}) ** 2))")

    def fftfreq(self, node: ast.Call) -> ast.expr | None:
        """``np.fft.fftfreq(n, d)`` -> its closed form: a frequency ladder, not a transform.

        The second half of the ladder is NEGATIVE -- bin ``k`` past the midpoint stands for ``k - n``
        -- which is the whole content of the function and the part a naive ``arange / (n * d)` drops.
        """
        if not node.args:
            return None
        count = ast.unparse(node.args[0])
        spacing = kwarg_value(node, "d", 1)
        step = "1.0" if spacing is None else ast.unparse(spacing)
        ladder = self.bind("fftfreq_k", parse_expr(f"np.arange({count})"))
        k = ast.unparse(ladder)
        return parse_expr(f"np.where({k} < ({count} + 1) // 2, {k}, {k} - {count}) / ({count} * {step})")

    def searchsorted(self, node: ast.Call) -> ast.expr | None:
        """``np.searchsorted(a, v, side)`` -> a binary search per element of ``v``.

        A binary search, not a linear count: numpy's is O(log n) per element and xsbench looks up a
        unionized grid of tens of thousands of edges. Counting would return the same indices and
        change the kernel's complexity class, which is the one thing a benchmark may not do.

        ``side='left'`` counts the entries STRICTLY below the value, ``'right'`` those at or below --
        one comparison apart, and that difference is exactly what a bin lookup's ``- 1`` relies on.
        """
        operands = [a for a in node.args[:2] if isinstance(a, ast.Name)]
        if len(operands) < 2:
            return None
        table, values = ast.unparse(operands[0]), ast.unparse(operands[1])
        if self.ranks.get(operands[0].id) != 1 or self.ranks.get(operands[1].id) != 1:
            return None
        side_node = kwarg_value(node, "side", 2)
        side = "left" if side_node is None else (side_node.value if isinstance(side_node, ast.Constant) else None)
        if side not in ("left", "right"):
            return None
        stem = self.temp("bisect")
        below = "<=" if side == "right" else "<"
        self.prelude.extend(
            ast.parse(
                f"{stem} = np.zeros({values}.shape[0], dtype=np.int64)\n"
                f"for {stem}_i in range({values}.shape[0]):\n"
                f"    {stem}_lo = 0\n"
                f"    {stem}_hi = {table}.shape[0]\n"
                f"    while {stem}_lo < {stem}_hi:\n"
                f"        {stem}_mid = ({stem}_lo + {stem}_hi) // 2\n"
                f"        if {table}[{stem}_mid] {below} {values}[{stem}_i]:\n"
                f"            {stem}_lo = {stem}_mid + 1\n"
                f"        else:\n"
                f"            {stem}_hi = {stem}_mid\n"
                f"    {stem}[{stem}_i] = {stem}_lo\n"
            ).body
        )
        return ast.Name(id=stem, ctx=ast.Load())


def indent_block(text: str) -> str:
    """Indent every line of a generated body by one level."""
    return "\n".join("    " + line for line in text.splitlines())


def names_a_clamp(node: ast.AST) -> bool:
    """Whether ``node`` contains a ``min``/``max`` -- an expression dace does not fold.

    Inlining one is what CREATES the second spelling this pass exists to remove. banded_mmt's
    reference already names its bounds (``a_start = max(i - a_lbound, 0)``), and inlining them put
    the clamp in two slices, where dace minted ``__sym_A_dense_slice`` and
    ``__sym_A_dense_slice_0`` -- two opaque symbols for one bound, which it then cannot prove equal
    to the ``__sym_expr_minus_expr`` it minted for the same difference on the other side. A name
    left standing becomes ONE minted symbol every occurrence shares. Plain arithmetic is different:
    dace evaluates it, so an inlined ``+``/``-``/``*`` chain folds to the same expression wherever
    it lands.
    """
    return any(
        isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in ("min", "max")
        for sub in ast.walk(node)
    )


def spell_aranges_with_named_lengths(fn_ast: ast.FunctionDef, known: set[str]) -> None:
    """``np.arange(lo, hi)`` -> ``np.arange(<n>) + lo`` where a local already NAMES ``hi - lo``.

    dace hoists each data-dependent slice/arange bound into its own opaque symbol, one per
    EXPRESSION: cp2k_grid_integrate's ``rel = np.arange(-span, span + 1)`` became
    ``__sym_span_plus_1 - __sym_neg_span`` while the buffer it feeds, ``np.empty((lp + 1, nrel))``
    with ``nrel = 2 * span + 1``, became ``__sym_nrel``. Both are the same length and dace has no
    way to know it. Spelling the arange with the name the kernel already computed leaves ONE
    symbol, and the write it feeds matches without anything having to be proved.

    Only where the name is bound EARLIER IN THE SAME BLOCK: a binding in another branch does not
    reach this arange, and one after it is not yet the length.
    """
    # One scan for every candidate, not one per candidate: _scan_size_assigns walks the whole
    # function, and densenet121 ran it 879 times for 62 s of a 134 s emit. A target set only
    # filters which assigns get recorded, so the answer per name is the same either way.
    once = once_bound_locals(fn_ast, known)
    first_rhs = scan_size_assigns(fn_ast, once)[0]
    lengths: dict[str, ast.expr] = {}
    for name in once:
        rhs = first_rhs.get(name)
        if rhs is not None and is_symbol_expr(rhs, {n.id for n in ast.walk(rhs) if isinstance(n, ast.Name)}):
            lengths[name] = rhs
    if not lengths:
        return
    for block in statement_lists(fn_ast):
        seen: dict[str, ast.expr] = {}
        for stmt in block:
            for node in ast.walk(stmt):
                assign = node if isinstance(node, ast.Assign) else None
                call = assign.value if assign is not None else None
                if assign is None or not (
                    isinstance(call, ast.Call)
                    and np_call_name(call) == "arange"
                    and len(call.args) == 2
                    and not call.keywords
                ):
                    continue
                extent = f"({ast.unparse(call.args[1])}) - ({ast.unparse(call.args[0])})"
                match = next((nm for nm, rhs in seen.items() if shape_exprs_equal(extent, ast.unparse(rhs))), None)
                if match is None:
                    continue
                lo = call.args[0]
                call.args = [ast.Name(id=match, ctx=ast.Load())]
                # ``arange(n) - span``, not ``arange(n) + -span``: the negation is one more node for
                # every consumer to carry, and dace spells the offset into every memlet that reads it.
                if isinstance(lo, ast.UnaryOp) and isinstance(lo.op, ast.USub):
                    assign.value = ast.BinOp(left=call, op=ast.Sub(), right=lo.operand)
                else:
                    assign.value = ast.BinOp(left=call, op=ast.Add(), right=lo)
            bound = stmt.targets[0] if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 else None
            if isinstance(bound, ast.Name) and bound.id in lengths:
                seen[bound.id] = lengths[bound.id]
    ast.fix_missing_locations(fn_ast)


def names_rebound(fn_ast: ast.FunctionDef) -> set[str]:
    """Names stored more than once in the function (an augmented target is a store too): their value depends
    on the path taken, so an alias reading one is only good up to that store, not past it."""
    counts: dict[str, int] = {}
    for node in ast.walk(fn_ast):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            counts[node.id] = counts.get(node.id, 0) + 1
    return {name for name, count in counts.items() if count > 1}


def bind_site(fn_ast: ast.FunctionDef, nm: str, rhs: ast.expr) -> tuple[list[ast.stmt], int] | None:
    """The statement list holding ``nm``'s one ``nm = rhs`` binding, and its index there."""
    for block in statement_lists(fn_ast):
        for idx, stmt in enumerate(block):
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == nm
                and stmt.value is rhs
            ):
                return block, idx
    return None


def rebind_boundary(block: list[ast.stmt], start: int, sources: set[str]) -> int:
    """Index of the first statement at ``start`` or later whose subtree stores a name in ``sources``,
    else ``len(block)``: a nested store counts, same as a store at this list's own level."""
    for idx in range(start, len(block)):
        if any(
            isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in sources
            for node in ast.walk(block[idx])
        ):
            return idx
    return len(block)


def splice_before_rebind(block: list[ast.stmt], start: int, nm: str, rhs: ast.expr, sources: set[str]) -> None:
    """Splice ``rhs`` into ``nm``'s uses from ``start`` up to the first later store of a name it
    reads: those uses run before that store, so they still read the value ``nm`` was bound to. A
    use before ``start``, at the boundary statement, or past it keeps reading ``nm`` itself -- inside
    a loop that use is either this iteration's own post-store read or a wrap-around from the one
    before, and either way the store has already run."""
    boundary = rebind_boundary(block, start, sources)
    for idx in range(start, boundary):
        block[idx] = SubstituteNames_({nm: rhs}).visit(block[idx])
        ast.fix_missing_locations(block[idx])


def name_loaded(fn_ast: ast.AST, nm: str) -> bool:
    """True iff some read of ``nm`` is still standing in the tree."""
    return any(
        isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == nm for node in ast.walk(fn_ast)
    )


def inline_symbol_aliases(fn_ast: ast.FunctionDef, symbols: set[str], known: set[str]) -> ast.FunctionDef:
    """Inline a scalar that is a pure symbolic expression over existing dc.symbols rather than
    promoting it: a minted second name for one quantity is one dace cannot prove equal.

    An alias whose expression reads a name rebound later (cegterg's ``nb1 = nbase`` ahead of
    ``nbase = nend``) is spliced flow-sensitively: only the uses that run before that later store
    get the expression, and ``nm``'s own binding survives for the uses that run after it.
    """
    shape_idents = (
        shape_ident_candidates(fn_ast, known)
        | mintable_int_locals(fn_ast, symbols, known)
        | constant_int_locals(fn_ast, known)
    )
    if not shape_idents:
        return fn_ast
    first_rhs, order, reassigned = scan_size_assigns(fn_ast, shape_idents)
    rebound = names_rebound(fn_ast)
    alias: dict[str, ast.AST] = {}
    flow_spliced: list[str] = []
    for nm in order:
        if nm in reassigned or names_a_clamp(first_rhs[nm]):
            continue
        if not is_symbol_expr(first_rhs[nm], symbols | set(alias)):
            continue
        # Folded at every splice, or a deep net nests one layer's extent inside the next until the
        # expression is hundreds of terms and dace's sympy stops finishing the parse.
        rhs = fold_expr(SubstituteNames_(alias).visit(copy.deepcopy(first_rhs[nm])))
        sources = {sub.id for sub in ast.walk(first_rhs[nm]) if isinstance(sub, ast.Name) and sub.id in rebound}
        if not sources:
            alias[nm] = rhs
            continue
        site = bind_site(fn_ast, nm, first_rhs[nm])
        if site is not None:
            block, idx = site
            splice_before_rebind(block, idx + 1, nm, rhs, sources)
            flow_spliced.append(nm)
    if alias:
        fn_ast = SubstituteNames_(alias).visit(fn_ast)
        fn_ast = DropAliasAssign(alias).visit(fn_ast)
    fully_spliced = {nm for nm in flow_spliced if not name_loaded(fn_ast, nm)}
    if fully_spliced:
        fn_ast = DropAliasAssign(fully_spliced).visit(fn_ast)
    if alias or fully_spliced:
        ast.fix_missing_locations(fn_ast)
    return fn_ast


def slice_bound_only_locals(fn_ast: ast.FunctionDef) -> set[str]:
    """Names every one of whose READS sits in a slice bound -- never in a shape, reshape or axis.

    Inlining one cannot change any array's rank or extent, only where a view starts and stops, so
    the wider atom set below is safe here in a way it is not for a name a shape reads.
    """
    inside: set[str] = set()
    outside: set[str] = set()

    def visit(node: ast.AST, in_slice: bool) -> None:
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                (inside if in_slice else outside).add(node.id)
            return
        for field, value in ast.iter_fields(node):
            here = in_slice or (isinstance(node, ast.Slice) and field in ("lower", "upper", "step"))
            if isinstance(value, ast.AST):
                visit(value, here)
                continue
            for child in field_nodes(value):
                if isinstance(child, ast.AST):
                    visit(child, here)

    visit(fn_ast, False)
    return inside - outside


def body_allocated_shape(body: list[ast.stmt], hret: str, values: set[str]) -> list[str] | None:
    """The shape the BODY allocates for what it writes into ``hret``, or ``None``.

    A kept helper's out-param is declared in the CALLER's vocabulary, because the caller is what
    allocates the buffer and the C and Fortran legs emit those extents as constants. Its body
    reaches the same extent through its OWN promoted symbols -- ``out = np.zeros((n, c, oh, ow))``
    against a declaration reading ``int_floor(227 - 10*dilation, stride) + 1`` -- and dace has no
    way to prove one equals the other, so the closing ``hret[:] = out`` is refused. Declaring the
    parameter with the body's spelling makes the two sides the same expression, and dace then
    SOLVES ``oh`` from the shape the caller passes, which is the caller-vocabulary form.
    """
    stores = [
        st
        for st in body
        if isinstance(st, ast.Assign)
        and len(st.targets) == 1
        and isinstance(st.targets[0], ast.Subscript)
        and isinstance(st.targets[0].value, ast.Name)
        and st.targets[0].value.id == hret
        and isinstance(st.targets[0].slice, ast.Slice)
        and st.targets[0].slice.lower is None
        and st.targets[0].slice.upper is None
    ]
    if len(stores) != 1:
        return None  # two writers spell two extents; neither is THE shape
    # ``acc / kernel_size`` is ``acc``'s shape: ``values`` names what carries a VALUE rather than
    # an extent, so peeling those leaves the one array the store is shaped by. Two of them left is
    # a broadcast this cannot size, and it declines rather than pick one.
    written = list({n.id for n in ast.walk(stores[0].value) if isinstance(n, ast.Name) and n.id not in values})
    if len(written) != 1:
        return None
    allocations = [
        shape_argument(st.value)
        for st in body
        if isinstance(st, ast.Assign)
        and len(st.targets) == 1
        and isinstance(st.targets[0], ast.Name)
        and st.targets[0].id == written[0]
        and shape_argument(st.value) is not None
    ]
    if len(allocations) != 1 or not isinstance(allocations[0], (ast.Tuple, ast.List)):
        return None
    return [ast.unparse(dim) for dim in allocations[0].elts]


class NameExtentExpression(ast.NodeTransformer):
    """Replace every expression spelled like one of ``minted``'s keys with that key's symbol name."""

    def __init__(self, minted: dict[str, str]) -> None:
        self.minted = minted

    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        self.generic_visit(node)
        name = self.minted.get(ast.unparse(node))
        if name is None:
            return node
        return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)


def with_named_floor_extents(
    owner: str,
    allocated: list[str],
    body: list[ast.stmt],
    symbol_names: list[str],
    taken: set[str],
) -> tuple[list[str], list[ast.stmt], list[str]]:
    """A kept helper's out-param extents, with every FLOOR-DIVIDED one carrying a symbol of its own.

    dace binds a nested ``@dc.program`` by handing sympy one equation per declared extent and
    solving for the callee's symbols. ``//`` reaches that solver as ``int_floor(a, b)``, a two-
    argument ``Function`` head sympy cannot invert, and it does not decline: matched against the
    caller's own ``int_floor`` it raises ``NotImplementedError: equal function with more than 1
    argument`` and the whole parse dies. conv_standard_1d_dilated_strided declares
    ``int_floor(length - 2 * k + 1, 2) + 1`` and conv_transpose2d_max_pool_hardtanh_mean_tanh's
    pooling helper ``int_floor(oh_ct - maxpool_kernel_size, maxpool_stride) + 1``; both died there.

    Such an equation is REDUNDANT in the first place -- ``length``, ``k`` and ``oh_ct`` are each
    already determined by an input parameter's own extent -- so naming the whole pooled extent
    costs no information and leaves the system linear. The name has to reach the BODY as well as
    the declaration: the body allocates the buffer this parameter is written from and slices the
    taps it pools, and the two sides must stay one expression or dace refuses the closing write.
    """
    minted: dict[str, str] = {}
    dims: list[str] = []
    for dim in allocated:
        if "//" not in dim:
            dims.append(dim)
            continue
        sym = minted.get(dim)
        if sym is None:
            sym = f"{owner}_extent{len(minted)}"
            while sym in taken:
                sym = f"{sym}_"
            minted[dim] = sym
            taken.add(sym)
        dims.append(sym)
    if not minted:
        return allocated, body, symbol_names
    named = NameExtentExpression(minted)
    body = [ast.fix_missing_locations(named.visit(stmt)) for stmt in body]
    return dims, body, [*symbol_names, *minted.values()]


def inline_slice_only_extents(fn_ast: ast.FunctionDef, symbols: set[str], known: set[str]) -> ast.FunctionDef:
    """Splice a slice bound's definition into the slice, so two spans that ARE one quantity share
    a spelling.

    dace hoists every data-dependent slice bound into an opaque symbol of its own, one per NAME. A
    tap loop names three of them for one span -- ``lo``, ``hi`` and the ``dyv = hi - lo`` it writes
    with -- and dace then has no way to see that ``ceiling(dyv * stride / stride)`` is the extent of
    the gather spelled ``hi - lo``. Splicing ``dyv`` back in leaves ``lo`` and ``hi`` as the only
    minted symbols, and both sides fold to the same difference.

    Deliberately narrower than :func:`inline_symbol_aliases`: only names read NOWHERE but in a
    slice bound, so a spliced expression can never reach an allocation or a contraction axis.
    """
    cand = slice_bound_only_locals(fn_ast) & once_bound_locals(fn_ast, known)
    if not cand:
        return fn_ast
    first_rhs, order, reassigned = scan_size_assigns(fn_ast, cand)
    atoms = symbols | mintable_int_locals(fn_ast, symbols, known)
    alias: dict[str, ast.AST] = {}
    for nm in order:
        if nm in reassigned or names_a_clamp(first_rhs[nm]):
            continue
        if is_symbol_expr(first_rhs[nm], atoms | set(alias)):
            alias[nm] = fold_expr(SubstituteNames_(alias).visit(copy.deepcopy(first_rhs[nm])))
    if not alias:
        return fn_ast
    fn_ast = SubstituteNames_(alias).visit(fn_ast)
    fn_ast = DropAliasAssign(alias).visit(fn_ast)
    ast.fix_missing_locations(fn_ast)
    return fn_ast


def is_shape_subscript(node: ast.AST) -> bool:
    """True iff node is <expr>.shape[k] -- a residual .shape read of a body-local transient's dimension."""
    return isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "shape"


def inline_transient_shape_scalars(fn_ast: ast.FunctionDef, known: set[str]) -> ast.FunctionDef:
    """Inline a transient's own .shape[k] dimension read into its uses -- dace forbids a name being both data and symbol."""
    cand = shape_ident_candidates(fn_ast, known)
    if not cand:
        return fn_ast
    first_rhs, order, reassigned = scan_size_assigns(fn_ast, cand)
    alias: dict[str, ast.AST] = {}
    for nm in order:
        if nm not in reassigned and is_shape_subscript(first_rhs[nm]):
            alias[nm] = copy.deepcopy(first_rhs[nm])
    if not alias:
        return fn_ast
    fn_ast = SubstituteNames_(alias).visit(fn_ast)
    fn_ast = DropAliasAssign(alias).visit(fn_ast)
    ast.fix_missing_locations(fn_ast)
    return fn_ast


def plan_size_promotion(
    fn_ast: ast.FunctionDef, known: set[str], symbols: set[str] | None = None
) -> tuple[list[str], list[tuple[str, str]], OrderedSet[str]]:
    """Plan promotion of body-computed size scalars to dace symbols; returns (order, symbol_defs, reassigned)."""
    cand = shape_ident_candidates(fn_ast, known) | mintable_int_locals(fn_ast, symbols or set(), known)
    if not cand:
        return [], [], OrderedSet()
    body_assigned = {
        a.targets[0].id
        for a in ast.walk(fn_ast)
        if isinstance(a, ast.Assign) and len(a.targets) == 1 and isinstance(a.targets[0], ast.Name)
    }
    # Transitive closure: a promoted def's operands must be symbols too (m = min(max_iter, n) drags in n).
    first_rhs, order, reassigned = scan_size_assigns(fn_ast, cand)
    changed = True
    while changed:
        changed = False
        for nm in list(order):
            # ``h__ssa3.shape[2]`` reads a DIMENSION: dragging h__ssa3 in makes an array alias
            # (``h = x``) look like a symbol expression, since ``x`` is a known name.
            bases = shape_base_ids(first_rhs[nm])
            for sub in ast.walk(first_rhs[nm]):
                if not isinstance(sub, ast.Name) or id(sub) in bases:
                    continue
                if sub.id not in known and sub.id not in cand and sub.id in body_assigned:
                    cand.add(sub.id)
                    changed = True
        if changed:
            first_rhs, order, reassigned = scan_size_assigns(fn_ast, cand)
    # Drop the names whose size is not symbolic -- and, transitively, whatever depended on them --
    # rather than abandoning promotion for the WHOLE kernel. The closure above follows every name in
    # a candidate's right-hand side, including positions that are not sizes at all (np.full's dtype
    # argument ``np.maximum(__hcall4, 0.0).dtype``). A dropped name keeps its data-dependent shape.
    while True:
        allowed = known | cand
        unpromotable = {nm for nm in order if not is_symbol_expr(first_rhs[nm], allowed)}
        unpromotable |= cand - set(order)  # a candidate with no definition has nothing to bind
        if not unpromotable:
            break
        cand -= unpromotable
        if not cand:
            return [], [], OrderedSet()
        first_rhs, order, reassigned = scan_size_assigns(fn_ast, cand)
    symbol_defs = [(nm, ast.unparse(first_rhs[nm])) for nm in order]
    return order, symbol_defs, reassigned


class SplitReassignedSize(ast.NodeTransformer):
    """Split a size symbol the body also reassigns: every use, allocation shapes included, reads <name>_iter.

    numpy sizes an allocation by the CURRENT value, and dace promotes a scalar extent per version.
    """

    def __init__(self, names: Iterable[str]) -> None:
        self.names = set(names)
        self.defined_: set[str] = set()  # first assignment per name = the (dropped) def
        self._droppable: set[int] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Record which assignments are droppable BEFORE rewriting anything.

        Only a statement of the function body itself is the symbol's definition. One inside an
        ``if``/``for`` is a CONDITIONAL binding -- one of several -- and dropping it does two wrong
        things: it loses the value on that path, and when it is the branch's only statement it
        leaves an empty block, which is not valid Python. The emitter's own ``ast.parse`` self-check
        then fails and the kernel gets no program at all, so the ratchet never even sees it
        (conv_transpose2d_add_min_gelu_multiply, conv_transpose3d_max_max_sum -- both a
        ``(0 if k - pad >= 0 else ...)`` ternary desugared into a two-branch assignment).

        A conditional binding renames to ``<name>_iter`` like any reassignment instead; the prologue
        seeds ``<name>_iter = <name>`` from the caller-bound symbol, so an unassigned path still
        reads the value it read before.
        """
        self._droppable = {
            id(stmt)
            for stmt in node.body
            if isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id in self.names
        }
        self.generic_visit(node)
        return node

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in self.names:
            nm = node.targets[0].id
            if nm not in self.defined_ and id(node) in self._droppable:
                self.defined_.add(nm)
                return None  # drop the defining assignment; the symbol value is caller-bound
        self.generic_visit(node)  # a reassignment: target + rhs uses rename to <name>_iter
        return node

    def visit_Name(self, node: ast.Name):
        if node.id in self.names:
            node.id = f"{node.id}_iter"
        return node


@functools.lru_cache(maxsize=None, typed=True)
def sympy_reserved(name: str) -> bool:
    """True when dace's parser resolves ``name`` to a sympy CALLABLE instead of a free symbol.

    ``poly``, ``symbols``, ``trace``, ``im``, ``sign`` and friends are sympy functions, so a kernel
    argument spelled that way is not a variable to dace: the moment the name reaches a symbolic
    context (a memlet subset, a shape, a promoted scalar) sympify gets the function object back and
    the parse dies as ``SympifyError: cannot sympify object of type <class 'function'>``, nowhere
    near the argument that caused it. ``sympy.abc._clash`` only shields one-letter and greek names.

    The probe has to be a COMPOUND expression: ``pystr_to_symbolic`` short-circuits a bare name
    straight to ``symbol()`` and would call every name safe.
    """
    try:
        from dace.symbolic import pystr_to_symbolic  # deferred: dace is not a translator dependency
    except ImportError:
        return False  # no dace, no sympy namespace to collide with
    try:
        expr = pystr_to_symbolic(f"{name} + 1")
    except Exception:  # noqa: BLE001 -- any sympify failure means the name is unusable as a symbol
        return True
    return not any(str(s) == name for s in expr.free_symbols)


class RenameNames(ast.NodeTransformer):
    """Rewrite renamed identifiers wherever they appear -- loads, stores and arguments alike."""

    def __init__(self, renames: dict[str, str]) -> None:
        self.renames = renames

    def visit_Name(self, node: ast.Name):
        node.id = self.renames.get(node.id, node.id)
        return node

    def visit_arg(self, node: ast.arg):
        node.arg = self.renames.get(node.arg, node.arg)
        return node


class SubstituteNames(ast.NodeTransformer):
    """Replace each Name in ``values`` by its literal. Used on a symbol RECIPE, which the caller
    evaluates in its own namespace -- a name that only exists inside the emitted module has to be
    gone by then, not merely defined here."""

    def __init__(self, values: dict[str, ast.expr]) -> None:
        self.values = values

    def visit_Name(self, node: ast.Name) -> ast.AST:
        replacement = self.values.get(node.id)
        return ast.copy_location(copy.deepcopy(replacement), node) if replacement is not None else node


def bound_names(body: list[ast.stmt]) -> OrderedSet[str]:
    """The names the body BINDS -- the only ones a rename may touch.

    A reserved name that is merely CALLED (``sqrt(x)``, ``exp(x)``, ``log(x)``) is resolved by
    dace's own replacement table and renaming it would break the call.
    """
    names: OrderedSet[str] = OrderedSet()
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                names.add(node.id)
    return names


def names_logical_sparse(kir: KernelIR) -> bool:
    """True when the body still spells a LOGICAL sparse matrix (``A @ x``), i.e. the kir is raw.

    The frontend expands ``A`` into its physical CSR buffers in the SIGNATURE, but only
    :func:`numpyto_common.lowering.lower` rewrites the BODY onto those buffers; a raw kir therefore
    reaches dace with a signature and a body that disagree, and dace answers ``Use of undefined
    variable "A"``. A buffer-style kernel (spmv) names no logical matrix and must NOT be lowered:
    its data-dependent slice is expressible through dace's symbolic shapes, and lowering it would
    make a variable-length copy dace cannot allocate.
    """
    if not kir.sparse:
        return False
    logical = set(kir.sparse)
    return any(isinstance(n, ast.Name) and n.id in logical for n in ast.walk(kir.tree))


def called_helpers(body: list[ast.stmt], helpers: list[KernelIR]) -> OrderedSet[str]:
    """Kept-helper names ``body`` still calls. A specialised helper is spelled ``<name>__s<N>``,
    so the match is the name or that prefix -- never a substring, which would claim ``relu_scale``."""
    names = OrderedSet(h.kernel_name for h in helpers)
    called: OrderedSet[str] = OrderedSet()
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                fid = node.func.id
                if any(fid == n or fid.startswith(f"{n}__") for n in names):
                    called.add(fid)
    return called


def contiguous_subscript(node: ast.Subscript) -> bool:
    """Whether ``node`` selects a CONTIGUOUS block, so its strides are the ones a plain parameter
    of that shape declares.

    C order: leading integer indices, then at most one bounded slice, then whole axes.
    ``Q[k, :, :]`` walks a plane with strides ``(N, 1)``; ``Q[:, :, k]`` walks the same shape with
    strides ``(N * m, m)``, and a step or an inserted axis is neither.
    """
    elements = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
    spanning = False
    for element in elements:
        if inserts_axis(element):
            return False
        if isinstance(element, ast.Slice):
            if element.step is not None:
                return False
            if spanning and not (element.lower is None and element.upper is None):
                return False
            spanning = True
        elif spanning:
            return False  # an index BEHIND a kept axis strides over it
    return True


def sliced_extents(node: ast.Subscript, shapes: dict[str, list[str]], known: set[str]) -> list[str] | None:
    """The extents a subscript leaves behind, by numpy's rank rules: a slice KEEPS an axis, an
    index DROPS it. ``None`` when the base or a bound is not one this can spell.

    Narrower than :meth:`ResolveShapeReads.sliced` on purpose, and for the opposite reason: the
    INDEX expression decides no extent, so an index by a loop variable -- ``Q[:, :, k]``, every
    one of these call sites -- has to leave the shape known, and that method declines it.
    """
    if not isinstance(node.value, ast.Name):
        return None
    base = shapes.get(node.value.id)
    elements = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
    if base is None or len(elements) > len(base):
        return None
    tokens: list[str] = []
    for axis, element in enumerate(elements):
        if not isinstance(element, ast.Slice):
            continue
        if element.step is not None:
            return None
        if element.lower is None and element.upper is None:
            tokens.append(base[axis])
            continue
        bounds = [b for b in (element.lower, element.upper) if b is not None]
        if any(not is_symbol_expr(b, known) for b in bounds):
            return None
        upper = ast.unparse(element.upper) if element.upper is not None else base[axis]
        if element.lower is None or (isinstance(element.lower, ast.Constant) and element.lower.value == 0):
            tokens.append(upper)
        else:
            tokens.append(f"{upper} - ({ast.unparse(element.lower)})")
    return tokens + base[len(elements) :]


def materialize_strided_helper_args(
    body: list[ast.stmt],
    written_by: dict[str, list[bool]],
    shapes: dict[str, list[str]],
    dtype_of: dict[str, str],
    known: set[str],
) -> list[ast.stmt]:
    """``body`` with every non-contiguous slice argument of a kept-helper call copied through a
    contiguous temp, allocations first.

    dace solves a callee's symbols from what the call passes and equates STRIDES as well as shapes.
    ``bratu_jvp(u, Q[:, :, k], ...)`` hands a rank-3 array's trailing-index view -- strides
    ``(N * (m + 1), m + 1)`` -- to a parameter declared ``[N, N]``, whose strides are ``(N, 1)``:
    the two equations say ``__SOLVE_N = N`` and ``__SOLVE_N = (m + 1) * N`` at once, sympy returns
    nothing, and the frontend reports "Cannot infer values for symbols in inference". Accepting the
    view would make the callee walk the wrong elements, so the view is copied, as
    ``build_callsite_stmts`` does for an array-returning call.

    ``written_by`` says, per call-site position, whether the callee WRITES that parameter; one that
    it does is copied back after the call, or the sweep would be a silent dropped result.

    The temp is allocated at FUNCTION scope, not beside the call: these calls sit in loops, and one
    name with one shape is what dace wants. An extent naming anything but a declared array, scalar
    or symbol is not loop-invariant, and declines the rewrite rather than sizing a buffer off a
    loop variable.
    """
    allocations: list[ast.stmt] = []
    counter = itertools.count()

    def materialized(stmt: ast.stmt) -> list[ast.stmt]:
        pre: list[ast.stmt] = []
        post: list[ast.stmt] = []
        for call in [n for n in ast.walk(stmt) if isinstance(n, ast.Call)]:
            if not isinstance(call.func, ast.Name):
                continue
            flags = written_by.get(call.func.id)
            if flags is None or len(flags) != len(call.args):
                continue
            for index, arg in enumerate(call.args):
                if not isinstance(arg, ast.Subscript) or contiguous_subscript(arg):
                    continue
                if not isinstance(arg.value, ast.Name) or arg.value.id not in dtype_of:
                    continue
                tokens = sliced_extents(arg, shapes, known)
                if not tokens or any(i not in known for t in tokens for i in IDENT_RE.findall(t)):
                    continue
                name = f"__hslice_{next(counter)}"
                dtype = dtype_of[arg.value.id]
                allocations.append(ast.parse(f"{name} = np.empty(({', '.join(tokens)},), dtype=np.{dtype})").body[0])
                pre.append(ast.parse(f"{name}[:] = {ast.unparse(arg)}").body[0])
                if flags[index]:
                    post.append(ast.parse(f"{ast.unparse(arg)} = {name}").body[0])
                call.args[index] = ast.Name(id=name, ctx=ast.Load())
        return pre + [stmt] + post

    def rewrite(stmts: list[ast.stmt]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for stmt in stmts:
            for field, value in ast.iter_fields(stmt):
                if isinstance(value, list) and value and all(isinstance(v, ast.stmt) for v in value):
                    setattr(stmt, field, rewrite(value))
            out.extend(materialized(stmt))
        return out

    rewritten = rewrite(body)
    return allocations + rewritten


@dataclasses.dataclass(slots=True)
class RenderedProgram:
    """One ``@dc.program``: its signature and body, plus what the MODULE has to declare for it.

    A module holds the kernel program and one program per kept helper, so the declarations are
    pooled: symbols, pinned constants and the renames a sympy collision forced are module-level
    facts, while ``params`` and ``body`` are the program's own.
    """

    name: str
    params: list[str]
    body: list[ast.stmt]
    symbol_names: list[str]
    #: Per-dimension binding recipe, evaluated by the CALLER. Only the kernel program has one --
    #: a helper's extents come from the shapes its call site passes, which dace resolves itself.
    symbol_defs: list[tuple[str, str]]
    renames: dict[str, str]
    pinned: dict[str, PinnedValue]
    needs_complex: bool


def exits_with_valueless_return(body: list[ast.stmt]) -> bool:
    """Whether ``body`` ends in a ``return`` that carries no value."""
    return bool(body) and isinstance(body[-1], ast.Return) and body[-1].value is None


def without_valueless_returns(body: list[ast.stmt]) -> list[ast.stmt]:
    """``body``, in TAIL position, with every ``return`` that carries no value structured away.

    :func:`numpyto_common.frontend._rewrite_returns_to_outparam` closes a promoted-return helper
    with ``hret[:] = expr`` plus a bare ``return``, which is what the C and Fortran legs emit as a
    ``void`` out-param procedure. dace lowers any ``return`` into a ReturnBlock, and its codegen
    emits a nested program's blocks INLINE in the caller's function -- so that bare return becomes
    a C ``return;`` that leaves the CALLER. Every statement after the call site is skipped and the
    kernel computes a wrong answer in silence, which is how eigh_test, nbody, channel_flow and
    cp2k_density_matrix_trs4 all stopped agreeing with numpy.

    Two exact rewrites, both of which say what falling off the end already says:

    * a return in tail position, and anything after it in the same list, is dropped;
    * a guard that exits (``if c: <A>; return`` with ``<B>`` after it) becomes ``if c: <A> else:
      <B>``, which puts ``<A>`` and ``<B>`` back in tail position for the recursion.

    A return this cannot reach -- inside a loop, or under a guard that already carries an ``else``
    -- is left alone rather than guessed at.
    """
    kept: list[ast.stmt] = []
    for index, stmt in enumerate(body):
        if isinstance(stmt, ast.Return) and stmt.value is None:
            return kept
        rest = body[index + 1 :]
        guard_exits = isinstance(stmt, ast.If) and not stmt.orelse and exits_with_valueless_return(stmt.body)
        if isinstance(stmt, ast.If) and (not rest or guard_exits):
            trailing = stmt.orelse if not rest else rest
            arm = without_valueless_returns(stmt.body)
            stmt.body = arm if arm else [ast.copy_location(ast.Pass(), stmt)]
            stmt.orelse = without_valueless_returns(trailing)
            kept.append(stmt)
            return kept
        kept.append(stmt)
    return kept


def render_program(
    kir: KernelIR,
    fn_name: str | None = None,
    helpers: Sequence[KernelIR] = (),
    nested: bool = False,
) -> RenderedProgram:
    """Lower ``kir``'s body into the form dace's frontend parses, and return it with its signature.

    Shared by the kernel and by every kept helper: a helper is a ``@dc.program`` of its own, so it
    needs the same desugaring, the same shape-read resolution and the same symbol promotion, and
    running it through a second code path would let the two drift.

    ``helpers`` is the whole kept-helper closure, not ``kir.helpers``: a helper calls a SIBLING and
    carries no list of its own. It is what :func:`materialize_strided_helper_args` reads to tell a
    call to one from any other call in the body.

    ``nested`` marks a kept helper, whose symbols dace SOLVES from the shapes its call site passes.
    The kernel program's own symbols are bound by recipe instead, so the two differ in what a
    declared extent may spell -- see :func:`with_named_floor_extents`.
    """
    if names_logical_sparse(kir):
        kir = lower(kir)
    kir = freeze_pinned_extent_scalars(kir)
    kir = freeze_shape_only_parameters(kir)
    name = fn_name or kir.kernel_name
    arrays = {a.name: a for a in kir.arrays}
    scalars = {s.name: s for s in kir.scalars}
    symbol_names = [s.name for s in kir.symbols]
    # Sparse kirs carry size symbols only in array shapes; collect free idents so each is declared as a dc.symbol.
    arr_shapes = {a.name: [str(s) for s in a.shape] for a in kir.arrays}
    known = set(arrays) | set(scalars)
    shape_idents: set[str] = set()
    for toks in arr_shapes.values():
        for tok in toks:
            for ident in IDENT_RE.findall(tok):
                shape_idents.add(ident)
                if ident not in known and ident not in symbol_names:
                    symbol_names.append(ident)
    # A scalar param used as an array shape (e.g. ``Nt`` sizing ``KE[Nt + 1]``) must be a dc.symbol:
    # a dace shape annotation cannot reference a runtime scalar, and a name cannot be both. Promote it
    # to a module-level symbol and drop it from the scalar params below (the caller binds it as a symbol).
    # Ordered: the loop below appends into ``symbol_names``, which IS the emitted dc.symbol
    # declaration block, so these have to keep the parameter order ``scalars`` came in.
    # A scalar used ONLY as a body extent -- lenet's ``C_before_fc1`` in
    # ``np.reshape(x, (N, C_before_fc1))`` -- appears in no declared array shape, so the scan above
    # never sees it and it stays a runtime scalar. DaCe cannot take a data descriptor as an extent:
    # the frontend tries to mint a symbol of that name and collides with the descriptor already
    # bound to it. Normalized on a COPY so both reshape spellings reach ``shape_argument`` in the
    # one form it reads. A rebound name is excluded -- a dc.symbol is immutable, and a name cannot
    # be both symbol and data.
    body_probe = NormalizeReshape().visit(copy.deepcopy(kir.tree))
    rebound = {n.id for n in ast.walk(body_probe) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    rebound |= {
        n.value.id
        for n in ast.walk(body_probe)
        if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name)
    }
    body_shape_idents: set[str] = set()
    for node in ast.walk(body_probe):
        shape_arg = shape_argument(node)
        if shape_arg is None:
            continue
        elements = shape_arg.elts if isinstance(shape_arg, (ast.Tuple, ast.List)) else [shape_arg]
        for element in elements:
            for sub in ast.walk(element):
                if isinstance(sub, ast.Name) and sub.id not in rebound:
                    body_shape_idents.add(sub.id)
    shape_scalars = OrderedSet(s for s in scalars if s in shape_idents or s in body_shape_idents)
    for s in shape_scalars:
        if s not in symbol_names:
            symbol_names.append(s)

    # Program signature: arrays + scalars in original input_args order; symbols are module-level.
    params: list[str] = []
    for arg in kir.input_args:
        if arg in arrays:
            params.append(f"{arg}: {array_annotation(arrays[arg])}")
        elif arg in scalars and arg not in shape_scalars:
            params.append(f"{arg}: {dace_dtype(scalars[arg].dtype)}")
        # symbols (and scalars promoted to symbols): skip (declared at module scope below)

    needs_complex = any(dace_dtype(a.dtype) == "dc_complex_float" for a in kir.arrays) or any(
        dace_dtype(s.dtype) == "dc_complex_float" for s in kir.scalars
    )

    # Desugar the body with the same pass numba/pythran use for feature parity; falls back to verbatim on parse failure.
    fn_ast = copy.deepcopy(kir.tree)
    fn_ast.name = kir.kernel_name
    try:
        desugared = desugar_for_python_backend(ast.unparse(fn_ast), kir, backend="dace")
        fn_ast = next(n for n in ast.parse(desugared).body if isinstance(n, ast.FunctionDef))
    except Exception as error:  # noqa: BLE001 -- keep the verbatim body if desugar fails
        logging.getLogger(__name__).warning(
            "dace desugar fell back to the verbatim body for %s (%s): %s: %s",
            kir.kernel_name,
            kir.source_path,
            type(error).__name__,
            error,
        )
        fn_ast = kir.tree
    # Rewrite leaked np_float/np_complex tokens to the dace precision global the module binds.
    framework_dtype = RewriteFrameworkDtype()
    fn_ast = framework_dtype.visit(fn_ast)
    # ``np.asarray`` has no dace replacement; on an array it is numpy's own identity, so it goes.
    fn_ast = DropIdentityAsarray(rank_table(fn_ast, {a.name: len(a.shape) for a in kir.arrays})).visit(fn_ast)
    # dace's frontend has no conditional expression (RHS or nested value): lower both to if/else.
    # Array branches of different dtype kinds bind one name, so both take numpy's join first.
    declared_kinds = {a.name: kind for a in kir.arrays if (kind := kind_of_dtype_str(a.dtype)) is not None}
    ternary_ranks = rank_table(fn_ast, {a.name: len(a.shape) for a in kir.arrays})
    fn_ast = DesugarTernary(dtype_table_(fn_ast, declared_kinds), ternary_ranks).visit(fn_ast)
    # dace's frontend takes one comparator per Compare: split a chained range test into its links.
    fn_ast = DesugarChainedCompare().visit(fn_ast)
    # dace names a method call by its receiver chain: a call/subscript receiver is refused outright.
    fn_ast = BindMethodReceiver().visit(fn_ast)
    # One variable per loop NAME in dace, so nested loops sharing one -- two ``for _ in range(...)``
    # -- share a counter and the outer never ends. Before every pass below that reads a loop target.
    uniquify_nested_loop_targets(fn_ast)
    ast.fix_missing_locations(fn_ast)
    # numpy infers a reshape's -1 from the size; dace takes the shape literally, so spell it out.
    fn_ast = ResolveInferredReshape(arr_shapes).visit(fn_ast)
    # dace unwraps a reshape's varargs then iterates them; a lone -1 is numpy's ravel.
    fn_ast = NormalizeReshape().visit(fn_ast)
    # dace refuses a store through a reshape it had to materialize; spell the copy numpy makes.
    fn_ast = MaterializeWrittenReshape().visit(fn_ast)
    # dace has no np.outer and rejects negative-stride subscripts; rewrite both to forms dace accepts.
    fn_ast = DesugarUnreplacedCalls().visit(fn_ast)
    # An einsum that contracts nothing is a broadcast product; dace's GEMM path mints a K=1 MatMul
    # whose views collapse to shapes its dispatch refuses. Say the product directly.
    fn_ast = DesugarContractionFreeEinsum().visit(fn_ast)
    fn_ast = DesugarReverseSlice().visit(fn_ast)
    # dace's frontend rejects element iteration over an array value: rewrite to an indexed range form.
    fn_ast = DesugarArrayIteration(
        lambda array: ast.parse(arr_shapes[array][0], mode="eval").body if arr_shapes.get(array) else None,
        lambda target, ordinal: f"__hpcagent_bench_idx{ordinal}",
    ).visit(fn_ast)
    # dace rejects a reversed dynamic-length slice (a View edge); snapshot it into a fixed-extent workspace first.
    arr_dtypes = {a.name: dace_dtype(a.dtype) for a in kir.arrays}
    fn_ast = MaterializeDynamicFlip(arr_shapes, arr_dtypes, set(symbol_names)).visit(fn_ast)
    ast.fix_missing_locations(fn_ast)
    # dace cannot codegen a chained assignment: one binding per target, the value evaluated once.
    chain_ranks = {**{a.name: len(a.shape) for a in kir.arrays}, **dict.fromkeys([*scalars, *symbol_names], 0)}
    fn_ast = dace_chained_assign_split(chain_ranks).visit(fn_ast)
    fn_ast = DropRedundantSliceStore().visit(fn_ast)
    ast.fix_missing_locations(fn_ast)
    # dace does not lower a point-wise fancy-index WRITE; it answers garbage rather than refusing.
    scatter_ranks = rank_table(fn_ast, {a.name: len(a.shape) for a in kir.arrays})
    scatter_ranks.update(loop_target_ranks(fn_ast))
    fn_ast = PointwiseScatterToLoop(scatter_ranks).visit(fn_ast)
    ast.fix_missing_locations(fn_ast)
    # Turn __hpcagent_bench_zeros__() markers into np.zeros/np.ones with the declared initial value.
    zeros_locals = kir.zeros_locals
    zeros_fills = kir.zeros_fills
    local_dtypes = kir.local_dtypes
    default_dtype = kir.float_precision or "float64"
    resolve_zeros = ResolveZeros(zeros_locals, zeros_fills, local_dtypes, default_dtype)
    fn_ast = resolve_zeros.visit(fn_ast)
    declare_unallocated_zeros_locals(
        fn_ast, zeros_locals, resolve_zeros.allocated, zeros_fills, local_dtypes, default_dtype, set(kir.input_args)
    )
    # np.empty's dace replacement has no dtype default (unlike zeros/ones/full): a bare call means
    # numpy's own float64 default, so fill in the precision-driven float global explicitly.
    fn_ast = AnnotateEmptyDtype(dace_dtype(default_dtype)).visit(fn_ast)
    # A builtin used as a dtype reaches dace's descriptor as a str, which its dtype property rejects.
    fn_ast = RewriteBuiltinDtype(dace_dtype(default_dtype)).visit(fn_ast)
    # A promoted return is a PARAMETER here; re-allocating it in the body would leave it unwritten.
    out_params = {a: [t.replace(" ", "") for t in arr_shapes[a]] for a in kir.input_args if a in arrays}
    fn_ast = FillOutputParamRealloc(out_params).visit(fn_ast)
    ast.fix_missing_locations(fn_ast)
    # dace has no runtime .shape: rewrite arr.shape[k] to the symbolic dim and drop redundant/illegal symbol recomputes.
    # Tuple assignment first, so the shape passes below see the subscript spelling they resolve.
    fn_ast = SplitTupleAssign().visit(fn_ast)
    ast.fix_missing_locations(fn_ast)
    # Before every .shape pass: what dace has no replacement for becomes a callback, and the
    # lowerings below spell their extents as .shape reads for those passes to resolve like any other.
    declared_ranks = {nm: len(dims) for nm, dims in arr_shapes.items()}
    complex_arrays = {nm for nm, dt in arr_dtypes.items() if "complex" in dt}
    fn_ast = LowerCallsDaceCannotReplace(rank_table(fn_ast, declared_ranks), complex_arrays).visit(fn_ast)
    ast.fix_missing_locations(fn_ast)
    fn_ast = ShapeToSymbol(arr_shapes).visit(fn_ast)
    # ... and every remaining .shape read, including on a transient: one unresolved read makes the
    # enclosing size expression non-symbolic, and promotion is all-or-nothing.
    # A declared scalar or size symbol is rank 0 -- it broadcasts against anything and decides no
    # extent -- so it has to be KNOWN, now that one unknown operand poisons the whole expression.
    value_shapes = {**arr_shapes, **{nm: [] for nm in list(scalars) + symbol_names}}
    fn_ast = ResolveShapeReads(value_shapes).visit(fn_ast)
    ast.fix_missing_locations(fn_ast)
    # dace sizes np.where from its branches: give a two-scalar where the condition's shape, or it refuses.
    fn_ast = BroadcastScalarWhere(value_shapes).visit(fn_ast)
    ast.fix_missing_locations(fn_ast)
    # Inline a shape scalar that's a pure symbolic alias of an existing dc.symbol, rather than promoting a fresh one.
    # Induction variables count as symbols here and NOWHERE else: they are atoms the inliner may fold
    # through, but promoting one to a dc.symbol the caller binds would fix it at one iteration.
    loop_syms = set(loop_induction_symbols(fn_ast))
    # Before the inliner runs, not after: an identity int() cast makes an alias unrecognisable, and
    # the whole point of the inliner is that one quantity keeps one spelling.
    fn_ast = StripIdentityIntCasts(set(symbol_names) | loop_syms).visit(fn_ast)
    ast.fix_missing_locations(fn_ast)
    # Before any inlining: the length name this reads has to still be standing.
    spell_aranges_with_named_lengths(fn_ast, set(arrays) | set(scalars) | set(symbol_names))
    fn_ast = inline_symbol_aliases(
        fn_ast, set(symbol_names) | loop_syms, set(arrays) | set(scalars) | set(symbol_names)
    )
    # Inline a transient's own .shape read used to size an accumulator (dace forbids name-as-both).
    fn_ast = inline_transient_shape_scalars(fn_ast, set(arrays) | set(scalars) | set(symbol_names))
    # Name any compound shape expression first, so promotion has a single name to work on.
    fn_ast = hoist_compound_extents(fn_ast, set(arrays) | set(scalars) | set(symbol_names))
    # Again over the names promotion is ABOUT to mint, to a FIXED POINT: each inlined helper
    # recopies the previous layer's extents, and inlining one SPLICES its definition into the shape
    # arguments, exposing names that were in no shape before (resnet's ``__inl1_kh = 7``).
    known = set(arrays) | set(scalars) | set(symbol_names)
    previous: set[str] = set()
    for unused in range(INLINE_ALIAS_ROUNDS):
        promotable, unused, unused = plan_size_promotion(fn_ast, known, set(symbol_names))
        if set(promotable) == previous:
            break  # nothing new was exposed: another round would substitute the same names again
        previous = set(promotable)
        fn_ast = inline_symbol_aliases(fn_ast, set(symbol_names) | set(promotable) | loop_syms, known)
    # AFTER the alias inliner: the tap-loop span reaches here as the name ``span_h``, and only the
    # inlining above turns it into the ``A * stride + 1`` this matches on.
    # Integer scalar PARAMETERS count as symbols here: dace promotes each to its own ``__sym_x``
    # the moment a slice bound reads it, so a span built from one is a symbolic expression whether
    # or not this emitter calls it a symbol. Leaving ``stride`` out left ``span_h = (oh - 1) *
    # stride + 1`` un-spliced, and dace then compared ``ceiling(__sym_span_h/__sym_stride)``
    # against ``oh`` with no way to cancel the two minted names.
    int_scalars = {n for n, d in scalars.items() if d.dtype.startswith(("int", "uint"))}
    fn_ast = inline_slice_only_extents(fn_ast, set(symbol_names) | loop_syms | int_scalars, known)
    fn_ast = DivisibleStridedSpan().visit(fn_ast)
    ast.fix_missing_locations(fn_ast)
    # dace forbids a data-dependent array shape; promote body-computed size scalars to dc.symbols the caller binds.
    promoted, symbol_defs, reassigned = plan_size_promotion(
        fn_ast, set(arrays) | set(scalars) | set(symbol_names), set(symbol_names)
    )
    for nm in promoted:
        if nm not in symbol_names:
            symbol_names.append(nm)
    if reassigned:
        fn_ast = SplitReassignedSize(reassigned).visit(fn_ast)
        ast.fix_missing_locations(fn_ast)
        # Seeded from the recipe, not the symbol: a symbol read only here is an ABI slot the caller must fill.
        recipes = dict(symbol_defs)
        fn_ast.body[0:0] = [ast.parse(f"{nm}_iter = {recipes[nm]}").body[0] for nm in reassigned]
    fn_ast = DropSymbolAssign(symbol_names).visit(fn_ast)
    ast.fix_missing_locations(fn_ast)
    # Last, after promotion: a name that became a dc.symbol is no longer a container, and neither
    # rewrite has anything to say about one. See dace issues 05 and 06 for both causes.
    declared_floats = {d.name for d in (*kir.arrays, *kir.scalars) if not d.dtype.startswith(("int", "uint", "bool"))}
    floats = float_names(fn_ast, declared_floats)
    fn_ast = CopyScalarAlias(value_shapes, floats, set(symbol_names) | set(scalars)).visit(fn_ast)
    widen_int_seeds(fn_ast, floats, set(symbol_names))
    ast.fix_missing_locations(fn_ast)
    # After every pass that emits ``t op= v`` and the size planners that read it as a mutation;
    # before the view passes, which must see the store it is.
    aug_seed = {**chain_ranks, **dict.fromkeys(symbol_names, 0), **loop_target_ranks(fn_ast)}
    fn_ast = DesugarAugAssign(settled_ranks(fn_ast, rank_table(fn_ast, aug_seed))).visit(fn_ast)
    ast.fix_missing_locations(fn_ast)
    # Last, over the settled body: dace makes a View node per binding and refuses to reassign one.
    # A rebound view name gets a fresh name per binding where the live ranges are disjoint, and a
    # copy where they are not -- a merge or a loop needs a phi, which a rename is not. One
    # descriptor also cannot hold two shapes, so a re-allocation gets its own name too.
    symbol_set = frozenset(symbol_names)
    copy_view_bindings(fn_ast, mixed_view_names(fn_ast, symbol_set), symbol_set)
    copy_view_bindings(fn_ast, set(version_rebound_views(fn_ast, symbol_set)), symbol_set)
    copy_view_bindings(fn_ast, views_of_written_bases(fn_ast), symbol_set)
    version_reallocations(fn_ast)
    # Widest last: a name bound to a computed value in two arms of a branch is one descriptor dace
    # refuses to rebind, and the narrower predicates above see neither binding. Only the bindings
    # with disjoint live ranges are renamed -- a phi is declined, not invented.
    version_rebound_names(fn_ast, value_binding)
    ast.fix_missing_locations(fn_ast)
    body = list(fn_ast.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    body = without_valueless_returns(body)
    # After the docstring, so the allocations do not displace it, and after every pass above, so
    # the argument expressions they rewrite have settled.
    written_by = {
        h.kernel_name: [any(a.name == p and a.is_output for a in h.arrays) for p in h.abi_param_order()]
        for h in helpers
    }
    if written_by:
        local_shapes = {nm: list(dims) for nm, dims in kir.zeros_locals.items()}
        slice_dtypes = {a.name: a.dtype for a in kir.arrays}
        slice_dtypes.update({nm: kir.local_dtypes.get(nm, default_dtype) for nm in local_shapes})
        body = materialize_strided_helper_args(
            body,
            written_by,
            {**arr_shapes, **local_shapes},
            slice_dtypes,
            set(arrays) | set(scalars) | set(symbol_names) | set(kir.pinned_consts) | set(kir.inlined_consts),
        )
    # A kept helper's out-param: redeclare it with the extent its OWN body allocates, so the
    # closing ``hret[:] = out`` compares one expression with itself. See body_allocated_shape.
    hret = kir.return_kind if kir.return_kind in arrays else ""
    if hret:
        resolvable = set(arrays) | set(scalars) | set(symbol_names)
        allocated = body_allocated_shape(body, hret, set(scalars) | set(symbol_names))
        if allocated and all(i in resolvable for d in allocated for i in IDENT_RE.findall(d)):
            if nested:
                # Only a NESTED program: the kernel's own symbols are bound by recipe from the
                # harness (``symbol_defs``), and a name minted here would have none.
                allocated, body, symbol_names = with_named_floor_extents(
                    name, allocated, body, symbol_names, resolvable | set(bound_names(body))
                )
            rebuilt = array_annotation(dataclasses.replace(arrays[hret], shape=tuple(allocated)))
            params = [f"{hret}: {rebuilt}" if p.split(":", 1)[0].strip() == hret else p for p in params]
    if kir.return_kind or reassigned:
        # A HELPER's symbol the settled program never names -- retired by the redeclaration above,
        # or cancelled out of an extent by extent_without_dead_symbols -- is not a symbol dace can
        # solve OR accept: passing a keyword the callee does not take is a DaceSyntaxError. A
        # reassigned size seeded from its recipe retires its own symbol the same way.
        prunable = set(symbol_names) if kir.return_kind else set(reassigned)
        used = {i for param in params for i in IDENT_RE.findall(param.split(":", 1)[1])}
        used |= {n.id for n in ast.walk(ast.Module(body=body, type_ignores=[])) if isinstance(n, ast.Name)}
        while True:  # a kept recipe reads its operands, so they stay bound
            kept = {n for n, unused in symbol_defs if n in used or n not in prunable}
            reached = {i for n, e in symbol_defs if n in kept for i in IDENT_RE.findall(e)} - used
            if not reached:
                break
            used |= reached
        symbol_names = [s for s in symbol_names if s in used or s not in prunable]
        symbol_defs = [(n, e) for n, e in symbol_defs if n in used or n not in prunable]

    # A bound name that collides with a sympy callable is not a variable to dace (see
    # sympy_reserved). Rename every one of them and record the map: the emitted program is the only
    # place the new spelling exists, so the caller has to rewrite its keyword arguments to match.
    param_names = [p.split(":", 1)[0].strip() for p in params]
    candidates = OrderedSet([*param_names, *symbol_names, *bound_names(body)])
    renames = {n: f"__{n}" for n in candidates if sympy_reserved(n)}
    if renames:
        body = [RenameNames(renames).visit(stmt) for stmt in body]
        params = [f"{renames.get(n, n)}: {p.split(':', 1)[1].strip()}" for n, p in zip(param_names, params)]
        symbol_names = [renames.get(n, n) for n in symbol_names]
        # The recipe is evaluated by the CALLER over the renamed keyword arguments, so its free
        # names have to be renamed with them or the eval below raises NameError on the old spelling.
        symbol_defs = [
            (renames.get(n, n), ast.unparse(RenameNames(renames).visit(ast.parse(e, mode="eval")).body))
            for n, e in symbol_defs
        ]

    # A PINNED CONFIG knob is a constant, not a symbol: the C leg spells it ``constexpr int64_t
    # max_iter = 100``, and lowering having promoted it (it sizes a workspace) must not turn it into
    # a dc.symbol here. Nothing binds such a symbol -- ``bind_free_symbols`` recovers a symbol from
    # an array's shape or from a recipe, and a config knob is neither -- so gmres' compiled SDFG
    # died on "Missing program argument". Substituted into the recipes too, because the CALLER
    # evaluates those outside this module, where the name does not exist.
    pinned = {n: v for n, v in (kir.pinned_consts or {}).items() if n in symbol_names}
    if pinned:
        symbol_names = [n for n in symbol_names if n not in pinned]
        literals: dict[str, ast.expr] = {n: ast.Constant(value=v) for n, v in pinned.items()}
        symbol_defs = [
            (n, ast.unparse(SubstituteNames(literals).visit(ast.parse(e, mode="eval")).body)) for n, e in symbol_defs
        ]
        # The SIGNATURE counts as a use, not only the body: seissol's ``nb`` sizes ``Q[batch, nb, 9]``
        # and appears nowhere else, so a body-only scan dropped both its symbol declaration and its
        # constant, leaving the annotation reading a name the module never binds.
        named = {node.id for stmt in body for node in ast.walk(stmt) if isinstance(node, ast.Name)}
        named |= {ident for param in params for ident in IDENT_RE.findall(param)}
        pinned = {n: v for n, v in pinned.items() if n in named}

    return RenderedProgram(
        name=name,
        params=params,
        body=body,
        symbol_names=symbol_names,
        symbol_defs=symbol_defs,
        renames=renames,
        pinned=pinned,
        needs_complex=needs_complex or framework_dtype.used_complex,
    )


def folded_with_constants(text: str, pinned: dict[str, PinnedValue]) -> str:
    """``text`` with every pinned knob replaced by its value, then folded.

    One canonical form for both sides of a lookup. ``c_out_per_group = out_channels //
    conv_transpose_groups`` and a ``bias`` declared ``out_channels`` long are the same extent
    whenever the knob is 1, and only substituting the knob makes the two texts say so.

    Through sympy, not the syntactic folder: the two spellings of one extent differ by arithmetic
    the identities do not reach -- ``1 * (kernel_size - 1) + output_padding + 1`` against
    ``kernel_size + output_padding`` -- and a key that is not canonical answers for nothing. The
    result is never emitted, only compared, so ``floor`` appearing in it costs nothing.
    """
    substituted = IDENT_RE.sub(lambda m: str(pinned[m.group()]) if m.group() in pinned else m.group(), text)
    canonical = sympify_shape(substituted)
    return str(canonical) if canonical is not None else fold_shape_expr(substituted)


def caller_side_recipe(owner: ast.FunctionDef, arg: ast.expr, pinned: dict[str, PinnedValue]) -> str:
    """The folded expression a call ARGUMENT stands for, or ``""`` when there is not one.

    An expression argument is its own recipe; a bare Name is one only through the owner's single
    assignment to it. Anything bound more than once has no one recipe and is declined.
    """
    if not isinstance(arg, ast.Name):
        return folded_with_constants(ast.unparse(arg), pinned)
    bound = [
        st.value
        for st in ast.walk(owner)
        if isinstance(st, ast.Assign)
        and len(st.targets) == 1
        and isinstance(st.targets[0], ast.Name)
        and st.targets[0].id == arg.id
    ]
    if bound:
        return folded_with_constants(ast.unparse(bound[0]), pinned) if len(bound) == 1 else ""
    # A name the owner never assigns has no recipe: it IS the caller's symbol, and which of the two
    # spellings survives is the ALIAS pass's call, made with the captured names in hand. Answering
    # here as well put BOTH directions in one map -- ``channels -> c`` from the alias pass beside
    # ``c -> channels`` from this one -- and with_helper_vocabulary then moved the two sides apart,
    # respelling the descriptors caller->helper while RenameNames took the body helper->caller:
    # max_pooling_2d declared [batch_size, c, h, w] over a body computing in channels/height/width.
    return ""


@dataclasses.dataclass(slots=True)
class HelperBinding:
    """What one call site says about a kept helper's symbols. See :func:`helper_call_bindings`."""

    aliases: dict[str, str] = dataclasses.field(default_factory=dict)
    constants: dict[str, PinnedValue] = dataclasses.field(default_factory=dict)
    collapse: dict[str, str] = dataclasses.field(default_factory=dict)
    expressions: dict[str, str] = dataclasses.field(default_factory=dict)
    pinned: dict[str, PinnedValue] = dataclasses.field(default_factory=dict)


def captured_parameter_names(hkir: KernelIR, abi: list[str], args: list[ast.expr]) -> list[str]:
    """The helper parameters whose NAME the descriptors already spell for a DIFFERENT quantity.

    A helper's descriptors are written in the CALLER's vocabulary while its body speaks its own
    parameter names, and the two vocabularies can use one word twice. ``_maxpool3d(x, kernel_size,
    stride, ...)`` is called with the POOL window for ``kernel_size`` and receives an input
    declared ``(D - 1) * stride + 1 * (kernel_size - 1) + 1`` -- the CONV kernel and the CONV
    stride. C and Fortran evaluate that extent at the call site, where the caller's meaning is the
    only one in scope; a dace program turns it into a module-level ``dc.symbol`` the callee's
    parameter of the same name then shadows, so one symbol stands for two extents.

    Only a name bound to something other than the caller's own name for it qualifies -- a
    parameter handed the caller's identically-named symbol is the same quantity twice.
    """
    extents = {s.name for s in hkir.symbols} | {d.name for d in hkir.scalars}
    spelled = {ident for arr in hkir.arrays for dim in arr.shape for ident in IDENT_RE.findall(str(dim))}
    return sorted(
        pname
        for pname, arg in zip(abi, args)
        if pname in extents and pname in spelled and not (isinstance(arg, ast.Name) and arg.id == pname)
    )


def helper_call_bindings(owner: ast.FunctionDef, hkir: KernelIR, pinned: dict[str, PinnedValue]) -> HelperBinding:
    """What the helper's call site says about its symbols.

    * ALIASES ``{caller's name for an extent: the helper's own name for it}``. A helper's
      descriptors spell its extents in the CALLER's vocabulary, because the C and Fortran legs emit
      those extents as constants and the caller is where the constant is known. A dace program is
      shape-generic instead: ``_conv2d``'s body names ``n``, ``h``, ``w``, and a signature naming
      ``batch_size``, ``height``, ``width`` for the same dimensions hands the frontend two symbol
      sets it cannot prove equal -- "could not broadcast [batch_size, 3, height, width] into
      [n, 3, h, w]". Adopting the helper's own name makes each extent inferable from its argument,
      unless that name is one :func:`captured_parameter_names` reports, in which case the caller's
      spelling is kept and a helper left with no un-captured spelling is refused outright.
    * CONSTANTS ``{the helper's name: the pinned value}``, for a symbol the call binds to one of
      the kernel's pinned config knobs. Passed as a symbol it stays free while the callee is
      parsed, so ``(length + 2 * padding - kernel_size) // stride + 1`` never folds to ``length``
      and the write into a ``[n, c, length]`` out-param is refused. A pinned knob is a
      compile-time constant in the helper for the same reason it is one in the kernel.
    * EXPRESSIONS ``{the caller's expression for an extent: the helper's own name for it}``. A
      helper's descriptors spell an extent the way the CALLER computes it, and the helper has a
      parameter standing for that same quantity -- ``weight``'s ``out_channels //
      conv_transpose_groups`` against the ``c_out_per_group`` the call passes for it. dace cannot
      prove one equals the other, so a write into ``out[:, g * c_out_per_group : ...]`` is refused
      for a contribution the weight sized. Respelling the shape with the parameter makes dace
      SOLVE it from the argument instead.
    * COLLAPSE ``{a later helper name for an extent: the first}``, for the parameters a call site
      binds to ONE caller symbol. ``_conv2d(..., kh, kw)`` called with ``kernel_size`` twice has a
      square kernel at THIS call site; leaving the second name standing declares the return shape
      in terms of ``kh`` while the body still computes in ``kw``, which the frontend reads as two
      unequal extents -- "could not broadcast [.., -kw + w + 1] into [.., -kh + w + 1]". A SCALAR
      parameter collapses the same way, onto the symbol its argument already names.
    """
    sites = sum(
        1
        for node in ast.walk(owner)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == hkir.kernel_name
    )
    for node in ast.walk(owner):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == hkir.kernel_name):
            continue
        abi = hkir.abi_param_order()
        if node.keywords or len(node.args) != len(abi):
            return HelperBinding()
        own = {s.name for s in hkir.symbols}
        scalar_names = {d.name for d in hkir.scalars}
        captured = captured_parameter_names(hkir, abi, node.args)
        binding = HelperBinding(pinned=dict(pinned))
        bound_to: dict[str, list[str]] = {}
        for pname, arg in zip(abi, node.args):
            # A bare Name only: the helper's name stands for THIS extent, and an expression is not
            # one the helper has a name for.
            if pname not in own or not isinstance(arg, ast.Name):
                continue
            if arg.id in pinned:
                binding.constants[pname] = pinned[arg.id]
            elif arg.id in own:
                bound_to.setdefault(arg.id, []).append(pname)
        for caller_name, names in bound_to.items():
            # A CAPTURED name cannot be the one kept: the descriptors already spell the caller's
            # quantity with it, so keeping it would leave one symbol standing for two extents.
            usable = [pname for pname in names if pname not in captured] or names
            binding.aliases[caller_name] = usable[0]
            for pname in names:
                if pname != usable[0]:
                    binding.collapse[pname] = usable[0]
        # A SCALAR parameter handed the very caller symbol one of the helper's own symbols already
        # stands for is that symbol under a second name. ``_conv_transpose2d`` takes ``stride`` by
        # value while its return descriptor spells the same quantity ``conv_transpose_stride``, so
        # the body mints ``oh`` from a runtime scalar while the out-param is declared from a
        # symbol, and dace refuses the write between two extents it cannot relate. The symbol is
        # the canonical one -- a runtime scalar cannot size a dace descriptor at all.
        #
        # ONE call site, because respelling a SHAPE costs nothing at a second site (dace re-solves a
        # declared extent per call) while retiring a runtime scalar spends the value that site would
        # have passed. A symbol the descriptors state OUTRIGHT is no exception, though it was read
        # as one: the reading was that dace solves such a symbol from the argument while the body's
        # own name rides along as a keyword, so both are bound -- but BOUND IS NOT EQUAL, and
        # ``_scale`` declared ``[N]`` over a body allocating ``np.empty(n)`` had its closing write
        # refused ("could not broadcast input array from shape [n] into shape [N]").
        for pname, arg in zip(abi, node.args if sites == 1 else []):
            if pname not in scalar_names or not isinstance(arg, ast.Name):
                continue
            canonical = binding.aliases.get(arg.id, arg.id if arg.id in own else None)
            if canonical is not None and canonical != pname:
                binding.collapse[pname] = canonical
        for pname in captured:
            if pname in binding.collapse:
                continue  # respelled onto a name the descriptors do not already spell
            if pname in binding.constants and pinned.get(pname) == binding.constants[pname]:
                continue  # both spellings are the same pinned knob, so one value serves both
            if common_frontend.HELPERS_KEPT_DISABLED:
                # The inlined form is where a refusal would have LANDED, and this helper is still
                # here: it resisted inlining, so refusing again only loses the kernel. Emit what
                # the emitter emitted before the capture was recognised.
                continue
            raise NotImplementedError(
                f"helper {hkir.kernel_name!r} takes {pname!r}, which its own descriptors already "
                f"spell for the caller's {pname!r}; one dc.symbol cannot carry both extents, so "
                f"the helper must be inlined into its caller"
            )
        # Scalars too, not only symbols: an extent the CALL SITE computes arrives as an integer
        # scalar parameter (``c_out_per_group``) and only becomes a dc.symbol later, when
        # render_program sees it size an array.
        extents = own | scalar_names
        ambiguous: set[str] = set()
        for pname, arg in zip(abi, node.args):
            if pname not in extents or pname in binding.constants or pname in binding.collapse:
                continue
            # A bare Name is the caller's own local for the quantity, so its DEFINITION is the
            # expression a descriptor would have been written with.
            recipe = caller_side_recipe(owner, arg, pinned)
            if not recipe:
                continue
            # A recipe that folds to ONE name the helper already holds is that name: with
            # ``groups`` pinned to 1, ``c_out_per_group = out_channels // groups`` IS
            # ``out_channels``, and keeping both leaves the body writing ``c_out_per_group``
            # columns into a ``bias`` declared ``out_channels`` long.
            if IDENT_RE.fullmatch(recipe) and recipe in own and recipe != pname:
                binding.collapse[pname] = recipe
            elif recipe in binding.expressions:
                # TWO parameters computed the same way. Picking either respells an extent with a
                # name the body may not use for it -- ``_maxpool3d``'s input width ``w`` was
                # declared ``ow``, the POOLED width, because a pinned kernel size made the two
                # recipes fold alike. An ambiguous key answers for neither.
                ambiguous.add(recipe)
            else:
                binding.expressions[recipe] = pname
        for recipe in ambiguous:
            binding.expressions.pop(recipe, None)
        # An alias of a name to ITSELF retires the parameter it names; it is only here so a second
        # parameter bound to the same argument collapses onto it.
        binding.aliases = {caller: own_name for caller, own_name in binding.aliases.items() if caller != own_name}
        return binding
    return HelperBinding()


def transitive_rename(mapping: dict[str, str]) -> dict[str, str]:
    """``mapping`` with every value chased through the map to its FINAL name.

    Two independent facts about the same call site can each rename the same extent, one onto
    the other's result: resnet101's third ``_conv2d`` bottleneck conv has its scalar ``h``
    collapse onto the caller's own ``__inl3_oh2`` (a scalar bound to another symbol), while
    ``__inl3_oh2`` itself collapses onto ``sh1`` (its own recipe folding to a single name). A
    one-hop rename then leaves ``__inl3_oh2`` stranded in the body -- the very name the second
    fact just retired -- as a free variable no ``dc.symbol`` declares.
    """
    resolved: dict[str, str] = {}
    for name in mapping:
        target = mapping[name]
        seen = {name}
        while target in mapping and target not in seen:
            seen.add(target)
            target = mapping[target]
        resolved[name] = target
    return resolved


def with_helper_vocabulary(hkir: KernelIR, binding: HelperBinding) -> KernelIR:
    """``hkir`` with every aliased caller symbol respelled as the helper's own and then retired, and
    every call-pinned symbol recorded as one of the helper's own constants.

    A copy: the same KernelIR feeds the C and Fortran legs, where the caller's vocabulary is the
    correct one. Only the shapes move -- the body already speaks the helper's names, which is what
    made the two sets disagree in the first place. A COLLAPSED name is the exception: the body is
    the only place it stands, so the rename lands there.
    """
    if not (binding.aliases or binding.constants or binding.collapse or binding.expressions):
        return hkir
    respelling = transitive_rename({**binding.aliases, **binding.collapse})
    collapse = {name: respelling[name] for name in binding.collapse}

    def respell(token: str) -> str:
        # Whole-dimension first: a match on the caller's recipe replaces the extent outright, and
        # respelling its identifiers one at a time would leave a different expression behind. Only
        # a COMPOUND extent: a bare symbol is a name the helper already has, and rewriting it to a
        # parameter that happens to equal it at this call site renames the wrong axis --
        # ``x``'s own ``c_in`` became the group width ``c_in // groups`` reaches when groups is 1.
        if len(IDENT_RE.findall(str(token))) > 1 or not IDENT_RE.fullmatch(str(token).strip()):
            named = binding.expressions.get(folded_with_constants(str(token), binding.pinned))
            if named is not None:
                return named
        return IDENT_RE.sub(lambda m: respelling.get(m.group(), m.group()), str(token))

    tree = hkir.tree
    if binding.collapse:
        tree = copy.deepcopy(hkir.tree)
        tree.body = [RenameNames(collapse).visit(stmt) for stmt in tree.body]
    retired = {*binding.aliases, *binding.collapse}
    arrays = [dataclasses.replace(a, shape=tuple(respell(dim) for dim in a.shape)) for a in hkir.arrays]
    return dataclasses.replace(
        hkir,
        tree=tree,
        arrays=arrays,
        symbols=[s for s in hkir.symbols if s.name not in retired],
        scalars=[s for s in hkir.scalars if s.name not in retired],
        input_args=[n for n in hkir.input_args if n not in retired],
        pinned_consts={**hkir.pinned_consts, **binding.constants},
    )


def with_solvable_extents(hkir: KernelIR) -> KernelIR:
    """``hkir`` with every parameter extent dace cannot SOLVE replaced by one symbol of its own.

    dace binds a nested ``@dc.program`` by solving the callee's symbols from the shapes its call
    site passes, and a compound extent contributes ONE equation however many symbols it spells.
    ``sgs_apply``'s three CSR parameters are declared ``(3 * NX - 2) * (3 * NY - 2) * (3 * NZ - 2)``
    and ``NX * NY * NZ + 1`` over a body that names only ``N``: three equations for four unknowns,
    and sympy answered with a one-parameter family of quadratics -- "Ambiguous values for symbols
    in inference". A name the callee's body never reads decides nothing the callee computes, so the
    whole extent becomes one symbol and the system is square again.

    Only an extent naming something nothing else SUPPLIES is rewritten. An identifier the body
    reads, a pinned config knob, and an identifier some other parameter declares on its own (a
    bare ``H`` beside an ``H - 2``) are all determined already, and respelling those would retire a
    name the body or a sibling extent still needs.
    """
    body_names = {n.id for n in ast.walk(hkir.tree) if isinstance(n, ast.Name)}
    bare = {str(d).strip() for a in hkir.arrays for d in a.shape if IDENT_RE.fullmatch(str(d).strip())}
    supplied = body_names | bare | set(hkir.pinned_consts) | set(hkir.inlined_consts)
    taken = {a.name for a in hkir.arrays} | {s.name for s in hkir.scalars} | {s.name for s in hkir.symbols}
    taken |= body_names | supplied
    minted: dict[str, str] = {}
    arrays: list[ArrayDesc] = []
    for arr in hkir.arrays:
        dims: list[str] = []
        for dim in arr.shape:
            idents = IDENT_RE.findall(str(dim))
            if not idents or all(i in supplied for i in idents):
                dims.append(str(dim))
                continue
            key = fold_shape_expr(str(dim))
            name = minted.get(key)
            if name is None:
                name = f"{hkir.kernel_name}_extent{len(minted)}"
                while name in taken:
                    name = f"{name}_"
                minted[key] = name
                taken.add(name)
            dims.append(name)
        arrays.append(dataclasses.replace(arr, shape=tuple(dims)))
    if not minted:
        return hkir
    # A symbol the rewrite left in no shape is one dace can neither solve nor accept: passing a
    # keyword the callee does not take is a DaceSyntaxError, so retire it here.
    standing = {i for a in arrays for d in a.shape for i in IDENT_RE.findall(str(d))} | body_names
    return dataclasses.replace(hkir, arrays=arrays, symbols=[s for s in hkir.symbols if s.name in standing])


def inferred_symbols(rendered: RenderedProgram) -> set[str]:
    """The rendered program's symbols dace reads off an ARGUMENT's shape rather than being passed.

    Every identifier in a parameter annotation qualifies: the annotation is the callee's declared
    shape, and dace solves it against the shape the call site actually passes. Passing one of these
    explicitly is refused ("Invalid keyword argument"), so the two sets have to be exact.
    """
    named: set[str] = set()
    for param in rendered.params:
        annotation = param.partition(":")[2]
        named.update(IDENT_RE.findall(annotation))
    return {s for s in rendered.symbol_names if s in named}


def declared_extents(rendered: RenderedProgram) -> list[str]:
    """Every per-dimension extent expression in the program's parameter annotations."""
    out: list[str] = []
    for param in rendered.params:
        unused, unused, annotation = param.partition(":")
        opened = annotation.find("[")
        if opened < 0 or not annotation.rstrip().endswith("]"):
            continue  # a scalar parameter declares no extent
        out.extend(split_top_level(annotation[opened + 1 : annotation.rstrip().rfind("]")]))
    return out


def split_top_level(text: str) -> list[str]:
    """``text`` cut on the commas that are not inside brackets, which is one entry per dimension."""
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def extent_pins(extent: str, symbol: str) -> bool:
    """Whether ``extent`` determines ``symbol`` -- two values for it give two different extents.

    A symbol that cancels out (``(ci + 1) * 4 - ci * 4``, ``__sym_span - k + k``) appears in the
    annotation and constrains nothing, so dace has no equation to solve and reports the argument as
    missing. Folded rather than compared verbatim, because the cancellation is what has to be seen.
    """
    substituted = [IDENT_RE.sub(lambda m: value if m.group() == symbol else m.group(), extent) for value in ("1", "2")]
    return fold_shape_expr(substituted[0]) != fold_shape_expr(substituted[1])


def unsolvable_signature_symbols(rendered: RenderedProgram) -> list[str]:
    """The symbols in ``rendered``'s signature that dace cannot recover from the arguments.

    dace binds a nested program's shape symbols by matching each argument's real shape against the
    declared annotation, so a symbol reaches the callee only if some extent SOLVES for it. Solving
    is modelled the way substitution works: an extent pins a symbol once every other symbol in it
    is already pinned, iterated to a fixed point, so a helper whose extents form a solvable system
    is accepted and only a genuinely free or cancelling symbol is named here.
    """
    inferred = inferred_symbols(rendered)
    extents = declared_extents(rendered)
    idents = [(extent, {s for s in inferred if s in set(IDENT_RE.findall(extent))}) for extent in extents]
    pinned: set[str] = set()
    while True:
        found = {
            next(iter(unpinned))
            for extent, symbols in idents
            if len(unpinned := symbols - pinned) == 1 and extent_pins(extent, next(iter(unpinned)))
        }
        if not found - pinned:
            return sorted(inferred - pinned)
        pinned |= found


def symbolic_float_arguments(owner: ast.FunctionDef, hkir: KernelIR, symbols: set[str]) -> list[str]:
    """The call site's scalar arguments whose value is a float built from a shape symbol.

    dace types a nested call's scalar argument through its symbolic layer, and a float over a symbol
    (``1.0 / (0.016 / N / ...)``) arrives as a ``sympy.Float`` its dtype table has no entry for.
    """
    bound = {
        target.id: node.value
        for node in ast.walk(owner)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    passed = {
        arg.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == hkir.kernel_name
        for arg in node.args
        if isinstance(arg, ast.Name)
    }
    named: list[str] = []
    for arg in sorted(passed):
        value = bound.get(arg)
        if value is None:
            continue
        nodes = list(ast.walk(value))
        floats = any(isinstance(n, ast.Constant) and isinstance(n.value, float) for n in nodes)
        symbolic = any(isinstance(n, ast.Name) and n.id in symbols for n in nodes)
        if floats and symbolic:
            named.append(arg)
    return named


def refuse_unsound_callee(
    rendered: RenderedProgram, name: str, owner: ast.FunctionDef, hkir: KernelIR, symbols: set[str]
) -> None:
    """Raise unless ``rendered`` is a form dace can be shown to call correctly.

    THE GATE on the kept-helper form. A helper emitted as its own ``@dc.program`` is only sound
    when dace can bind its signature from the call site, and the emitter answers that POSITIVELY:
    what it cannot prove solvable is refused here, ``emit_with_inline_fallback`` re-renders the
    kernel with the helper inlined, and the frontend sees a form that has always worked. A blocklist
    of kernel names would go stale on the next kernel; this degrades for kernels nobody has seen.

    Three conditions, each read off the signature and the call site alone:

    * every symbol dace must solve from an argument shape is solvable -- see
      :func:`unsolvable_signature_symbols`;
    * every extent is integral, since an extent carrying a float literal reaches dace's symbolic
      layer as a ``sympy.Float`` and dies in a ``KeyError`` on its dtype table;
    * no scalar argument is a float built from a shape symbol, for the same dtype table -- see
      :func:`symbolic_float_arguments`.

    Over-refusing is SAFE here and under-refusing is not: a refusal costs the kept-helper form for
    one kernel and the inlined form is emitted instead, while a miss is a program the frontend
    rejects at parse time or, worse, one it accepts and computes wrongly.

    Only while a fallback REMAINS. Inlining does not dissolve every helper -- one whose form has no
    inlinable shape stays a program of its own under ``without_kept_helpers`` -- so refusing on the
    retry too would answer with no program at all, which costs the kernel its DaCe column entirely
    rather than degrading it. On the retry the best-effort form is emitted and the frontend gives
    the verdict.

    What the gate does NOT prove, and what a repair of this path has to fix, measured per kernel
    against the corpus:

    * the OUT-PARAM extent can disagree with what the body stores into it, because the return
      classification reads a shape that the specialised body then contradicts
      (``cp2k_density_matrix_trs4``: declared ``[n_block_rows + 1]``, body writes ``[n_block_rows]``;
      ``lenet``'s ``maxpool2d`` declares ``int_floor(H - 4, 2)`` and the body writes ``H_out``).
      Catching it needs shape inference over the body, which the emitter does not have;
    * a helper's extents can be spelled in the CALLER's vocabulary where no single call-site name
      recovers the helper's own (``mamba2_return_y``: ``(batch_size, n_heads, n_chunks + 1,
      n_chunks + 1)`` against a body naming ``span``);
    * ``gromacs/nbnxm``'s ``_inner_4x4`` loses ``ci`` -- it appears only as ``(ci + 1) * 4 -
      ci * 4``, which this gate catches, but the argument it should have been is a real omission;
    * ``conv_standard_1d_dilated_strided`` reaches the frontend with a two-argument ``np.equal``
      that has no dace replacement, which is a lowering gap and not a signature one;
    * ``matmul_avg_pool_gelu_scale_max``'s ``_avgpool1d_taps`` takes its ``kernel_size`` and
      ``stride`` as runtime ``dc.int64`` scalars and then SIZES a tap span with them, so the extent
      is data-dependent inside the body while the signature itself is solvable.
    """
    if common_frontend.HELPERS_KEPT_DISABLED:
        return
    unsolvable = unsolvable_signature_symbols(rendered)
    if unsolvable:
        raise NotImplementedError(
            f"program {name!r} declares {unsolvable}, which no argument's shape solves for; dace "
            f"cannot bind them at the call site"
        )
    for extent in declared_extents(rendered):
        if FLOAT_LITERAL_RE.search(extent):
            raise NotImplementedError(
                f"program {name!r} declares the extent {extent!r}, which is not integral; a float "
                f"extent reaches dace's symbolic layer as a sympy.Float"
            )
    symbolic_floats = symbolic_float_arguments(owner, hkir, symbols)
    if symbolic_floats:
        raise NotImplementedError(
            f"program {name!r} is called with {symbolic_floats}, each a float built from a shape "
            f"symbol; dace has no dtype for the sympy.Float that reaches it"
        )


def bind_helper_call(node: ast.Call, hkir: KernelIR, rendered: RenderedProgram) -> None:
    """Rewrite one call to a kept helper onto the signature :func:`render_program` gave it.

    The call arrives in ABI order -- references then scalars, each sorted by name -- which is the
    order the C and Fortran legs emit and has nothing to do with the dace program's parameter
    order. Rebuild it by NAME: the rendered parameters positionally in their own order, the
    body-only symbols as keywords, and the shape-inferred symbols dropped.
    """
    abi = hkir.abi_param_order()
    if node.keywords or len(node.args) != len(abi):
        raise ValueError(f"call to {hkir.kernel_name!r} passes {len(node.args)} arguments for the ABI order {abi}")
    arg_of = dict(zip(abi, node.args))
    emitted_from = {emitted: original for original, emitted in rendered.renames.items()}
    inferred = inferred_symbols(rendered)
    args: list[ast.expr] = []
    for param in rendered.params:
        pname = param.split(":", 1)[0].strip()
        original = emitted_from.get(pname, pname)
        if original not in arg_of:
            raise ValueError(f"{hkir.kernel_name!r} takes {pname!r}, which its call site does not pass")
        args.append(arg_of[original])
    # A symbol the emitter MINTED (a hoisted compound extent) has no argument slot; its recipe
    # names the helper's own parameters, so the call site's arguments spell it. Recipes are in
    # dependency order, so each is resolved against the ones already bound.
    bound: dict[str, ast.expr] = dict(arg_of)
    for sym, recipe in rendered.symbol_defs:
        bound.setdefault(sym, SubstituteNames(bound).visit(ast.parse(recipe, mode="eval")).body)
    keywords: list[ast.keyword] = []
    for sym in rendered.symbol_names:
        if sym in inferred:
            continue  # dace solves it from the argument shape; passing it too is an error there
        original = emitted_from.get(sym, sym)
        if original not in bound:
            raise ValueError(
                f"{hkir.kernel_name!r} needs {sym!r}, which appears in no parameter shape and which "
                f"its call site does not pass; dace cannot bind it"
            )
        keywords.append(ast.keyword(arg=sym, value=bound[original]))
    node.args = args
    node.keywords = keywords


def bind_helper_calls(
    body: list[ast.stmt], rendered_by_name: dict[str, RenderedProgram], kir_by_name: dict[str, KernelIR]
) -> None:
    """Rewrite every call in ``body`` that names one of the rendered helpers."""
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in rendered_by_name:
                bind_helper_call(node, kir_by_name[node.func.id], rendered_by_name[node.func.id])


def returns_removed(stmts: list[ast.stmt], tail: list[ast.stmt], name: str) -> list[ast.stmt]:
    """``stmts`` followed by ``tail``, with every VALUELESS ``return`` gone.

    A branch that returns drops ``tail``, which is exactly what the return said; a branch that
    falls through carries it, so the statements after an escaping ``if`` move into both of its arms
    rather than staying where the return would have skipped them. See :func:`body_without_returns`.

    A return that carries a VALUE stays: a rank-0 helper is emitted as a by-value ``@dc.program``
    and dace binds its result at the call site, so the return is the helper's answer rather than
    the out-param form's terminator. The statements after it are dead in python too, and go.
    """
    out: list[ast.stmt] = []
    for index, stmt in enumerate(stmts):
        if isinstance(stmt, ast.Return):
            if stmt.value is not None:
                out.append(stmt)
            return out
        if not any(isinstance(node, ast.Return) for node in ast.walk(stmt)):
            out.append(stmt)
            continue
        if not isinstance(stmt, ast.If):
            raise NotImplementedError(
                f"program {name!r} returns from inside a {type(stmt).__name__.lower()}, which has no "
                f"fall-through arm to carry the statements after it"
            )
        rest = returns_removed(stmts[index + 1 :], tail, name)
        body = returns_removed(stmt.body, rest, name)
        orelse = returns_removed(stmt.orelse, copy.deepcopy(rest), name)
        out.append(rebuilt_if(stmt, body, orelse))
        return out
    return out + tail


def rebuilt_if(stmt: ast.If, body: list[ast.stmt], orelse: list[ast.stmt]) -> ast.If:
    """``stmt`` with new branches, located where the original was."""
    node = ast.If(test=stmt.test, body=body or [ast.Pass()], orelse=orelse)
    ast.copy_location(node, stmt)
    return ast.fix_missing_locations(node)


def without_returns(hkir: KernelIR, name: str) -> KernelIR:
    """``hkir`` with no VALUELESS ``return`` left in its body, for a program dace calls as a callee.

    A ``return`` inside a nested ``@dc.program`` returns from the CALLER: dace splices the callee's
    ``ReturnBlock`` into the caller's own control flow, so every statement after the CALL is
    unreachable and its outputs keep whatever the driver allocated, with nothing raised. A return
    in tail position of the whole body is dead and goes; an earlier one becomes the branch it
    already was, with the statements it skipped moved under the arm that reaches them. A by-value
    return is the helper's ANSWER and stays -- see :func:`returns_removed`.

    A copy: the same KernelIR feeds the C and Fortran legs, where a return is a return.
    """
    tree = copy.deepcopy(hkir.tree)
    tree.body = returns_removed(tree.body, [], name)
    return dataclasses.replace(hkir, tree=tree)


def render_helper_closure(kir: KernelIR, main: RenderedProgram) -> list[tuple[KernelIR, RenderedProgram]]:
    """Render every kept helper the kernel reaches, callees included, in definition-before-use order.

    A helper is free to call a SIBLING, so the set is closed by walking what each rendered body
    still calls. Python binds the name at call time, but emitting a callee first keeps the module
    readable and matches what the C leg's prototypes buy there.
    """
    by_name = {h.kernel_name: h for h in kir.helpers}
    ordered: list[tuple[KernelIR, RenderedProgram]] = []
    done: OrderedSet[str] = OrderedSet()

    def visit(owner: ast.FunctionDef, name: str) -> None:
        if name in done:
            return
        done.add(name)
        hkir = by_name[name]
        binding = helper_call_bindings(owner, hkir, kir.pinned_consts or {})
        # Vocabulary first: a caller recipe that names one of the helper's own parameters is a
        # better spelling for an extent than a symbol minted for it.
        settled = with_solvable_extents(with_helper_vocabulary(hkir, binding))
        # Returns go before render_program, not after: the duplication this can make of the
        # statements past a branch rebinds a view, and the passes that version such a rebinding run
        # in there.
        rendered = render_program(without_returns(settled, name), name, kir.helpers, nested=True)
        refuse_unsound_callee(rendered, name, owner, hkir, {sy.name for sy in kir.symbols})
        for callee in called_helpers(rendered.body, kir.helpers):
            visit(hkir.tree, callee)
        ordered.append((hkir, rendered))

    for name in called_helpers(main.body, kir.helpers):
        visit(kir.tree, name)
    return ordered


def substituted_extent(token: object, substitutions: dict[str, str]) -> str:
    """``token`` with every substituted symbol spelled as the expression its call site binds it to."""
    if not substitutions:
        return str(token)
    return IDENT_RE.sub(
        lambda m: f"({substitutions[m.group()]})" if m.group() in substitutions else m.group(), str(token)
    )


def allocation_shapes(body: list[ast.stmt]) -> dict[str, list[str]]:
    """``{local: its allocation's per-axis extents}`` for every ``<local> = np.empty(...)`` in ``body``."""
    found: dict[str, list[str]] = {}
    for stmt in body:
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
            continue
        shape = shape_argument(stmt.value)
        if shape is None:
            continue
        elements = shape.elts if isinstance(shape, (ast.Tuple, ast.List)) else [shape]
        found[stmt.targets[0].id] = [ast.unparse(element) for element in elements]
    return found


def full_slice_target(target: ast.expr) -> str | None:
    """The array name a ``<name>[:] = ...`` statement writes WHOLE, or ``None`` for anything else."""
    if isinstance(target, ast.Name):
        return target.id
    if not (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)):
        return None
    element = target.slice
    if isinstance(element, ast.Slice) and element.lower is None and element.upper is None and element.step is None:
        return target.value.id
    return None


def output_write_extents(
    main: RenderedProgram, declared: dict[str, tuple[str, ...]], pinned: dict[str, PinnedValue] | None = None
) -> list[tuple[str, str, list[str], list[str]]]:
    """``(array, workspace, declared extents, workspace extents)`` per whole-array copy in ``main``.

    One entry per ``<declared array>[:] = <workspace>`` the kernel performs, with the pinned knobs
    already substituted into both sides. The two extent lists are the same quantity spelled by the
    MANIFEST and by the reference's own body, which is what makes them comparable at all.
    """
    literals = {name: str(value) for name, value in (pinned or {}).items()}
    allocated = allocation_shapes(main.body)
    pairs: list[tuple[str, str, list[str], list[str]]] = []
    for stmt in main.body:
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.value, ast.Name)):
            continue
        name = full_slice_target(stmt.targets[0])
        source = allocated.get(stmt.value.id)
        if name is None or source is None or name not in declared:
            continue
        target = [substituted_extent(token, literals) for token in declared[name]]
        source = [substituted_extent(token, literals) for token in source]
        if len(target) != len(source):
            continue  # a rank difference is a broadcast the frontend decides, not a spelling
        pairs.append((name, stmt.value.id, target, source))
    return pairs


def emit_dace(kir: KernelIR, fn_name: str | None = None) -> str:
    """Return the source of a ``<short>_dace.py`` module for ``kir``.

    A kept helper is emitted as its own ``@dc.program`` above the kernel, not inlined into it.
    dace's frontend BINDS a nested program call and rebinds the callee's shape symbols per call
    site, so one shape-generic helper serves call sites of different extents -- which is the whole
    reason the un-inlined form exists. Inlining would specialize the helper to one call site's
    shapes and recopy its body once per call.

    A symbol of the callee's that appears in a parameter's declared SHAPE is inferred from the
    argument and must not be passed; one that appears only in the callee's BODY is a required
    argument, passed BY KEYWORD (positionally, dace's ``closure_resolver`` indexes its
    parameter-name list with the argument's position and raises ``IndexError``).
    """
    main = render_program(kir, fn_name or kir.kernel_name, kir.helpers)
    helpers = render_helper_closure(kir, main)
    rendered_by_name = {h.kernel_name: r for h, r in helpers}
    kir_by_name = {h.kernel_name: h for h, unused in helpers}
    for unused, rendered in [*helpers, (None, main)]:
        bind_helper_calls(rendered.body, rendered_by_name, kir_by_name)
    programs = [r for unused, r in helpers] + [main]

    out: list[str] = []
    out.append('"""DaCe program auto-generated from the numpy reference by numpyto_c.dace_emit."""')
    out.append("import numpy as np")
    out.append("import dace as dc")
    imp = "dc_float, dc_complex_float" if any(r.needs_complex for r in programs) else "dc_float"
    out.append(f"from hpcagent_bench.frameworks.dace_framework import {imp}")
    # BOTH spellings: the lowering emits bare `sqrt(x)` for a desugared numpy ufunc and keeps a
    # QUALIFIED `math.sqrt(x)` the reference wrote by hand, and the name-import alone makes the
    # second one a DaceSyntaxError ('Use of undefined variable "math"').
    out.append("import math")
    out.append("from math import sin, cos, log, exp, pow, sqrt")
    out.append("")
    # Pooled across the programs: a declaration is a module-level fact, and a helper shares the
    # kernel's spelling for a quantity they both take.
    pinned: dict[str, PinnedValue] = {}
    for rendered in programs:
        pinned.update(rendered.pinned)
    for const_name, const_value in pinned.items():
        # Module scope, which the dace frontend reads as a compile-time constant, so the body keeps
        # the manifest's spelling instead of an inlined literal.
        out.append(f"{const_name} = {const_value!r}")
    if pinned:
        out.append("")
    symbol_names: list[str] = []
    for rendered in programs:
        symbol_names.extend(n for n in rendered.symbol_names if n not in symbol_names)
    if symbol_names:
        # One declaration per symbol: dtype and sign are per-symbol facts the generator spelling
        # this replaced could carry neither of. The assumption reaches sympy through
        # dace.symbol's ``**assumptions`` and decides comparisons the solver would otherwise
        # leave symbolic -- so only what is proven is declared, never what is merely likely.
        desc_of = {s.name: s for s in kir.symbols}
        for helper, unused in helpers:
            desc_of.update({s.name: s for s in helper.symbols if s.name not in desc_of})
        # Re-derived rather than read off the descriptor: the promotions above append emit-local
        # names no :func:`stamp_symbol_assumptions` pass has seen.
        dims = shape_dimension_symbols([*kir.arrays, *(a for h, unused in helpers for a in h.arrays)])
        signs = dict(kir.symbol_signs)
        for helper, unused in helpers:
            signs.update({n: v for n, v in helper.symbol_signs.items() if n not in signs})
        for sym_name in symbol_names:
            desc = desc_of.get(sym_name)
            dtype = dace_dtype(desc.dtype) if desc else "dc.int64"
            # A promoted scalar has no descriptor; its sign comes from the manifest binding
            # the frontend carried over (``conv_padding: 0`` -> nonnegative).
            sign = "positive" if sym_name in dims else (desc.assumption if desc else signs.get(sym_name, ""))
            assumption = f", {sign}=True" if sign else ""
            out.append(f"{sym_name} = dc.symbol('{sym_name}', dtype={dtype}{assumption})")
        out.append("")
    if main.symbol_defs:
        # Per-dimension binding recipe: caller evaluates these in order at call time. See
        # sparse_oracle._run_dace. The KERNEL's alone -- a helper's extents come from the shapes
        # its call site passes, which dace resolves without the caller knowing they exist.
        out.append(f"__hpcagent_bench_symbol_defs__ = {main.symbol_defs!r}")
        out.append("")
    if main.renames:
        # ``{manifest name: emitted name}``. See dace_framework.call_args, the one place that
        # applies it -- everything downstream of there already speaks the emitted spelling.
        out.append(f"__hpcagent_bench_renames__ = {main.renames!r}")
        out.append("")
    # Which of the module's programs is the KERNEL. Kept helpers are @dc.programs too, so a reader
    # can no longer take the sole one, and the name matches neither the file stem (lenet ->
    # lenet5) nor a fixed word (nussinov -> kernel).
    out.append(f"__hpcagent_bench_program__ = {main.name!r}")
    out.append("")
    for rendered in programs:
        out.append("")
        out.append("@dc.program")
        out.append(f"def {rendered.name}({', '.join(rendered.params)}):")
        if not rendered.body:
            out.append("    pass")
        else:
            for stmt in rendered.body:
                for line in ast.unparse(stmt).splitlines():
                    out.append("    " + line)
    return "\n".join(out) + "\n"
