# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scalarize compile-time tuples so no tuple value ever reaches an emitter.

C has no tuple, Fortran has no tuple, and none of these tuples needs one: the ports build them
from source literals (``stride = (stride, stride)``), fix their length in the source, and index
them with constants. The whole layer is compile-time, so it folds away rather than lowering to a
struct / derived type / ``std::tuple`` -- a runtime aggregate would be an emitter feature and an
ABI question in exchange for nothing the kernel actually computes with.

What folds, in one forward pass that interprets the tuple-valued names:

* ``t = (a, b)``, ``t = u + v``, ``t = tuple(<expr> for i in range(K))`` -- bound, not emitted;
* ``t[K]`` -> the K-th element expression, ``t[i:j]`` -> a shorter tuple, ``len(t)`` -> a constant;
* ``x.shape`` (rank known) -> ``(x.shape[0], ..., x.shape[r-1])``, so ``x.shape[2:]`` has a
  compile-time LENGTH while its extents stay symbolic -- that is what makes the group-norm idiom
  ``x.reshape((n, g, c // g) + x.shape[2:])`` a plain rank-3 reshape;
* ``isinstance(x, T)`` -> a constant, from the kind the pass already knows for ``x``;
* ``x is None`` -> a constant once ``x`` is known bound;
* a static ``if`` / ``IfExp`` -> its live branch, which is what exposes the tuple binding beneath;
* ``slice(a, b)`` inside a subscript index -> a real ``ast.Slice``.

Anything undecidable is left exactly as it was: this pass narrows, it never guesses.

Entry point: :func:`desugar_tuples`.
"""
import ast
import copy
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from numpyto_common.numpy_desugar import expr_rank
from numpyto_common.ordered import OrderedSet

#: Module aliases a kernel may spell numpy as.
NUMPY_MODULES = frozenset({"np", "numpy"})

#: Type name as written in an ``isinstance`` -> the concrete runtime types it accepts. Provenance is
#: tracked because ``isinstance(np.int64(3), int)`` is FALSE: a fold that ignored that would emit a
#: kernel taking a branch the numpy oracle never takes. ``int`` accepts a bool since Python's bool
#: subclasses int. ``float`` deliberately does NOT list ``np_float`` even though np.float64 happens
#: to subclass float -- np.float32 does not, and this pass cannot tell them apart.
TYPE_ACCEPTS: Dict[str, Tuple[str, ...]] = {
    "int": ("py_int", "py_bool"),
    "bool": ("py_bool", ),
    "float": ("py_float", ),
    "complex": ("py_complex", ),
    "str": ("py_str", ),
    "tuple": ("tuple", ),
    "list": ("list", ),
    "integer": ("np_int", ),
    "signedinteger": ("np_int", ),
    "unsignedinteger": ("np_int", ),
    "floating": ("np_float", ),
    "complexfloating": ("np_complex", ),
    "bool_": ("np_bool", ),
    "number": ("np_int", "np_float", "np_complex"),
    "ndarray": ("ndarray", ),
}

#: Value kind -> the concrete runtime types a value of that kind may actually have. A DECLARED
#: scalar parameter is ``integer``/``floating``: the harness hands a preset symbol over as a Python
#: int but an ``init.scalars`` entry as a numpy scalar, and nothing here can tell which. That is why
#: a bare ``isinstance(knob, int)`` stays unfolded while ``isinstance(knob, (int, np.integer))``
#: folds -- the second is true either way, the first is not.
KIND_TYPES: Dict[str, Tuple[str, ...]] = {
    "int": ("py_int", ),
    "float": ("py_float", ),
    "complex": ("py_complex", ),
    "bool": ("py_bool", ),
    "str": ("py_str", ),
    "integer": ("py_int", "np_int"),
    "floating": ("py_float", "np_float"),
    "np_int": ("np_int", ),
    "np_float": ("np_float", ),
    "tuple": ("tuple", ),
    "list": ("list", ),
    "array": ("ndarray", ),
    "none": ("none", ),
}

#: Builtin scalar casts -> the kind of their result. ``int(x)`` yields a genuine Python int, which is
#: what makes the inliner's materialised ``__inlN_stride = int(stride)`` decidable.
CAST_KINDS: Dict[str, str] = {"int": "int", "float": "float", "complex": "complex", "bool": "bool"}

#: numpy scalar constructors -> the kind of their result.
NUMPY_CAST_KINDS: Dict[str, str] = {
    **{
        f"int{w}": "np_int"
        for w in (8, 16, 32, 64)
    },
    **{
        f"uint{w}": "np_int"
        for w in (8, 16, 32, 64)
    },
    **{
        f"float{w}": "np_float"
        for w in (16, 32, 64)
    },
    "bool_": "bool",
}


def decide_isinstance(kind: str, names: List[str]) -> Optional[bool]:
    """``isinstance(<kind>, (names...))``, or ``None`` when the value's provenance leaves it open."""
    accepted = {t for name in names for t in TYPE_ACCEPTS[name]}
    hits = {t in accepted for t in KIND_TYPES[kind]}
    return hits.pop() if len(hits) == 1 else None


class Env:
    """Names bound to a compile-time tuple, plus the names known to be bound at all (so
    ``x is None`` decides). Copied per branch; never shared across a loop body."""

    def __init__(self,
                 tuples: Optional[Dict[str, List[ast.expr]]] = None,
                 bound: Optional[Set[str]] = None,
                 kinds: Optional[Dict[str, str]] = None) -> None:
        self.tuples: Dict[str, List[ast.expr]] = dict(tuples or {})
        self.bound: Set[str] = set(bound or ())
        #: Local -> value kind, so the inliner's ``__inlN_stride = int(stride)`` stays decidable.
        self.kinds: Dict[str, str] = dict(kinds or {})

    def copy(self) -> "Env":
        return Env(self.tuples, self.bound, self.kinds)

    def invalidate(self, names) -> None:
        for name in names:
            self.tuples.pop(name, None)
            self.kinds.pop(name, None)
            self.bound.discard(name)


def is_index_element(node: ast.AST) -> bool:
    """An element that can only be part of a SUBSCRIPT index: a slice, ``None``, or an ellipsis."""
    if isinstance(node, ast.Slice):
        return True
    if isinstance(node, ast.Constant):
        return node.value is None or node.value is Ellipsis
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "slice"


def _target_names(target: ast.AST) -> OrderedSet:
    """The name(s) one assignment TARGET actually binds -- a bare Name, every element of a
    Tuple/List/Starred pattern, or (for a Subscript/Attribute LValue) the base object being
    written THROUGH, e.g. ``a`` in ``a[i] = x``.

    Does NOT recurse into a Subscript's INDEX or an Attribute's name: those are read, not bound.
    ``out[lo:hi:stride[0]] += tap`` (a strided-slice tap loop) walks the whole target with a bare
    ``ast.walk`` and used to pick up ``stride`` from inside the slice STEP as if this statement
    rebound it -- which then made ``assigned_names`` invalidate ``stride``'s compile-time tuple
    tracking (see below) on every loop this statement sits in, so a stride that had already folded
    to a literal outside the loop reverted to an un-foldable ``stride[0]`` the moment the SAME
    variable was also used as a slice step somewhere inside it."""
    if isinstance(target, ast.Name):
        return OrderedSet((target.id, ))
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        names = OrderedSet()
        for elt in target.elts:
            names |= _target_names(elt)
        return names
    if isinstance(target, (ast.Subscript, ast.Attribute)):
        return _target_names(target.value)
    return OrderedSet()


def assigned_names(node: ast.AST) -> OrderedSet:
    """Every name the subtree can rebind -- the conservative kill set for a branch or loop body."""
    out = OrderedSet()
    for sub in ast.walk(node):
        targets: List[ast.AST] = []
        if isinstance(sub, ast.Assign):
            targets = list(sub.targets)
        elif isinstance(sub, (ast.AugAssign, ast.AnnAssign)):
            targets = [sub.target]
        elif isinstance(sub, ast.For):
            targets = [sub.target]
        for tgt in targets:
            out |= _target_names(tgt)
    return out


def const_value(node: ast.AST) -> Any:
    """The literal a node denotes, or :data:`NO_VALUE` when it is not a literal."""
    return node.value if isinstance(node, ast.Constant) else NO_VALUE


#: Distinct from ``None``, which is itself a foldable literal.
NO_VALUE = object()


def const_int(node: ast.AST) -> Optional[int]:
    v = const_value(node)
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def static_bool(node: ast.AST) -> Optional[bool]:
    v = const_value(node)
    return None if v is NO_VALUE else bool(v)


def type_names(node: ast.AST) -> Optional[List[str]]:
    """The bare type names an ``isinstance`` second argument denotes (``int``, ``np.integer``,
    ``(int, np.integer)``), or ``None`` when it names something this pass does not model."""
    if isinstance(node, ast.Tuple):
        parts = [type_names(e) for e in node.elts]
        return None if any(p is None for p in parts) else [n for p in parts for n in p]
    if isinstance(node, ast.Name):
        return [node.id] if node.id in TYPE_ACCEPTS else None
    if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in NUMPY_MODULES
            and node.attr in TYPE_ACCEPTS):
        return [node.attr]
    return None


class TupleDesugar:
    """The forward interpreter. ``int_scalars`` / ``float_scalars`` / ``arrays`` are the kernel's
    declared parameters, which is what makes an ``isinstance`` guard on a knob decidable."""

    def __init__(self,
                 int_scalars: FrozenSet[str] = frozenset(),
                 float_scalars: FrozenSet[str] = frozenset(),
                 arrays: FrozenSet[str] = frozenset(),
                 ranks: Optional[Dict[str, int]] = None) -> None:
        self.int_scalars = frozenset(int_scalars)
        self.float_scalars = frozenset(float_scalars)
        self.arrays = frozenset(arrays)
        #: Declared array -> rank, so ``x.ndim`` is a literal and the broadcast shapes built from it
        #: (``(1,) * (x.ndim - 2)``) have a compile-time length.
        self.ranks: Dict[str, int] = dict(ranks or {})
        #: Counter for the locals :meth:`capture_self_reference` mints.
        self.aliases = 0

    # -- kinds ------------------------------------------------------------- #
    def kind(self, node: ast.AST, env: Env) -> Optional[str]:
        """The value kind of an expression, or ``None`` when this pass cannot tell."""
        if self.tuple_of(node, env) is not None:
            return "tuple"
        if isinstance(node, ast.List):
            return "list"
        if isinstance(node, ast.Constant):
            v = node.value
            if v is None:
                return "none"
            for py_type, name in ((bool, "bool"), (int, "int"), (float, "float"), (complex, "complex"), (str, "str")):
                if isinstance(v, py_type):
                    return name
            return None
        if isinstance(node, ast.Call):
            return self.cast_kind(node.func) if len(node.args) == 1 else None
        if isinstance(node, ast.Name):
            if node.id in env.kinds:
                return env.kinds[node.id]
            if node.id in self.arrays:
                return "array"
            if node.id in self.int_scalars:
                return "integer"
            if node.id in self.float_scalars:
                return "floating"
        return None

    def cast_kind(self, func: ast.AST) -> Optional[str]:
        """The kind produced by a scalar cast (``int(x)``, ``np.float32(x)``)."""
        if isinstance(func, ast.Name):
            return CAST_KINDS.get(func.id)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in NUMPY_MODULES:
            return NUMPY_CAST_KINDS.get(func.attr) or CAST_KINDS.get(func.attr)
        return None

    # -- tuple recognition -------------------------------------------------- #
    def tuple_of(self, node: ast.AST, env: Env) -> Optional[List[ast.expr]]:
        """The element expressions of a compile-time tuple, or ``None``."""
        if isinstance(node, ast.Tuple):
            return list(node.elts)
        if isinstance(node, ast.List) and node.elts and all(is_index_element(e) for e in node.elts):
            # An INDEX list only (``[slice(None)] * x.ndim``, built to be mutated then tupled). A
            # data list is left alone: it may be feeding ``np.array([...])``, which wants the List.
            return list(node.elts)
        if isinstance(node, ast.Name) and node.id in env.tuples:
            return list(env.tuples[node.id])
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = self.tuple_of(node.left, env), self.tuple_of(node.right, env)
            return None if left is None or right is None else left + right
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            # `is not None`, never `or`: a 0-repeat is the empty tuple, which is falsy but correct.
            repeated = self.repeat(node.left, node.right, env)
            return repeated if repeated is not None else self.repeat(node.right, node.left, env)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("tuple", "list") and len(
                node.args) == 1 and not node.keywords:
            inner = self.tuple_of(node.args[0], env)
            if inner is not None:
                return inner
            unrolled = self.unroll(node.args[0], env)
            if unrolled is not None:
                return unrolled
            # ``tuple(range(2, x.ndim))`` -- the ports spell "every axis from here on" this way.
            bounds = self.range_bounds(node.args[0], env)
            return None if bounds is None else [ast.Constant(value=i) for i in range(*bounds)]
        # A bare comprehension: ``[<expr> for i in range(K)]`` used directly (not wrapped in
        # ``tuple(...)``/``list(...)``), same trip-count requirement as the wrapped form.
        if isinstance(node, (ast.GeneratorExp, ast.ListComp)):
            return self.unroll(node, env)
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            sliced = self.slice_of(node, env)
            if sliced is not None:
                return sliced
        return self.shape_tuple(node)

    def slice_of(self, node: ast.Subscript, env: Env) -> Optional[List[ast.expr]]:
        """``x.shape[2:]`` -- a SLICE of a compile-time tuple is still one.

        The ports spell "the trailing axes" this way, and group-norm's
        ``x.reshape((n, g, c // g) + x.shape[2:])`` is the live case: without this the concat could
        not collapse, so ``y`` had no compile-time rank and the next line's ``y.ndim`` never folded
        either -- which surfaces much later as "axis must be a compile-time integer".

        Every present bound must fold to an int; the length is what has to be compile-time, not the
        extents, which stay symbolic.
        """
        elements = self.tuple_of(node.value, env)
        if elements is None:
            return None
        bounds: List[Optional[int]] = []
        for part in (node.slice.lower, node.slice.upper, node.slice.step):
            if part is None:
                bounds.append(None)
                continue
            folded = const_int(self.fold(copy.deepcopy(part), env))
            if folded is None:
                return None
            bounds.append(folded)
        return [copy.deepcopy(e) for e in elements[slice(*bounds)]]

    def shape_tuple(self, node: ast.AST) -> Optional[List[ast.expr]]:
        """``x.shape`` -> per-axis ``x.shape[i]``, once the rank is known. Length compile-time,
        extents still symbolic -- exactly what a reshape needs and a tuple concat can consume."""
        if not (isinstance(node, ast.Attribute) and node.attr == "shape"):
            return None
        rank = self.rank(node.value)
        if rank is None:
            return None
        return [
            ast.Subscript(value=copy.deepcopy(node), slice=ast.Constant(value=i), ctx=ast.Load()) for i in range(rank)
        ]

    def repeat(self, seq: ast.AST, count: ast.AST, env: Env) -> Optional[List[ast.expr]]:
        """``(1,) * K`` -> K copies. The ports pad a broadcast shape out to an array's rank this way."""
        elts = self.tuple_of(seq, env)
        times = const_int(count)
        return None if elts is None or times is None or times < 0 else [copy.deepcopy(e) for e in elts] * times

    def rank(self, node: ast.AST) -> Optional[int]:
        """The rank of an array expression. Shared with the rest of the pipeline so that
        ``np.expand_dims(x, axis=1).shape`` resolves by the same rules the emitter uses."""
        return expr_rank(node, self.ranks)

    def unroll(self, node: ast.AST, env: Env) -> Optional[List[ast.expr]]:
        """``<expr> for i in range(K)`` -> the K substituted element expressions. The trip count must
        be a literal; a symbolic one would need a runtime tuple, which is the thing we are removing."""
        if not isinstance(node, (ast.GeneratorExp, ast.ListComp)) or len(node.generators) != 1:
            return None
        gen = node.generators[0]
        if gen.ifs or gen.is_async or not isinstance(gen.target, ast.Name):
            return None
        bounds = self.range_bounds(gen.iter, env)
        if bounds is None:
            return None
        lo, hi, step = bounds
        return [
            self.fold(_substitute(copy.deepcopy(node.elt), gen.target.id, ast.Constant(value=i)), env)
            for i in range(lo, hi, step)
        ]

    def range_bounds(self, node: ast.AST, env: Env) -> Optional[Tuple[int, int, int]]:
        """``range(...)`` with literal arguments, as ``(start, stop, step)``."""
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range" and node.args
                and not node.keywords):
            return None
        args = [const_int(self.fold(copy.deepcopy(a), env)) for a in node.args]
        if any(a is None for a in args) or len(args) > 3:
            return None
        if len(args) == 1:
            return 0, args[0], 1
        return args[0], args[1], (args[2] if len(args) == 3 else 1)

    # -- expression folding ------------------------------------------------- #
    def fold(self, node: ast.expr, env: Env) -> ast.expr:
        return _Folder(self, env).visit(node)

    # -- statements --------------------------------------------------------- #
    def run(self, stmts: List[ast.stmt], env: Env, linear: bool = True) -> List[ast.stmt]:
        out: List[ast.stmt] = []
        for stmt in stmts:
            out.extend(self.statement(stmt, env, linear))
        return out

    def statement(self, stmt: ast.stmt, env: Env, linear: bool) -> List[ast.stmt]:
        if isinstance(stmt, ast.If):
            return self.branch(stmt, env, linear)
        if isinstance(stmt, (ast.For, ast.While)):
            return self.loop(stmt, env)
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            return self.assign(stmt, env, linear)
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0],
                                                                                  (ast.Tuple, ast.List)):
            unpacked = self.unpack(stmt, env, linear)
            if unpacked is not None:
                return unpacked
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Subscript):
            element = self.assign_element(stmt, env, linear)
            if element is not None:
                return element
        for field, value in ast.iter_fields(stmt):
            if isinstance(value, ast.expr):
                setattr(stmt, field, self.fold(value, env))
            elif isinstance(value, list):
                setattr(stmt, field, [self.fold(v, env) if isinstance(v, ast.expr) else v for v in value])
        env.invalidate(assigned_names(stmt))
        return [stmt]

    def assign(self, stmt: ast.Assign, env: Env, linear: bool) -> List[ast.stmt]:
        name = stmt.targets[0].id
        stmt.value = self.fold(stmt.value, env)
        elts = self.tuple_of(stmt.value, env)
        # Recording a binding is only sound where the assignment dominates every later read of the
        # name in this list -- true down a straight line, false across loop iterations.
        if elts is not None and linear:
            elts, captured = self.capture_self_reference(name, elts, stmt, env)
            env.tuples[name] = elts
            env.bound.add(name)
            self.ranks.pop(name, None)  # now a tuple, so any array rank it carried is stale
            return captured
        env.invalidate([name])
        if not isinstance(stmt.value, ast.Constant) or stmt.value.value is not None:
            env.bound.add(name)
        value_kind = self.kind(stmt.value, env)
        if value_kind is not None and linear:
            env.kinds[name] = value_kind
        self.track_rank(name, stmt.value, linear)
        return [stmt]

    def capture_self_reference(self, name: str, elements: List[ast.expr], at: ast.stmt,
                               env: Env) -> Tuple[List[ast.expr], List[ast.stmt]]:
        """``x = (x, x)`` -- what every port's ``_as_tuple(x, 2)`` folds to. Its elements mean x's
        PREVIOUS value, but recording them as they stand leaves the name denoting both that value
        and the tuple, so a later read of an element folds straight back into the tuple. Capture the
        old value in a fresh local and point the elements at that instead.
        """
        reads_itself = any(
            isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load) for e in elements
            for n in ast.walk(e))
        if not reads_itself:
            return elements, []
        self.aliases += 1
        alias = f"__ta{self.aliases}_{name}"
        captured = [_substitute(copy.deepcopy(e), name, ast.Name(id=alias, ctx=ast.Load())) for e in elements]
        bind = ast.copy_location(
            ast.Assign(targets=[ast.Name(id=alias, ctx=ast.Store())], value=ast.Name(id=name, ctx=ast.Load())), at)
        env.bound.add(alias)
        if name in env.kinds:
            env.kinds[alias] = env.kinds[name]
        return captured, [bind]

    def unpack(self, stmt: ast.Assign, env: Env, linear: bool) -> Optional[List[ast.stmt]]:
        """``oh, ow = output_size`` -> one binding per name, folded like any other assignment.

        Neither backend has a tuple to unpack, and the adaptive-pool ports spell their output size
        this way. Returns ``None`` -- leaving the statement exactly as it is -- when the right-hand
        side is not a compile-time tuple of matching length, which is the only case where splitting
        it would be a guess.
        """
        targets = stmt.targets[0].elts
        if not all(isinstance(t, ast.Name) for t in targets):
            return None
        elements = self.tuple_of(self.fold(stmt.value, env), env)
        if elements is None or len(elements) != len(targets):
            return None
        out: List[ast.stmt] = []
        for target, value in zip(targets, elements):
            one = ast.copy_location(ast.Assign(targets=[target], value=copy.deepcopy(value)), stmt)
            out.extend(self.assign(one, env, linear))
        return out

    def assign_element(self, stmt: ast.Assign, env: Env, linear: bool) -> Optional[List[ast.stmt]]:
        """``slices[k] = <index>`` on a bound index list -> rebind the list, emit nothing.

        The ports build a full-slice list, overwrite ONE axis, then tuple it
        (``slices = [slice(None)] * x.ndim; slices[dim] = slice(a, b); x[tuple(slices)]``). Returns
        ``None`` when this is an ordinary array store, which must be left exactly as it is.
        """
        target = stmt.targets[0]
        if not (isinstance(target.value, ast.Name) and target.value.id in env.tuples and linear):
            return None
        index = const_int(self.fold(copy.deepcopy(target.slice), env))
        elements = env.tuples[target.value.id]
        if index is None or not -len(elements) <= index < len(elements):
            return None
        elements[index] = self.fold(stmt.value, env)
        return []

    def track_rank(self, name: str, value: ast.expr, linear: bool) -> None:
        """Record the target's rank from the ALREADY-FOLDED RHS. Folding is what makes it knowable:
        ``y = x.reshape((n, g, c // g) + x.shape[2:])`` has no compile-time rank until the concat
        collapses, and without ``y``'s rank the next line's ``y.ndim`` never folds either.

        A rebinding inside a branch or loop (``linear=False``) may or may not run, so the name
        afterwards holds either value -- but if BOTH readings have the same rank, that IS the rank.
        Dropping it unconditionally is what lost max-pool's accumulator: ``out`` is allocated rank 4
        and then rebound as ``out = np.maximum(out, ...)`` inside the tap loops, which cost every
        LATER statement its rank, and group-norm's ``tuple(range(2, y.ndim))`` two helpers on could
        no longer fold.
        """
        rank = expr_rank(value, self.ranks)
        if not linear:
            if rank is None or self.ranks.get(name) != rank:
                self.ranks.pop(name, None)
            return
        if rank is None:
            self.ranks.pop(name, None)
        else:
            self.ranks[name] = rank

    def branch(self, stmt: ast.If, env: Env, linear: bool) -> List[ast.stmt]:
        stmt.test = self.fold(stmt.test, env)
        taken = static_bool(stmt.test)
        if taken is not None:
            return self.run(stmt.body if taken else stmt.orelse, env, linear)
        stmt.body = self.run(stmt.body, env.copy(), linear=False)
        stmt.orelse = self.run(stmt.orelse, env.copy(), linear=False)
        env.invalidate(assigned_names(stmt))
        return [stmt]

    def loop(self, stmt: ast.stmt, env: Env) -> List[ast.stmt]:
        for field in ("iter", "test"):
            value = vars(stmt).get(field)
            if isinstance(value, ast.expr):
                setattr(stmt, field, self.fold(value, env))
        # A body read can come from the PREVIOUS iteration, so kill what the body rebinds first.
        inner = env.copy()
        inner.invalidate(assigned_names(stmt))
        stmt.body = self.run(stmt.body, inner, linear=False)
        stmt.orelse = self.run(stmt.orelse, inner.copy(), linear=False)
        env.invalidate(assigned_names(stmt))
        return [stmt]


#: Marks a tuple this pass materialised from a binding. Its elements were folded when the binding
#: was recorded and must not be folded again: ``p = (p,)`` binds ``p`` to a tuple whose element IS
#: ``p``, so re-entering it would substitute forever and yield ``((p,),)``.
FOLDED_TUPLE = "tuple_desugar_folded"


class _Folder(ast.NodeTransformer):
    """Bottom-up expression rewrite against one :class:`Env`."""

    def __init__(self, interp: TupleDesugar, env: Env) -> None:
        self.interp = interp
        self.env = env

    def visit(self, node: ast.AST) -> ast.AST:
        return node if vars(node).get(FOLDED_TUPLE) else super().visit(node)

    def _materialize(self, elts: List[ast.expr], at: ast.AST) -> ast.Tuple:
        out = ast.copy_location(ast.Tuple(elts=[copy.deepcopy(e) for e in elts], ctx=ast.Load()), at)
        setattr(out, FOLDED_TUPLE, True)
        return out

    def visit_Name(self, node: ast.Name) -> ast.AST:
        elts = self.env.tuples.get(node.id)
        if elts is None or not isinstance(node.ctx, ast.Load):
            return node
        return self._materialize(elts, node)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        # After folding, never before: the index is often ``(slice(None),) + built`` and only
        # becomes a tuple of slice() calls once the concatenation has folded.
        node.slice = _slice_calls_to_slices(node.slice)
        elts = self.interp.tuple_of(node.value, self.env)
        if elts is None:
            return node
        if isinstance(node.slice, ast.Slice):
            sliced = self._slice_elements(elts, node.slice)
            return node if sliced is None else self._materialize(sliced, node)
        index = const_int(node.slice)
        if index is None or not -len(elts) <= index < len(elts):
            return node
        return ast.copy_location(copy.deepcopy(elts[index]), node)

    def _slice_elements(self, elts: List[ast.expr], sl: ast.Slice) -> Optional[List[ast.expr]]:
        """``t[i:j:k]`` with literal bounds -> the selected elements. ``x.shape[2:]`` is the reason
        this exists; an unbounded or symbolic bound leaves the subscript alone."""
        bounds = []
        for part in (sl.lower, sl.upper, sl.step):
            if part is None:
                bounds.append(None)
                continue
            value = const_int(part)
            if value is None:
                return None
            bounds.append(value)
        return [copy.deepcopy(e) for e in elts[slice(*bounds)]]

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, (ast.Add, ast.Mult)):
            elts = self.interp.tuple_of(node, self.env)
            if elts is not None and (isinstance(node.left, ast.Tuple) or isinstance(node.right, ast.Tuple)):
                return self._materialize(elts, node)
        folded = _fold_int_arithmetic(node)
        return node if folded is None else ast.copy_location(folded, node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        value = const_int(node.operand)
        if value is None or not isinstance(node.op, (ast.USub, ast.UAdd)):
            return node
        return ast.copy_location(ast.Constant(value=-value if isinstance(node.op, ast.USub) else value), node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if isinstance(node.func, ast.Name) and node.func.id == "isinstance" and len(node.args) == 2:
            folded = self._isinstance(node)
            if folded is not None:
                return folded
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and not node.keywords:
            if node.func.id == "tuple" and len(node.args) == 1:
                elts = self.interp.tuple_of(node, self.env)
                if elts is not None:
                    return self._materialize(elts, node)
            if node.func.id == "len" and len(node.args) == 1:
                elts = self.interp.tuple_of(node.args[0], self.env)
                if elts is not None:
                    return ast.copy_location(ast.Constant(value=len(elts)), node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "ndim" and isinstance(node.func.value, ast.Name)
                and node.func.value.id in NUMPY_MODULES and len(node.args) == 1):
            ndim = self._ndim(node.args[0])
            if ndim is not None:
                return ast.copy_location(ast.Constant(value=ndim), node)
        return node

    def _ndim(self, node: ast.expr) -> Optional[int]:
        """A value's rank: 0 for a scalar, the declared rank for an array, else undecidable."""
        if self.interp.kind(node, self.env) in ("int", "float", "complex", "bool"):
            return 0
        return self.interp.rank(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if node.attr != "ndim":
            return node
        ndim = self._ndim(node.value)
        return node if ndim is None else ast.copy_location(ast.Constant(value=ndim), node)

    def _isinstance(self, node: ast.Call) -> Optional[ast.AST]:
        """``isinstance(x, T)`` decided from x's kind -- the exact Python answer, never a guess: a
        fold that disagreed with the numpy reference would emit a kernel the oracle never runs."""
        names = type_names(node.args[1])
        kind = self.interp.kind(self.visit(node.args[0]), self.env)
        if names is None or kind is None:
            return None
        decided = decide_isinstance(kind, names)
        return None if decided is None else ast.copy_location(ast.Constant(value=decided), node)

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if len(node.ops) != 1:
            return node
        left, right = node.left, node.comparators[0]
        if isinstance(node.ops[0], (ast.Is, ast.IsNot)):
            decided = self._is_none(left, right)
            if decided is not None:
                return ast.copy_location(ast.Constant(value=decided is isinstance(node.ops[0], ast.Is)), node)
            return node
        decide = _EXACT_COMPARE.get(type(node.ops[0]))
        if decide is not None:
            lv, rv = const_value(left), const_value(right)
            # Literal-vs-literal only, and only for types whose comparison is exact -- a float
            # compare folded here would bake in a rounding decision the backend might make
            # differently. Ordering counts: a padding folded to its manifest value leaves
            # ``if 3 > 0:`` guarding a rebinding, and an unfolded guard is a second buffer shape.
            if lv is not NO_VALUE and rv is not NO_VALUE and isinstance(lv,
                                                                        (bool, int)) and isinstance(rv, (bool, int)):
                return ast.copy_location(ast.Constant(value=decide(lv, rv)), node)
            if isinstance(node.ops[0], (ast.Eq, ast.NotEq)) and isinstance(lv, str) and isinstance(rv, str):
                return ast.copy_location(ast.Constant(value=decide(lv, rv)), node)
        return node

    def _is_none(self, left: ast.expr, right: ast.expr) -> Optional[bool]:
        """Whether ``left is right`` is statically a None-identity, or ``None`` if undecidable."""
        if not (isinstance(right, ast.Constant) and right.value is None):
            return None
        if isinstance(left, ast.Constant):
            return left.value is None
        kind = self.interp.kind(left, self.env)
        if kind is not None:
            return kind == "none"
        return False if isinstance(left, ast.Name) and left.id in self.env.bound else None

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        self.generic_visit(node)
        taken = static_bool(node.test)
        return node if taken is None else (node.body if taken else node.orelse)


#: Comparisons folded on literal operands, by operator. Exact for integers, bools and (equality
#: only) strings; ordering on floats is deliberately absent.
_EXACT_COMPARE = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
}

#: Integer operations folded on literal operands. Integers only -- a float fold would bake in a
#: rounding decision the backend is entitled to make differently.
INT_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
}


def _fold_int_arithmetic(node: ast.BinOp) -> Optional[ast.Constant]:
    """``0 + 2`` -> ``2``. A comprehension unroll leaves index arithmetic on literals, and a tuple
    subscript needs a literal index to select an element."""
    op = INT_OPS.get(type(node.op))
    left, right = const_int(node.left), const_int(node.right)
    if op is None or left is None or right is None or (right == 0 and isinstance(node.op, (ast.FloorDiv, ast.Mod))):
        return None
    return ast.Constant(value=op(left, right))


def _slice_calls_to_slices(index: ast.AST) -> ast.AST:
    """``a[slice(i, j)]`` / ``a[(slice(None), slice(i, j))]`` -> real slice syntax.

    The ports build index tuples out of ``slice`` objects because a tuple is the only way to
    concatenate index positions in Python; once concatenated the calls are plain slices again."""
    elts = index.elts if isinstance(index, ast.Tuple) else [index]
    converted = [_as_slice(e) for e in elts]
    if all(c is None for c in converted):
        return index
    out = [orig if new is None else new for orig, new in zip(elts, converted)]
    return ast.copy_location(ast.Tuple(elts=out, ctx=ast.Load()), index) if isinstance(index, ast.Tuple) else out[0]


def _as_slice(node: ast.AST) -> Optional[ast.Slice]:
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "slice"
            and 1 <= len(node.args) <= 3 and not node.keywords):
        return None
    args = [None if isinstance(a, ast.Constant) and a.value is None else a for a in node.args]
    lower, upper, step = (None, args[0], None) if len(args) == 1 else (args + [None, None])[:3]
    return ast.copy_location(ast.Slice(lower=lower, upper=upper, step=step), node)


def _substitute(node: ast.expr, name: str, value: ast.expr) -> ast.expr:
    """``node`` with every load of ``name`` replaced by ``value`` (the comprehension unroll)."""

    class _Sub(ast.NodeTransformer):

        def visit_Name(self, inner: ast.Name) -> ast.AST:
            if inner.id == name and isinstance(inner.ctx, ast.Load):
                return ast.copy_location(copy.deepcopy(value), inner)
            return inner

    return ast.fix_missing_locations(_Sub().visit(node))


def desugar_tuples(fn: ast.FunctionDef,
                   int_scalars: FrozenSet[str] = frozenset(),
                   float_scalars: FrozenSet[str] = frozenset(),
                   arrays: FrozenSet[str] = frozenset(),
                   ranks: Optional[Dict[str, int]] = None) -> None:
    """Fold every compile-time tuple out of ``fn`` in place (see the module docstring)."""
    # The marker guards ONE run against re-entering a tuple that run just materialised. Left
    # standing it also freezes that tuple's ELEMENTS against the next run, and this pass runs a
    # second time once helper calls are spliced -- which is how a ``np.full`` shape kept an
    # unfolded ``padding[0]`` while the statement beside it folded the same read away.
    for node in ast.walk(fn):
        vars(node).pop(FOLDED_TUPLE, None)
    interp = TupleDesugar(int_scalars, float_scalars, arrays, ranks)
    env = Env(bound=set(int_scalars) | set(float_scalars) | set(arrays))
    fn.body = interp.run(fn.body, env)
    _drop_dead_none_bindings(fn)
    ast.fix_missing_locations(fn)


def _drop_dead_none_bindings(fn: ast.FunctionDef) -> None:
    """Drop ``name = None`` where nothing reads ``name`` any more, OR where the very next
    statement in the same block unconditionally rebinds it.

    Folding ``if stride is None: stride = kernel_size`` leaves the helper's ``stride = None`` default
    behind, and ``None`` has no C spelling -- the emitter would refuse the kernel over a statement
    that no longer does anything. The "no readers anywhere" check alone misses the common case: the
    call-site argument was itself a literal ``None`` (matmul_max_pool_sum_scale calling its own
    ``_maxpool1d(..., None, ...)``), so inlining binds the reassigned param through a fresh
    ``__inl1_stride = None`` init (see ``_InlineHelpers``'s ``reassigned_params``) that sits directly
    beside the now-unconditional ``__inl1_stride = kernel_size`` the guard folded to -- and
    ``__inl1_stride`` genuinely IS read later on, just never through this dead write. Adjacency makes
    that safe to see without a full liveness pass: nothing between the two statements can read the
    ``None``, so whatever the second statement writes is the only value that ever reaches a reader.
    """
    read = OrderedSet(n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))

    def none_bind_target(stmt: ast.stmt) -> Optional[str]:
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Constant) and stmt.value.value is None):
            return stmt.targets[0].id
        return None

    def rebinds(stmt: ast.stmt, name: str) -> bool:
        return (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == name)

    def prune(stmts: List[ast.stmt]) -> List[ast.stmt]:
        out: List[ast.stmt] = []
        for i, stmt in enumerate(stmts):
            name = none_bind_target(stmt)
            if name is not None and (name not in read or (i + 1 < len(stmts) and rebinds(stmts[i + 1], name))):
                continue
            out.append(stmt)
        return out

    for node in ast.walk(fn):
        for field, value in ast.iter_fields(node):
            if isinstance(value, list) and any(isinstance(v, ast.stmt) for v in value):
                setattr(node, field, prune(value))
