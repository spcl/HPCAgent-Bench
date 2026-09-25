"""Shapes and dtypes recovered from a kernel's ``initialize()`` and from numpy constructor calls."""

import ast
import pathlib
from collections.abc import Iterator, Mapping

from hpcagent_bench.translators.numpyto_common.frontend.manifest import as_block, as_list
from hpcagent_bench.translators.numpyto_common.frontend.module_constants import inline_module_constants
from hpcagent_bench.translators.numpyto_common.frontend.shape_arith import (
    collect_inlined_scalar_defs,
    substitute_inlined_scalar_defs,
)


#: Numpy dtype identifiers recognised by ``dtype_from_constructor``.
NP_DTYPE_NAMES: dict[str, str] = {
    "float64": "float64",
    "float32": "float32",
    "float16": "float16",
    "float128": "float128",
    "longdouble": "float128",
    "double": "float64",
    "single": "float32",
    "half": "float16",
    "int64": "int64",
    "int32": "int32",
    "int16": "int16",
    "int8": "int8",
    "intp": "int64",
    "uint64": "uint64",
    "uint32": "uint32",
    "uint16": "uint16",
    "uint8": "uint8",
    "complex64": "complex64",
    "complex128": "complex128",
    "complex256": "complex256",
    "bool_": "bool",
    "bool": "bool",
    # ``hpcagent_bench.frameworks.framework`` aliases that the legacy
    # mandelbrot kernels import (``np_complex``, ``np_float``). Both are
    # precision-following: resolve to the natural float64 / complex128 here
    # and let the precision pass narrow them to float32 / complex64 for an
    # fp32 run. (Hardcoding float32 truncated the fp64 grid to single
    # precision -- the mandelbrot1 boundary then drifted ~4e-4.)
    "np_float": "float64",
    "np_complex": "complex128",
}


#: The same two aliases, as the set of names a reference may rebind off the framework module.
FRAMEWORK_DTYPE_ALIASES = frozenset(("np_float", "np_complex"))


def dtype_from_constructor(rhs: ast.AST) -> str | None:
    """Inspect a constructor call's ``dtype=`` kwarg or astype receiver
    and return the matching internal dtype tag (e.g. ``float64``).

    Recognises ``dtype=np.complex128`` / ``dtype=np_complex`` /
    ``dtype=data.dtype`` (the latter resolves to the source's dtype
    if recorded in ``so_far_dtypes``) and the ``.astype(dtype)`` form.
    """
    if isinstance(rhs, ast.Call):
        # ``foo.astype(dtype)`` -- recurse with the receiver.
        if isinstance(rhs.func, ast.Attribute) and rhs.func.attr == "astype" and rhs.args:
            inner = dtype_from_dtype_arg(rhs.args[0])
            if inner is not None:
                return inner
        for kw in rhs.keywords:
            if kw.arg == "dtype":
                t = dtype_from_dtype_arg(kw.value)
                if t is not None:
                    return t
        # ``np.int64(4)`` -- the dtype IS the callee, with no dtype= kwarg to read. A scalar built
        # this way carries its width nowhere else, so missing it left the emitter to fall back to
        # the run's float type: compute's integer a/b/c became ``const double``, which the harness
        # then called with int64 arguments.
        if not rhs.keywords:
            t = dtype_from_dtype_arg(rhs.func)
            if t is not None:
                return t
    return None


def dtype_from_dtype_arg(node: ast.AST) -> str | None:
    """Resolve a ``dtype=`` kwarg expression to an internal dtype tag.

    Handles three shapes:
    * ``np.complex128`` (Attribute on Name)
    * ``np_complex`` / ``np_float`` (bare module-aliased Name)
    * ``data.dtype`` (Attribute ``.dtype`` -- mirrors source array;
      caller should look it up; here we return ``None`` so the
      caller falls back to its own dtype-tracking table).
    """
    if isinstance(node, ast.Attribute) and node.attr in NP_DTYPE_NAMES:
        return NP_DTYPE_NAMES[node.attr]
    if isinstance(node, ast.Name) and node.id in NP_DTYPE_NAMES:
        return NP_DTYPE_NAMES[node.id]
    return None


def initialize_function(
    numpy_py: pathlib.Path, info: Mapping[str, object]
) -> tuple[ast.Module, ast.FunctionDef] | None:
    """The companion ``<module>.py``'s ``init.func_name`` function, parsed, or ``None``."""
    func_name = as_block(info.get("init")).get("func_name")
    if func_name is None:
        return None
    path = numpy_py.with_name(numpy_py.stem.removesuffix("_numpy") + ".py")
    if not path.exists():
        return None
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return tree, node
    return None


def named_assigns(init_fn: ast.FunctionDef) -> Iterator[tuple[str, ast.expr]]:
    """``(name, value)`` for each top-level ``name = value`` of ``init_fn``."""
    for stmt in init_fn.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            yield stmt.targets[0].id, stmt.value


def return_targets_(init_fn: ast.FunctionDef) -> list[str]:
    """The names the last ``return`` of ``init_fn`` hands back, in order."""
    for stmt in reversed(init_fn.body):
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            if isinstance(stmt.value, ast.Tuple):
                return [ast.unparse(e) for e in stmt.value.elts]
            if isinstance(stmt.value, ast.Name):
                return [stmt.value.id]
            return []
    return []


def kernel_array_args_(info: Mapping[str, object]) -> list[str]:
    """``input_args`` that are arrays, in order: what an ``initialize()`` return lines up with."""
    array_args = {str(a) for a in as_list(info.get("array_args"))}
    return [a for a in (str(a) for a in as_list(info.get("input_args"))) if a in array_args]


def dtypes_from_initialize(numpy_py: pathlib.Path, info: Mapping[str, object]) -> dict[str, str]:
    """dtype tags of the arrays ``initialize()`` builds, keyed by its locals and, when its return
    lines up one-to-one with the kernel's array arguments, by those argument names too (a
    misaligned return is skipped: the manifest's ``init.dtypes`` is authoritative there)."""
    loaded = initialize_function(numpy_py, info)
    if loaded is None:
        return {}
    tree_, init_fn = loaded
    dtypes: dict[str, str] = {}
    for name, value in named_assigns(init_fn):
        dt = dtype_from_constructor(value)
        if dt is not None:
            dtypes[name] = dt
    return_targets = return_targets_(init_fn)
    kernel_array_args = kernel_array_args_(info)
    if return_targets and len(kernel_array_args) == len(return_targets):
        for kernel_name, ret_name in zip(kernel_array_args, return_targets):
            if ret_name in dtypes and kernel_name not in dtypes:
                dtypes[kernel_name] = dtypes[ret_name]
    return dtypes


def shapes_from_initialize(numpy_py: pathlib.Path, info: Mapping[str, object]) -> dict[str, str]:
    """Shape expressions of the arrays the companion ``initialize()`` builds (see
    :func:`shape_from_constructor` for the recognised constructors), keyed by its locals and by the
    kernel array arguments its return lines up with positionally."""
    loaded = initialize_function(numpy_py, info)
    if loaded is None:
        return {}
    tree, init_fn = loaded
    # Its own module constants and single-assignment scalar locals (``H_out = H - K + 1``) are
    # substituted into the harvested shapes, so none survives as a phantom ABI symbol.
    inline_module_constants(tree, init_fn, [])
    init_scalar_defs = collect_inlined_scalar_defs(init_fn, prefix=None)
    # ``sizes = [S0, S1]`` so ``sizes[0]`` in a shape resolves to ``S0``.
    list_locals = {
        name: [ast.unparse(e) for e in value.elts]
        for name, value in named_assigns(init_fn)
        if isinstance(value, ast.List)
    }
    shapes: dict[str, str] = {}
    for name, value in named_assigns(init_fn):
        shape = shape_from_constructor(value, shapes)
        if shape is None:
            continue
        for lst_name, elts in list_locals.items():
            for i, elt in enumerate(elts):
                shape = shape.replace(f"{lst_name}[{i}]", elt)
        if init_scalar_defs:
            shape = substitute_inlined_scalar_defs((shape,), init_scalar_defs)[0]
        shapes[name] = shape
    for kernel_name, ret_name in zip(kernel_array_args_(info), return_targets_(init_fn)):
        if ret_name in shapes and kernel_name not in shapes:
            shapes[kernel_name] = shapes[ret_name]
    return shapes


SHAPE_FIRST_ARG = {
    "empty",
    "zeros",
    "ones",
    "ndarray",
    "full",
    "identity",
    # numpy.random plus ``rng = default_rng(...); rng.random(shape, ...)``:
    "rand",
    "random",
    "randn",
    "standard_normal",
    "uniform",
    # integer generators (``rng.integers(low, high, size=...)`` /
    # legacy ``np.random.randint(low, high, size=...)``) carry the shape in
    # ``size`` exactly like the float distributions below.
    "integers",
    "randint",
}


#: numpy.random distribution generators with a ``(low, high, ..., size)``
#: signature -- the shape is the ``size`` arg, never the leading params.
DIST_FUNCS = {
    "uniform",
    "normal",
    "exponential",
    "poisson",
    "beta",
    "gamma",
    "binomial",
    "lognormal",
    "laplace",
    "logistic",
    "integers",
    "randint",
}


SHAPE_SECOND_ARG = {"fromfunction"}


#: Constructors that spread axis lengths across SEPARATE positional args
#: (``np.random.rand(M, N)``); every other shape-first ctor takes one shape arg.
AXES_AS_ARGS = {"rand", "randn"}


#: Constructors whose result shares the FIRST positional arg's shape.
SHARE_SHAPE_OF_FIRST = {"copy", "asarray", "ascontiguousarray", "array", "ravel", "flatten", "abs", "absolute"}


def elementwise_operands(node: ast.AST) -> list[ast.expr] | None:
    """Operands of a shape-preserving wrapper -- a comparison, arithmetic, a unary op, or
    ``np.where`` / ``clip`` / ``minimum`` / ``maximum`` -- whose array operand carries the shape
    (``(rng.random((N, N)) < 0.15).astype(int)``); ``None`` for anything else."""
    if isinstance(node, ast.Compare):
        return [node.left, *node.comparators]
    if isinstance(node, ast.BinOp):
        return [node.left, node.right]
    if isinstance(node, ast.UnaryOp):
        return [node.operand]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in ("np", "numpy")
        and node.func.attr in ("where", "clip", "minimum", "maximum")
    ):
        return list(node.args)
    return None


def call_name(func: ast.expr) -> str | None:
    return func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else None)


def shape_first_arg(node: ast.Call, attr: str) -> str | None:
    """Shape of a shape-first constructor or random generator: ``size=`` wins, a distribution
    (``uniform(low, high, size)``) takes its third positional, ``rand``/``randn`` spread one axis per
    argument, every other constructor takes its first argument."""
    size = next((kw.value for kw in node.keywords if kw.arg == "size"), None)
    if size is not None:
        return unparse_shape_arg(size)
    if attr in DIST_FUNCS:
        return unparse_shape_arg(node.args[2]) if len(node.args) >= 3 else None
    if (
        attr in AXES_AS_ARGS
        and len(node.args) >= 2
        and all(
            isinstance(a, (ast.Constant, ast.Name, ast.BinOp, ast.Subscript, ast.Call, ast.UnaryOp)) for a in node.args
        )
    ):
        return "(" + ", ".join(ast.unparse(a) for a in node.args) + ")"
    return unparse_shape_arg(node.args[0])


def shape_from_constructor(node: ast.AST, so_far: dict[str, str]) -> str | None:
    """``"(N, M)"`` shape text of one array-building expression, or ``None``.

    Sees through ``.astype(...)`` and shape-preserving wrappers; ``x.copy()``, ``*_like(x)`` and
    copies take ``x``'s shape from ``so_far``; generators on a ``default_rng()`` object count as
    ``np.random`` calls.
    """
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "astype":
        return shape_from_constructor(node.func.value, so_far)
    operands = elementwise_operands(node)
    if operands is not None:
        return next((s for op in operands if (s := shape_from_constructor(op, so_far)) is not None), None)
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    # ``arr.copy()``, not ``np.copy(arr)``.
    if (
        isinstance(func, ast.Attribute)
        and func.attr == "copy"
        and isinstance(func.value, ast.Name)
        and func.value.id != "np"
    ):
        return so_far.get(func.value.id)
    attr = call_name(func)
    if attr is None:
        return None
    if (attr.endswith("_like") or attr in SHARE_SHAPE_OF_FIRST) and node.args and isinstance(node.args[0], ast.Name):
        return so_far.get(node.args[0].id)
    if attr in SHAPE_FIRST_ARG and (node.args or any(kw.arg == "size" for kw in node.keywords) or attr in DIST_FUNCS):
        return shape_first_arg(node, attr)
    if attr in SHAPE_SECOND_ARG and len(node.args) >= 2:
        return unparse_shape_arg(node.args[1])
    shape = next((kw.value for kw in node.keywords if kw.arg == "shape"), None)
    return unparse_shape_arg(shape) if shape is not None else None


def unparse_shape_arg(node: ast.AST) -> str | None:
    """Turn a shape AST (tuple / single symbol) into ``"(N,M)"`` text."""
    if isinstance(node, ast.Tuple):
        return "(" + ", ".join(ast.unparse(e) for e in node.elts) + ")"
    if isinstance(node, ast.Name):
        return f"({node.id},)"
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return f"({node.value},)"
    # A single EXPRESSION axis length -- ``np.random.rand(R + 1)`` (stencil
    # weights) / ``np.zeros(n - 1)``: a 1-D array whose length is the unparsed
    # arithmetic. Without this the BinOp dropped to the wrong ``(N,)`` fallback.
    if isinstance(node, (ast.BinOp, ast.Subscript, ast.Call, ast.UnaryOp)):
        return f"({ast.unparse(node)},)"
    return None
