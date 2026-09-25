"""numpy call and attribute vocabularies the JAX passes classify expressions by."""

BOOL_FUNCS = (
    "logical_and",
    "logical_or",
    "logical_not",
    "logical_xor",
    "less",
    "greater",
    "less_equal",
    "greater_equal",
    "equal",
    "not_equal",
    "isfinite",
    "isnan",
    "isinf",
)


SHAPE_FUNCS = (
    "zeros",
    "ones",
    "empty",
    "full",
    "reshape",
    "arange",
    "zeros_like",
    "ones_like",
    "broadcast_to",
    "tile",
    "repeat",
    "linspace",
    "histogram",
)


ARRAY_ATTRS = (
    "shape",
    "size",
    "ndim",
    "T",
    "dtype",
    "real",
    "imag",
    "max",
    "min",
    "mean",
    "sum",
    "std",
    "var",
    "dot",
    "copy",
    "flatten",
    "ravel",
    "reshape",
    "transpose",
    "conj",
    "prod",
    "argmax",
    "argmin",
)


# Shape/count funcs whose first positional arg is the data array, not a dim.
LEADING_DATA_FUNCS = (
    "reshape",
    "histogram",
    "tile",
    "repeat",
    "broadcast_to",
    "zeros_like",
    "ones_like",
    "empty_like",
    "full_like",
)


#: builtins that return a concrete scalar from concrete args (so ``abs(iord)`` /
#: ``int(x)`` stay static) -- read by :func:`is_static_expr`.
STATIC_BUILTINS = {"abs", "int", "float", "min", "max", "round", "len", "bool", "sum"}
