"""Hoist registered numpy calls out of expressions into temporaries."""

import ast

from hpcagent_bench.translators.numpyto_common import dtypes
from hpcagent_bench.translators.numpyto_common.lib_nodes.call_args import const_axis, kwarg_or_pos, read_axis_keepdims
from hpcagent_bench.translators.numpyto_common.lib_nodes.constructors import arange_count
from hpcagent_bench.translators.numpyto_common.lib_nodes.dims import call_to_str
from hpcagent_bench.translators.numpyto_common.lib_nodes.extents import (
    INT_PRESERVING_ELEMENTWISE,
    all_integer_operands,
    broadcast_extents,
    concat_operands_axis,
    iter_extent_of_,
    sum_width_tokens,
)
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import attr_call, const_int, reads_complex
from hpcagent_bench.translators.numpyto_common.lib_nodes.matmul_hoist import MatmulHoister
from hpcagent_bench.translators.numpyto_common.lib_nodes.registry import ELEMENTWISE_SHAPE_OPS, NP_CALL_EXPANDERS
from hpcagent_bench.translators.numpyto_common.lib_nodes.repeat import diff_operand
from hpcagent_bench.translators.numpyto_common.subscripts import has_slice_subscript


def numpy_call_key(call: ast.Call) -> tuple[str, str] | None:
    """The expander registry key for a call: ``np.<name>`` -> ``("np", name)``, ``np.linalg.<name>`` ->
    ``("np", "linalg.<name>")``, ``<module>.<name>`` -> ``(module, name)``; None for anything else."""
    func = call.func
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            return ("np" if func.value.id == "np" else func.value.id, func.attr)
        if (
            isinstance(func.value, ast.Attribute)
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "np"
        ):
            return ("np", f"{func.value.attr}.{func.attr}")
    return None


class CallHoister(ast.NodeTransformer):
    """Hoist any registered ``np.*`` call buried in an expression to a fresh
    temp ``__cb<n>``; the expander then lowers ``__cb<n> = call(...)``. A
    scalar-returning call (reduction/dot/std) hoists to a scalar local; an
    array-returning call (copy/outer/transpose) hoists to an array temp whose
    shape is inferred from its arguments.
    """

    def __init__(
        self,
        shape_table: dict[str, tuple[str, ...]],
        scalar_temps: dict[str, bool],
        array_temps: dict[str, tuple[str, ...]],
        counter: list[int],
        local_dtypes: dict[str, str] | None = None,
        dim_aliases: dict[str, str] | None = None,
        blas: bool = False,
    ) -> None:
        self.shape_table = shape_table
        self.scalar_temps = scalar_temps
        self.array_temps = array_temps
        self.counter = counter
        #: Both forwarded to the nested ``MatmulHoister`` (see its docstring).
        self.dim_aliases: dict[str, str] = dim_aliases or {}
        self.blas = blas
        # Side-effect dtype table (shared with the lowering pipeline)
        # so a ``__cb<n>`` whose RHS contains complex literals or
        # complex-typed Name references is tagged ``complex128``.
        self.local_dtypes: dict[str, str] = local_dtypes if local_dtypes is not None else {}
        self.pre_stmts: list[ast.stmt] = []
        #: Never populated on this class; forwarded to the nested ``MatmulHoister``,
        #: which treats ``None`` the same as an empty sparse-array table.
        self.sparse: dict[str, object] | None = None
        #: Axis/keepdims of the reduction call ``visit_Call`` is currently hoisting;
        #: read back by ``derive_output_shape`` within that same call.
        self._cur_axis: list[int] | None = None
        self._cur_keepdims: bool = False

    def infer_complex(self, expr: ast.AST) -> bool:
        """``True`` iff ``expr`` reads a complex value (skipping ``.shape`` reads)."""
        return reads_complex(expr, self.local_dtypes)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # ``np.repeat(src, np.diff(p))``: the count's telescoping sum (see
        # expand_repeat / _diff_operand) needs the ORIGINAL ``np.diff`` call
        # form. A plain ``generic_visit`` would recurse into it first -- ``np.diff``
        # is itself a registered call, so it would get hoisted into an opaque
        # ``__cb<n>`` temp before the repeat expander ever ran, losing the one
        # piece of syntax that proves the sum is derivable. Visit every other
        # child normally and leave that one argument untouched.
        if numpy_call_key(node) == ("np", "repeat") and len(node.args) >= 2 and diff_operand(node.args[1]) is not None:
            node.func = self.visit(node.func)
            node.args = [(a if i == 1 else self.visit(a)) for i, a in enumerate(node.args)]
            node.keywords = [self.visit(kw) for kw in node.keywords]
        else:
            self.generic_visit(node)
        # Hoist any matmul subexpressions inside the call args first:
        # ``np.maximum(input @ w1 + b1, 0)`` -> ``__mm1 = input @ w1; ...;
        # np.maximum(__mm1 + b1, 0)``, so the elementwise expander sees a bare
        # BinOp on Names, not a MatMult.
        mm = MatmulHoister(
            self.shape_table,
            self.array_temps,
            self.counter,
            local_dtypes=self.local_dtypes,
            sparse=self.sparse,
            dim_aliases=self.dim_aliases,
            blas=self.blas,
        )
        node.args = [mm.visit(a) for a in node.args]
        self.pre_stmts.extend(mm.pre_stmts)
        # Hoist a non-Name first arg of an array reduction (sum/max/min/mean/
        # prod/std/argmax/argmin) into a fresh temp: ``np.mean(a * b)`` ->
        # ``__cb<n> = a * b; np.mean(__cb<n>)`` so the reduction expander sees
        # a Name operand -- likewise ``np.argmax(np.abs(v))`` spills
        # ``np.abs(v)`` before the arg-reduction scaffold (which requires a
        # Name) runs. Shape-preserving index ops (roll/flip/transpose/reshape)
        # join the set too, so ls3df _hpsi's ``np.roll(psi_frag[f], m, axis)``
        # spills ``psi_frag[f]`` and hoists as ``np.roll(__cb<n>, m, axis)`` --
        # otherwise the whole-array roll stays buried in the broadcast BinOp
        # and the per-element scalarizer mangles it into a scalar-arg roll.
        key = numpy_call_key(node)
        if (
            key
            in (
                {
                    ("np", k)
                    for k in {
                        "sum",
                        "max",
                        "min",
                        "mean",
                        "prod",
                        "std",
                        "var",
                        "median",
                        "any",
                        "all",
                        "count_nonzero",
                        "argmax",
                        "argmin",
                        "repeat",
                        "transpose",
                        "reshape",
                        "triu",
                        "tril",
                        "flip",
                        "roll",
                        "copy",
                        "array",
                        "bincount",
                        "cumsum",
                        "cumprod",
                        "swapaxes",
                        "expand_dims",
                        "squeeze",
                        "moveaxis",
                    }
                }
                | {("np", "fft.fftn"), ("np", "fft.ifftn"), ("np", "fft.fft"), ("np", "fft.ifft")}
            )
            and node.args
            and not isinstance(node.args[0], ast.Name)
        ):
            first = node.args[0]
            ext = iter_extent_of_(first, self.shape_table)
            if ext is not None:
                self.counter[0] += 1
                temp = f"__cb{self.counter[0]}"
                shape = tuple(call_to_str(e) for e in ext)
                self.array_temps[temp] = shape
                self.shape_table[temp] = shape
                if self.infer_complex(first):
                    self.local_dtypes[temp] = "complex128"
                # When ``first`` carries slice-bearing Subscripts (maxpool's
                # ``np.max(x[:, 2i:2i+2, :], axis=(1, 2))``), the
                # post-LibNodeRewriter lift can no longer recover x's
                # per-statement shape -- by then x is overwritten with its
                # final shape. Emit the slice-LHS form instead: marker +
                # ``__cb[:, ...] = first``; slice-fusion lowers this into a
                # per-element copy later.
                if has_slice_subscript(first):
                    rank = len(shape)
                    slice_form = (
                        ast.Slice(lower=None, upper=None, step=None)
                        if rank == 1
                        else ast.Tuple(
                            elts=[ast.Slice(lower=None, upper=None, step=None) for unused in range(rank)],
                            ctx=ast.Load(),
                        )
                    )
                    marker = ast.Assign(
                        targets=[ast.Name(id=temp, ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Name(id="__hpcagent_bench_zeros__", ctx=ast.Load()), args=[], keywords=[]
                        ),
                    )
                    slice_lhs = ast.Subscript(
                        value=ast.Name(id=temp, ctx=ast.Load()), slice=slice_form, ctx=ast.Store()
                    )
                    slice_assign = ast.Assign(targets=[slice_lhs], value=first)
                    self.pre_stmts.append(marker)
                    self.pre_stmts.append(slice_assign)
                else:
                    # Synth: ``__cb<n> = first``. The LibNodeRewriter's
                    # _lower_prelude_calls step then turns this into a
                    # per-element copy via _WholeArrayAssignRewriter.
                    self.pre_stmts.append(ast.Assign(targets=[ast.Name(id=temp, ctx=ast.Store())], value=first))
                node.args[0] = ast.Name(id=temp, ctx=ast.Load())
        key = numpy_call_key(node)
        if key is None or key not in NP_CALL_EXPANDERS:
            return node
        # Stash axis/keepdims kwargs for the reduction case so
        # _derive_output_shape can compute the correct array shape.
        if key == ("np", "linalg.norm"):
            # ``linalg.norm``'s positional layout is ``(v, ord, axis, keepdims)``,
            # unlike a reduction's 2nd-positional ``axis`` -- strip a
            # positional/keyword ``ord`` before reading the axis (mirroring
            # ``expand_linalg_norm``), else a positional ord (``norm(a, 1)``) is
            # misread as ``axis=1`` and an axis-less vector norm is wrongly
            # hoisted as an array.
            norm_args = [node.args[0]] + list(node.args[2:]) if node.args else []
            norm_kwargs = [kw for kw in node.keywords if kw.arg != "ord"]
            self._cur_axis, self._cur_keepdims = read_axis_keepdims(norm_args, norm_kwargs)
        else:
            self._cur_axis, self._cur_keepdims = read_axis_keepdims(node.args, node.keywords)
        self.counter[0] += 1
        temp = f"__cb{self.counter[0]}"
        # Classify: scalar return vs array return.
        is_scalar = key[1] in {
            "sum",
            "max",
            "min",
            "mean",
            "prod",
            "std",
            "var",
            "dot",
            "vdot",
            "inner",
            "linalg.norm",
            "linalg.det",
            "argmax",
            "argmin",
            "any",
            "all",
            "count_nonzero",
            "median",
            "trace",
        }
        # ``np.inner`` is scalar ONLY for rank-1 x rank-1; higher ranks
        # contract the last axes into an array result.
        if key[1] == "inner":
            ranks = [len(self.shape_table.get(a.id, ())) for a in node.args if isinstance(a, ast.Name)]
            if any(r > 1 for r in ranks):
                is_scalar = False
        # Axis-aware reductions with axis specified return an array. ``var``
        # belongs here for the same reason ``std`` does -- they're one op
        # (``expand_var_or_std``, std is var plus a sqrt). Omitting it left
        # gpt2_block's layer-norm ``np.var(z, axis=-1, keepdims=True)``
        # classified scalar, so its temp was never sized or declared an array.
        if (
            is_scalar
            and key[1]
            in {
                "sum",
                "max",
                "min",
                "mean",
                "prod",
                "std",
                "var",
                "argmax",
                "argmin",
                "any",
                "all",
                "count_nonzero",
                "linalg.norm",
            }
            and self._cur_axis is not None
        ):
            is_scalar = False
        if is_scalar and node.args and isinstance(node.args[0], ast.Subscript):
            # np.dot on 1-D slices is scalar.
            ext = iter_extent_of_(node.args[0], self.shape_table)
            if ext is not None and len(ext) == 1 and key[1] == "dot":
                is_scalar = True
        if not is_scalar:
            # Array-returning: try to determine the output shape from args.
            shape = self.derive_output_shape(key, node.args, node.keywords)
            if shape is None:
                return node
            self.array_temps[temp] = shape
            self.shape_table[temp] = shape
            # ``argmax``/``argmin`` produce an INDEX array -> int64, not the
            # default double (so the buffer + any store into an int target is
            # an integer, matching numpy's intp result).
            if key[1] in {"argmax", "argmin"}:
                self.local_dtypes[temp] = "int64"
            # Propagate complex dtype when the call's argument tree
            # contains complex literals / complex-Name references.
            # ``np.exp(-2.0j * np.pi * ...)`` etc. land here.
            # Every ``np.fft.*`` transform RETURNS complex even from a real
            # input, so force the output temp complex regardless of operand.
            if self.infer_complex(node) or key[1] in {"fft.fftn", "fft.ifftn", "fft.fft", "fft.ifft"}:
                self.local_dtypes[temp] = "complex128"
            # Shape-preserving ops (``reshape`` / ``repeat`` / ``copy``
            # / ``transpose`` / ``flip``) inherit the source array's
            # dtype: ``Xiv = np.reshape(Xi, (xn * yn,))`` where ``Xi``
            # is int64 must keep Xiv as int64, not the default double.
            SHAPE_PRESERVING = {
                "reshape",
                "repeat",
                "copy",
                "array",
                "asarray",
                "ascontiguousarray",
                "transpose",
                "flip",
            }
            if key[1] in SHAPE_PRESERVING and node.args and temp not in self.local_dtypes:
                first = node.args[0]
                if isinstance(first, ast.Name):
                    src_dt = self.local_dtypes.get(first.id)
                    if src_dt:
                        self.local_dtypes[temp] = src_dt
            # An all-integer elementwise ufunc returns an INTEGER array in numpy --
            # declare the temp int64 so an exact int64 result is not round-tripped
            # through a double (which drops every bit above 2**53).
            if (
                key[1] in INT_PRESERVING_ELEMENTWISE
                and temp not in self.local_dtypes
                and all_integer_operands(node.args, self.local_dtypes)
            ):
                self.local_dtypes[temp] = "int64"
            # ``np.where`` promotes its two VALUE operands and ignores the condition's dtype, so an
            # integer select stays integer -- the last hop of bitonic_sort's comparator network,
            # whose int64 payload was otherwise handed back through a float temp.
            if (
                key[1] == "where"
                and len(node.args) == 3
                and temp not in self.local_dtypes
                and all_integer_operands(node.args[1:], self.local_dtypes)
            ):
                self.local_dtypes[temp] = "int64"
        else:
            self.scalar_temps[temp] = True
            if self.infer_complex(node):
                self.local_dtypes[temp] = "complex128"
            # A value-preserving scalar reduction (max / min / sum / prod) over an
            # INTEGER-tagged operand yields an integer -- inherit that dtype so the
            # accumulator temp is declared int, not the float default. Otherwise the
            # Fortran emit's running-max ``merge(int_elem, real_acc, ...)`` update is
            # a kind mismatch. mean/std/var/median are excluded: they produce a float
            # even from an int input.
            elif key[1] in {"max", "min", "sum", "prod"} and node.args and isinstance(node.args[0], ast.Name):
                src_dt = self.local_dtypes.get(node.args[0].id)
                if src_dt and dtypes.is_integer(src_dt):
                    self.local_dtypes[temp] = src_dt
        # Emit a ``__cb<n> = __hpcagent_bench_zeros__()`` marker first so the emit
        # walker can inline-declare the temp at the marker site -- required
        # when the temp's shape depends on an enclosing for-loop iter
        # (stockham_fft's ``R ** i``). The subsequent ``__cb<n> = call(...)``
        # then lowers into a per-element copy via the existing call-expansion path.
        if not is_scalar:
            self.pre_stmts.append(
                ast.Assign(
                    targets=[ast.Name(id=temp, ctx=ast.Store())],
                    value=ast.Call(func=ast.Name(id="__hpcagent_bench_zeros__", ctx=ast.Load()), args=[], keywords=[]),
                )
            )
        # Synthesise an Assign that the LibNodeRewriter will lower.
        self.pre_stmts.append(ast.Assign(targets=[ast.Name(id=temp, ctx=ast.Store())], value=node))
        return ast.Name(id=temp, ctx=ast.Load())

    def derive_output_shape(
        self, key: tuple[str, str], args: list[ast.expr], keywords: list[ast.keyword] | None = None
    ) -> tuple[str, ...] | None:
        op = key[1]
        # Tensor contractions (einsum/tensordot/inner): reuse the shared
        # output-extent resolver so the hoister can lift a contraction out of a
        # BinOp -- seissol's batched-GEMM-as-einsum ``Q[:] = Q +
        # np.einsum('dkl,blq,dqp->bkp', ...)``. A scalar-result contraction
        # ('ii->') yields a None extent, handled by the direct-assign expander
        # path instead.
        if op in {"einsum", "tensordot", "inner"} and len(args) >= 2:
            # ``tensordot``'s ``axes`` is frequently passed by KEYWORD
            # (``axes=([2], [0])``); dropping it here defaults to ``axes=2``
            # and yields a truncated/scalar extent (the >2-D temp mis-sized as 1-D).
            call = attr_call("np", op, list(args))
            call.keywords = list(keywords or [])
            ext = iter_extent_of_(call, self.shape_table)
            if ext is not None:
                return tuple(call_to_str(e) for e in ext)
        # ``np.bincount(idx, weights=w, minlength=M)`` -> a rank-1 result of exactly M slots (see
        # expand_bincount for why the data-dependent upper term is not the extent).
        if op == "bincount" and args:
            minlength = kwarg_or_pos(args, keywords or [], 2, "minlength")
            if minlength is not None:
                return (call_to_str(minlength),)
        # ``np.searchsorted(a, v)`` -> one index per element of the VALUES operand, so the temp
        # takes ``v``'s extent and not the sorted array's.
        if op == "searchsorted" and len(args) >= 2:
            ext = iter_extent_of_(args[1], self.shape_table)
            if ext is not None:
                return tuple(call_to_str(e) for e in ext)
        # ``np.pad`` -> source shape with each axis grown by ``2 * pad_width``.
        if op == "pad" and args:
            call = attr_call("np", "pad", list(args))
            call.keywords = list(keywords or [])
            ext = iter_extent_of_(call, self.shape_table)
            if ext is not None:
                return tuple(call_to_str(e) for e in ext)
        # Allocator-style calls: shape from the constructor arg.
        if op in {"linspace", "arange"}:
            # linspace(start, stop, n) -> (n,); arange(stop) -> (stop,);
            # arange(start, stop[, step]) -> its element count. The 3-arg form must go through
            # arange_count: `stop - start` ignores the step, which over-allocates for step > 1 and
            # is NEGATIVE for a step < 0 (see arange_count).
            if op == "linspace" and len(args) >= 3:
                return (call_to_str(args[2]),)
            if op == "arange":
                if len(args) == 1:
                    return (call_to_str(args[0]),)
                if len(args) == 2:
                    return (ast.unparse(ast.BinOp(left=args[1], op=ast.Sub(), right=args[0])),)
                if len(args) >= 3:
                    return (ast.unparse(arange_count(list(args[:3]))),)
        # ``np.fromfunction(lambda..., (N, M))`` -> the SECOND arg is the shape.
        if op == "fromfunction" and len(args) >= 2:
            sh = args[1]
            elts = sh.elts if isinstance(sh, (ast.Tuple, ast.List)) else [sh]
            return tuple(call_to_str(e) for e in elts)
        # ``np.histogram(a, bins, ...)`` returns ``hist`` of length
        # ``bins`` (the ``[0]`` Subscript unwrap selects it).
        if op == "histogram" and len(args) >= 2:
            return (call_to_str(args[1]),)
        # ``np.linalg.inv(A)`` returns the square inverse with A's
        # shape.
        if op == "linalg.inv" and args and isinstance(args[0], ast.Name):
            shape = self.shape_table.get(args[0].id)
            if shape:
                return tuple(shape)
        # ``np.linalg.solve(A, b)`` returns x with b's shape.
        if op == "linalg.solve" and len(args) >= 2 and isinstance(args[1], ast.Name):
            shape = self.shape_table.get(args[1].id)
            if shape:
                return tuple(shape)
        # Every ``np.fft.*`` transform is shape-preserving (the output has the
        # same shape as the input -- only the values change).
        if op in {"fft.fftn", "fft.ifftn", "fft.fft", "fft.ifft"} and args and isinstance(args[0], ast.Name):
            shape = self.shape_table.get(args[0].id)
            if shape:
                return tuple(shape)
        # ``np.fft.fftfreq(n, d=...)`` -> a 1-D frequency array of length ``n``
        # (the first positional arg is the sample count, not an array operand).
        if op == "fft.fftfreq" and args:
            return (call_to_str(args[0]),)
        # ``np.diag(v [, k])`` -- 1-D operand builds an ``(n+|k|, n+|k|)`` matrix,
        # 2-D operand extracts the diagonal. Reuses the ``iter_extent_of_`` rule
        # so the constructed-shape logic lives in one place; lets a Lanczos
        # ``T = np.diag(alphas) + np.diag(betas[1:], 1) + np.diag(betas[1:], -1)``
        # hoist each ``np.diag`` out of the BinOp into a correctly sized temp.
        if op == "diag" and args:
            call = attr_call("np", "diag", list(args))
            call.keywords = list(keywords or [])
            ext = iter_extent_of_(call, self.shape_table)
            if ext is not None:
                return tuple(call_to_str(e) for e in ext)
        # ``np.roll`` / ``np.linalg.cholesky`` / ``np.tril`` / ``np.triu`` all
        # return an array with the FIRST operand's shape -- so an inline
        # ``acc + np.roll(x, m, axis)`` (the periodic-stencil idiom) can be
        # hoisted out of the BinOp instead of reaching the emitter unlowered.
        if op in {"roll", "linalg.cholesky", "tril", "triu"} and args and isinstance(args[0], ast.Name):
            shape = self.shape_table.get(args[0].id)
            if shape:
                return tuple(shape)
        # ``swapaxes`` / ``expand_dims`` / ``squeeze`` / ``moveaxis`` -- the operand's extent with axes
        # swapped, moved, or a unit axis inserted / dropped. ``iter_extent_of_`` already computes all three, so route to
        # it rather than restating the axis arithmetic; without a branch here they fall through to
        # the elementwise case, which skips them (they are NON_ELEMENTWISE), and the None return
        # silently DECLINES to hoist -- leaving ``q @ np.swapaxes(k, -1, -2)`` for the emitter.
        if op in {"swapaxes", "expand_dims", "squeeze", "moveaxis"} and args:
            call = attr_call("np", op, list(args))
            call.keywords = list(keywords or [])
            ext = iter_extent_of_(call, self.shape_table)
            if ext is not None:
                return tuple(call_to_str(e) for e in ext)
        # ``np.reshape(a, shape)`` -- output extents are the shape arg, with a
        # single ``-1`` resolved to prod(source) / prod(other dims). Lets the
        # flattened-dot idiom ``a.ravel() @ a.ravel()`` (lowered to reshape)
        # hoist inline out of the matmul.
        if op == "reshape" and len(args) >= 2 and isinstance(args[0], ast.Name):
            src = self.shape_table.get(args[0].id)
            sh = args[1]
            elts = sh.elts if isinstance(sh, (ast.Tuple, ast.List)) else [sh]
            toks = [call_to_str(e) for e in elts]
            if src is not None:
                prod_src = "(" + ") * (".join(str(s) for s in src) + ")"
                if any(str(t).strip() == "-1" for t in toks):
                    others = [t for t in toks if str(t).strip() != "-1"]
                    if others:
                        denom = "(" + ") * (".join(str(t) for t in others) + ")"
                        neg = f"({prod_src}) / ({denom})"
                    else:
                        neg = f"({prod_src})"
                    toks = [neg if str(t).strip() == "-1" else str(t) for t in toks]
                return tuple(str(t) for t in toks)
        # ``np.concatenate((a, b, ...), axis=k)`` -> common shape, axis summed.
        if op == "concatenate" and args:
            try:
                names_, shapes, axis = concat_operands_axis(args, keywords, self.shape_table)
            except NotImplementedError:
                shapes = None
            if shapes:
                base = list(shapes[0])
                base[axis] = "(" + ") + (".join(s[axis] for s in shapes) + ")"
                return tuple(base)
        # Elementwise unary / binary share the operand shape; first array
        # operand (Name or Subscript-with-Slice) wins.
        if op in ELEMENTWISE_SHAPE_OPS and args:
            # Broadcast the extents of ALL operands, not just the first: the
            # hoisted temp for ``np.maximum(a(M,), B(N, M))`` must be the full
            # broadcast shape ``(N, M)``, matching the elementwise expander's own
            # broadcast iteration -- else a lower-rank first operand under-sizes
            # the temp.
            acc: tuple[ast.expr, ...] | None = None
            for arg in args:
                ext = iter_extent_of_(arg, self.shape_table)
                if ext is None:
                    continue
                acc = ext if acc is None else broadcast_extents(acc, ext)
            if acc is not None:
                return tuple(call_to_str(e) for e in acc)
        # ``np.hstack((a, b, c))`` -- horizontal stack along axis 1
        # for 2-D operands, axis 0 for 1-D operands. Sum the
        # concatenation-axis widths; the other axes are shared.
        if op == "hstack" and args:
            ops = list(args[0].elts) if (len(args) == 1 and isinstance(args[0], ast.Tuple)) else list(args)
            shapes = []
            for op_arg in ops:
                if not isinstance(op_arg, ast.Name):
                    return None
                s = self.shape_table.get(op_arg.id)
                if not s:
                    return None
                shapes.append(s)
            if not shapes:
                return None
            rank = len(shapes[0])
            if rank == 1:
                return (sum_width_tokens([s[0] for s in shapes]),)
            if rank == 2:
                return (shapes[0][0], sum_width_tokens([s[1] for s in shapes]))
            return None
        if op == "diff" and args and isinstance(args[0], ast.Name):
            shape = self.shape_table.get(args[0].id)
            if not shape:
                return None
            rank = len(shape)
            n_node = kwarg_or_pos(args, keywords, 1, "n")
            if n_node is not None and const_int(n_node) != 1:
                return None
            ax_node = kwarg_or_pos(args, keywords, 2, "axis")
            ax = rank - 1 if ax_node is None else const_axis(ax_node, rank)
            if ax is None:
                return None
            out = list(shape)
            ext = out[ax]
            out[ax] = str(int(ext) - 1) if ext.strip().isdigit() else f"({ext}) - 1"
            return tuple(out)
        if op in {"transpose", "triu", "flip"} and args and isinstance(args[0], ast.Name):
            shape = self.shape_table.get(args[0].id)
            if not shape:
                return None
            if op != "transpose":
                return tuple(shape)
            # ``np.transpose(A, axes)`` honours the perm (positional or
            # via the ``axes=`` keyword); without it, reverse axes.
            perm_arg = kwarg_or_pos(args, keywords, 1, "axes")
            if isinstance(perm_arg, (ast.Tuple, ast.List)):
                perm = [e.value for e in perm_arg.elts if isinstance(e, ast.Constant) and isinstance(e.value, int)]
                if len(perm) == len(shape):
                    return tuple(shape[p] for p in perm)
            return tuple(reversed(shape))
        # Axis-aware reductions: output shape = reduction axis removed (or size
        # 1 if keepdims). ``argmax``/``argmin`` with an axis return the index
        # array over the kept axes (same shape as a value reduction);
        # axis-aware ``linalg.norm`` is a per-line L2 reduction with the same
        # kept-axes shape; ``var`` sizes exactly like ``std`` (one op -- std is
        # var plus a sqrt).
        if op in {"sum", "max", "min", "mean", "prod", "std", "var", "argmax", "argmin", "linalg.norm"}:
            if args and isinstance(args[0], ast.Name):
                src_shape = self.shape_table.get(args[0].id)
                if src_shape:
                    # args doesn't carry keywords (those are on the parent
                    # call), so read the live axis/keepdims stash visit_Call set.
                    kw_axes, kw_keep = self._cur_axis, self._cur_keepdims
                    if kw_axes is None:
                        return None  # scalar -- not array-shape
                    # ``read_axis_keepdims`` returns a list or None; normalise
                    # to a set of resolved positive axes.
                    if isinstance(kw_axes, int):
                        kw_axes = [kw_axes]
                    resolved = []
                    for a in kw_axes:
                        na = a + len(src_shape) if a < 0 else a
                        if 0 <= na < len(src_shape):
                            resolved.append(na)
                    axes_set = set(resolved)
                    if kw_keep:
                        return tuple("1" if i in axes_set else s for i, s in enumerate(src_shape))
                    return tuple(s for i, s in enumerate(src_shape) if i not in axes_set)
        if op == "reshape" and len(args) >= 2:
            shape_arg = args[1]
            if isinstance(shape_arg, ast.Tuple):
                parts = []
                for e in shape_arg.elts:
                    if const_int(e) is not None:
                        parts.append(str(const_int(e)))
                    elif isinstance(e, ast.Name):
                        parts.append(e.id)
                    else:
                        parts.append(ast.unparse(e))
                # Resolve a ``-1`` placeholder (``x.reshape(batch, -1)``) to the source
                # element count over the product of the other target dims. ``/`` renders
                # as integer division in C/Fortran (both dims are integers).
                neg1 = [i for i, p in enumerate(parts) if p.strip() == "-1"]
                src = args[0]
                src_shape = self.shape_table.get(src.id) if isinstance(src, ast.Name) else None
                if len(neg1) == 1 and src_shape:
                    total = " * ".join(f"({t})" for t in src_shape)
                    others = [p for j, p in enumerate(parts) if j != neg1[0]]
                    denom = " * ".join(f"({p})" for p in others) if others else "1"
                    parts[neg1[0]] = f"({total}) / ({denom})"
                return tuple(parts)
        if op in {"outer", "add.outer"} and len(args) == 2:
            a_ext = iter_extent_of_(args[0], self.shape_table)
            b_ext = iter_extent_of_(args[1], self.shape_table)
            if a_ext is not None and b_ext is not None and len(a_ext) == 1 and len(b_ext) == 1:
                return (call_to_str(a_ext[0]), call_to_str(b_ext[0]))
        # ``np.diagonal(a)`` on a SQUARE rank-2 operand: one element per row. Without a size here the
        # hoister declines, so the diagonal stayed inline inside ``np.tanh(...)`` -- where the
        # elementwise scalariser has no cell to read and the call reached emit whole.
        if op == "diagonal" and len(args) == 1:
            d_ext = iter_extent_of_(args[0], self.shape_table)
            if d_ext is not None and len(d_ext) == 2 and ast.unparse(d_ext[0]) == ast.unparse(d_ext[1]):
                return (call_to_str(d_ext[0]),)
        # linalg ops that preserve their argument's shape.
        if op in {"linalg.cholesky", "linalg.inv"} and args and isinstance(args[0], ast.Name):
            shape = self.shape_table.get(args[0].id)
            if shape:
                return tuple(shape)
        return None
