"""The flat-grid FFT idiom (reshape + fftn + reshape) lowered to materialised temporaries."""

import ast

from hpcagent_bench.translators.numpyto_common.lowering.calls import match_fft, match_reshape


class FftGridReshapeRewriter(ast.NodeTransformer):
    """Lower the QE flat-grid FFT idiom into materialised reshape + fft temps.

    ``np.fft.ifftn(X.reshape((d1, d2, d3, -1)), axes=(0, 1, 2)).reshape(M, -1)``
    (optionally followed by ``[:, 0]``) is a 3-D DFT applied column-wise to an
    ``(M, C)`` array reinterpreted as a ``d1 x d2 x d3`` grid (vexx's
    ``invfft``/``fwfft``). ``X`` has a known 2-D shape ``(M, C)`` so the
    reshape ``-1`` is exactly ``C`` -- no general ``-1`` inference, no reliance
    on the unprovable ``d1*d2*d3 == M`` identity in symbolic form. Rewrite to::

        __g = np.reshape(X, (d1, d2, d3, C))     # split leading axis, keep C
        __f = np.fft.ifftn(__g, axes=(0, 1, 2))  # expand_dftn batches axis 3
        __o = np.reshape(__f, (M, C))            # flatten back
        <lhs> = __o            # or __o (1-D, length M) when the chain ends [:,0]

    Each temp's shape is registered so LibNodeRewriter's ``expand_reshape`` /
    ``expand_dftn`` (both C-order, matching numpy) expand them into loops.
    Runs before LibNodeRewriter."""

    def __init__(
        self, shape_table: dict[str, tuple[str, ...]], local_dtypes: dict[str, str], counter: list[int]
    ) -> None:
        self.shape_table = shape_table
        self.local_dtypes = local_dtypes
        self.counter = counter

    def src_MC(self, src: ast.AST):
        """Resolve the FFT input expression to ``(name, M, C)`` -- a bare Name
        to reshape and its leading/trailing extents. Handles a 2-D Name, a 1-D
        Name (C=1), and ``X[:, None]`` of a 1-D Name (C=1). Else ``None``."""
        if isinstance(src, ast.Name):
            shp = self.shape_table.get(src.id)
            if not shp:
                return None
            return src.id, shp[0], (shp[1] if len(shp) > 1 else "1")
        if (
            isinstance(src, ast.Subscript)
            and isinstance(src.value, ast.Name)
            and isinstance(src.slice, ast.Tuple)
            and len(src.slice.elts) == 2
            and isinstance(src.slice.elts[0], ast.Slice)
            and isinstance(src.slice.elts[1], ast.Constant)
            and src.slice.elts[1].value is None
        ):
            shp = self.shape_table.get(src.value.id)
            if not shp or len(shp) != 1:
                return None
            return src.value.id, shp[0], "1"
        return None

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        if len(node.targets) != 1:
            return node
        rhs = node.value
        # Optional trailing ``[:, k]`` column select.
        col_k = None
        if (
            isinstance(rhs, ast.Subscript)
            and isinstance(rhs.slice, ast.Tuple)
            and len(rhs.slice.elts) == 2
            and isinstance(rhs.slice.elts[0], ast.Slice)
            and isinstance(rhs.slice.elts[1], ast.Constant)
        ):
            col_k = rhs.slice.elts[1].value
            chain = rhs.value
        else:
            chain = rhs
        outer = match_reshape(chain)
        if outer is None:
            return node
        fft_node, out_elts = outer
        fft = match_fft(fft_node)
        if fft is None:
            return node
        fn_name, inner_call, fft_kw = fft
        inner = match_reshape(inner_call)
        if inner is None:
            return node
        src_expr, grid_elts = inner
        # grid shape = leading dims + trailing ``-1``; need >=2 dims and -1 last.
        if len(grid_elts) < 2 or ast.unparse(grid_elts[-1]).strip() != "-1":
            return node
        mc = self.src_MC(src_expr)
        if mc is None:
            return node
        src_name, M, C = mc
        grid_dims = [ast.unparse(e) for e in grid_elts[:-1]]
        grid_shape = tuple(grid_dims) + (C,)
        n = self.counter[0]
        self.counter[0] += 3
        g, f, o = f"__fg{n}", f"__ff{n}", f"__fo{n}"

        def tok_(t):
            return ast.parse(t, mode="eval").body

        def tuple_(toks):
            return ast.Tuple(elts=[tok_(t) for t in toks], ctx=ast.Load())

        reshape_g = ast.Assign(
            targets=[ast.Name(id=g, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="reshape", ctx=ast.Load()),
                args=[ast.Name(id=src_name, ctx=ast.Load()), tuple_(grid_shape)],
                keywords=[],
            ),
        )
        fft_call = ast.Assign(
            targets=[ast.Name(id=f, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="fft", ctx=ast.Load()),
                    attr=fn_name,
                    ctx=ast.Load(),
                ),
                args=[ast.Name(id=g, ctx=ast.Load())],
                keywords=fft_kw,
            ),
        )
        # Output reshape: drop the singleton column when the chain ends ``[:, 0]``
        # (only valid for C == 1, the single-column case); else keep (M, C).
        if col_k is not None:
            if str(C) != "1" or col_k != 0:
                return node
            out_shape = (M,)
        else:
            out_shape = (M, C)
        reshape_o = ast.Assign(
            targets=[ast.Name(id=o, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="reshape", ctx=ast.Load()),
                args=[ast.Name(id=f, ctx=ast.Load()), tuple_(out_shape)],
                keywords=[],
            ),
        )
        for nm, shp in ((g, grid_shape), (f, grid_shape), (o, out_shape)):
            self.shape_table[nm] = shp
            self.local_dtypes[nm] = "complex128"
        node.value = ast.Name(id=o, ctx=ast.Load())
        for s in (reshape_g, fft_call, reshape_o, node):
            ast.copy_location(s, node) if isinstance(s, ast.stmt) else None
        ast.fix_missing_locations(reshape_g)
        ast.fix_missing_locations(fft_call)
        ast.fix_missing_locations(reshape_o)
        return [reshape_g, fft_call, reshape_o, node]
