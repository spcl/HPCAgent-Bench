"""``eigh`` / ``eigvalsh`` lowered to a Jacobi eigen-solver loop nest."""

import ast
from collections.abc import Mapping

from hpcagent_bench.translators.numpyto_common import dtypes
from hpcagent_bench.translators.numpyto_common.numpy_desugar.common import eigh_call_kind, is_eigh_assign_target
from hpcagent_bench.translators.numpyto_common.numpy_desugar.kinds import dtype_kind, dtype_table_
from hpcagent_bench.translators.numpyto_common.numpy_desugar.linalg import cholesky_lines
from hpcagent_bench.translators.numpyto_common.numpy_desugar.ranks import expr_rank


def eigh_w_dtype(is_real: bool, names, array_dtypes: dict[str, str]) -> str | None:
    """Eigenvalue dtype for an ``eigh`` lowering, or ``None`` to let the loop use the input's
    own ``.dtype``.

    numpy's eigenvalues carry the REAL half of the operand dtype. A real operand says that
    itself (``c.dtype``), so this returns ``None`` and the lowering emits the runtime form.
    A COMPLEX operand cannot: ``.real.dtype`` is not something the DaCe frontend parses, so
    the width is resolved HERE from the declared array dtypes -- the same rule and the same
    fp64-when-unresolvable fallback :func:`fft_real_dtype` uses, which matters because a
    precision sweep remaps the dtype tables but never the literals in an emitted body.
    """
    if is_real:
        return None
    for nm in names:
        dtype = array_dtypes.get(nm) if nm else None
        if dtype is None:
            continue
        try:
            return f"np.{dtypes.real_component_dtype(dtype)}"
        except KeyError:
            return f"np.{dtypes.canonical(dtype)}"
    return "np.float64"


def eigh_jacobi_lines(
    w: str, y: str, c: str, n: str, p: str, is_real: bool = False, w_dtype: str | None = None
) -> list[str]:
    """Source lines diagonalising Hermitian ``n``x``n`` matrix ``c`` by cyclic
    complex Jacobi into eigenvalues ``w`` (ascending real, shape ``(n,)``) and
    eigenvectors ``y`` (unitary columns). Each sweep rotates every off-diagonal
    pair ``(pp, qq)`` to zero with a unitary ``J`` (phase ``apq/|apq|`` then a
    real symmetric Jacobi angle); the two-sided update ``A = J^H A J`` runs as
    explicit column/row loops. Selection-sort ascending at the end (matches
    numpy.linalg.eigh's order). ``n`` is explicit (not ``c.shape[0]``) so
    C/Fortran see the resolved dimension symbol, not a temp's ``.shape``.
    Validated against numpy to ~5e-15.

    ``is_real`` marks a PROVABLY non-complex ``c`` (the caller's own dtype
    table, not a runtime check): the ``.real``/``.imag`` accessors below are
    then never emitted -- ``np.real(x)`` on a real ``x`` is ``x`` and
    ``np.imag(x)`` is exactly ``0.0`` -- instead of leaving an accessor for
    DaCe to lower into an ADL-unreachable, unqualified ``real()``/``imag()``
    C++ call (a complex operand resolves through ``std::real`` by ADL; a bare
    ``double`` reaches no namespace at all)."""
    # Diagonalise ``c`` IN PLACE -- the caller always passes a fresh, disposable
    # matrix (the reduced ``L^-1 a L^-H`` or an ``ascontiguousarray`` copy), so no
    # extra working copy is needed (and the C/Fortran backends need not infer a
    # copy-temp's complex dtype).
    a, v = c, f"{p}_jv"
    # A real input's eigenvalues are its own dtype; a complex one's real half is not spellable
    # in emitted source, so the caller resolves it and passes w_dtype.
    wd_default = f"{c}.dtype"
    off_ap, app_ap, aqq_ap = f"{a}[{p}_pp, {p}_qq]", f"{a}[{p}_pp, {p}_pp]", f"{a}[{p}_qq, {p}_qq]"
    apq_ap, diag_ap = f"{p}_apq", f"{a}[{p}_i, {p}_i]"
    off_re = off_ap if is_real else f"{off_ap}.real"
    off_im = "0.0" if is_real else f"{off_ap}.imag"
    apq_re = apq_ap if is_real else f"{apq_ap}.real"
    apq_im = "0.0" if is_real else f"{apq_ap}.imag"
    app_re = app_ap if is_real else f"{app_ap}.real"
    aqq_re = aqq_ap if is_real else f"{aqq_ap}.real"
    diag_re = diag_ap if is_real else f"{diag_ap}.real"
    ephi_h = f"{p}_ephi" if is_real else f"np.conj({p}_ephi)"  # a real phase is +-1, its own conjugate
    return [
        # eigenvector accumulator V = I, as zeros + a diagonal loop (``np.eye``'s
        # C/Fortran expansion does not carry the complex dtype the way ``np.zeros``
        # does, so the accumulator would otherwise declare real).
        f"{v} = np.zeros(({n}, {n}), {c}.dtype)",
        f"for {p}_di in range({n}):",
        f"    {v}[{p}_di, {p}_di] = 1",
        f"for {p}_sw in range(80):",
        f"    {p}_off = 0.0",
        f"    for {p}_pp in range({n}):",
        f"        for {p}_qq in range({p}_pp + 1, {n}):",
        f"            {p}_off += {off_re} * {off_re} + {off_im} * {off_im}",
        f"    if {p}_off <= 1e-30:",
        f"        break",
        f"    for {p}_pp in range({n}):",
        f"        for {p}_qq in range({p}_pp + 1, {n}):",
        f"            {p}_apq = {a}[{p}_pp, {p}_qq]",
        f"            {p}_m = np.hypot({apq_re}, {apq_im})",
        f"            if {p}_m == 0.0:",
        f"                continue",
        f"            {p}_app = {app_re}",
        f"            {p}_aqq = {aqq_re}",
        f"            {p}_ephi = {p}_apq / {p}_m",
        f"            {p}_tau = ({p}_aqq - {p}_app) / (2.0 * {p}_m)",
        f"            {p}_ts = 1.0 if {p}_tau >= 0.0 else -1.0",
        f"            {p}_t = {p}_ts / (abs({p}_tau) + np.sqrt({p}_tau * {p}_tau + 1.0))",
        f"            {p}_c = 1.0 / np.sqrt({p}_t * {p}_t + 1.0)",
        f"            {p}_s = {p}_t * {p}_c",
        f"            for {p}_k in range({n}):",  # A @ J : columns pp, qq
        f"                {p}_akp = {a}[{p}_k, {p}_pp]",
        f"                {p}_akq = {a}[{p}_k, {p}_qq]",
        f"                {a}[{p}_k, {p}_pp] = {p}_c * {p}_akp - {p}_s * {ephi_h} * {p}_akq",
        f"                {a}[{p}_k, {p}_qq] = {p}_s * {p}_ephi * {p}_akp + {p}_c * {p}_akq",
        f"            for {p}_k in range({n}):",  # J^H @ A : rows pp, qq
        f"                {p}_apk = {a}[{p}_pp, {p}_k]",
        f"                {p}_aqk = {a}[{p}_qq, {p}_k]",
        f"                {a}[{p}_pp, {p}_k] = {p}_c * {p}_apk - {p}_s * {p}_ephi * {p}_aqk",
        f"                {a}[{p}_qq, {p}_k] = {p}_s * {ephi_h} * {p}_apk + {p}_c * {p}_aqk",
        f"            for {p}_k in range({n}):",  # V @ J : columns pp, qq
        f"                {p}_vkp = {v}[{p}_k, {p}_pp]",
        f"                {p}_vkq = {v}[{p}_k, {p}_qq]",
        f"                {v}[{p}_k, {p}_pp] = {p}_c * {p}_vkp - {p}_s * {ephi_h} * {p}_vkq",
        f"                {v}[{p}_k, {p}_qq] = {p}_s * {p}_ephi * {p}_vkp + {p}_c * {p}_vkq",
        # Eigenvalues carry the REAL half of the input dtype (numpy: eigh(complex64) -> float32,
        # eigh(float32) -> float32). A real input spells that as its own .dtype; a complex one
        # is resolved by the caller (:func:`eigh_w_dtype`), never assumed fp64 here.
        f"{w} = np.zeros({n}, {w_dtype or wd_default})",
        f"for {p}_i in range({n}):",
        f"    {w}[{p}_i] = {diag_re}",
        f"for {p}_i in range({n}):",  # selection-sort ascending, permuting eigenvectors
        f"    {p}_mn = {p}_i",
        f"    for {p}_j in range({p}_i + 1, {n}):",
        f"        if {w}[{p}_j] < {w}[{p}_mn]:",
        f"            {p}_mn = {p}_j",
        f"    if {p}_mn != {p}_i:",
        f"        {p}_tw = {w}[{p}_i]",
        f"        {w}[{p}_i] = {w}[{p}_mn]",
        f"        {w}[{p}_mn] = {p}_tw",
        f"        for {p}_k in range({n}):",
        f"            {p}_tv = {v}[{p}_k, {p}_i]",
        f"            {v}[{p}_k, {p}_i] = {v}[{p}_k, {p}_mn]",
        f"            {v}[{p}_k, {p}_mn] = {p}_tv",
        f"{y} = {v}",
    ]


def eigh_stmts(
    w: str,
    v: str,
    a: str,
    b: str | None,
    lo: str,
    hi: str,
    p: str,
    native_std: bool = False,
    is_real: bool = False,
    w_dtype: str | None = None,
) -> list[str]:
    """Source lines for ``w, v = eigh(a[, b])[subset lo:hi]`` (ascending).

    The generalized Hermitian problem ``a x = w b x`` reduces to standard form
    via the Cholesky factor of ``b`` (``b = L L^H``): ``C = L^-1 a L^-H`` is
    Hermitian with the same eigenvalues, and its eigenvectors back-transform
    as ``x = L^-H y``. ``cholesky``/``inv``/``@`` stay ``np.linalg``/matmul for
    native backends (numba/dace) and are lowered by :data:`LINALG_HOIST` for
    pythran. The standard eigh is the self-contained Jacobi above, unless
    ``native_std`` (backends whose ``np.linalg.eigh`` handles standard
    complex-Hermitian natively -- jax), which emits a single
    ``np.linalg.eigh`` call instead. Validated vs scipy ~1e-15.

    ``is_real`` -- see :func:`eigh_jacobi_lines` -- is only meaningful when
    ``b`` is None: a generalized problem's reduced ``C`` is complex the moment
    either operand is, so the caller must not set it with ``b`` present."""
    if b is not None:
        pre = [
            f"{p}_L = np.linalg.cholesky({b})",
            f"{p}_Li = np.linalg.inv({p}_L)",
            f"{p}_C = {p}_Li @ {a} @ {p}_Li.conj().T",
        ]
        cname = f"{p}_C"
    else:
        pre = [f"{p}_C = {a}.copy()"]
        cname = f"{p}_C"
    std = (
        [f"{p}_wa, {p}_ya = np.linalg.eigh({cname})"]
        if native_std
        else eigh_jacobi_lines(f"{p}_wa", f"{p}_ya", cname, f"{a}.shape[0]", p, is_real=is_real, w_dtype=w_dtype)
    )
    lines = pre + std
    xname = f"{p}_xa" if b is not None else f"{p}_ya"
    if b is not None:
        lines.append(f"{p}_xa = {p}_Li.conj().T @ {p}_ya")
    if lo == "None":  # whole spectrum -> bare name (a ``[None:None]`` slice trips the C lowering)
        lines += [f"{w} = {p}_wa", f"{v} = {xname}"]
    else:
        lines += [f"{w} = {p}_wa[{lo}:{hi}]", f"{v} = {xname}[:, {lo}:{hi}]"]
    return lines


def eigh_c_stmts(
    w: str,
    v: str | None,
    a: str,
    b: str | None,
    lo: str,
    hi: str,
    p: str,
    eigenvalues_only: bool = False,
    is_real: bool = False,
    w_dtype: str | None = None,
) -> list[str]:
    """Fully self-contained loop lowering of standard/generalized complex-
    Hermitian ``eigh`` for the C/Fortran backends, which have no ``np.linalg``
    and no matmul lowering for the ``L^-H`` conjugate-transpose operand. Emits
    explicit loops only: complex-Hermitian Cholesky ``b = L L^H``, the lower-
    triangular inverse ``L^-1`` by forward substitution, the two matmuls
    ``C = L^-1 a L^-H``, the cyclic complex Jacobi, and the back-transform
    ``x = L^-H y``. Matmul outputs are pre-zeroed and ``+=``-accumulated; a
    complex zero is ``z - z``. Validated vs scipy ~1e-15.

    ``eigenvalues_only`` (``np.linalg.eigvalsh``) binds only the ascending
    eigenvalue vector ``w``: the same Jacobi sweep runs, but the ``L^-H``
    back-transform and eigenvector output ``v`` are dropped (``v`` is
    ``None``). numpy has no generalized eigvalsh, so this path always has
    ``b`` None.

    ``is_real`` -- see :func:`eigh_jacobi_lines` -- only applies to the
    standard (``b`` is None) form: the generalized branch's ``Cm`` is always
    built to a complex-capable ``.dtype`` (``a.dtype``/``b.dtype`` propagate
    through the Cholesky/matmul lines unconditionally), so the caller must not
    set it with ``b`` present."""
    n = f"{a}.shape[0]"
    lines: list[str] = []
    if b is not None:
        L, Li = f"{p}_L", f"{p}_Li"
        lines += cholesky_lines(L, b, n, f"{p}c", hermitian=True)
        lines += [  # explicit lower-triangular inverse L^-1 by forward substitution
            f"{Li} = np.zeros(({n}, {n}), {b}.dtype)",
            f"for {p}_ij in range({n}):",
            f"    {Li}[{p}_ij, {p}_ij] = 1.0 / {L}[{p}_ij, {p}_ij]",
            f"    for {p}_ii in range({p}_ij + 1, {n}):",
            f"        {p}_acc = {L}[{p}_ii, {p}_ii] - {L}[{p}_ii, {p}_ii]",
            f"        for {p}_ik in range({p}_ij, {p}_ii):",
            f"            {p}_acc += {L}[{p}_ii, {p}_ik] * {Li}[{p}_ik, {p}_ij]",
            f"        {Li}[{p}_ii, {p}_ij] = -{p}_acc / {L}[{p}_ii, {p}_ii]",
        ]
        # ``Tm`` / ``Cm`` (not ``T`` / ``C``): Fortran is case-insensitive, so a
        # matrix named ``T`` would collide with the Jacobi rotation scalar ``t``
        # (tangent) and ``C`` with ``c`` (cosine) -- the emitter would silently
        # drop one declaration and the body would index a scalar.
        T, C = f"{p}_Tm", f"{p}_Cm"
        lines += [  # Tm = Li @ a
            f"{T} = np.zeros(({n}, {n}), {b}.dtype)",
            f"for {p}_ti in range({n}):",
            f"    for {p}_tj in range({n}):",
            f"        for {p}_tk in range({n}):",
            f"            {T}[{p}_ti, {p}_tj] += {Li}[{p}_ti, {p}_tk] * {a}[{p}_tk, {p}_tj]",
        ]
        lines += [  # C = T @ Li^H  (Li^H[k, l] = conj(Li[l, k]))
            f"{C} = np.zeros(({n}, {n}), {b}.dtype)",
            f"for {p}_ci in range({n}):",
            f"    for {p}_cj in range({n}):",
            f"        for {p}_ck in range({n}):",
            f"            {C}[{p}_ci, {p}_cj] += {T}[{p}_ci, {p}_ck] * np.conj({Li}[{p}_cj, {p}_ck])",
        ]
        cname = C
    else:
        # ``Cm`` (not ``C``): Fortran is case-insensitive, so a matrix named ``C``
        # collides with the Jacobi cosine scalar ``c`` -- the emitter would reject
        # the second declaration. (The generalized branch above uses ``Cm`` too.)
        cname = f"{p}_Cm"
        # Explicit ``np.zeros`` allocation + element copy (NOT ``np.ascontiguousarray``,
        # whose copy-loop lowering leaves the fresh RUNTIME-shaped target unallocated --
        # a NULL write in the Jacobi). Mirrors the generalized branch's ``np.zeros``
        # temps. The Jacobi mutates ``Cm`` in place, so the input ``a`` must not alias it.
        lines += [
            f"{cname} = np.zeros(({n}, {n}), {a}.dtype)",
            f"for {p}_ci in range({n}):",
            f"    for {p}_cj in range({n}):",
            f"        {cname}[{p}_ci, {p}_cj] = {a}[{p}_ci, {p}_cj]",
        ]
    lines += eigh_jacobi_lines(f"{p}_wa", f"{p}_ya", cname, n, p, is_real=is_real, w_dtype=w_dtype)
    if eigenvalues_only:  # eigvalsh: only the eigenvalue vector, no back-transform / U output
        lines.append(f"{w} = {p}_wa" if lo == "None" else f"{w} = {p}_wa[{lo}:{hi}]")
        return lines
    if b is not None:
        X = f"{p}_X"
        lines += [  # back-transform x = Li^H @ ya
            f"{X} = np.zeros(({n}, {n}), {b}.dtype)",
            f"for {p}_xi in range({n}):",
            f"    for {p}_xj in range({n}):",
            f"        for {p}_xk in range({n}):",
            f"            {X}[{p}_xi, {p}_xj] += np.conj({Li}[{p}_xk, {p}_xi]) * {p}_ya[{p}_xk, {p}_xj]",
        ]
        xname = X
    else:
        xname = f"{p}_ya"
    if lo == "None":  # whole spectrum -> bare name (a ``[None:None]`` slice trips the C lowering)
        lines += [f"{w} = {p}_wa", f"{v} = {xname}"]
    else:
        lines += [f"{w} = {p}_wa[{lo}:{hi}]", f"{v} = {xname}[:, {lo}:{hi}]"]
    return lines


def eigh_operand_is_real(a_node: ast.AST, b_node: ast.AST | None, dtypes: dict[str, str]) -> bool:
    """True iff every eigh operand present is PROVABLY non-complex per ``dtypes``
    (a name -> dtype-KIND table, :func:`dtype_kind`'s convention), so
    :func:`eigh_jacobi_lines` can drop its ``.real``/``.imag`` accessors. An
    UNKNOWN kind is not proof of real -- it keeps the historic, always-correct
    complex path, matching every other dtype-gated desugar in this module."""

    def known_real(node: ast.AST) -> bool:
        kind = dtype_kind(node, dtypes)
        return kind is not None and kind != "complex"

    return known_real(a_node) and (b_node is None or known_real(b_node))


class EighLoopRewriter(ast.NodeTransformer):
    """Rewrite ``w, v = eigh(a[, b], subset_by_index=[lo, hi])`` (np.linalg /
    scipy.linalg / an imported alias) to the fully self-contained loop lowering
    (:func:`eigh_c_stmts`) for the C/Fortran frontend, which has no ``np.linalg``.
    Applied to the whole module tree (helpers included) BEFORE kernel inlining, so
    the ``_sci_eigh`` alias import is still in scope. A non-Name operand is
    materialised first.

    ``dtypes`` is the declared KIND table (manifest arrays and preset scalars). This runs before
    helper inlining, so a helper's names are known only through ``kind_tables``
    (:func:`module_kind_tables`); everything else stays the safe unknown."""

    def __init__(
        self,
        alias_names: set,
        dtypes: dict[str, str],
        kind_tables: Mapping[str, dict[str, str]] | None = None,
        array_dtypes: dict[str, str] | None = None,
    ) -> None:
        self.alias_names = alias_names
        self.declared = dtypes
        self.dtypes = dtypes
        #: Per-function kinds (:func:`module_kind_tables`); ``None`` applies ``dtypes`` to every function.
        self.kind_tables = kind_tables
        #: Declared RAW array dtypes (widths, not kinds), for the eigenvalue dtype a complex
        #: operand cannot spell in emitted source (:func:`eigh_w_dtype`).
        self.array_dtypes = array_dtypes or {}
        self._ctr = 0

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Propagate the declared kinds across this function's own assignments, then rewrite it.

        The manifest names only the kernel's arrays, and an ``eigh`` operand is routinely a LOCAL
        built from them -- rayleigh_ritz_rotation's is ``M = Linv @ h_sub @ Linv.T``, three
        assignments and two factorisations away from anything declared. A table that stops at the
        declared names reads every such operand as unknown, so the real branch was unreachable for
        exactly the kernels that need it.

        With ``kind_tables`` a helper sees only what :func:`module_kind_tables` proved across its call
        sites: a helper parameter sharing a kernel array's name is a different value, and being wrong
        in the "real" direction DROPS an imaginary part. A function the tables miss starts from nothing.
        Without them the caller named no kernel, so the declared table applies everywhere."""
        outer = self.dtypes
        if self.kind_tables is None:
            self.dtypes = dtype_table_(node, self.declared)
        elif node.name in self.kind_tables:
            self.dtypes = self.kind_tables[node.name]
        else:
            self.dtypes = dtype_table_(node, {})
        self.generic_visit(node)
        self.dtypes = outer
        return node

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        if len(node.targets) != 1:
            return node
        hit = eigh_call_kind(node.value, self.alias_names)
        if hit is None:
            return node
        kind, a_node, b_node, kw = hit
        tgt = node.targets[0]
        # ``w, v = eigh(...)`` (eigenpair) or ``w = eigvalsh(...)`` (a single Name
        # target -- eigenvalues only, no eigenvector back-transform / U output).
        if isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2 and all(isinstance(e, ast.Name) for e in tgt.elts):
            w, v = tgt.elts[0].id, tgt.elts[1].id
        elif kind == "eigvalsh" and isinstance(tgt, ast.Name):
            w, v = tgt.id, None
        else:
            return node
        p = f"__eigh{self._ctr}"
        self._ctr += 1
        pre: list[str] = []

        def name_of(nd, tag):
            if isinstance(nd, ast.Name):
                return nd.id
            pre.append(f"{p}_{tag} = np.ascontiguousarray({ast.unparse(nd)})")
            return f"{p}_{tag}"

        aname = name_of(a_node, "a")
        bname = name_of(b_node, "b") if b_node is not None else None
        s = kw.get("subset_by_index")
        if isinstance(s, (ast.List, ast.Tuple)) and len(s.elts) == 2:
            lo, hi = ast.unparse(s.elts[0]), f"({ast.unparse(s.elts[1])}) + 1"
        else:
            lo, hi = "None", "None"
        # Standard form only (b_node None): a generalized C is complex-capable regardless of a/b's
        # own dtype (see _eigh_c_stmts).
        is_real = b_node is None and eigh_operand_is_real(a_node, b_node, self.dtypes)
        w_dtype = eigh_w_dtype(is_real, (aname, bname), self.array_dtypes)
        lines = pre + eigh_c_stmts(
            w, v, aname, bname, lo, hi, p, eigenvalues_only=(v is None), is_real=is_real, w_dtype=w_dtype
        )
        return [ast.copy_location(st, node) for st in ast.parse("\n".join(lines)).body]


class EighCallHoister(ast.NodeTransformer):
    """Materialise an ``eigh`` / ``eigvalsh`` call that appears NESTED in an
    expression -- ``float(np.linalg.eigvalsh(T).max()) + beta`` in LS3DF's Lanczos
    upper-bound -- into its own ``__eigv<k> = <call>`` statement, so the direct-assign
    :class:`EighLoopRewriter` can lower it. A call that is already the RHS of an
    eligible eigh-assign (:func:`is_eigh_assign_target`) is left in place. Runs on the
    whole module (helpers included) BEFORE the loop rewriter, mirroring its scope."""

    def __init__(self, alias_names: set) -> None:
        self.alias_names = alias_names
        self.pre: list[ast.stmt] = []
        self._ctr = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if eigh_call_kind(node, self.alias_names) is None:
            return node
        self._ctr += 1
        name = f"__eigv{self._ctr}"
        self.pre.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=node))
        return ast.Name(id=name, ctx=ast.Load())

    def flush(self, node: ast.stmt):
        # An eligible direct eigh-assign stays whole -- descending would hoist its own
        # RHS call and hide it from the loop rewriter.
        if is_eigh_assign_target(node, self.alias_names):
            return node
        saved = self.pre
        self.pre = []
        self.generic_visit(node)
        pre = self.pre
        self.pre = saved
        if not pre:
            return node
        for s in pre:
            ast.copy_location(s, node)
            ast.fix_missing_locations(s)
        return pre + [node]

    def visit_stmts(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for s in stmts:
            r = self.visit(s)
            if r is None:
                continue
            out.extend(r if isinstance(r, list) else [r])
        return out

    def visit_While(self, node: ast.While) -> ast.AST:
        # Do NOT hoist an eigh call out of the loop CONDITION: a ``__eigv`` temp
        # emitted before the loop would freeze a value the ``while`` test must
        # recompute each iteration (``while eigvalsh(A).max() > tol: A = update(A)``).
        # Leave ``node.test`` unvisited; only the body / else statements hoist locally.
        node.body = self.visit_stmts(node.body)
        node.orelse = self.visit_stmts(node.orelse)
        return node

    visit_Assign = flush
    visit_AugAssign = flush
    visit_Expr = flush
    visit_Return = flush
    visit_If = flush
    visit_For = flush


class EighInline(ast.NodeTransformer):
    """Lower ``w, v = eigh(a[, b], subset_by_index=[lo, hi])`` -- standard or
    generalized complex-Hermitian ``eigh`` (numpy or scipy, incl. an imported
    alias) -- to a Cholesky-reduced complex Jacobi loop nest (see
    :func:`eigh_stmts`). Handles the tuple-target eigenpair form and the
    eigenvalues-only single-target form (``np.linalg.eigvalsh`` or
    ``eigh(..., eigvals_only=True)``); a non-``Name`` operand is materialised first.
    Runs BEFORE :data:`LINALG_HOIST` so the cholesky/inv it emits are themselves
    lowered for pythran.

    ``dtypes`` is the per-function dtype-KIND table (:func:`dtype_table_`),
    consulted the same way :func:`hoist_cholesky` consults it for its ``hermitian``
    flag -- see :func:`eigh_operand_is_real`."""

    def __init__(
        self,
        ranks: dict[str, int],
        alias_names: set,
        dtypes: dict[str, str],
        array_dtypes: dict[str, str] | None = None,
    ) -> None:
        self.ranks = ranks
        self.alias_names = alias_names
        self.dtypes = dtypes
        #: Declared array dtypes, for the eigenvalue width a complex operand cannot spell.
        self.array_dtypes = array_dtypes or {}
        self.changed = False
        self._ctr = 0

    def subset(self, kw) -> tuple:
        """``subset_by_index=[lo, hi]`` (inclusive) -> slice bounds ``(lo, hi+1)``
        strings; whole spectrum -> ``('None', 'None')`` (a full ``[:]`` slice)."""
        s = kw.get("subset_by_index")
        if isinstance(s, (ast.List, ast.Tuple)) and len(s.elts) == 2:
            return ast.unparse(s.elts[0]), f"({ast.unparse(s.elts[1])}) + 1"
        return "None", "None"

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) != 1:
            return node
        hit = eigh_call_kind(node.value, self.alias_names)
        if hit is None:
            return node
        kind, a_node, b_node, kw = hit
        tgt = node.targets[0]
        evo = kw.get("eigvals_only")
        # ``eigvalsh`` (a distinct eigenvalues-only op) and ``eigh(..., eigvals_only=True)``
        # (scipy's flag) both bind a single eigenvalue vector -- no eigenvectors.
        eigvals_only = kind == "eigvalsh" or (isinstance(evo, ast.Constant) and evo.value is True)
        # Target: ``w, v = eigh(...)`` (tuple) or ``w = eigvalsh(...)`` / ``w = eigh(..., eigvals_only=True)``.
        if isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2 and all(isinstance(e, ast.Name) for e in tgt.elts):
            w, v = tgt.elts[0].id, tgt.elts[1].id
        elif isinstance(tgt, ast.Name) and eigvals_only:
            w, v = tgt.id, None
        else:
            return node
        if expr_rank(a_node, self.ranks) not in (2, None) or (
            b_node is not None and expr_rank(b_node, self.ranks) not in (2, None)
        ):
            return node
        p = f"__eigh{self._ctr}"
        self._ctr += 1
        pre: list[str] = []

        def name_of(nd, tag):
            if isinstance(nd, ast.Name):
                return nd.id
            pre.append(f"{p}_{tag} = np.ascontiguousarray({ast.unparse(nd)})")
            return f"{p}_{tag}"

        aname = name_of(a_node, "a")
        bname = name_of(b_node, "b") if b_node is not None else None
        lo, hi = self.subset(kw)
        vtmp = v if v is not None else f"{p}_vdrop"
        # Standard form only (b_node None): a generalized C is complex-capable regardless of a/b's
        # own dtype (see _eigh_stmts).
        is_real = b_node is None and eigh_operand_is_real(a_node, b_node, self.dtypes)
        w_dtype = eigh_w_dtype(is_real, (aname, bname), self.array_dtypes)
        lines = pre + eigh_stmts(w, vtmp, aname, bname, lo, hi, p, is_real=is_real, w_dtype=w_dtype)
        self.changed = True
        return [ast.copy_location(s, node) for s in ast.parse("\n".join(lines)).body]

    def visit_AugAssign(self, node):
        return node

    def visit_Return(self, node):
        return node

    def visit_Expr(self, node):
        return node
