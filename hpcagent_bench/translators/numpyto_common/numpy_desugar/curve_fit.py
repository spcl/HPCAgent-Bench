"""``scipy.optimize.curve_fit`` lowered to a Levenberg-Marquardt loop nest."""

import ast

from hpcagent_bench.translators.numpyto_common.numpy_desugar.common import DesugarError, as_stmts
from hpcagent_bench.translators.numpyto_common.numpy_desugar.constants import fd_step, working_float_dtype
from hpcagent_bench.translators.numpyto_common.numpy_desugar.lists import fold_list_accumulators


def curve_fit_lm_lines(
    popt: str, f: str, x: str, y: str, p0: str, pfx: str, iters: int, precision: str | None = None
) -> list[str]:
    """Source lines for a naive Levenberg-Marquardt fit replacing ``curve_fit``.

    ``scipy.optimize.curve_fit(f, x, y, p0=...)`` with no bounds/sigma is an
    unconstrained nonlinear least-squares fit; MINPACK's ``lmdif`` (what scipy
    calls) is a trust-region LM over a forward-difference Jacobian. Emits the
    textbook damped-normal-equations form::

        r = f(x, p) - y                       # residual
        J[:, c] = (f(x, p + h e_c) - f(x, p)) / h   # forward-difference Jacobian
        (J^T J + lam * diag(J^T J)) dp = -J^T r
        accept dp if it lowers ||r||^2 (lam /= 10), else keep p (lam *= 10)

    A rejected step isn't retried within the trip -- the next trip re-solves at
    the same p with a larger lam, keeping the nest a flat fixed-trip loop with
    no inner convergence search.

    Two choices make the result agree with scipy's to the harness's 1e-9, not
    just the same optimum to fitting accuracy:

    * Step ``h = sqrt(eps) * |p_j|`` is MINPACK's own (:func:`fd_step`, over
      the WORKING precision -- an fp64 step vanishes at fp32). A
      finite-difference Jacobian shifts the stationary point from ``J^T r=0``
      to ``J~^T r=0``; sharing the step makes both solvers inherit the SAME
      shift.
    * A fixed trip count well past convergence, not a dynamic break -- static
      backends want a static trip count, and surplus trips are no-ops once the
      step is rejected at the ``lam`` ceiling.

    ``np.linalg.solve`` is left to the existing Gauss-Jordan expander; the
    damping keeps the system positive definite, so pivoting never meets a
    singular column in practice.

    Generated names differ by more than case: Fortran identifiers are
    case-insensitive, so ``_J`` beside ``_j`` would collide into one symbol.
    """
    n, m = f"{p0}.shape[0]", f"{y}.shape[0]"
    i, c, a, b, it = f"{pfx}_i", f"{pfx}_c", f"{pfx}_a", f"{pfx}_b", f"{pfx}_it"
    # Every array below is this fit's own scratch, so it is allocated at the WORKING precision
    # the step size already tracks -- not a hardcoded fp64, which would run an fp32 fit at
    # double width and then narrow on the store into popt's fp32 target.
    wf = f"np.{working_float_dtype(precision)}"
    return [
        f"{popt} = np.zeros(({n},), dtype={wf})",
        f"{pfx}_jac = np.zeros(({m}, {n}), dtype={wf})",
        f"{pfx}_ata = np.zeros(({n}, {n}), dtype={wf})",
        f"{pfx}_atad = np.zeros(({n}, {n}), dtype={wf})",
        f"{pfx}_grad = np.zeros(({n},), dtype={wf})",
        f"{pfx}_rhs = np.zeros(({n},), dtype={wf})",
        f"{pfx}_pt = np.zeros(({n},), dtype={wf})",
        f"{pfx}_r = np.zeros(({m},), dtype={wf})",
        f"for {i} in range({n}):",
        f"    {popt}[{i}] = {p0}[{i}]",
        f"{pfx}_lam = 0.001",
        f"{pfx}_f0 = {f}({x}, {popt})",
        f"{pfx}_ssq = 0.0",
        f"for {i} in range({m}):",
        f"    {pfx}_r[{i}] = {pfx}_f0[{i}] - {y}[{i}]",
        f"    {pfx}_ssq = {pfx}_ssq + {pfx}_r[{i}] * {pfx}_r[{i}]",
        f"for {it} in range({iters}):",
        f"    for {c} in range({n}):",
        f"        {pfx}_h = {fd_step(precision)} * np.abs({popt}[{c}])",
        f"        if {pfx}_h == 0.0:",
        f"            {pfx}_h = {fd_step(precision)}",
        f"        for {i} in range({n}):",
        f"            {pfx}_pt[{i}] = {popt}[{i}]",
        f"        {pfx}_pt[{c}] = {popt}[{c}] + {pfx}_h",
        f"        {pfx}_fp = {f}({x}, {pfx}_pt)",
        f"        for {i} in range({m}):",
        f"            {pfx}_jac[{i}, {c}] = ({pfx}_fp[{i}] - {pfx}_f0[{i}]) / {pfx}_h",
        f"    for {a} in range({n}):",
        f"        {pfx}_ga = 0.0",
        f"        for {i} in range({m}):",
        f"            {pfx}_ga = {pfx}_ga + {pfx}_jac[{i}, {a}] * {pfx}_r[{i}]",
        f"        {pfx}_grad[{a}] = {pfx}_ga",
        f"        for {b} in range({n}):",
        f"            {pfx}_ab = 0.0",
        f"            for {i} in range({m}):",
        f"                {pfx}_ab = {pfx}_ab + {pfx}_jac[{i}, {a}] * {pfx}_jac[{i}, {b}]",
        f"            {pfx}_ata[{a}, {b}] = {pfx}_ab",
        f"    for {a} in range({n}):",
        f"        for {b} in range({n}):",
        f"            {pfx}_atad[{a}, {b}] = {pfx}_ata[{a}, {b}]",
        f"        {pfx}_atad[{a}, {a}] = {pfx}_ata[{a}, {a}] + {pfx}_lam * {pfx}_ata[{a}, {a}]",
        f"        {pfx}_rhs[{a}] = -{pfx}_grad[{a}]",
        f"    {pfx}_step = np.linalg.solve({pfx}_atad, {pfx}_rhs)",
        f"    for {i} in range({n}):",
        f"        {pfx}_pt[{i}] = {popt}[{i}] + {pfx}_step[{i}]",
        f"    {pfx}_ft = {f}({x}, {pfx}_pt)",
        f"    {pfx}_ssqt = 0.0",
        f"    for {i} in range({m}):",
        f"        {pfx}_ssqt = {pfx}_ssqt + ({pfx}_ft[{i}] - {y}[{i}]) * ({pfx}_ft[{i}] - {y}[{i}])",
        f"    if {pfx}_ssqt < {pfx}_ssq:",
        f"        for {i} in range({n}):",
        f"            {popt}[{i}] = {pfx}_pt[{i}]",
        f"        for {i} in range({m}):",
        f"            {pfx}_f0[{i}] = {pfx}_ft[{i}]",
        f"            {pfx}_r[{i}] = {pfx}_ft[{i}] - {y}[{i}]",
        f"        {pfx}_ssq = {pfx}_ssqt",
        f"        {pfx}_lam = {pfx}_lam * 0.1",
        f"        if {pfx}_lam < 1e-14:",
        f"            {pfx}_lam = 1e-14",
        f"    else:",
        f"        {pfx}_lam = {pfx}_lam * 10.0",
        f"        if {pfx}_lam > 10000000000.0:",
        f"            {pfx}_lam = 10000000000.0",
    ]


#: ``curve_fit`` keywords the LM lowering reproduces exactly. ``maxfev`` bounds
#: MINPACK's eval budget; the fixed trip count converges far inside it, so
#: honouring the number is meaningless. A keyword that CHANGES the objective
#: (bounds/sigma/absolute_sigma) or derivative (jac) is refused instead of
#: silently fitting something else.
CURVE_FIT_IGNORED_KW = frozenset({"maxfev", "p0", "method", "full_output"})


#: Fixed LM trip count. The fit converges well inside it -- 200 trips reproduce
#: the 100-trip parameters bit-for-bit, each surplus trip a rejected step (lambda
#: at ceiling) -- so the margin costs time, not accuracy, and buys insensitivity
#: to the starting guess.
CURVE_FIT_ITERS = 100


def curve_fit_call(node: ast.AST) -> ast.Call | None:
    """A ``curve_fit(...)`` call -- bare, ``scipy.optimize.``- or ``optimize.``-
    qualified -- else None."""
    if not isinstance(node, ast.Call):
        return None
    fn = node.func
    if isinstance(fn, ast.Name) and fn.id == "curve_fit":
        return node
    if isinstance(fn, ast.Attribute) and fn.attr == "curve_fit":
        return node
    return None


def curve_fit_guess(call: ast.Call) -> ast.expr | None:
    """``p0`` of a ``curve_fit`` call, by keyword or as the fourth positional argument."""
    guess = next((kw.value for kw in call.keywords if kw.arg == "p0"), None)
    if guess is None and len(call.args) >= 4:
        return call.args[3]
    return guess


class CurveFitRewriter(ast.NodeTransformer):
    """``popt, pcov = curve_fit(f, x, y, p0=g)`` -> a naive LM loop nest.

    Static backends have no scipy; the fit is an unconstrained nonlinear
    least-squares problem over a smooth analytic model, so it lowers to plain
    loops + arithmetic (:func:`curve_fit_lm_lines`), leaving the linear solve
    to the existing ``np.linalg.solve`` expander.

    Model ``f`` is a nested/module-level ``def f(grid, *p)``: curve_fit calls
    it as ``f(x, *popt)``, so its varargs tuple IS the parameter vector.
    Rebinding ``*p`` to a single ndarray parameter matches curve_fit's own
    contract, turning ``f`` into an ordinary one-array-in-one-array-out helper
    the inliner already handles (``npeaks`` resolves free in the inlined-into
    scope). Negative constant indices into ``p`` (``p[-1]``, the shared
    baseline) are rewritten against the now-known parameter count, which the
    emitters cannot fold themselves.

    ``pcov`` is NOT computed: the corpus kernel binds it to ``_`` and never
    reads it. A live ``pcov`` raises rather than silently emitting nothing.
    """

    def __init__(self, tree: ast.Module, kernel: ast.FunctionDef, precision: str | None = None) -> None:
        self.tree = tree
        self.kernel = kernel
        self.ctr = 0
        self.changed = False
        #: Working float precision, for the LM's finite-difference step (:func:`fd_step`).
        self.precision = precision
        #: ``(popt_name, parameter-count expression)`` per lowered fit.
        self.fitted: list[tuple[str, str]] = []

    def find_model(self, name: str) -> ast.FunctionDef | None:
        for scope in (self.kernel, self.tree):
            for node in ast.walk(scope):
                if isinstance(node, ast.FunctionDef) and node.name == name and node is not self.kernel:
                    return node
        return None

    @staticmethod
    def rebind_varargs(fdef: ast.FunctionDef, nexpr: str) -> None:
        """``def f(grid, *p)`` -> ``def f(grid, p)`` with ``p[-k]`` -> ``p[nexpr - k]``."""
        vp = fdef.args.vararg
        if vp is None:
            return
        fdef.args.args.append(ast.arg(arg=vp.arg))
        fdef.args.vararg = None
        for sub in ast.walk(fdef):
            if (
                isinstance(sub, ast.Subscript)
                and isinstance(sub.value, ast.Name)
                and sub.value.id == vp.arg
                and isinstance(sub.slice, ast.UnaryOp)
                and isinstance(sub.slice.op, ast.USub)
                and isinstance(sub.slice.operand, ast.Constant)
            ):
                sub.slice = ast.parse(f"{nexpr} - {sub.slice.operand.value}", mode="eval").body
        ast.fix_missing_locations(fdef)

    def visit_Assign(self, node: ast.Assign):
        call = curve_fit_call(node.value)
        if call is None or len(node.targets) != 1:
            return node
        tgt = node.targets[0]
        if isinstance(tgt, ast.Tuple):
            if len(tgt.elts) != 2 or not all(isinstance(e, ast.Name) for e in tgt.elts):
                raise DesugarError(f"curve_fit: unsupported target {ast.unparse(tgt)}")
            if tgt.elts[1].id != "_":
                raise DesugarError(
                    "curve_fit: the pcov covariance output is not computed by the LM "
                    f"lowering, but {tgt.elts[1].id!r} binds it"
                )
            popt = tgt.elts[0].id
        elif isinstance(tgt, ast.Name):
            popt = tgt.id
        else:
            raise DesugarError(f"curve_fit: unsupported target {ast.unparse(tgt)}")
        for kw in call.keywords:
            if kw.arg not in CURVE_FIT_IGNORED_KW:
                raise DesugarError(
                    f"curve_fit: keyword {kw.arg!r} changes the fit; the LM lowering "
                    "only reproduces the unweighted, unbounded, FD-Jacobian form"
                )
        p0 = curve_fit_guess(call)
        if len(call.args) < 3 or p0 is None:
            raise DesugarError("curve_fit: need f, xdata, ydata and an explicit p0")
        f, x, y = call.args[0], call.args[1], call.args[2]
        if not all(isinstance(v, ast.Name) for v in (f, x, y, p0)):
            raise DesugarError("curve_fit: f / xdata / ydata / p0 must be plain names")
        model = self.find_model(f.id)
        if model is None:
            raise DesugarError(f"curve_fit: model {f.id!r} is not a def in this module")
        pfx = f"__lm{self.ctr}"
        self.ctr += 1
        nexpr = f"{p0.id}.shape[0]"
        self.rebind_varargs(model, nexpr)
        self.fitted.append((popt, nexpr))
        self.changed = True
        lines = curve_fit_lm_lines(popt, f.id, x.id, y.id, p0.id, pfx, CURVE_FIT_ITERS, self.precision)
        return ast.parse("\n".join(lines)).body


class NegParamIndexFold(ast.NodeTransformer):
    """``popt[-k]`` -> ``popt[<len> - k]`` for a fitted parameter vector.

    The kernel reads the fitted baseline as ``offset[0] = popt[-1]``. ``popt``
    is created by the LM lowering with a symbolic length, so the emitters (which
    fold a negative index only against a STATIC extent) cannot resolve it.
    """

    def __init__(self, name: str, nexpr: str) -> None:
        self.name = name
        self.nexpr = nexpr

    def visit_Subscript(self, node: ast.Subscript):
        self.generic_visit(node)
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == self.name
            and isinstance(node.slice, ast.UnaryOp)
            and isinstance(node.slice.op, ast.USub)
            and isinstance(node.slice.operand, ast.Constant)
        ):
            node.slice = ast.parse(f"{self.nexpr} - {node.slice.operand.value}", mode="eval").body
            ast.fix_missing_locations(node)
        return node


def rewrite_curve_fit(tree: ast.Module, kernel: ast.FunctionDef, precision: str | None = None) -> None:
    """Lower every ``curve_fit`` in ``kernel`` to a naive LM loop nest, in place.

    Runs BEFORE helper inlining (like the eigh rewriter) so the model ``def``
    is still distinct to rebind; the LM's calls to it are inlined afterwards by
    the ordinary helper machinery.

    ``precision`` is the working float type, needed HERE at the source rewrite
    because the LM's finite-difference step is a numerical constant baked into
    the emitted body -- ``apply_precision`` later remaps dtype tables only and
    cannot reach a literal (see :func:`fd_step`). ``None`` keeps the fp64 rule.
    """
    fits = [n for n in ast.walk(kernel) if isinstance(n, ast.Call) and curve_fit_call(n) is not None]
    if not fits:
        return
    # The LM lines index the p0 vector, so it must be an array first -- even a bare display.
    guesses = [curve_fit_guess(call) for call in fits]
    fold_list_accumulators(kernel, frozenset(guess.id for guess in guesses if isinstance(guess, ast.Name)))
    rw = CurveFitRewriter(tree, kernel, precision)
    kernel.body = [s for stmt in kernel.body for s in as_stmts(rw.visit(stmt))]
    if not rw.changed:
        return
    # ``offset[0] = popt[-1]`` reads the fitted vector's tail; resolve it against
    # the parameter count now that the vector is an array of known length.
    for popt, nexpr in rw.fitted:
        NegParamIndexFold(popt, nexpr).visit(kernel)
    ast.fix_missing_locations(kernel)
