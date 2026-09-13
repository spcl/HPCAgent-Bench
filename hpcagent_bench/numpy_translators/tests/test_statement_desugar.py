"""Chained assignment and array iteration keep numpy's meaning on the native, dace and jax backends.

``a = b = v`` evaluates ``v`` once and, for an array, binds ONE buffer to both names; ``for x in arr``
walks the leading axis. The three backends share one pass for each (``numpyto_common.statement_desugar``).
"""

import ast
import json
import pathlib
import tempfile
import textwrap
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np

from _native_tu import build_run_c
from _op_oracle import _bench_info as bench_info
from numpyto_c.dace_emit import emit_dace
from numpyto_c.emit import emit_c
from numpyto_common.frontend import parse_kernel
from numpyto_common.ir import KernelIR
from numpyto_common.lowering import lower
from numpyto_common.statement_desugar import DesugarArrayIteration, SplitChainedAssign
from numpyto_jax.core import emit_jax

#: Every array in these kernels is ``(N,)``, and ``N`` is this.
EXTENT = 4

#: Bumps the counter it is handed, so each call shows in the counter.
BUMP = "def bump(c):\n    c[0] = c[0] + 1.0\n    return c[0]\n\n\n"


def kernel(inputs: list[str], *body: str, helpers: str = "") -> str:
    """Numpy source of ``k(*inputs, out, N)``, one body line per entry of ``body``."""
    params = ", ".join([*inputs, "out", "N"])
    lines = "".join(f"    {line}\n" for line in body)
    return f"import numpy as np\n\n\n{helpers}def k({params}):\n{lines}"


def emitted(source: str, inputs: list[str], emit: Callable[[KernelIR], str]) -> str:
    """``emit`` applied to the parsed kernel ``k`` of ``source``."""
    arrays = [*inputs, "out"]
    info = bench_info("k", inputs, ["out"], dict.fromkeys(arrays, "(N,)"), {"N": EXTENT}, None)
    with tempfile.TemporaryDirectory() as scratch:
        folder = pathlib.Path(scratch)
        (folder / "k_numpy.py").write_text(source)
        (folder / "bench_info.json").write_text(json.dumps(info))
        return emit(parse_kernel(folder / "k_numpy.py", folder / "bench_info.json"))


def native(kir: KernelIR) -> str:
    return emit_c(lower(kir), fn_name="k")


def run_native(text: str, inputs: dict[str, list[float]]) -> list[float]:
    """Build the emitted C ``k`` with a driver that passes ``inputs`` and a zero ``out``; return ``out``."""
    declarations = "".join(
        "    double " + name + "[N] = {" + ", ".join(map(str, values)) + "};\n" for name, values in inputs.items()
    )
    call = ", ".join([*inputs, "out", "N"])
    driver = (
        "#include <stdio.h>\n"
        "int main(void) {\n"
        f"    enum {{ N = {EXTENT} }};\n"
        f"{declarations}"
        "    double out[N] = {0};\n"
        f"    k({call});\n"
        '    for (int i = 0; i < N; ++i) printf("%.17g\\n", out[i]);\n'
        "    return 0;\n"
        "}\n"
    )
    run = build_run_c(text, driver)
    assert run.returncode == 0, run.stderr
    return [float(value) for value in run.stdout.split()]


def run_jax(source: str, inputs: dict[str, list[float]], jit: bool) -> list[float]:
    """Translate ``k`` to jax, call it on ``inputs`` and a zero ``out``, and return ``out``."""
    namespace: dict[str, Any] = {}
    exec(compile(emit_jax(source, "k", jit=jit), "<jax>", "exec"), namespace)
    arguments = [jnp.asarray(values) for values in inputs.values()]
    return np.asarray(namespace["k"](*arguments, jnp.zeros(EXTENT), EXTENT)).tolist()


def kernel_program(text: str) -> ast.FunctionDef:
    """The emitted ``@dc.program`` named ``k``."""
    (program,) = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == "k"]
    return program


# --------------------------------------------------------------------------- #
# Native lowering: the repeated right-hand side gave each name its own buffer #
# and ran the value once per target.                                          #
# --------------------------------------------------------------------------- #


def test_a_write_through_one_chained_array_name_is_seen_through_the_other_natively() -> None:
    """numpy binds ``a`` and ``b`` to ONE zeros buffer, so ``b[0]`` reads the 1.0 written through ``a``."""
    text = emitted(kernel([], "a = b = np.zeros(N)", "a[0] = 1.0", "out[0] = b[0]"), [], native)
    assert text.count("malloc(") == 1, text
    assert run_native(text, {}) == [1.0, 0.0, 0.0, 0.0]


def test_a_chained_call_runs_once_natively() -> None:
    """``x = y = bump(cnt)`` calls ``bump`` once: the counter reads 1 and both names hold it."""
    source = kernel(["cnt"], "x = y = bump(cnt)", "out[0] = x + y", "out[1] = cnt[0]", helpers=BUMP)
    text = emitted(source, ["cnt"], native)
    assert text.count("bump(cnt") == 1, text
    assert run_native(text, {"cnt": [0.0] * EXTENT}) == [2.0, 1.0, 0.0, 0.0]


def test_a_chained_scalar_that_reads_its_own_target_binds_the_old_value_natively() -> None:
    """``s = t = s + 1.0`` with ``s == 2.0`` binds both to 3.0; a repeated sum would read the new ``s``."""
    text = emitted(kernel([], "s = 2.0", "s = t = s + 1.0", "out[0] = s", "out[1] = t"), [], native)
    assert run_native(text, {}) == [3.0, 3.0, 0.0, 0.0]


def test_an_array_alias_keeps_the_shared_buffer_after_the_other_name_is_rebound_natively() -> None:
    """Rebinding ``a`` leaves ``b`` on the buffer ``a`` wrote 1.0 into."""
    source = kernel([], "a = b = np.zeros(N)", "a[0] = 1.0", "a = np.ones(N)", "out[0] = b[0]", "out[1] = a[0]")
    assert run_native(emitted(source, [], native), {}) == [1.0, 1.0, 0.0, 0.0]


def test_an_array_alias_rebound_inside_a_loop_starts_from_the_shared_buffer_natively() -> None:
    """``v`` reads the 2.0 written through ``p`` on the first iteration, then its own rebinding."""
    source = kernel(
        [], "p = v = np.zeros(N)", "p[0] = 2.0", "for i in range(3):", "    out[i] = v[0]", "    v = p + 1.0"
    )
    assert run_native(emitted(source, [], native), {}) == [2.0, 3.0, 3.0, 0.0]


def test_a_chained_literal_gives_each_scalar_its_own_value_natively() -> None:
    text = emitted(kernel([], "s0 = s1 = 0.0", "s0 += 1.0", "out[0] = s0", "out[1] = s1"), [], native)
    assert run_native(text, {}) == [1.0, 0.0, 0.0, 0.0]


def test_array_enumerate_and_zip_iteration_walk_the_declared_extent_natively() -> None:
    source = kernel(
        ["a", "b"],
        "for v in a:",
        "    out[0] += v",
        "for i, v in enumerate(a, start=1):",
        "    out[1] += i * v",
        "for x, y in zip(a, b):",
        "    out[2] += x * y",
    )
    inputs = {"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0, 6.0, 7.0, 8.0]}
    assert run_native(emitted(source, ["a", "b"], native), inputs) == [10.0, 30.0, 70.0, 0.0]


# --------------------------------------------------------------------------- #
# dace: emitted text, since dace itself is not JIT-run here.                  #
# --------------------------------------------------------------------------- #


def test_dace_walks_an_array_through_an_index_over_its_declared_extent() -> None:
    """dace's frontend rejects ``for v in a``: the loop runs over ``range(N)`` and binds ``v`` from ``a``."""
    program = kernel_program(emitted(kernel(["a"], "for v in a:", "    out[0] += v"), ["a"], emit_dace))
    (loop,) = [node for node in ast.walk(program) if isinstance(node, ast.For)]
    declared = program.args.args[0].annotation
    assert declared is not None and isinstance(declared, ast.Subscript)
    assert ast.unparse(loop.iter) == f"range({ast.unparse(declared.slice)})"
    assert ast.unparse(loop.body[0]) == f"v = a[{ast.unparse(loop.target)}]"


def test_dace_binds_a_chained_array_to_one_name() -> None:
    """``a = b = np.zeros(N)`` is one container under ``a``; ``b`` never names a second one."""
    program = kernel_program(emitted(kernel([], "a = b = np.zeros(N)", "a[0] = 1.0", "out[0] = b[0]"), [], emit_dace))
    assert "b" not in {node.id for node in ast.walk(program) if isinstance(node, ast.Name)}
    assert "out[0] = a[0]" in [ast.unparse(stmt) for stmt in program.body]


# --------------------------------------------------------------------------- #
# jax: run the translation. Functional updates rebind the written name, so a  #
# temp shared by two names loses the write.                                   #
# --------------------------------------------------------------------------- #


def test_a_write_through_one_chained_array_name_is_seen_through_the_other_in_jax() -> None:
    source = kernel([], "a = b = np.zeros(N)", "a[0] = 1.0", "out[0] = b[0]")
    assert run_jax(source, {}, jit=False) == [1.0, 0.0, 0.0, 0.0]


def test_traced_jax_walks_an_array_by_index() -> None:
    source = kernel(["a"], "for v in a:", "    out[0] += v")
    assert "in a:" not in emit_jax(source, "k", jit=True)
    assert run_jax(source, {"a": [1.0, 2.0, 3.0, 4.0]}, jit=True) == [10.0, 0.0, 0.0, 0.0]


# --------------------------------------------------------------------------- #
# The shared passes on source text: what each backend above is handed.       #
# --------------------------------------------------------------------------- #


def function_tree(body: str) -> ast.Module:
    return ast.parse("def k():\n" + textwrap.indent(body, "    "))


def function_text(body: str) -> str:
    return ast.unparse(function_tree(body))


def split(body: str, repeat_literals: bool = False, seed_ranks: dict[str, int] | None = None) -> str:
    """``k``'s source after the chained-assign split, temps named ``t0``, ``t1``, ..."""
    tree = function_tree(body)
    SplitChainedAssign(lambda ordinal: f"t{ordinal}", repeat_literals, seed_ranks).visit(tree)
    return ast.unparse(tree)


def test_a_chained_scalar_is_evaluated_once_into_a_temp() -> None:
    assert split("s = t = x * 2.0\n", seed_ranks={"x": 0}) == function_text("t0 = x * 2.0\ns = t0\nt = t0\n")


def test_a_chained_literal_goes_through_the_temp() -> None:
    assert split("s = t = 0.0\n") == function_text("t0 = 0.0\ns = t0\nt = t0\n")


def test_a_chained_literal_repeats_where_the_backend_asks_for_it() -> None:
    assert split("s = t = 0.0\n", repeat_literals=True) == function_text("s = 0.0\nt = 0.0\n")


def test_a_chained_call_is_evaluated_once() -> None:
    """A helper call may act on its arguments, so it runs once and the other name is renamed to its value."""
    assert split("x = y = bump(c)\nout = x + y\n") == function_text("x = bump(c)\nout = x + x\n")


def test_a_chained_array_is_one_name() -> None:
    body = "a = b = np.zeros(n)\na[0] = 1.0\nout[0] = b[0]\n"
    assert split(body) == function_text("a = np.zeros(n)\na[0] = 1.0\nout[0] = a[0]\n")


def test_an_in_place_update_through_an_array_alias_updates_the_shared_name() -> None:
    body = "a = b = np.zeros(n)\nb += 1.0\nout[0] = a[0]\n"
    assert split(body) == function_text("a = np.zeros(n)\na += 1.0\nout[0] = a[0]\n")


def test_an_alias_takes_the_buffer_where_the_shared_name_is_rebound() -> None:
    body = "a = b = np.zeros(n)\na[0] = 1.0\na = np.ones(n)\nout[0] = b[0]\n"
    assert split(body) == function_text("a = np.zeros(n)\na[0] = 1.0\nb = a\na = np.ones(n)\nout[0] = b[0]\n")


def test_an_alias_rebound_inside_a_loop_takes_the_buffer_before_the_loop() -> None:
    body = "p = v = np.zeros(n)\nfor i in range(3):\n    out[i] = v[0]\n    v = p + 1.0\n"
    expected = "p = np.zeros(n)\nv = p\nfor i in range(3):\n    out[i] = v[0]\n    v = p + 1.0\n"
    assert split(body) == function_text(expected)


def test_an_alias_read_after_its_block_takes_the_buffer_at_the_end_of_the_block() -> None:
    body = "if c:\n    a = b = np.zeros(n)\n    a[0] = 1.0\nout[0] = b[0]\n"
    expected = "if c:\n    a = np.zeros(n)\n    a[0] = 1.0\n    b = a\nout[0] = b[0]\n"
    assert split(body) == function_text(expected)


def test_a_later_chained_target_indexes_with_the_name_bound_before_it() -> None:
    """Targets bind left to right, so ``x[b]`` sees the new ``b``, which is ``a``."""
    assert split("a = b = x[b] = np.zeros(n)\n") == function_text("a = np.zeros(n)\nx[a] = a\n")


def test_array_iteration_walks_the_leading_extent_and_records_the_array_it_walks() -> None:
    tree = function_tree("for v in a:\n    s += v\nfor w in names:\n    s += w\n")
    sut = DesugarArrayIteration(
        lambda array: ast.Name(id="N", ctx=ast.Load()) if array == "a" else None,
        lambda target, ordinal: f"i{ordinal}",
    )
    sut.visit(tree)
    expected = "for i0 in range(N):\n    v = a[i0]\n    s += v\nfor w in names:\n    s += w\n"
    assert ast.unparse(tree) == function_text(expected)
    assert sut.var_to_array == {"v": "a"}
