"""A comprehension / lambda / walrus / except-handler bound name must not leak
into ``_promote_free_names_to_params``'s free-name promotion.

The ``declared`` seeding walk only bound ``ast.Assign`` / ``ast.AugAssign`` /
``ast.For`` targets, so any OTHER binding form -- a comprehension's
``generator.target``, a ``lambda``'s arguments, a walrus (``ast.NamedExpr``)
target, or an ``except ... as name`` handler -- read as a genuinely FREE name
and got promoted to a real C-ABI scalar parameter the harness never passes,
shifting every later positional argument by one slot.

``distribution_search_numpy.py`` hits this for real::

    [int(round(fr * size)) for fr in (...)]

which leaked ``fr`` as a trailing parameter (harmless only because ``fr``
happens to sort last alphabetically among this kernel's params, and both
native emitters already fail earlier on unrelated constructs). The other
cases below are not hit by any live kernel yet, so they are pinned directly
against the pass rather than through the full emit pipeline.
"""
import ast

from numpyto_common.ir import KernelIR
from numpyto_common.lowering import _promote_free_names_to_params


def _kir(body: str, input_args: list) -> KernelIR:
    tree = ast.parse(body).body[0]
    return KernelIR(tree=tree, kernel_name="k", input_args=list(input_args))


def test_listcomp_target_not_promoted_to_param():
    kir = _kir(
        "def k(xs, size, out):\n"
        "    out[0] = sum(int(round(fr * size)) for fr in xs)\n",
        ["xs", "size", "out"],
    )
    _promote_free_names_to_params(kir)
    assert "fr" not in kir.input_args
    assert [s.name for s in kir.symbols] == []


def test_setcomp_and_dictcomp_targets_not_promoted():
    kir = _kir(
        "def k(xs, out):\n"
        "    d = {v: v * 2 for v in xs}\n"
        "    s = {w for w in xs}\n"
        "    out[0] = len(d) + len(s)\n",
        ["xs", "out"],
    )
    _promote_free_names_to_params(kir)
    assert "v" not in kir.input_args
    assert "w" not in kir.input_args


def test_lambda_args_not_promoted():
    kir = _kir(
        "def k(xs, out):\n"
        "    f = lambda z: z + 1\n"
        "    out[0] = f(xs[0])\n",
        ["xs", "out"],
    )
    _promote_free_names_to_params(kir)
    assert "z" not in kir.input_args


def test_walrus_target_not_promoted():
    kir = _kir(
        "def k(xs, out):\n"
        "    if (n := len(xs)) > 0:\n"
        "        out[0] = n\n",
        ["xs", "out"],
    )
    _promote_free_names_to_params(kir)
    assert "n" not in kir.input_args


def test_except_handler_name_not_promoted():
    kir = _kir(
        "def k(xs, out):\n"
        "    try:\n"
        "        out[0] = xs[0]\n"
        "    except IndexError as err:\n"
        "        out[0] = 0\n",
        ["xs", "out"],
    )
    _promote_free_names_to_params(kir)
    assert "err" not in kir.input_args


# --- end-to-end pin: the real kernel this was found on --------------------- #


def test_distribution_search_no_longer_leaks_fr():
    """distribution_search's ABI must equal the manifest binding exactly, with
    no resurrected ``fr`` comprehension variable trailing the real params."""
    from _bench_yaml import kir_for
    from hpcagent_bench.spec import BenchSpec
    from hpcagent_bench.support.bindings import binding_from_spec

    kir = kir_for("distribution_search", do_lower=True)
    order = kir.param_order()
    assert "fr" not in order
    assert order == ["backward_target", "forward_target", "p", "V"]

    spec = BenchSpec.load("distribution_search")
    canonical = [a.name for a in binding_from_spec(spec).args]
    assert order == canonical
