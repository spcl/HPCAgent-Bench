"""A reference may rebind the framework precision globals; no backend may translate that.

``hpcagent_bench.frameworks.framework`` exports ``np_float`` / ``np_complex``, which
``Framework.set_datatype`` rewrites per run. A reference that imports those NAMES snapshots their
value at first import, so a process that runs fp64 and then fp32 keeps computing in whichever it
imported under -- measured, and silently wrong for mandelbrot1, mandelbrot2 and cloudsc. Reading
them off the module inside the kernel fixes that, but leaves an assignment whose right-hand side is
an attribute access, and every native emitter died on it with
``NotImplementedError: expression Attribute``.

The statement has no runtime meaning for a translated backend: ``np_float`` is resolved as a dtype
NAME by ``_NP_DTYPE_NAMES`` and narrowed to the run precision by the precision pass. So the shared
frontend drops it for every backend at once.
"""

import ast

from numpyto_common.frontend import _strip_framework_dtype_rebinding


def _fn(src: str) -> ast.FunctionDef:
    return ast.parse(src).body[0]


def test_separate_rebindings_are_dropped():
    fn = _fn(
        "def k(a):\n"
        "    np_float = framework.np_float\n"
        "    np_complex = framework.np_complex\n"
        "    return a.astype(np_float)\n"
    )
    _strip_framework_dtype_rebinding(fn)
    body = ast.unparse(fn)
    assert "framework" not in body
    assert "astype(np_float)" in body, "the dtype NAME must survive; only the assignment goes"


def test_a_tuple_rebinding_is_dropped():
    """The spelling mandelbrot uses."""
    fn = _fn("def k(a):\n    np_float, np_complex = framework.np_float, framework.np_complex\n    return np_complex\n")
    _strip_framework_dtype_rebinding(fn)
    assert "framework" not in ast.unparse(fn)


def test_an_ordinary_assignment_to_the_same_name_survives():
    """Anti-vacuity: only a rebinding read off a MODULE goes. Dropping any assignment that merely
    mentions the name would delete real computation."""
    fn = _fn("def k(a):\n    np_float = np.float32\n    return a.astype(np_float)\n")
    _strip_framework_dtype_rebinding(fn)
    assert "np_float = np.float32" in ast.unparse(fn)


def test_an_unrelated_attribute_assignment_survives():
    """A different name read off the module is somebody else's statement, not this rule's."""
    fn = _fn("def k(a):\n    scale = framework.scale\n    return a * scale\n")
    _strip_framework_dtype_rebinding(fn)
    assert "scale = framework.scale" in ast.unparse(fn)
