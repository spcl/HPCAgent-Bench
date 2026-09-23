# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""experiments/prerender_cpf.sbatch's ``inner`` mode carries the SAME dace-resolves-inside-DACE_TREE
assert as canon_column.sh's ``inner`` mode (both guard against the container's own /opt/dace
silently winning), fixed the same way: compared through ``os.path.realpath`` rather than as raw
strings, so a trailing slash or a `//` in an otherwise-correct DACE_TREE is not mistaken for a wrong
tree. See tests/test_canon_column_dace_tree_assert.py for the fuller case set; this file only proves
the SAME fix landed here too.
"""

import pathlib
import subprocess

from hpcagent_bench import paths

SCRIPT = paths.ROOT / "experiments" / "prerender_cpf.sbatch"


def _stub_dace_tree(tmp_path: pathlib.Path, name: str = "dace-stub") -> pathlib.Path:
    dace_tree = tmp_path / name
    (dace_tree / "dace").mkdir(parents=True)
    (dace_tree / "dace" / "__init__.py").write_text("")
    return dace_tree


def _stub_opt(tmp_path: pathlib.Path, *, decoy_dace: bool) -> pathlib.Path:
    """An OPT tree with a stub ``hpcagent_bench.cpf_prerender`` (exits 0 immediately) so a run that
    gets past the assert completes cleanly. ``decoy_dace=True`` ships its own ``dace`` package,
    standing in for the image's /opt/dace: OPT is the second PYTHONPATH entry `inner` builds (after
    DACE_TREE), so it is what Python's import machinery falls through to when DACE_TREE has no dace
    package of its own."""
    opt_dir = tmp_path / "opt"
    if decoy_dace:
        (opt_dir / "dace").mkdir(parents=True)
        (opt_dir / "dace" / "__init__.py").write_text("")
    pkg = opt_dir / "hpcagent_bench"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "cpf_prerender.py").write_text("if __name__ == '__main__':\n    raise SystemExit(0)\n")
    return opt_dir


def _run_inner(tmp_path: pathlib.Path, dace_tree: str, opt_dir: pathlib.Path) -> subprocess.CompletedProcess[str]:
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    return subprocess.run(
        ["bash", str(SCRIPT), "inner", str(cache), "someview", "somekernel", "cpu", dace_tree, str(opt_dir)],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_prerender_inner_refuses_when_dace_resolves_outside_dace_tree(tmp_path: pathlib.Path) -> None:
    dace_tree = tmp_path / "no-dace-mounted-here"
    dace_tree.mkdir()
    opt_dir = _stub_opt(tmp_path, decoy_dace=True)

    result = _run_inner(tmp_path, str(dace_tree), opt_dir)

    assert result.returncode == 1, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert f"prerender: dace does not resolve to {dace_tree}" in result.stderr, result.stderr


def test_prerender_inner_accepts_a_correct_dace_tree_with_a_trailing_slash(tmp_path: pathlib.Path) -> None:
    dace_tree = _stub_dace_tree(tmp_path)
    opt_dir = _stub_opt(tmp_path, decoy_dace=False)

    result = _run_inner(tmp_path, str(dace_tree) + "/", opt_dir)

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "does not resolve" not in result.stderr, result.stderr
