# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""canon_column.sh's ``inner`` mode refuses to run a column against the WRONG dace before it times a
single kernel (the image ships its own dace at /opt/dace as an editable install, and a PYTHONPATH
prepend that silently lost -- or was never set right -- would file every row of the column under
that copy instead of the tree the run is pinned to, indistinguishable from a real measurement).

Two properties, each load-bearing on its own:

* the assert actually FIRES when dace resolves outside DACE_TREE (a decoy ``dace`` package reachable
  from elsewhere on PYTHONPATH stands in for /opt/dace here, since the real image is not available
  outside a container) -- exit 1, with a message naming DACE_TREE;
* the assert does not misfire on a CORRECT tree spelled awkwardly (a trailing slash, a `//`, or a
  path given relative to the process's own CWD) -- the raw string-compare bug this replaces failed
  every one of those for a genuinely right tree, which would have refused a good run for no reason.
"""

import os
import pathlib
import subprocess

from hpcagent_bench import paths

CANON_COLUMN = paths.ROOT / "experiments" / "canon_column.sh"


def _stub_dace_tree(tmp_path: pathlib.Path, name: str = "dace-stub") -> pathlib.Path:
    """A trivial, importable ``dace`` package under its own directory."""
    dace_tree = tmp_path / name
    (dace_tree / "dace").mkdir(parents=True)
    (dace_tree / "dace" / "__init__.py").write_text("")
    return dace_tree


def _stub_opt(tmp_path: pathlib.Path, *, decoy_dace: bool) -> pathlib.Path:
    """An ``opt`` tree with a no-op ``scripts/cache_env.sh`` and a stub ``hpcagent_bench.cli`` that
    answers ``preflight`` and ``run-framework`` well enough for `inner` to run one kernel to
    completion. ``decoy_dace=True`` additionally ships its OWN ``dace`` package -- opt is the SECOND
    entry on the PYTHONPATH `inner` builds (after DACE_TREE), so when DACE_TREE has no dace package
    of its own (this test's stand-in for DACE_TREE never having been mounted at all), Python's
    import machinery falls through to this one instead, exactly the /opt/dace failure mode the
    real assert exists to catch."""
    opt_dir = tmp_path / "opt"
    (opt_dir / "scripts").mkdir(parents=True)
    (opt_dir / "scripts" / "cache_env.sh").write_text("# stub cache_env.sh for this test, no-op\n")
    if decoy_dace:
        (opt_dir / "dace").mkdir()
        (opt_dir / "dace" / "__init__.py").write_text("")
    pkg = opt_dir / "hpcagent_bench"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "cli.py").write_text(
        "import sys\n\n"
        "if __name__ == '__main__':\n"
        "    if sys.argv[1:2] == ['preflight']:\n"
        "        raise SystemExit(0)\n"
        "    csv = None\n"
        "    args = sys.argv[1:]\n"
        "    if '--csv' in args:\n"
        "        csv = args[args.index('--csv') + 1]\n"
        "    if csv:\n"
        "        with open(csv, 'w') as fh:\n"
        "            fh.write('framework,preset,datatype,kernel,impl,status,validated,median_ms,failure,error\\n')\n"
        "            fh.write('stubcol,fuzzed,float64,onlykernel,ref,ok,True,1.5,,\\n')\n"
        "    raise SystemExit(0)\n"
    )
    return opt_dir


def _run_inner(
    tmp_path: pathlib.Path, dace_tree: str, opt_dir: pathlib.Path, *, cwd: pathlib.Path | None = None
) -> subprocess.CompletedProcess[str]:
    out_root = tmp_path / "out"
    out_root.mkdir(exist_ok=True)
    env = dict(os.environ, SLURM_PROCID="0", SLURM_NTASKS="1", DACE_TREE=dace_tree)
    return subprocess.run(
        ["bash", str(CANON_COLUMN), "inner", "stubcol", str(out_root), "onlykernel", "fuzzed", str(opt_dir)],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_inner_refuses_when_dace_resolves_outside_dace_tree(tmp_path: pathlib.Path) -> None:
    dace_tree = tmp_path / "no-dace-mounted-here"
    dace_tree.mkdir()  # exists, but has no dace/ package of its own -- the broken-mount case
    opt_dir = _stub_opt(tmp_path, decoy_dace=True)

    result = _run_inner(tmp_path, str(dace_tree), opt_dir)

    assert result.returncode == 1, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert f"dace does not resolve to {dace_tree}" in result.stderr, result.stderr
    assert not (tmp_path / "out" / "stubcol.rank0.csv").exists(), "refused before running any kernel"


def test_inner_proceeds_when_dace_resolves_inside_dace_tree(tmp_path: pathlib.Path) -> None:
    dace_tree = _stub_dace_tree(tmp_path)
    opt_dir = _stub_opt(tmp_path, decoy_dace=False)

    result = _run_inner(tmp_path, str(dace_tree), opt_dir)

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "does not resolve" not in result.stderr, result.stderr
    assert (tmp_path / "out" / "stubcol.rank0.csv").exists()


def test_inner_accepts_a_correct_dace_tree_with_a_trailing_slash(tmp_path: pathlib.Path) -> None:
    """The raw string-compare bug this replaces built ``sys.argv[1]`` as ``f"{DACE_TREE}/dace/__init__.py"``
    -- a DACE_TREE already ending in ``/`` turned that into a `//`, a different STRING from the same
    file's ``dace.__file__``, and refused an otherwise-correct run."""
    dace_tree = _stub_dace_tree(tmp_path)
    opt_dir = _stub_opt(tmp_path, decoy_dace=False)

    result = _run_inner(tmp_path, str(dace_tree) + "/", opt_dir)

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "does not resolve" not in result.stderr, result.stderr


def test_inner_accepts_a_correct_dace_tree_given_as_a_relative_path(tmp_path: pathlib.Path) -> None:
    """DACE_TREE given relative to the process's own CWD (an interactive ``sbatch --export``, or a
    caller that never absolutized it) resolves to the SAME file as dace's own absolute
    ``__file__``; only a realpath-based compare sees that, a raw string compare never would."""
    dace_tree = _stub_dace_tree(tmp_path)
    opt_dir = _stub_opt(tmp_path, decoy_dace=False)

    result = _run_inner(tmp_path, dace_tree.name, opt_dir, cwd=tmp_path)

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "does not resolve" not in result.stderr, result.stderr
