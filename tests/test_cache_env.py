# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``scripts/cache_env.sh`` is the ONE place the cache layout is derived (see the launch contract,
``experiments/env.sh``): every submitter sources it instead of naming a cache path itself. These
tests exercise the real script through bash, with a throwaway ``SCRATCH``, rather than
reimplementing its arithmetic in Python -- the property under test is what the shell actually
resolves, not what this file assumes it resolves to.

Two roots exist on purpose (see the script's own header): ``FAST_SCRATCH`` (iopsstor, weights) and
``SCRATCH`` (general scratch, JIT build output). The tests below hold both apart, and hold the
second one to what ``.cache/README.md`` documents as the default layout -- a script and a doc that
drift apart is a submitter silently landing on the wrong path.
"""

import pathlib
import re
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "cache_env.sh"
README = (REPO / ".cache" / "README.md").read_text()


def run(script: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Source ``cache_env.sh`` then run ``script``, in a minimal environment this test controls."""
    full_env = {"PATH": "/usr/bin:/bin", **env}
    return subprocess.run(
        ["bash", "-c", f'set -uo pipefail; . "{SCRIPT}"; {script}'],
        env=full_env,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("var", ["JIT_CACHE_ROOT", "HPCAGENT_BENCH_CPF_PRERENDER_DIR", "HPCAGENT_BENCH_TOOLS_DIR"])
def test_every_jit_side_cache_var_lives_under_scratch(tmp_path: pathlib.Path, var: str) -> None:
    """None of these may drift onto ``/capstor``, ``$HOME`` or a hardcoded user path: they must all
    resolve somewhere under the ``SCRATCH`` this test hands them, and nowhere else."""
    scratch = tmp_path / "scratch"
    proc = run(f'echo "${var}"', {"SCRATCH": str(scratch), "USER": "tester"})
    assert proc.returncode == 0, proc.stderr
    resolved = pathlib.Path(proc.stdout.strip())
    assert resolved.is_relative_to(scratch), f"{var}={resolved} escaped SCRATCH={scratch}"


def test_hf_home_lives_under_fast_scratch_not_under_the_jit_cache_root(tmp_path: pathlib.Path) -> None:
    """The two roots are split on purpose (weights on iopsstor, JIT build output on the general
    scratch): a change that folds HF_HOME under JIT_CACHE_ROOT would serve model weights off the
    slower filesystem the script's own header says this split exists to avoid."""
    fast, scratch = tmp_path / "fast", tmp_path / "scratch"
    proc = run('echo "$HF_HOME"', {"FAST_SCRATCH": str(fast), "SCRATCH": str(scratch), "USER": "tester"})
    assert proc.returncode == 0, proc.stderr
    resolved = pathlib.Path(proc.stdout.strip())
    assert resolved.is_relative_to(fast), f"HF_HOME={resolved} did not land under FAST_SCRATCH={fast}"
    assert not resolved.is_relative_to(scratch), f"HF_HOME={resolved} leaked under SCRATCH={scratch}"


def test_missing_scratch_fails_loudly_instead_of_silently_writing_home_or_tmp(tmp_path: pathlib.Path) -> None:
    """JIT_CACHE_ROOT has no FAST_SCRATCH-style unconditional default: an unset SCRATCH must abort
    sourcing the file, not quietly resolve under $HOME or /tmp where a later job would never find it."""
    home = tmp_path / "home"
    home.mkdir()
    proc = run('echo "should not print: $JIT_CACHE_ROOT"', {"HOME": str(home)})
    assert proc.returncode != 0, proc.stdout
    assert "set SCRATCH" in proc.stderr, proc.stderr
    assert "should not print" not in proc.stdout
    assert not any(home.iterdir()), "cache_env.sh wrote into $HOME despite SCRATCH being unset"


def test_missing_scratch_still_fails_loudly_even_when_jit_cache_root_is_explicitly_overridden_absent(
    tmp_path: pathlib.Path,
) -> None:
    """An explicitly EMPTY override is still a missing setting (see the account resolver's own
    "absent, not defaulted" rule): it must not be treated as "no override, fall back silently"."""
    proc = run('echo "$JIT_CACHE_ROOT"', {"JIT_CACHE_ROOT": "", "USER": "tester"})
    assert proc.returncode != 0, proc.stdout
    assert "set SCRATCH" in proc.stderr, proc.stderr


def test_an_explicit_jit_cache_root_override_is_honoured_without_scratch(tmp_path: pathlib.Path) -> None:
    """A caller that already resolved its own root (tests, a rerun pinned to a frozen cache) must
    still be able to override, exactly like the account resolver's HPCAGENT_BENCH_ACCOUNT escape hatch."""
    override = tmp_path / "frozen-cache"
    proc = run('echo "$JIT_CACHE_ROOT"', {"JIT_CACHE_ROOT": str(override), "USER": "tester"})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(override)


def test_cache_mkdirs_creates_directories_only_under_the_resolved_roots(tmp_path: pathlib.Path) -> None:
    """The helper submitters call after sourcing the file must not reach outside FAST_SCRATCH/SCRATCH --
    a stray mkdir under $HOME or /tmp would be invisible until a shared host filled up."""
    fast, scratch, home = tmp_path / "fast", tmp_path / "scratch", tmp_path / "home"
    home.mkdir()
    proc = run("hpcagent_bench_cache_mkdirs", {"FAST_SCRATCH": str(fast), "SCRATCH": str(scratch), "HOME": str(home)})
    assert proc.returncode == 0, proc.stderr
    assert (fast / ".hpcagentbench-cache" / "hf").is_dir(), "HF_HOME (under FAST_SCRATCH) was not created"
    assert (scratch / ".hpcagentbench-cache").is_dir(), "JIT_CACHE_ROOT (under SCRATCH) was not created"
    assert (scratch / ".hpcagentbench-cache" / ".cpf-prerender").is_dir()
    assert (scratch / ".hpcagentbench-cache" / "tools").is_dir()
    assert not any(home.iterdir()), "hpcagent_bench_cache_mkdirs wrote into $HOME"


def table_default(var: str) -> str:
    """The literal ``Default`` column of ``var``'s row in one of the README's markdown tables."""
    for line in README.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 3 or f"`{var}`" not in cells[1]:
            continue
        default = cells[2]
        assert default.startswith("`") and default.endswith("`"), default
        return default.strip("`")
    raise AssertionError(f"README no longer documents a default for {var!r}; update this test or the README")


def resolve_documented_default(var: str, scratch: pathlib.Path, cache_root: str) -> str:
    """Expand a documented default (``${SCRATCH}/...`` or ``${HPCAGENT_BENCH_CACHE}/...``) against
    the concrete roots a test run resolved, so this stays a substring check, not string surgery."""
    documented = table_default(var)
    return documented.replace("${HPCAGENT_BENCH_CACHE}", cache_root).replace("${SCRATCH}", str(scratch))


@pytest.mark.parametrize(
    "var",
    [
        "HPCAGENT_BENCH_CACHE",
        "HPCAGENT_BENCH_CPF_PRERENDER_DIR",
        "HPCAGENT_BENCH_TOOLS_DIR",
        "HPCAGENT_BENCH_RUNS_ROOT",
        "HPCAGENT_BENCH_RESULTS_DIR",
        "HPCAGENT_BENCH_GENERATED_CACHE_HOST",
        "HPCAGENT_BENCH_PACK_ROOT",
        "HPCAGENT_BENCH_PIP_CACHE_DIR",
        "HPCAGENT_BENCH_SPACK_BUILDCACHE_DIR",
        "HPCAGENT_BENCH_TMP_DIR",
    ],
)
def test_every_cache_root_default_matches_the_cache_readme(tmp_path: pathlib.Path, var: str) -> None:
    """.cache/README.md's table is the operator-facing contract; a rename in the script that
    nobody updates the README for silently makes the doc describe a directory that does not
    exist."""
    scratch = tmp_path / "scratch"
    cache_root = str(scratch / ".hpcagentbench-cache")
    expected = resolve_documented_default(var, scratch, cache_root)
    proc = run(f'echo "${var}"', {"SCRATCH": str(scratch), "USER": "tester"})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == expected


def test_the_weights_dir_default_matches_the_cache_readme(tmp_path: pathlib.Path) -> None:
    fast, scratch = tmp_path / "fast", tmp_path / "scratch"
    documented = table_default("HPCAGENT_BENCH_WEIGHTS_DIR")
    proc = run(
        'echo "$HPCAGENT_BENCH_WEIGHTS_DIR"', {"FAST_SCRATCH": str(fast), "SCRATCH": str(scratch), "USER": "tester"}
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == documented.replace("${FAST_SCRATCH}", str(fast))


def test_hpcagent_bench_cache_and_jit_cache_root_are_the_same_root_by_default(tmp_path: pathlib.Path) -> None:
    """The pre-unification name (JIT_CACHE_ROOT) and the canonical one must always agree: nothing
    that already pins JIT_CACHE_ROOT (a rerun frozen to a cache, a test) may silently drift onto a
    different tree than something that pins the new name."""
    scratch = tmp_path / "scratch"
    proc = run('echo "$HPCAGENT_BENCH_CACHE"; echo "$JIT_CACHE_ROOT"', {"SCRATCH": str(scratch), "USER": "tester"})
    assert proc.returncode == 0, proc.stderr
    cache_line, jit_line = proc.stdout.splitlines()
    assert cache_line == jit_line == str(scratch / ".hpcagentbench-cache")


def test_setting_hpcagent_bench_cache_sets_jit_cache_root_too(tmp_path: pathlib.Path) -> None:
    override = tmp_path / "pinned-cache"
    proc = run('echo "$JIT_CACHE_ROOT"', {"HPCAGENT_BENCH_CACHE": str(override), "USER": "tester"})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(override)


def test_setting_jit_cache_root_sets_hpcagent_bench_cache_too(tmp_path: pathlib.Path) -> None:
    override = tmp_path / "pinned-jit"
    proc = run('echo "$HPCAGENT_BENCH_CACHE"', {"JIT_CACHE_ROOT": str(override), "USER": "tester"})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(override)


def test_hf_home_follows_the_weights_dir_when_only_that_is_set(tmp_path: pathlib.Path) -> None:
    """HF_HOME must derive from HPCAGENT_BENCH_WEIGHTS_DIR, never carry an independent spelling of
    the same path: a caller that only points the named weights variable somewhere must not have
    HF_HOME quietly stay on the unconditional FAST_SCRATCH default instead."""
    weights = tmp_path / "weights-only"
    scratch = tmp_path / "scratch"
    proc = run(
        'echo "$HF_HOME"', {"HPCAGENT_BENCH_WEIGHTS_DIR": str(weights), "SCRATCH": str(scratch), "USER": "tester"}
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"{weights}/hf"


@pytest.mark.parametrize(
    "var",
    [
        "HPCAGENT_BENCH_CPF_PRERENDER_DIR",
        "HPCAGENT_BENCH_TOOLS_DIR",
        "HPCAGENT_BENCH_RUNS_ROOT",
        "HPCAGENT_BENCH_RESULTS_DIR",
        "HPCAGENT_BENCH_GENERATED_CACHE_HOST",
        "HPCAGENT_BENCH_PACK_ROOT",
        "HPCAGENT_BENCH_PIP_CACHE_DIR",
        "HPCAGENT_BENCH_SPACK_BUILDCACHE_DIR",
        "HPCAGENT_BENCH_TMP_DIR",
        "HPCAGENT_BENCH_BASE_IMAGES_DIR",
        "HPCAGENT_BENCH_CE_IMAGES_DIR",
    ],
)
def test_every_derived_dir_lives_under_the_cache_root_by_default(tmp_path: pathlib.Path, var: str) -> None:
    """The unification promise: nothing named here may land outside the one root unless the
    caller overrides that specific variable (the next test)."""
    scratch = tmp_path / "scratch"
    proc = run(f'echo "${var}"', {"SCRATCH": str(scratch), "USER": "tester"})
    assert proc.returncode == 0, proc.stderr
    resolved = pathlib.Path(proc.stdout.strip())
    assert resolved.is_relative_to(scratch / ".hpcagentbench-cache"), f"{var}={resolved} escaped the cache root"


@pytest.mark.parametrize(
    "var",
    [
        "HPCAGENT_BENCH_CPF_PRERENDER_DIR",
        "HPCAGENT_BENCH_TOOLS_DIR",
        "HPCAGENT_BENCH_RUNS_ROOT",
        "HPCAGENT_BENCH_RESULTS_DIR",
        "HPCAGENT_BENCH_GENERATED_CACHE_HOST",
        "HPCAGENT_BENCH_PACK_ROOT",
        "HPCAGENT_BENCH_PIP_CACHE_DIR",
        "HPCAGENT_BENCH_SPACK_BUILDCACHE_DIR",
        "HPCAGENT_BENCH_TMP_DIR",
        "HPCAGENT_BENCH_BASE_IMAGES_DIR",
        "HPCAGENT_BENCH_CE_IMAGES_DIR",
    ],
)
def test_every_derived_dir_honours_its_own_override(tmp_path: pathlib.Path, var: str) -> None:
    scratch = tmp_path / "scratch"
    override = tmp_path / f"pinned-{var}"
    proc = run(f'echo "${var}"', {"SCRATCH": str(scratch), "USER": "tester", var: str(override)})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(override)


def test_two_users_on_the_same_host_resolve_to_distinct_fast_scratch_roots() -> None:
    """FAST_SCRATCH's own unconditional default is keyed by $USER; two accounts must never be handed
    the same weights directory, which would let one user's job load (or evict) another's checkpoint."""
    first = run('echo "$FAST_SCRATCH"', {"USER": "alice", "SCRATCH": "/nonexistent-a"})
    second = run('echo "$FAST_SCRATCH"', {"USER": "bob", "SCRATCH": "/nonexistent-b"})
    assert first.returncode == 0 and second.returncode == 0
    assert first.stdout.strip() != second.stdout.strip()
    assert "alice" in first.stdout
    assert "bob" in second.stdout


def test_edf_mounts_follow_scratch_and_fast_scratch_instead_of_naming_a_filesystem(tmp_path: pathlib.Path) -> None:
    """The mounts an EDF writer binds are the top-level filesystems of the two roots, so moving
    scratch to another filesystem moves the mount with it; one filesystem holding both roots is
    mounted once."""
    proc = run(
        'echo "$HPCAGENT_BENCH_DATA_ROOTS"; hpcagent_bench_edf_mounts',
        {"SCRATCH": "/fsa/scratch/cscs/tester/x86_64", "FAST_SCRATCH": "/fsb/scratch/cscs/tester", "USER": "tester"},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == ["/fsa /fsb", '"/fsa/:/fsa/", "/fsb/:/fsb/"']
    same = run(
        "hpcagent_bench_edf_mounts",
        {"SCRATCH": "/fsa/scratch/cscs/tester/x86_64", "FAST_SCRATCH": "/fsa/fast/tester", "USER": "tester"},
    )
    assert same.returncode == 0, same.stderr
    assert same.stdout.strip() == '"/fsa/:/fsa/"'
