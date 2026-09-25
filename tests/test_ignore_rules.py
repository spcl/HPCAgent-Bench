# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The repo's .gitignore files and .dockerignore keep tool output out, and keep sources in.

Each group of ``.gitignore`` rules gets one representative generated path that must be ignored,
and the sources that sit beside generated files (numpy references, hand-written C references,
manifests, build scripts) must stay trackable. No tracked file may match an ignore rule: a hand
override at a generated name carries its own ``!`` line instead of a silent ``git add -f``.
``.dockerignore`` is checked the same way for secrets, caches, run outputs and hidden tests.
"""

import re
import subprocess

import pytest

from hpcagent_bench import paths

KERNEL = "hpcagent_bench/benchmarks/scientific_computing/dense_linear_algebra/gemm"
LLR = "hpcagent_bench/benchmarks/loop_level_reasoning/argmax_value"

#: One generated path per .gitignore group, plus the repo tools' own outputs found in used trees.
GIT_IGNORED = [
    "hpcagent_bench/__pycache__/cli.cpython-312.pyc",
    "dist/hpcagent_bench-0.1.0.tar.gz",
    "hpcagent_bench.egg-info/PKG-INFO",
    ".venv/bin/python",
    ".pytest_cache/v/cache/nodeids",
    ".ruff_cache/CACHEDIR.TAG",
    ".mypy_cache/3.12/x.json",
    ".coverage.nid001.123",
    "collected.txt",
    ".openblas-cache/openblas-openmp/install/include/cblas.h",
    ".cache/generated/x.json",
    f"{KERNEL}/.cache/gemm_dace.sdfg",
    ".dacecache/gemm/build/libgemm.so",
    "hpcagent_bench/.hpcagent_bench_cache/csr.npz",
    f"{KERNEL}/gemm_dace.py",
    f"{KERNEL}/gemm_jax.py",
    f"{KERNEL}/gemm_cpp.py",
    f"{LLR}/argmax_value_numba_np.py",
    f"{KERNEL}/cpp_backend/gemm_fp64.c",
    f"{KERNEL}/cpp_backend/gemm_fp64_binding.json",
    f"{KERNEL}/cpp_backend/gemm_fp32_pluto_input_kernel.hu",
    f"{KERNEL}/cpp_backend/build/libgemm_c.so",
    f"{KERNEL}/cpp_backend/.ppcg_transform_x1y2/gemm_fp32_pluto_input_host.cu",
    "results/hpcagent_bench.db",
    "results/plots/heatmap.pdf",
    "hpcagent_bench0.db",
    "hpcagent_bench/native_runs/run1/gemm/submission.c",
    ".perf_reports/opt_report/gemm.txt",
    "hf_dataset/data.parquet",
    "harbor-runs/job1/result.json",
    "tasks/gemm/task.toml",
    "regrades/regrade-0.db",
    "worklist.jsonl",
    "ci-mi200-652111.out",
    "core_nid002536_17693",
    "judge-agent-amd.sqsh",
    "containers/agent/skills/opt-reports/SKILL.md",
    "shared/prompt-repo.md",
    ".env",
    "id_ed25519",
    "experiments/.env.cpf-llr-focus40-qwen38-c",
    "experiments/.rendered/arm.env",
    "experiments/layers/site.env",
    "experiments/problems-harness20.jsonl",
    "experiments/problems-harness20.kernels.resolved.txt",
    "experiments/owed/promote-owed-llr-0924.jsonl",
    "experiments/beverin-services-645720.out",
    "experiments/beverin-services-645720.err",
    "experiments/mwd-final-regrades-tol0925/x.db",
]

#: Sources that live next to generated files and must stay trackable.
GIT_KEPT = [
    f"{KERNEL}/gemm_numpy.py",
    f"{KERNEL}/gemm.yaml",
    f"{LLR}/argmax_value_reference.c",
    "hpcagent_bench/core_dumps.py",
    "hpcagent_bench/native_runs/.gitkeep",
    "results/.gitkeep",
    "results/plots/.gitkeep",
    ".perf_reports/.gitkeep",
    ".cache/README.md",
    "tests/data/llr40/data/rows.csv",
    "experiments/layers/site-cscs.env",
    "experiments/kernels-harness20.txt",
    "containers/images/build_common.sh",
    "containers/tools/lib/helper.sh",
]

#: One path per .dockerignore group; every one must stay out of an image build context.
DOCKER_EXCLUDED = [
    "hpcagent_bench/harness/hidden_tests/seeds.py",
    ".env",
    "experiments/.env.arm",
    "experiments/layers/site.env",
    "scripts/cscs/env.toml",
    "containers/judge/server.pem",
    ".git/HEAD",
    "hpcagent_bench/__pycache__/cli.cpython-312.pyc",
    f"{KERNEL}/.cache/gemm_dace.sdfg",
    ".dacecache/gemm/build/libgemm.so",
    "hpcagent_bench/.hpcagent_bench_cache/csr.npz",
    f"{KERNEL}/cpp_backend/build/libgemm_c.so",
    "containers/images/judge.sqsh",
    "results/hpcagent_bench.db",
    "hpcagent_bench0.db",
    "hpcagent_bench/native_runs/run1/gemm/submission.c",
    "experiments/.rendered/arm.env",
    "experiments/owed/list.txt",
    "experiments/problems-harness20.jsonl",
    "core_nid002536_17693",
    "hf_dataset/data.parquet",
]

#: Paths the Dockerfiles COPY; excluding them would break an image build.
DOCKER_KEPT = [
    "hpcagent_bench/cli.py",
    "hpcagent_bench/core_dumps.py",
    "hpcagent_bench/native_runs/.gitkeep",
    f"{KERNEL}/gemm_numpy.py",
    "pyproject.toml",
    "README.md",
    "scripts/install_dace.sh",
    "containers/agent/harness/pins.env",
]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=paths.ROOT, capture_output=True, text=True, check=True).stdout


def git_ignored(path: str) -> bool:
    """``git check-ignore`` on a path that need not exist; exit 1 means "not ignored"."""
    proc = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", path], cwd=paths.ROOT, capture_output=True, check=False
    )
    assert proc.returncode in (0, 1), proc.stderr
    return proc.returncode == 0


def docker_pattern(pattern: str) -> re.Pattern[str]:
    """A .dockerignore pattern as a regex over a context-relative path (Go filepath.Match plus ``**``)."""
    out = ""
    for token in re.split(r"(\*\*/|\*\*|\*|\?|\[[^]]*\])", pattern.strip("/")):
        match token:
            case "**/":
                out += "(?:.*/)?"
            case "**":
                out += ".*"
            case "*":
                out += "[^/]*"
            case "?":
                out += "[^/]"
            case _ if token.startswith("["):
                out += token
            case _:
                out += re.escape(token)
    return re.compile(out)


def docker_excluded(path: str) -> bool:
    """Docker's rule: the last pattern matching the path or one of its parent dirs decides."""
    lines = (paths.ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    rules = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    parts = path.split("/")
    prefixes = ["/".join(parts[: i + 1]) for i in range(len(parts))]
    excluded = False
    for rule in rules:
        negated = rule.startswith("!")
        regex = docker_pattern(rule.removeprefix("!"))
        if any(regex.fullmatch(prefix) for prefix in prefixes):
            excluded = not negated
    return excluded


@pytest.mark.parametrize("path", GIT_IGNORED)
def test_generated_path_is_gitignored(path: str) -> None:
    assert git_ignored(path), f"{path} is tool output but git would offer to commit it"


@pytest.mark.parametrize("path", GIT_KEPT)
def test_source_path_is_not_gitignored(path: str) -> None:
    assert not git_ignored(path), f"{path} is a source but an ignore rule hides it from git add"


def test_no_tracked_file_matches_an_ignore_rule() -> None:
    """A tracked file under an ignore rule is invisible to reviewers of the rules; hand overrides
    at generated names get an explicit ``!`` line in .gitignore instead."""
    assert git("ls-files", "-i", "-c", "--exclude-standard").split() == []


def test_every_override_negation_names_a_tracked_file() -> None:
    """A ``!`` line for a removed override would silently re-admit a generated file at that name."""
    lines = (paths.ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    overrides = [line.removeprefix("!/") for line in lines if line.startswith("!/hpcagent_bench/benchmarks/")]
    assert overrides
    tracked = set(git("ls-files", "--", *overrides).split())
    assert sorted(set(overrides) - tracked) == []


@pytest.mark.parametrize("path", DOCKER_EXCLUDED)
def test_path_is_kept_out_of_image_contexts(path: str) -> None:
    assert docker_excluded(path), f"{path} would be sent to an image build context"


@pytest.mark.parametrize("path", DOCKER_KEPT)
def test_copied_source_reaches_image_contexts(path: str) -> None:
    assert not docker_excluded(path), f"{path} is COPYed by a Dockerfile but .dockerignore drops it"


def test_the_docker_matcher_follows_the_last_matching_rule() -> None:
    """The matcher above is what the docker cases rest on: ``**/`` reaches any depth, a pattern
    matching a parent dir excludes its contents, and a later ``!`` re-admits."""
    assert docker_pattern("**/*.pem").fullmatch("a/b/c.pem")
    assert docker_pattern("**/*.pem").fullmatch("c.pem")
    assert not docker_pattern("*.out").fullmatch("experiments/x.out")
    assert docker_pattern("hpcagent_bench/native_runs/*").fullmatch("hpcagent_bench/native_runs/.gitkeep")
    assert docker_excluded("hpcagent_bench/native_runs/r/k/x.c")
    assert not docker_excluded("hpcagent_bench/native_runs/.gitkeep")
