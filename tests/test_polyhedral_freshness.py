# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A polyhedral transform is reused only when it was built from the same scop bytes and tool invocation, and a
failed Polygeist build never leaves a partial output where the column compiles from."""

import os
import pathlib
import subprocess
from typing import Any

import pytest

from hpcagent_bench.categories import Language
from hpcagent_bench.frameworks.errors import NotSupportedByFramework
from hpcagent_bench.polyhedral import pluto as pluto_transform
from hpcagent_bench.polyhedral import polygeist as polygeist_transform
from hpcagent_bench.polyhedral import ppcg as ppcg_transform

SCOP = """#include <stdint.h>
void mm_fp64(const int64_t N, double (*restrict A)[N], double (*restrict C)[N]) {
#pragma scop
  for (int64_t i = 0; i < N; i++)
    for (int64_t j = 0; j < N; j++)
      C[i][j] += A[i][j];
#pragma endscop
}
"""


def write_scop(cpp_backend: pathlib.Path) -> pathlib.Path:
    cpp_backend.mkdir(parents=True, exist_ok=True)
    scop = cpp_backend / "mm_fp64_pluto_input.c"
    scop.write_text(SCOP)
    return scop


def rewrite_with_an_older_mtime(scop: pathlib.Path, newer: pathlib.Path) -> None:
    """New bytes in ``scop`` whose mtime is still older than ``newer`` (a restored checkout, a coarse clock)."""
    scop.write_text(SCOP.replace("+=", "-="))
    older = newer.stat().st_mtime_ns - 10**9
    os.utime(scop, ns=(older, older))


def counting_polycc(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    runs: list[int] = []

    def fake(cmd: Any, **kwargs: Any) -> subprocess.CompletedProcess:  # noqa: ANN401
        runs.append(1)
        pathlib.Path(cmd[cmd.index("-o") + 1]).write_text(f"/* transform {len(runs)} */\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(pluto_transform, "polycc_exe", lambda: "/usr/bin/polycc")
    monkeypatch.setattr(pluto_transform, "assert_affine", lambda *a, **k: None)
    monkeypatch.setattr(pluto_transform, "run_bounded", fake)
    return runs


def test_a_rewritten_scop_is_retransformed_by_polycc_even_with_an_older_mtime(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scop = write_scop(tmp_path)
    runs = counting_polycc(monkeypatch)
    [out] = pluto_transform.transformed_sources(tmp_path, "mm")
    pluto_transform.transformed_sources(tmp_path, "mm")
    assert len(runs) == 1, "an unchanged scop was transformed again"

    rewrite_with_an_older_mtime(scop, out)
    pluto_transform.transformed_sources(tmp_path, "mm")

    assert len(runs) == 2, "a changed scop reused the transform of its old bytes"
    assert out.read_text() == "/* transform 2 */\n"


def test_changed_polycc_args_retransform(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_scop(tmp_path)
    runs = counting_polycc(monkeypatch)
    pluto_transform.transformed_sources(tmp_path, "mm")

    monkeypatch.setattr(pluto_transform, "POLYCC_ARGS", (*pluto_transform.POLYCC_ARGS, "--noprevector"))
    pluto_transform.transformed_sources(tmp_path, "mm")

    assert len(runs) == 2, "a transform built under other polycc args was reused"


def test_a_rewritten_scop_is_retransformed_by_ppcg_even_with_an_older_mtime(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scop = write_scop(tmp_path)
    runs: list[int] = []

    def fake_ppcg(scop: pathlib.Path, backend: Language, timeout: float | None = None) -> Any:  # noqa: ANN401
        runs.append(1)
        host, device = ppcg_transform.transformed_paths(scop, backend)
        host.write_text(f"/* host {len(runs)} */\n")
        device.write_text("__global__ void kernel0(void) {}\n")
        return ["ppcg"], subprocess.CompletedProcess(["ppcg"], 0, "", "")

    monkeypatch.setattr(ppcg_transform, "ppcg_lookup", lambda: ("/usr/bin/true", ""))
    monkeypatch.setattr(ppcg_transform, "hipify_exe", lambda: "/usr/bin/true")
    monkeypatch.setattr(ppcg_transform, "assert_affine", lambda *a, **k: None)
    monkeypatch.setattr(ppcg_transform, "run_ppcg", fake_ppcg)
    host, _ = ppcg_transform.transformed_sources(tmp_path, "mm", Language.HIP)
    ppcg_transform.transformed_sources(tmp_path, "mm", Language.HIP)
    assert len(runs) == 1, "an unchanged scop was transformed again"

    rewrite_with_an_older_mtime(scop, host)
    ppcg_transform.transformed_sources(tmp_path, "mm", Language.HIP)

    assert len(runs) == 2, "a changed scop reused the transform of its old bytes"


def test_a_failed_polygeist_build_leaves_the_published_outputs_whole(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last pipeline step dies mid-write: the column's ``.ll`` keeps the previous build, and no intermediate
    lands in ``cpp_backend``."""
    scop = write_scop(tmp_path)
    (tmp_path / "mm_fp64_pluto_binding.json").write_text("{}")
    ll = tmp_path / f"{scop.stem}_polygeist_gpu.ll"
    shim = tmp_path / f"{scop.stem}_polygeist_gpu_shim.c"
    ll.write_text("; previous build\n")
    shim.write_text("/* previous shim */\n")
    before = set(tmp_path.iterdir())

    def fake_tool(cmd: list[str], scop: pathlib.Path, base: str) -> subprocess.CompletedProcess:
        if cmd[0] == "clang":
            return subprocess.CompletedProcess(cmd, 0, "/resource\n", "")
        out = pathlib.Path(cmd[cmd.index("-o") + 1])
        if cmd[0] == "cgeist":
            out.write_text("module {\n  func.func @mm_fp64(%arg0: i64) {\n    return\n  }\n}\n")
        elif cmd[0] == "mlir-translate":
            out.write_text("; half writ")
            raise NotSupportedByFramework(polygeist_transform.FRAMEWORK, base, "mlir-translate killed")
        else:
            out.write_text("gpu.launch_func\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(polygeist_transform, "run_tool", fake_tool)
    monkeypatch.setattr(polygeist_transform.flags, "detect_gfx", lambda: "gfx90a")
    with pytest.raises(NotSupportedByFramework, match="killed"):
        polygeist_transform.transform_scop(scop, "mm")

    assert ll.read_text() == "; previous build\n"
    assert shim.read_text() == "/* previous shim */\n"
    assert set(tmp_path.iterdir()) == before, "the failed build left files in cpp_backend"


if __name__ == "__main__":
    import tempfile

    for test in (
        test_a_rewritten_scop_is_retransformed_by_polycc_even_with_an_older_mtime,
        test_changed_polycc_args_retransform,
        test_a_rewritten_scop_is_retransformed_by_ppcg_even_with_an_older_mtime,
        test_a_failed_polygeist_build_leaves_the_published_outputs_whole,
    ):
        with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as patch:
            test(pathlib.Path(tmp), patch)
