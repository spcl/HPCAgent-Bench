# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""node_monitor.sh's CSV shape and monitor_report.py's parser for it.

The judge nodes run many concurrent grading builds; gpu_pct alone (a cross-GPU
average) hides per-GPU imbalance that a judge-to-GPU pinning decision needs to see.
node_monitor.sh appends gpu0_pct..gpu(N-1)_pct after the original 8 columns, and
monitor_report.py must keep reading both the old 8-column files (already on disk
from past runs) and the new extended ones -- pinned here so neither format regresses.
"""
import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "containers/cluster/example-script"
SCRIPT = EXAMPLE / "node_monitor.sh"
REPORT = EXAMPLE / "monitor_report.py"

BASH = shutil.which("bash")

OLD_HEADER = "ts,cpu_pct,load1,mem_used_mib,mem_total_mib,gpu_pct,vram_used_mib,vram_total_mib"

# node_monitor.sh needs these on PATH to run at all -- including "bash" itself, since
# the fake smi tools below are `#!/usr/bin/env bash` scripts and env resolves that
# through PATH too. The real smi tools are deliberately excluded so a test controls
# exactly which GPU tool (if any) the script finds, regardless of what happens to be
# installed on the machine running the suite.
BASE_TOOLS = ("hostname", "mkdir", "awk", "grep", "wc", "tr", "sleep", "date", "cat", "bash")


def restricted_bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "restricted_bin"
    bin_dir.mkdir(exist_ok=True)
    for tool in BASE_TOOLS:
        real = shutil.which(tool)
        assert real, f"{tool} missing from host PATH, cannot build a restricted PATH for the test"
        (bin_dir / tool).symlink_to(real)
    return bin_dir


def fake_smi(tmp_path: Path, name: str, body: str) -> Path:
    """A directory holding one fake smi executable that always answers with ``body``,
    so node_monitor.sh's ``command -v`` probe finds a controlled tool instead of
    whatever rocm-smi/amd-smi happen to be installed on the test host."""
    bin_dir = tmp_path / f"fake_{name}"
    bin_dir.mkdir()
    tool = bin_dir / name
    tool.write_text(f"#!/usr/bin/env bash\n{body}\n")
    tool.chmod(0o755)
    return bin_dir


def run_monitor(tmp_path: Path, path_dirs: list[Path], interval: str = "0.2", role: str = "judge") -> str:
    """Runs node_monitor.sh for a bit over one sample interval, stops it, and returns
    the one CSV file it wrote."""
    out_dir = tmp_path / "monitor"
    env = dict(os.environ)
    env["PATH"] = ":".join(str(d) for d in path_dirs)
    env["ROLE"] = role
    env["OUT_DIR"] = str(out_dir)
    env["INTERVAL"] = interval
    proc = subprocess.Popen([BASH, str(SCRIPT)], env=env)
    try:
        time.sleep(float(interval) * 2 + 0.5)
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    csv_files = list(out_dir.glob("*.csv"))
    assert len(csv_files) == 1, f"expected exactly one CSV, got {csv_files}"
    return csv_files[0].read_text()


ROCM_JSON = ('{"card0": {"GPU use (%)": "10", "VRAM Total Used Memory (B)": "1048576", '
             '"VRAM Total Memory (B)": "10485760"}, '
             '"card1": {"GPU use (%)": "90", "VRAM Total Used Memory (B)": "2097152", '
             '"VRAM Total Memory (B)": "10485760"}}')

AMDSMI_JSON = ('[{"gpu": 0, "usage": {"gfx_activity": {"value": 15}}}, '
               '{"gpu": 1, "usage": {"gfx_activity": {"value": 55}}}, '
               '{"gpu": 2, "usage": {"gfx_activity": {"value": 35}}}]')


def test_no_gpu_tool_keeps_the_old_8_column_header(tmp_path):
    text = run_monitor(tmp_path, [restricted_bin(tmp_path)])
    lines = text.strip().splitlines()
    assert lines[0] == OLD_HEADER
    row = lines[1].split(",")
    assert len(row) == 8
    assert row[5:8] == ["", "", ""]  # gpu_pct, vram_used_mib, vram_total_mib all blank


def test_rocm_smi_appends_one_column_per_card_after_the_fixed_8(tmp_path):
    rocm_bin = fake_smi(tmp_path, "rocm-smi", f"echo '{ROCM_JSON}'")
    text = run_monitor(tmp_path, [rocm_bin, restricted_bin(tmp_path)])
    lines = text.strip().splitlines()
    header = lines[0].split(",")
    assert header == OLD_HEADER.split(",") + ["gpu0_pct", "gpu1_pct"]
    row = lines[1].split(",")
    assert len(row) == len(header)
    assert row[5] == "50.0"  # gpu_pct: cross-GPU average of 10 and 90
    assert row[8:10] == ["10", "90"]  # gpu0_pct, gpu1_pct: per-card, positional order preserved


def test_amd_smi_fallback_also_appends_per_gpu_columns(tmp_path):
    amdsmi_bin = fake_smi(tmp_path, "amd-smi", f"echo '{AMDSMI_JSON}'")
    text = run_monitor(tmp_path, [amdsmi_bin, restricted_bin(tmp_path)])
    lines = text.strip().splitlines()
    header = lines[0].split(",")
    assert header == OLD_HEADER.split(",") + ["gpu0_pct", "gpu1_pct", "gpu2_pct"]
    row = lines[1].split(",")
    assert row[8:11] == ["15", "55", "35"]
    assert row[6:8] == ["", ""]  # amd-smi metric -u carries no vram fields, same as before


def test_header_is_fixed_for_the_life_of_the_csv_file(tmp_path):
    """A relaunch that appends to an existing CSV must not rewrite its header, even if
    the GPU tool available at relaunch time differs (e.g. rocm-smi now installed)."""
    out_dir = tmp_path / "monitor"
    out_dir.mkdir()
    (out_dir / "judge-nid001.csv").write_text(OLD_HEADER + "\n2026-01-01T00:00:00Z,1.0,0.1,1,2,,,\n")
    rocm_bin = fake_smi(tmp_path, "rocm-smi", f"echo '{ROCM_JSON}'")
    env = dict(os.environ)
    env["PATH"] = f"{rocm_bin}:{restricted_bin(tmp_path)}"
    env["ROLE"] = "judge"
    env["OUT_DIR"] = str(out_dir)
    env["INTERVAL"] = "0.2"
    proc = subprocess.Popen([BASH, str(SCRIPT)], env=env)
    try:
        time.sleep(0.9)
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    lines = (out_dir / "judge-nid001.csv").read_text().strip().splitlines()
    assert lines[0] == OLD_HEADER  # unchanged: new columns from a mid-run tool swap would misalign old rows


def monitor_report():
    spec = importlib.util.spec_from_file_location("monitor_report", REPORT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolves its `from __future__` string
    # annotations via sys.modules[cls.__module__]; skipping this registration crashes
    # NodeStats's class body with an AttributeError on a None module.
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


def test_old_8_column_files_still_report_gpu_mean_and_no_balance_section(tmp_path, capsys):
    write_csv(tmp_path / "agent-nid001.csv", OLD_HEADER, [
        "2026-01-01T00:00:00Z,50.0,1.0,10000,20000,30.0,1000,10000",
        "2026-01-01T00:00:05Z,90.0,1.5,10500,20000,35.0,1000,10000",
    ])
    mod = monitor_report()
    node = mod.compute_node_stats(tmp_path / "agent-nid001.csv")
    assert node.gpu_mean == pytest.approx(32.5)
    assert node.per_gpu_mean == {}
    assert node.gpu_imbalance is None
    mod.print_gpu_balance([node])
    assert capsys.readouterr().out == ""  # nothing printed: old format carries no per-GPU columns


def test_extended_format_reports_per_gpu_mean_and_imbalance_spread(tmp_path, capsys):
    header = OLD_HEADER + ",gpu0_pct,gpu1_pct,gpu2_pct,gpu3_pct"
    write_csv(tmp_path / "judge-nid002.csv", header, [
        "2026-01-01T00:00:00Z,20.0,1.0,10000,20000,55.0,1000,10000,10.0,90.0,50.0,70.0",
        "2026-01-01T00:00:05Z,25.0,1.1,10500,20000,60.0,1000,10000,20.0,100.0,60.0,60.0",
    ])
    mod = monitor_report()
    node = mod.compute_node_stats(tmp_path / "judge-nid002.csv")
    assert node.per_gpu_mean == {
        "gpu0_pct": pytest.approx(15.0),
        "gpu1_pct": pytest.approx(95.0),
        "gpu2_pct": pytest.approx(55.0),
        "gpu3_pct": pytest.approx(65.0),
    }
    assert node.gpu_imbalance == pytest.approx(80.0)  # gpu1 (95.0) - gpu0 (15.0): the imbalance a pinning
    # decision needs to see, invisible in gpu_mean (57.5)
    mod.print_gpu_balance([node])
    printed = capsys.readouterr().out
    assert "spread=" in printed and "80.0" in printed
    assert "gpu0=15.0" in printed and "gpu1=95.0" in printed


def test_gpu_columns_are_read_header_driven_not_by_position(tmp_path):
    """A file whose per-GPU columns are NOT contiguous with the fixed 8 (e.g. reordered
    by hand) still parses correctly -- gpu_columns() keys off the header, not offsets."""
    mod = monitor_report()
    assert mod.gpu_columns(["ts", "gpu2_pct", "cpu_pct", "gpu0_pct",
                            "gpu1_pct"]) == ["gpu0_pct", "gpu1_pct", "gpu2_pct"]
    assert mod.gpu_columns(list(OLD_HEADER.split(","))) == []


def test_report_cli_prints_gpu_balance_only_for_extended_files(tmp_path):
    write_csv(tmp_path / "agent-old.csv", OLD_HEADER, [
        "2026-01-01T00:00:00Z,50.0,1.0,10000,20000,30.0,1000,10000",
    ])
    header = OLD_HEADER + ",gpu0_pct,gpu1_pct"
    write_csv(tmp_path / "judge-new.csv", header, [
        "2026-01-01T00:00:00Z,20.0,1.0,10000,20000,55.0,1000,10000,10.0,100.0",
    ])
    proc = subprocess.run([sys.executable, str(REPORT), str(tmp_path)], capture_output=True, text=True, check=True)
    assert "gpu balance" in proc.stdout
    assert "judge   new" in proc.stdout
    after_heading = proc.stdout.split("gpu balance", 1)[1]
    balance_section = after_heading.split("\n\n", 1)[0]  # up to the next blank-line-separated section
    assert "spread=" in balance_section and "90.0" in balance_section
    assert "gpu0=10.0" in balance_section and "gpu1=100.0" in balance_section
    assert "agent" not in balance_section  # agent-old.csv has no per-GPU columns
