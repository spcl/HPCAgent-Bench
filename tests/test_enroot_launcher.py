# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The enroot fallback launcher: its entrypoint and its environment tunnel.

enroot strips SLURM_* inside the container, even when passed explicitly with --env, so
scripts/cscs/enroot_srun.sh forwards the task's variables under an HBFWD_ prefix and
scripts/cscs/enroot_start.conf restores the real names after the hooks have run. WHICH variables
is scripts/cscs/enroot_forward.sh, tested here by calling it, and the launcher itself is run end to
end against fake srun/enroot binaries to pin what enroot is actually handed. If the restore
fails, every rank reads SLURM_PROCID as unset and believes it is rank 0: the first smoke run of
the canon columns had four ranks each run every kernel and write one CSV concurrently.

The entrypoint broke twice on the way to working, and both are pinned here:
  * it used bash builtins, but enroot parses the file with POSIX sh ("local: not in a function");
  * an apostrophe in a comment ended the single-quoted body that embedded it.
WHAT EACH KIND OF TEST CAN AND CANNOT CATCH. The behavioural tests source the real conf and call
rc(), so they pin the restore LOGIC -- names restored, no leftovers, no value executed. They do NOT
reproduce the container's shell: the container is Ubuntu, where /bin/sh is dash, but on beverin's
login node /bin/sh is bash and dash is not installed, and bash accepts `local` and `compgen`
without complaint. So a bash-only construct passes the behavioural tests here and still fails in
the container. test_the_entrypoint_uses_no_bash_only_construct is the guard for THAT, and it is the
one that caught the regression when the bash version was reintroduced.
"""

import pathlib
import shutil
import subprocess

import pytest

from hpcagent_bench import paths

CSCS = paths.ROOT / "scripts" / "cscs"
CONF = CSCS / "enroot_start.conf"
LAUNCHER = CSCS / "enroot_srun.sh"

SH = shutil.which("dash") or shutil.which("sh")


def _run_rc(env: dict[str, str], command: str) -> subprocess.CompletedProcess:
    """Source the conf in POSIX sh and call rc(), as enroot does, with a clean environment."""
    script = f'. "{CONF}"; rc /usr/bin/env'
    return subprocess.run(
        [SH, "-c", script], env={"PATH": "/usr/bin:/bin", **env}, capture_output=True, text=True, check=False
    )


def test_the_entrypoint_parses_as_posix_sh() -> None:
    result = subprocess.run([SH, "-n", str(CONF)], capture_output=True, text=True)
    assert result.returncode == 0, f"{CONF.name} is not valid POSIX sh: {result.stderr}"


@pytest.mark.parametrize("bashism", ["local ", "compgen", "${!", "[[", "declare "])
def test_the_entrypoint_uses_no_bash_only_construct(bashism: str) -> None:
    """`sh -n` only checks syntax; `local` and `${!` parse fine and fail at RUN time under sh."""
    code = "\n".join(l for l in CONF.read_text().splitlines() if not l.lstrip().startswith("#"))
    assert bashism not in code, f"{CONF.name} uses bash-only {bashism!r}; enroot runs it with sh"


def test_a_tunnelled_slurm_variable_is_restored_under_its_real_name() -> None:
    result = _run_rc({"HBFWD_SLURM_PROCID": "5", "HBFWD_SLURM_NTASKS": "8"}, "env")
    assert result.returncode == 0, result.stderr
    got = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert got.get("SLURM_PROCID") == "5", f"SLURM_PROCID not restored: {got}"
    assert got.get("SLURM_NTASKS") == "8"


def test_no_tunnel_variable_is_left_behind() -> None:
    """A leftover HBFWD_ copy is harmless today and a surprise for anything that scans the env."""
    result = _run_rc({"HBFWD_SLURM_PROCID": "5"}, "env")
    assert "HBFWD_" not in result.stdout, "tunnel variables leaked into the container environment"


def test_a_hostile_value_is_restored_verbatim_and_never_executed(tmp_path: pathlib.Path) -> None:
    """The restore uses eval. Only NAMES may reach it; a value containing command substitution,
    quotes or spaces must arrive unchanged, and must not run."""
    marker = tmp_path / "executed"
    hostile = f"a \"b\" 'c' $(touch {marker}) `touch {marker}`; touch {marker}"
    result = _run_rc({"HBFWD_NCCL_SOCKET_IFNAME": hostile}, "env")
    got = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert not marker.exists(), "a forwarded VALUE was executed as shell code"
    assert got.get("NCCL_SOCKET_IFNAME") == hostile, f"value altered in transit: {got.get('NCCL_SOCKET_IFNAME')!r}"


def test_the_launcher_parses() -> None:
    """Its per-task body is one single-quoted string, so a stray apostrophe anywhere inside it
    ends the string and the file stops parsing."""
    result = subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


FORWARD_LIB = CSCS / "enroot_forward.sh"


def _forwardable(mode: str, *names: str) -> dict[str, bool]:
    """Ask the real hb_forwardable, in bash, as the launcher's per-task body does."""
    script = (
        f'. "{FORWARD_LIB}"; for n in "$@"; do if hb_forwardable "$n"; then echo "$n yes"; else echo "$n no"; fi; done'
    )
    out = subprocess.run(
        ["bash", "-c", script, "_", *names],
        env={"PATH": "/usr/bin:/bin", "HPCAGENT_BENCH_ENROOT_FORWARD": mode},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {n: v == "yes" for n, v in (line.split() for line in out.splitlines())}


RANK_NEEDS = [
    "SLURM_PROCID",
    "SLURM_NTASKS",
    "SLURM_LOCALID",
    "PMIX_RANK",
    "ROCR_VISIBLE_DEVICES",
    "MASTER_ADDR",
    "NCCL_SOCKET_IFNAME",
    "HPCAGENT_BENCH_CACHE",
    "SCRATCH",
    "HF_HOME",
    "CANON_OPT_REPORTS",
]
HOST_TOOLCHAIN = [
    "PATH",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PYTHONPATH",
    "PYTHONHOME",
    "MODULEPATH",
    "LMOD_CMD",
    "CPATH",
    "PKG_CONFIG_PATH",
    "VIRTUAL_ENV",
    "SHLVL",
]
NEVER = [
    "HBFWD_SLURM_PROCID",
    "HB_IMAGE",
    "ENROOT_CACHE_PATH",
    "OCI_ANNOTATION_com__hooks__cxi__enabled",
    "BASH_FUNC_module%%",
]


@pytest.mark.parametrize("mode", ["rank", "all"])
def test_what_a_rank_needs_is_forwarded(mode: str) -> None:
    got = _forwardable(mode, *RANK_NEEDS)
    assert all(got.values()), f"{mode}: not forwarded: {[n for n, ok in got.items() if not ok]}"


@pytest.mark.parametrize("mode", ["rank", "all"])
def test_the_hosts_toolchain_is_never_forwarded(mode: str) -> None:
    """The host's search paths would replace the image's and load host libraries and host Python
    packages into the container; the image's (or the EDF's) values must be the ones that arrive."""
    got = _forwardable(mode, *HOST_TOOLCHAIN, *NEVER)
    assert not any(got.values()), f"{mode}: forwarded from the host: {[n for n, ok in got.items() if ok]}"


def test_all_mode_carries_what_a_run_cluster_role_step_reads() -> None:
    """A role step re-enters run_cluster.sh and reads what the batch step computed. Under pyxis all
    of it arrived; under the rank allowlist none of it does, and the judge starts with no RUN_DIR."""
    step_vars = [
        "RUN_DIR",
        "RUN_ROOT",
        "JUDGE_BASE_URL",
        "JUDGE_NODELIST",
        "AGENT_MAX_TOKENS",
        "INFERENCE_NODES",
        "CLUSTER_ENV_FILE",
        "LITELLM_MASTER_KEY",
        "HOME",
        "USER",
    ]
    assert all(_forwardable("all", *step_vars).values())
    assert not any(_forwardable("rank", "RUN_DIR", "JUDGE_BASE_URL", "LITELLM_MASTER_KEY").values())


FAKE_SRUN = """#!/bin/bash
# Drop srun options; run the trailing `bash -c body` like one task would.
while [ $# -gt 0 ] && [ "$1" != bash ]; do shift; done
SLURM_PROCID=3 SLURM_NTASKS=4 exec "$@"
"""
FAKE_ENROOT = """#!/bin/bash
printf '%s\n' "$@" > "$HB_TEST_OUT/argv"
env > "$HB_TEST_OUT/env"
"""


def _launch(
    tmp_path: pathlib.Path, mode: str, hooks: str = "", extra_mount: str = "", extra_env: dict[str, str] | None = None
) -> tuple[list[str], dict[str, str]]:
    """Run the real enroot_srun.sh with fake srun/enroot on PATH; return enroot's argv and env."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, body in (("srun", FAKE_SRUN), ("enroot", FAKE_ENROOT)):
        (bindir / name).write_text(body)
        (bindir / name).chmod(0o755)
    image = tmp_path / "image.sqsh"
    image.write_text("")
    edf = tmp_path / "role.judge-node.toml"
    mounts = ", ".join(f'"{m}"' for m in (f"{tmp_path}:{tmp_path}", extra_mount) if m)
    edf.write_text(
        f'image = "{image}"\nmounts = [{mounts}]\nworkdir = "{tmp_path}"\n'
        '[env]\nNCCL_SOCKET_IFNAME = "from-edf"\nNCCL_NET = "AWS Libfabric"\nNCCL_NET_PLUGIN = "ofi"\n'
        '[annotations]\ncom.hooks.cxi.enabled = "true"\n'
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    env = {
        "PATH": f"{bindir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "USER": "tester",
        "SCRATCH": str(scratch),
        "HB_TEST_OUT": str(tmp_path),
        "SLURM_JOB_ID": "1",
        "HPCAGENT_BENCH_ENROOT_FORWARD": mode,
        "HPCAGENT_BENCH_COMM_HOOKS": hooks,
        "RUN_DIR": "/run/dir",
        "LITELLM_MASTER_KEY": "sk-SECRET value $(touch pwned)",
        "NCCL_SOCKET_IFNAME": "from-host",
        "LD_LIBRARY_PATH": "/host/lib",
        **(extra_env or {}),
    }
    result = subprocess.run(
        ["bash", str(LAUNCHER), str(edf), "--ntasks=4", "--", "echo", "hi"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    argv = (tmp_path / "argv").read_text().splitlines()
    got = dict(l.split("=", 1) for l in (tmp_path / "env").read_text().splitlines() if "=" in l)
    return argv, got


def test_no_forwarded_value_is_ever_on_the_command_line(tmp_path: pathlib.Path) -> None:
    """argv is readable by every process on the node, and run_cluster.sh forwards the inference key.
    enroot is handed NAMES; the values travel in the environment."""
    argv, env = _launch(tmp_path, "all")
    assert not any("SECRET" in a for a in argv), f"a forwarded value reached argv: {argv}"
    assert "HBFWD_LITELLM_MASTER_KEY" in argv
    assert env.get("HBFWD_LITELLM_MASTER_KEY") == "sk-SECRET value $(touch pwned)"
    assert not (tmp_path / "pwned").exists()


def test_the_task_rank_reaches_enroot(tmp_path: pathlib.Path) -> None:
    argv, env = _launch(tmp_path, "rank")
    assert "HBFWD_SLURM_PROCID" in argv and env.get("HBFWD_SLURM_PROCID") == "3"
    assert "HBFWD_RUN_DIR" not in argv, "rank mode forwarded a run_cluster variable"


def test_canon_opt_reports_reaches_the_container_under_the_default_rank_mode(tmp_path: pathlib.Path) -> None:
    """Regression for the smoke run that submitted a canon column with OPT_REPORTS=1 and got an
    empty reports/ directory: submit-canon-llr40.sh exports CANON_OPT_REPORTS into the batch job's
    environment, canon_column.sh's `outer` mode never re-exports it, and it is `inner` mode --
    running INSIDE the enroot container -- that reads it to decide whether to pass --opt-reports to
    run-framework. Under the default 'rank' forward mode CANON_OPT_REPORTS was not on the allowlist,
    so it was silently dropped at the enroot boundary: inside the container it read as unset, the
    column ran with reports off, and nothing said so -- the run just quietly produced no reports."""
    _, env = _launch(tmp_path, "rank", extra_env={"CANON_OPT_REPORTS": "1"})
    assert env.get("HBFWD_CANON_OPT_REPORTS") == "1", "CANON_OPT_REPORTS never reached the container's environment"


def test_the_edf_wins_over_the_host_and_host_paths_stay_out(tmp_path: pathlib.Path) -> None:
    argv, _ = _launch(tmp_path, "all")
    assert "NCCL_SOCKET_IFNAME=from-edf" in argv
    assert "HBFWD_NCCL_SOCKET_IFNAME" not in argv, "a key the EDF sets was also forwarded from the host"
    assert "HBFWD_LD_LIBRARY_PATH" not in argv


def test_an_edf_given_by_path_is_used_and_its_mounts_and_image_arrive(tmp_path: pathlib.Path) -> None:
    argv, _ = _launch(tmp_path, "rank")
    assert f"{tmp_path}:{tmp_path}" in argv, (
        "a read-write mount must stay the two-field form enroot binds submounts with"
    )
    assert str(tmp_path / "image.sqsh") in argv


def test_a_read_only_edf_mount_becomes_a_created_read_only_bind(tmp_path: pathlib.Path) -> None:
    """enroot turns --mount colons into fstab spaces: "src:dst:ro" handed through verbatim read "ro"
    as the filesystem type and created no missing target, so the agent's tools never mounted."""
    argv, _ = _launch(tmp_path, "rank", extra_mount=f"{tmp_path}/image.sqsh:/opt/tools/one:ro")
    assert f"{tmp_path}/image.sqsh:/opt/tools/one:none:x-create=file,bind,ro,nosuid,nodev,private" in argv


@pytest.mark.parametrize("hooks, expected", [("off", "false"), ("on", "true")])
def test_comm_hooks_switch_reaches_the_hook_annotations(tmp_path: pathlib.Path, hooks: str, expected: str) -> None:
    _, env = _launch(tmp_path, "rank", hooks)
    assert env.get("OCI_ANNOTATION_com__hooks__cxi__enabled") == expected
    assert env.get("OCI_ANNOTATION_com__hooks__netstack__source") == "host"


def _exported_annotations(edf: pathlib.Path) -> dict[str, str]:
    """Run ONLY the launcher's EDF-parsing block against a real EDF and collect its exports."""
    text = LAUNCHER.read_text()
    start = text.index("import shlex, sys, tomllib")
    end = text.index("\nPY\n", start)
    code = text[start:end]
    py = shutil.which("python3.11") or shutil.which("python3")
    out = subprocess.run([py, "-", str(edf)], input=code, capture_output=True, text=True, check=True).stdout
    found = {}
    for line in out.splitlines():
        if line.startswith("export OCI_ANNOTATION_"):
            key, _, value = line[len("export ") :].partition("=")
            found[key] = value.strip("'")
    return found


RENDERED = sorted(pathlib.Path.home().joinpath(".edf").glob("hpcagent-bench-*-mi300-latest.toml"))


@pytest.mark.skipif(not RENDERED, reason="no rendered EDFs in ~/.edf to parse")
@pytest.mark.parametrize("edf", RENDERED, ids=lambda p: p.stem)
@pytest.mark.parametrize(
    "variable, expected",
    [
        ("OCI_ANNOTATION_com__hooks__netstack__source", "host"),
        ("OCI_ANNOTATION_com__hooks__cxi__enabled", "true"),
        ("OCI_ANNOTATION_com__hooks__aws_ofi_nccl__enabled", "true"),
        ("OCI_ANNOTATION_com__hooks__aws_ofi_nccl__variant", "rocm6"),
    ],
)
def test_every_hook_annotation_in_the_edf_reaches_the_hooks(edf: pathlib.Path, variable: str, expected: str) -> None:
    """Dotted TOML keys parse as NESTED tables. Iterating the top level exported one variable named
    after "com" and none of these, so the aws_ofi_nccl hook -- which requires exactly "true" --
    exited silently and RCCL ran every multi-node collective over TCP."""
    exported = _exported_annotations(edf)
    assert exported.get(variable) == expected, (
        f"{edf.name}: {variable} is {exported.get(variable)!r}, expected {expected!r}; exported: {sorted(exported)}"
    )


@pytest.mark.parametrize("hooks, forced", [("off", False), ("on", True)])
def test_the_hook_plugin_is_forced_only_when_the_hook_runs(tmp_path: pathlib.Path, hooks: str, forced: bool) -> None:
    """NCCL_NET/NCCL_NET_PLUGIN name the plugin aws_ofi_nccl mounts. Forced with the hook off, RCCL
    refuses to initialize even on one node: sglang TP4 died with "Failed to initialize any NET plugin"."""
    argv, _ = _launch(tmp_path, "rank", hooks)
    assert ("NCCL_NET=AWS Libfabric" in argv) is forced, argv
    assert ("NCCL_NET_PLUGIN=ofi" in argv) is forced, argv
    assert "NCCL_SOCKET_IFNAME=from-edf" in argv, "only the plugin selectors may be dropped"
