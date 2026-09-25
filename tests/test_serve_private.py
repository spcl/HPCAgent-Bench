# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The private serving launcher, inference/serve-private.sbatch, for both presets.

Runs the launcher's refusal and dry-run paths outside Slurm against stub srun/sbatch/curl/ip: the key
reaches sglang only through --config, every server binds 127.0.0.1 last (ACCESS=alps: the node's hsn0
address, metrics off), each preset refuses the other partition and serves its own flags, and
MODE=serve prints a loopback tunnel or an endpoint.json but never the key.
"""

import json
import pathlib
import re
import subprocess

import pytest

from tests.env_render import rendered

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "containers" / "inference" / "serve-private.sbatch"
#: The qwen38 campaign base whose serving flags the mi300 preset mirrors.
CAMPAIGN_BASE = "llrbase-c:qwen38"
KEY = "0123456789abcdef" * 4
PRESETS = ("mi300", "mi200")
#: The partition each preset must refuse.
OTHER_PARTITION = {"mi300": "mi200", "mi200": "mi300"}
#: How many servers each preset's default LEGS start in smoke mode.
DEFAULT_LEG_COUNT = {"mi300": 1, "mi200": 3}
SECRETS = {"sglang-auth.yaml", "api.key", "auth.header"}
HSN0_ADDRESS = "172.28.9.16"
#: `ip -4 -o addr show dev hsn0` on a beverin node, verbatim.
HSN0_LINE = f"3: hsn0    inet {HSN0_ADDRESS}/16 scope global hsn0\\       valid_lft forever preferred_lft forever"


def launch(
    tmp_path: pathlib.Path, preset: str | None, key_mode: int = 0o600, **extra: str
) -> subprocess.CompletedProcess[str]:
    """Run the launcher outside Slurm, with no inherited environment and stubs that record any call."""
    stub = tmp_path / "bin"
    stub.mkdir(exist_ok=True)
    for name in ("srun", "sbatch", "curl"):
        (stub / name).write_text(f'#!/bin/sh\necho {name} >> "{tmp_path}/stub-calls"\n', encoding="utf-8")
        (stub / name).chmod(0o755)
    (stub / "ip").write_text('#!/bin/sh\nprintf "%s\\n" "$HSN0_LINE"\n', encoding="utf-8")
    (stub / "ip").chmod(0o755)
    key = tmp_path / "endpoint.key"
    key.write_text(KEY + "\n", encoding="utf-8")
    key.chmod(key_mode)
    for repo in ("Qwen--Qwen3.8-27B", "Qwen--Qwen3.8-27B-FP8"):
        (tmp_path / "hf" / "hub" / f"models--{repo}").mkdir(parents=True, exist_ok=True)
    env = {
        "PATH": f"{stub}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "USER": "serve-private-test-user",
        "SCRATCH": str(tmp_path),
        "REPO": str(ROOT),
        "KEY_FILE": str(key),
        "HF_HOME": str(tmp_path / "hf"),
        "RUN_ROOT": str(tmp_path / "runs"),
        "DRY_RUN": "1",
        "HSN0_LINE": HSN0_LINE,
        **({} if preset is None else {"PRESET": preset}),
        **extra,
    }
    return subprocess.run(["bash", str(LAUNCHER)], capture_output=True, text=True, check=False, env=env, cwd=tmp_path)


def assert_untouched(tmp_path: pathlib.Path, done: subprocess.CompletedProcess[str]) -> None:
    """No stub ran, no run dir exists, and the key appears in no output."""
    assert not (tmp_path / "stub-calls").exists()
    assert not (tmp_path / "runs").exists()
    assert KEY not in done.stdout + done.stderr


def argv_lines(stdout: str) -> list[str]:
    return [line.removeprefix("argv: ") for line in stdout.splitlines() if line.startswith("argv: ")]


def flags(words: list[str]) -> dict[str, str]:
    """Each --flag mapped to the word after it, or to "" when the next word is another flag."""
    following = [*words[1:], ""]
    return {word: ("" if nxt.startswith("--") else nxt) for word, nxt in zip(words, following) if word.startswith("--")}


def campaign_sglang_flags() -> dict[str, str]:
    """SGLANG_EXTRA_ARGS of the qwen38 campaign, with ${SCRIPT_DIR} expanded as sourcing does."""
    found = re.findall(r'^SGLANG_EXTRA_ARGS="([^"]*)"$', rendered(CAMPAIGN_BASE), re.MULTILINE)
    assert len(found) == 1, CAMPAIGN_BASE
    return flags(found[0].replace("${SCRIPT_DIR}", str(ROOT / "experiments")).split())


def test_the_launcher_refuses_to_start_without_a_preset(tmp_path: pathlib.Path) -> None:
    done = launch(tmp_path, None)
    assert done.returncode == 2
    assert "PRESET must be mi300 or mi200" in done.stderr
    assert_untouched(tmp_path, done)


def test_the_launcher_refuses_an_unknown_mode(tmp_path: pathlib.Path) -> None:
    done = launch(tmp_path, "mi300", MODE="public")
    assert done.returncode == 2
    assert "MODE must be smoke or serve" in done.stderr
    assert_untouched(tmp_path, done)


@pytest.mark.parametrize("preset", PRESETS)
def test_each_preset_refuses_a_job_on_the_other_partition(tmp_path: pathlib.Path, preset: str) -> None:
    other = OTHER_PARTITION[preset]
    done = launch(tmp_path, preset, SLURM_JOB_ID="1", SLURM_JOB_PARTITION=other)
    assert done.returncode == 2
    assert f"PRESET={preset} runs on partition {preset}, not '{other}'" in done.stderr
    assert_untouched(tmp_path, done)


@pytest.mark.parametrize("preset", PRESETS)
def test_each_preset_reads_its_own_key_file_by_default(tmp_path: pathlib.Path, preset: str) -> None:
    done = launch(tmp_path, preset, KEY_FILE="")
    assert done.returncode == 2
    assert f"no key file {tmp_path}/.config/hpcagent-bench/{preset}-endpoint.key" in done.stderr
    assert_untouched(tmp_path, done)


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("mode", [0o644, 0o640, 0o400])
def test_the_launcher_refuses_a_key_file_that_is_not_mode_600_before_touching_anything(
    tmp_path: pathlib.Path, preset: str, mode: int
) -> None:
    done = launch(tmp_path, preset, key_mode=mode)
    assert done.returncode == 2
    assert f"is mode {mode:o}, not 600" in done.stderr
    assert_untouched(tmp_path, done)


@pytest.mark.parametrize("preset", PRESETS)
def test_each_preset_passes_the_key_only_through_a_config_file_in_a_mode_700_run_dir(
    tmp_path: pathlib.Path, preset: str
) -> None:
    done = launch(tmp_path, preset)
    assert done.returncode == 0, done.stderr
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert run_dir.stat().st_mode & 0o777 == 0o700
    assert f"config:   {run_dir}/sglang-auth.yaml (mode 600)" in done.stdout
    argvs = argv_lines(done.stdout)
    assert len(argvs) == DEFAULT_LEG_COUNT[preset]
    for argv in argvs:
        assert f"--config {run_dir}/sglang-auth.yaml" in argv
        assert "--api-key" not in argv
    assert KEY not in done.stdout + done.stderr
    assert [path.name for path in run_dir.iterdir() if path.name in SECRETS] == []
    assert not (tmp_path / "stub-calls").exists()


@pytest.mark.parametrize("preset", PRESETS)
def test_every_leg_of_each_preset_binds_loopback_last_on_the_command_line(tmp_path: pathlib.Path, preset: str) -> None:
    done = launch(tmp_path, preset, LEGS="tp4:0.80 tp2:0.85", EXTRA_ARGS="--log-level debug")
    assert done.returncode == 0, done.stderr
    argvs = argv_lines(done.stdout)
    assert [argv.split()[-4:] for argv in argvs] == [
        ["--host", "127.0.0.1", "--port", "30000"],
        ["--host", "127.0.0.1", "--port", "30001"],
    ]
    assert "--tp-size 2 --mem-fraction-static 0.85" in argvs[1]
    assert "0.0.0.0" not in LAUNCHER.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "extra",
    [
        "--host 0.0.0.0",
        "--port=8000",
        "--api-key secret",
        "--config /tmp/c.yaml",
        "--tokenizer-worker-num 2",
        "--enable-metrics",
    ],
)
def test_the_launcher_refuses_extra_args_that_name_the_host_port_key_config_tokenizer_workers_or_metrics(
    tmp_path: pathlib.Path, extra: str
) -> None:
    """sglang enforces the key only with one tokenizer worker and never on /metrics."""
    done = launch(tmp_path, "mi200", EXTRA_ARGS=extra)
    assert done.returncode == 2
    assert "EXTRA_ARGS may not set" in done.stderr


def test_the_launcher_refuses_a_malformed_leg(tmp_path: pathlib.Path) -> None:
    done = launch(tmp_path, "mi200", LEGS="tp4:0.80 tp3:0.80")
    assert done.returncode == 2
    assert "leg 'tp3:0.80'" in done.stderr


def test_the_mi300_preset_refuses_a_leg_wider_than_its_four_gpus(tmp_path: pathlib.Path) -> None:
    done = launch(tmp_path, "mi300", LEGS="tp8:0.306")
    assert done.returncode == 2
    assert "leg 'tp8:0.306' needs 8 GPUs; mi300 nodes have 4" in done.stderr
    assert_untouched(tmp_path, done)


def test_the_mi300_preset_serves_the_qwen38_campaign_flags_on_fp8_weights_with_aiter(tmp_path: pathlib.Path) -> None:
    done = launch(tmp_path, "mi300")
    assert done.returncode == 0, done.stderr
    (argv,) = argv_lines(done.stdout)
    served = flags(argv.split())
    campaign = campaign_sglang_flags()
    assert {name: served.get(name) for name in campaign} == campaign
    assert (served["--attention-backend"], served["--mem-fraction-static"]) == ("aiter", "0.306")
    assert (served["--model-path"], served["--tp-size"]) == ("Qwen/Qwen3.8-27B-FP8", "4")
    assert "--disable-custom-all-reduce" not in served
    assert "image:    hpcagent-bench-sglang-mi300-latest\n" in done.stdout
    assert "env:      SGLANG_USE_AITER=1 SGLANG_SET_CPU_AFFINITY=0\n" in done.stdout


def test_the_mi200_preset_serves_bf16_weights_with_triton_attention_aiter_off_and_no_custom_all_reduce(
    tmp_path: pathlib.Path,
) -> None:
    done = launch(tmp_path, "mi200")
    assert done.returncode == 0, done.stderr
    served = flags(argv_lines(done.stdout)[0].split())
    assert served["--attention-backend"] == "triton"
    assert "--disable-custom-all-reduce" in served
    assert (served["--model-path"], served["--tp-size"], served["--mem-fraction-static"]) == (
        "Qwen/Qwen3.8-27B",
        "4",
        "0.80",
    )
    shared = {name: value for name, value in campaign_sglang_flags().items() if name != "--mem-fraction-static"}
    shared.pop("--attention-backend")
    assert {name: served.get(name) for name in shared} == shared
    assert "image:    hpcagent-bench-sglang-mi200-latest\n" in done.stdout
    assert "env:      SGLANG_USE_AITER=0 SGLANG_SET_CPU_AFFINITY=0\n" in done.stdout


@pytest.mark.parametrize("preset", PRESETS)
def test_serve_mode_starts_one_server_and_prints_a_loopback_tunnel_but_never_the_key(
    tmp_path: pathlib.Path, preset: str
) -> None:
    done = launch(
        tmp_path,
        preset,
        MODE="serve",
        API_PORT="30123",
        HPCAGENT_BENCH_SSH_JUMP="jumphost",
        HPCAGENT_BENCH_LOGIN_HOST="loginhost",
    )
    assert done.returncode == 0, done.stderr
    assert len(argv_lines(done.stdout)) == 1
    # The hosts come from the site layer (experiments/layers/site-*.env), never from the script.
    tunnels = [line.strip() for line in done.stdout.splitlines() if "ssh -N -J jumphost,loginhost" in line]
    assert len(tunnels) == 1
    assert "-L 127.0.0.1:30123:127.0.0.1:30123 serve-private-test-user@" in tunnels[0]
    assert f"scp loginhost:{tmp_path}/endpoint.key " in done.stdout
    assert "Authorization: Bearer %s" in done.stdout
    assert KEY not in done.stdout + done.stderr
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert [path.name for path in run_dir.iterdir() if path.name in SECRETS] == []
    assert not (tmp_path / "stub-calls").exists()


def test_the_launcher_refuses_an_unknown_access(tmp_path: pathlib.Path) -> None:
    done = launch(tmp_path, "mi200", ACCESS="public")
    assert done.returncode == 2
    assert "ACCESS must be tunnel or alps, got 'public'" in done.stderr
    assert_untouched(tmp_path, done)


@pytest.mark.parametrize("preset", PRESETS)
def test_alps_access_binds_every_leg_to_this_nodes_hsn0_address_last(tmp_path: pathlib.Path, preset: str) -> None:
    done = launch(tmp_path, preset, ACCESS="alps", LEGS="tp4:0.80 tp2:0.85")
    assert done.returncode == 0, done.stderr
    assert [argv.split()[-4:] for argv in argv_lines(done.stdout)] == [
        ["--host", HSN0_ADDRESS, "--port", "30000"],
        ["--host", HSN0_ADDRESS, "--port", "30001"],
    ]
    assert f"access:   alps, --host {HSN0_ADDRESS}\n" in done.stdout


@pytest.mark.parametrize("preset", PRESETS)
def test_alps_access_serves_no_metrics(tmp_path: pathlib.Path, preset: str) -> None:
    """sglang answers /metrics without the key, and every Alps node can reach an hsn0 bind."""
    done = launch(tmp_path, preset, ACCESS="alps", LEGS="tp4:0.80 tp2:0.85")
    assert done.returncode == 0, done.stderr
    argvs = argv_lines(done.stdout)
    assert len(argvs) == 2
    assert [argv for argv in argvs if "--enable-metrics" in argv.split()] == []


def test_alps_access_refuses_a_node_without_an_hsn0_address_before_touching_anything(tmp_path: pathlib.Path) -> None:
    done = launch(tmp_path, "mi200", ACCESS="alps", HSN0_LINE="", SLURMD_NODENAME="nid002536")
    assert done.returncode == 2
    assert "ACCESS=alps: nid002536 has no hsn0 IPv4 address" in done.stderr
    assert_untouched(tmp_path, done)


@pytest.mark.parametrize("preset", PRESETS)
def test_alps_serve_mode_publishes_the_url_model_and_key_path_but_never_the_key(
    tmp_path: pathlib.Path, preset: str
) -> None:
    done = launch(tmp_path, preset, ACCESS="alps", MODE="serve", API_PORT="30123", SLURMD_NODENAME="nid002536")
    assert done.returncode == 0, done.stderr
    published = [line.removeprefix("endpoint.json: ") for line in done.stdout.splitlines() if "endpoint.json: " in line]
    assert len(published) == 1, done.stdout
    assert json.loads(published[0]) == {
        "url": f"http://{HSN0_ADDRESS}:30123/v1",
        "served_model": "hpcagent-bench-vllm",
        "key_file": f"{tmp_path}/endpoint.key",
        "node": "nid002536",
        "job_id": "dry-run",
    }
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert f"source {ROOT}/containers/inference/alps-endpoint.sh {run_dir}/endpoint.json" in done.stdout
    assert "ssh -N -J" not in done.stdout
    assert KEY not in done.stdout + done.stderr


def test_alps_serve_mode_deletes_endpoint_json_with_the_secrets_when_the_job_exits(tmp_path: pathlib.Path) -> None:
    """A Daint job that finds endpoint.json must be able to trust the server behind it is still there."""
    done = launch(tmp_path, "mi200", ACCESS="alps", MODE="serve")
    assert done.returncode == 0, done.stderr
    assert "endpoint.json: {" in done.stdout
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert [path.name for path in run_dir.iterdir() if path.name in SECRETS | {"endpoint.json"}] == []


def test_alps_serve_mode_refuses_a_served_model_that_would_break_endpoint_json(tmp_path: pathlib.Path) -> None:
    done = launch(tmp_path, "mi200", ACCESS="alps", MODE="serve", SERVED_MODEL='qwen"38')
    assert done.returncode == 2
    assert "may not contain a quote or a backslash" in done.stderr
    assert_untouched(tmp_path, done)
