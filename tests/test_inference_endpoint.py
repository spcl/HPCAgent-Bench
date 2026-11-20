# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A setup served by another job's keyed endpoint (``INFERENCE_SOURCE=endpoint``): ``serve-only.sbatch ACCESS=alps``
serves the experiment's model with the key, publishes ``endpoint.json`` only after its 401/200 check and deletes it
first on exit; the setup side accepts only this user's mode-600 files, plain http bases and its own checkpoint, and
the key never passes through python or any output."""

import json
import os
import pathlib
import subprocess
import sys
import types

import pytest

from hpcagent_bench import paths
from tests.fresh_module import fresh

ROOT = paths.ROOT
CLUSTER = ROOT / "hpcagent_bench" / "cluster"
SERVE_ONLY = CLUSTER / "serve-only.sbatch"
KEY = "0123456789abcdef" * 4
KIMI = "moonshotai/Kimi-K2.7-Code"
SECRETS = ("sglang-auth.yaml", "api.key", "auth.header")
#: What the launcher's host interpreter needs from the test's own environment to import this checkout.
INHERITED = ("PYTHONPATH", "VIRTUAL_ENV")


@pytest.fixture(name="service")
def service_fixture() -> types.ModuleType:
    return fresh("inference_service")


def private_file(path: pathlib.Path, text: str) -> pathlib.Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)
    return path


def serving_job(tmp_path: pathlib.Path, **overrides: str) -> dict[str, str]:
    """What a two-pipeline serving job's batch shell hands ``--write-endpoint``."""
    return {
        "VLLM_REPLICA_URLS": "http://nid001:8000/v1,http://nid005:8000/v1",
        "VLLM_SERVED_MODEL": "hpcagent-bench-vllm",
        "VLLM_MODEL": KIMI,
        "INFERENCE_ENGINE": "sglang",
        "INFERENCE_CE_ENV": "hpcagent-bench-sglang-mi300-latest-v10",
        "INFERENCE_KEY_FILE": str(private_file(tmp_path / "kimi.key", KEY + "\n")),
        "SLURM_JOB_NODELIST": "nid[001-008]",
        "SLURM_JOB_ID": "4242",
        **overrides,
    }


def endpoint_setup(endpoint: pathlib.Path, **overrides: str) -> dict[str, str]:
    """An mi200 setup's environment reading ``endpoint``."""
    return {
        "INFERENCE_SOURCE": "endpoint",
        "INFERENCE_NODES": "0",
        "INFERENCE_ENDPOINT_FILE": str(endpoint),
        "VLLM_MODEL": KIMI,
        **overrides,
    }


def test_a_published_endpoint_resolves_to_every_pipeline_and_the_key_file(
    service: types.ModuleType, tmp_path: pathlib.Path
) -> None:
    published = tmp_path / "endpoint.json"
    service.write_endpoint(published, serving_job(tmp_path))
    assert published.stat().st_mode & 0o777 == 0o600
    exported = service.endpoint_launcher_env(service.served_endpoint(endpoint_setup(published)))
    assert exported == {
        "VLLM_MASTER_HOST": "",
        "VLLM_BASE_URL": "http://nid001:8000/v1",
        "VLLM_REPLICA_URLS": "http://nid001:8000/v1,http://nid005:8000/v1",
        "VLLM_SERVED_MODEL": "hpcagent-bench-vllm",
        "INFERENCE_KEY_FILE": str(tmp_path / "kimi.key"),
    }
    # alps-endpoint.sh reads the same file
    assert {"url", "served_model", "key_file"} <= json.loads(published.read_text(encoding="utf-8")).keys()


def test_the_run_records_what_served_it_and_never_the_key(service: types.ModuleType, tmp_path: pathlib.Path) -> None:
    published = tmp_path / "endpoint.json"
    service.write_endpoint(published, serving_job(tmp_path))
    service.record(tmp_path / "run", endpoint_setup(published))
    text = (tmp_path / "run" / service.RECORD_NAME).read_text(encoding="utf-8")
    assert json.loads(text) == {
        "source": "endpoint",
        "engine": "sglang",
        "ce_env": "hpcagent-bench-sglang-mi300-latest-v10",
        "model": KIMI,
        "job_id": "4242",
        "nodes": "nid[001-008]",
        "replicas": ["http://nid001:8000/v1", "http://nid005:8000/v1"],
    }
    assert KEY not in text + published.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("job", "setup", "reason"),
    [
        ({}, {"INFERENCE_NODES": "8"}, "starts no server"),
        ({}, {"VLLM_MODEL": "openai/gpt-oss-120b"}, "serves moonshotai/Kimi-K2.7-Code"),
        ({"VLLM_REPLICA_URLS": "http://0.0.0.0:8000/v1"}, {}, "on a named host"),  # noqa: S104 -- the refused bind
        ({"VLLM_REPLICA_URLS": "https://nid001:8000/v1"}, {}, "is not http://"),
    ],
)
def test_an_endpoint_that_is_not_the_setups_is_refused(
    service: types.ModuleType, tmp_path: pathlib.Path, job: dict[str, str], setup: dict[str, str], reason: str
) -> None:
    published = tmp_path / "endpoint.json"
    with pytest.raises(SystemExit, match=reason):
        service.write_endpoint(published, serving_job(tmp_path, **job))
        service.served_endpoint(endpoint_setup(published, **setup))


@pytest.mark.parametrize("which", ["endpoint file", "key file"])
def test_an_endpoint_or_key_file_others_can_touch_is_refused(
    service: types.ModuleType, tmp_path: pathlib.Path, which: str
) -> None:
    """Whoever can rewrite endpoint.json can point the agents at another server; a readable key is a shared key."""
    published = tmp_path / "endpoint.json"
    service.write_endpoint(published, serving_job(tmp_path))
    (published if which == "endpoint file" else tmp_path / "kimi.key").chmod(0o644)
    with pytest.raises(SystemExit, match=f"the {which} .* must be a file owned by you, mode 600"):
        service.served_endpoint(endpoint_setup(published))


def test_the_launcher_reads_the_key_from_the_named_file_in_its_own_shell() -> None:
    """The exported block names the key FILE; the key never passes through python's stdout."""
    script = (CLUSTER / "run_cluster.sh").read_text(encoding="utf-8")
    branch = script[script.index('elif [[ "${INFERENCE_SOURCE}" == "endpoint" ]]; then') :]
    branch = branch[: branch.index("\nelse\n")]
    assert '--export)" || exit 1' in branch
    assert 'VLLM_API_KEY="$(tr -d \'\\n\' <"${INFERENCE_KEY_FILE}")"' in branch
    # the keyed server: the key by --config, the bind the node's own address, never the wildcard
    assert '"${sgl_auth[@]}"' in script
    assert '--host "${sgl_host}" --port "${VLLM_PORT}"' in script


# serve-only.sbatch ACCESS=alps, outside Slurm against stub scontrol / srun / curl


def stubs(tmp_path: pathlib.Path) -> pathlib.Path:
    """scontrol names 8 nodes; srun answers the arch check and, as the server step, records its environment and argv
    and copies endpoint.json once it appears; curl answers 401 without the key's header file and 200 with it."""
    stub = tmp_path / "bin"
    stub.mkdir()
    arch = next(
        line.split("=", 1)[1]
        for line in (ROOT / "containers" / "images" / "gpu_arch.env").read_text(encoding="ascii").splitlines()
        if line.startswith("GPU_ARCH_mi300=")
    )
    scripts = {
        "scontrol": "#!/bin/bash\nfor i in 1 2 3 4 5 6 7 8; do echo nid00$i; done\n",
        "srun": f"""#!/bin/bash
case "$*" in
    *"cat /opt/gpu-arch"*) echo {arch} ;;
    *rocminfo*) echo "  Name:                    {arch}" ;;
    *--vllm-node*)
        env > "{tmp_path}/server-step.env"
        printf '%s\\n' "$@" > "{tmp_path}/server-step.argv"
        echo "max_total_num_tokens=2000000" > "$RUN_DIR/server-0.log"  # the pool a real server states
        for _ in $(seq 300); do
            [[ -f "$RUN_DIR/endpoint.json" ]] && cp "$RUN_DIR/endpoint.json" "{tmp_path}/published.json" && exit 0
            sleep 0.1
        done
        exit 1 ;;
esac
""",
        "curl": f"""#!/bin/bash
out="" code="" keyed=0 url=""
while (( $# )); do
    case "$1" in
        -o) out="$2"; shift ;;
        -w) code=1; shift ;;
        -H) [[ "$2" == @*auth.header ]] && keyed=1; shift ;;
        http*) url="$1" ;;
    esac
    shift
done
echo "$url keyed=$keyed" >> "{tmp_path}/curl.log"
if [[ -n "$code" ]]; then (( keyed )) && printf 200 || printf 401; exit 0; fi
(( keyed )) || exit 22
[[ -z "$out" ]] || echo '{{}}' > "$out"
""",
    }
    for name, text in scripts.items():
        (stub / name).write_text(text, encoding="utf-8")
        (stub / name).chmod(0o755)
    return stub


def serve_only(
    tmp_path: pathlib.Path, model: str, key_mode: int = 0o600, **extra: str
) -> subprocess.CompletedProcess[str]:
    key = private_file(tmp_path / "kimi.key", KEY + "\n")
    key.chmod(key_mode)
    edf = tmp_path / "sglang.toml"
    edf.write_text('image = "/nonexistent.sqsh"\n', encoding="utf-8")
    overrides = private_file(tmp_path / "serve-overrides.env", f"INFERENCE_CE_ENV={edf}\n")
    env = {
        "PATH": f"{stubs(tmp_path)}:{pathlib.Path(sys.executable).parent}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "SCRATCH": str(tmp_path),
        "HPCAGENT_BENCH_SITE_ENV": str(private_file(tmp_path / "site.env", "")),
        "HPCAGENT_BENCH_HOST_PYTHON": sys.executable,
        **{name: os.environ[name] for name in INHERITED if name in os.environ},
        "ACCESS": "alps",
        "MODEL": model,
        "SERVE_BASE_ENV": f"mlgpu40:{model}",
        "SERVE_ENV_FILE": str(overrides),
        "KEY_FILE": str(key),
        **extra,
    }
    return subprocess.run(["bash", str(SERVE_ONLY)], capture_output=True, text=True, check=False, env=env, cwd=tmp_path)


def test_an_alps_serving_job_publishes_every_pipeline_after_its_key_check_and_deletes_it_first(
    tmp_path: pathlib.Path,
) -> None:
    done = serve_only(
        tmp_path,
        "kimi27sglang",
        SLURM_JOB_ID="4242",
        SLURM_JOB_NODELIST="nid[001-008]",
        SLURM_JOB_PARTITION="mi300",
        SLURM_CPUS_ON_NODE="8",
        GPUS_PER_NODE="4",
        CLUSTER_SCRIPT_DIR=str(CLUSTER),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    run_dir = tmp_path / "inference-server" / "4242"
    assert run_dir.stat().st_mode & 0o777 == 0o700
    published = json.loads((tmp_path / "published.json").read_text(encoding="utf-8"))
    assert published["replicas"] == ["http://nid001:8000/v1", "http://nid005:8000/v1"]
    assert (published["model"], published["engine"]) == (KIMI, "sglang")
    assert published["key_file"] == str(tmp_path / "kimi.key")
    # both pipelines answered 401 without the key before the file appeared
    log = (tmp_path / "curl.log").read_text(encoding="utf-8")
    assert all(f"{url[:-3]}/v1/chat/completions keyed=0" in log for url in published["replicas"])
    served = (run_dir / "serve.env").read_text(encoding="utf-8")
    assert f"INFERENCE_SGLANG_AUTH_CONFIG={run_dir}/sglang-auth.yaml" in served
    assert "--enable-metrics" not in served.split("SGLANG_EXTRA_ARGS=")[-1].splitlines()[0]
    for leak in (done.stdout, done.stderr, served, (tmp_path / "server-step.env").read_text(encoding="utf-8")):
        assert KEY not in leak
    assert [name for name in ("endpoint.json", *SECRETS) if (run_dir / name).exists()] == []


@pytest.mark.parametrize(
    ("model", "key_mode", "reason"),
    [
        ("oss120b", 0o600, "ACCESS=alps: vllm serves /metrics and other routes without the key"),
        ("kimi27sglang", 0o644, "is mode 644, not 600"),
    ],
)
def test_alps_access_refuses_an_open_engine_or_a_shared_key_before_submitting(
    tmp_path: pathlib.Path, model: str, key_mode: int, reason: str
) -> None:
    done = serve_only(tmp_path, model, key_mode, SUBMIT="0")
    assert done.returncode == 2
    assert reason in done.stderr
    assert not (tmp_path / "curl.log").exists()
    assert not (tmp_path / "inference-server").exists()
