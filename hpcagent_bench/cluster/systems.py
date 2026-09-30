# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``hpcagent-bench job submit``: a helper job's ``sbatch`` line for the system it runs on.

The node shape of a job (tasks per node, cores per task, GPUs per node or per task, partition, account) differs
between machines, so the samples in ``docs/jobs/`` carry no shape of their own that a site must edit: this module
resolves each field and passes it as an ``sbatch`` option, which overrides the script's ``#SBATCH`` line.
A field's value is, in order of precedence:

1. its command-line flag (``--cpus-per-task 72``);
2. its environment variable (:data:`FIELDS`), which the site layer (``experiments/layers/site.env``) also
   provides, since a layer keeps a value the environment already holds;
3. the system's entry in ``systems.yaml`` (Beverin and Daint.Alps ship; ``$HPCAGENT_BENCH_SYSTEMS_FILE`` adds
   or replaces others). The system is ``--system``, else ``$HPCAGENT_BENCH_SYSTEM``, else the entry whose
   ``cluster`` is ``$SLURM_CLUSTER_NAME``, else the first entry (Beverin).

``--gpus-per-task`` given by the caller replaces the system's ``--gpus-per-node`` and vice versa: Slurm takes one.
"""

import argparse
import os
import pathlib
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence

import yaml

from hpcagent_bench import paths

__all__ = ["FIELDS", "SYSTEMS_FILE", "load_systems", "main", "resolve", "sbatch_command", "site_environment"]

#: field -> (sbatch option, environment variable that sets it).
FIELDS: dict[str, tuple[str, str]] = {
    "nodes": ("--nodes", "HPCAGENT_BENCH_JOB_NODES"),
    "partition": ("--partition", "SBATCH_PARTITION"),
    "account": ("--account", "SBATCH_ACCOUNT"),
    "ntasks_per_node": ("--ntasks-per-node", "HPCAGENT_BENCH_JOB_NTASKS_PER_NODE"),
    "cpus_per_task": ("--cpus-per-task", "HPCAGENT_BENCH_JOB_CPUS_PER_TASK"),
    "gpus_per_node": ("--gpus-per-node", "HPCAGENT_BENCH_JOB_GPUS_PER_NODE"),
    "gpus_per_task": ("--gpus-per-task", "HPCAGENT_BENCH_JOB_GPUS_PER_TASK"),
    "time": ("--time", "HPCAGENT_BENCH_JOB_TIME"),
}
#: Slurm takes one of these, so an explicit one displaces the system's other.
GPU_FIELDS = ("gpus_per_node", "gpus_per_task")
PACKAGED = pathlib.Path(__file__).with_name("systems.yaml")
SYSTEMS_FILE = "HPCAGENT_BENCH_SYSTEMS_FILE"


def load_systems(environ: Mapping[str, str]) -> dict[str, dict[str, object]]:
    """The packaged systems, then the file named by ``$HPCAGENT_BENCH_SYSTEMS_FILE`` over them."""
    systems: dict[str, dict[str, object]] = yaml.safe_load(PACKAGED.read_text(encoding="utf-8")) or {}
    extra = environ.get(SYSTEMS_FILE)
    if extra:
        systems |= yaml.safe_load(pathlib.Path(extra).read_text(encoding="utf-8")) or {}
    return systems


def site_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """``environ`` with the site layer applied under it (``scripts/site_env.sh``: the layer keeps what the
    environment already holds), so a layer's ``SBATCH_PARTITION`` counts as if it were exported."""
    script = paths.repo_root() / "scripts" / "site_env.sh"
    if not script.is_file():
        return dict(environ)
    done = subprocess.run(
        ["bash", "-c", '. "$1" >/dev/null 2>&1; env -0', "_", str(script)],
        capture_output=True,
        env=dict(environ),
        check=False,
    )
    pairs = (entry.partition("=") for entry in done.stdout.decode(errors="replace").split("\0") if "=" in entry)
    return {key: value for key, _, value in pairs} or dict(environ)


def choose_system(name: str | None, systems: Mapping[str, Mapping[str, object]], environ: Mapping[str, str]) -> str:
    """``name``, else ``$HPCAGENT_BENCH_SYSTEM``, else the system of ``$SLURM_CLUSTER_NAME``, else the first one."""
    chosen = name or environ.get("HPCAGENT_BENCH_SYSTEM") or ""
    if not chosen:
        cluster = environ.get("SLURM_CLUSTER_NAME", "")
        chosen = next((key for key, entry in systems.items() if cluster and entry.get("cluster") == cluster), "")
    chosen = chosen or next(iter(systems))  # the first entry of systems.yaml
    if chosen not in systems:
        raise SystemExit(f"job submit: no system {chosen!r}; known: {', '.join(sorted(systems))}")
    return chosen


def resolve(
    system: str | None, flags: Mapping[str, str | None], environ: Mapping[str, str]
) -> tuple[str, dict[str, str]]:
    """``(system, {field: value})``: a flag over an environment variable over the system's entry."""
    systems = load_systems(environ)
    name = choose_system(system, systems, environ)
    explicit = {
        field: value
        for field, (_option, variable) in FIELDS.items()
        if (value := flags.get(field) or environ.get(variable))
    }
    defaults = {field: str(value) for field, value in systems[name].items() if field in FIELDS}
    if any(field in explicit for field in GPU_FIELDS):
        defaults = {field: value for field, value in defaults.items() if field not in GPU_FIELDS}
    return name, {**defaults, **explicit}


def sbatch_command(script: str, values: Mapping[str, str], script_args: Sequence[str], nice: str = "") -> list[str]:
    """``sbatch`` with every resolved field as an option, then ``script`` and its arguments."""
    options = [f"{FIELDS[field][0]}={value}" for field, value in values.items()]
    return ["sbatch", *options, *([f"--nice={nice}"] if nice else []), script, *script_args]


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="hpcagent-bench job submit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--system", help="a system of systems.yaml (default: $HPCAGENT_BENCH_SYSTEM, else by cluster)")
    for field, (option, variable) in FIELDS.items():
        parser.add_argument(option, dest=field, metavar="VALUE", help=f"overrides ${variable} and the system's value")
    parser.add_argument("--dry-run", action="store_true", help="print the sbatch command and submit nothing")
    parser.add_argument("script", type=pathlib.Path, help="the job script, e.g. docs/jobs/grade-under.sbatch")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="the script's own arguments")
    args = parser.parse_args(list(argv))
    environ = site_environment(os.environ)
    name, values = resolve(args.system, vars(args), environ)
    command = sbatch_command(str(args.script), values, args.script_args, environ.get("HPCAGENT_BENCH_NICE", ""))
    print(f"# system {name}: {shlex.join(command)}", file=sys.stderr)
    return 0 if args.dry_run else subprocess.run(command, check=False).returncode
