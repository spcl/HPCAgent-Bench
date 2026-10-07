# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The ``sbatch`` options of a job, for the system it runs on: ``hpcagent-bench job submit`` (every job script:
the helper samples, the container jobs) and ``hpcagent-bench job options`` (the experiment submitter,
``cluster/submit.sh``).

The node shape of a job (tasks per node, cores per task, GPUs per node or per task, partition, account) differs
between machines, so this module resolves each field and passes it as an ``sbatch`` option. A field's value is, in
order of precedence:

1. its command-line flag (``--cpus-per-task 72``);
2. Slurm's own input variable (``SBATCH_PARTITION``, ``SBATCH_ACCOUNT``), which beats a script's ``#SBATCH`` line
   under plain ``sbatch`` too;
3. the job script's leading ``#SBATCH`` line (``job submit`` only): a field the script pins stays as it is. An image
   build or verify job pins its role's ``images.env`` partition the same way, unless ``--system`` names a machine;
4. its other environment variable (:data:`FIELDS`), which the site layer (``experiments/layers/site.env``) also
   provides, since a layer keeps a value the environment already holds;
5. the system's entry in ``systems.yaml`` (``$HPCAGENT_BENCH_SYSTEMS_FILE`` adds or replaces entries). The system is
   ``--system``, else ``$HPCAGENT_BENCH_SYSTEM``, else the entry whose ``cluster`` is ``$SLURM_CLUSTER_NAME``, else
   none: a cluster with no entry runs from flags and the environment alone. Without ``--system``, a partition set
   above that another entry of the same cluster names picks that entry (``--partition mi200`` on Beverin is
   ``beverin-mi200``).

Each pair of :data:`PAIRS` says one thing two ways, so it comes whole from the highest source that sets either
field: an explicit ``--gpus-per-task`` displaces the system's ``--gpus-per-node``, a script's ``--ntasks`` the
system's ``--ntasks-per-node``. A job names the fields it cannot run without (:data:`JOB_REQUIRED`); a missing one
is an error that names its flag and its environment variable. Two more values resolve the same way and are not
``sbatch`` options (:data:`EXTRAS`): the hardware, which names the GPU generation whose images and serving layers an
experiment uses, and the partition's longest time limit, which caps a scaled experiment's wall clock. Neither has a
default.
"""

import argparse
import os
import pathlib
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import yaml

from hpcagent_bench import paths

__all__ = [
    "EXTRAS",
    "FIELDS",
    "GPU_FIELDS",
    "JOB_FIELDS",
    "JOB_REQUIRED",
    "PACKAGED",
    "PAIRS",
    "SHORT",
    "SYSTEMS_FILE",
    "TASK_FIELDS",
    "Resolved",
    "choose_system",
    "image_partition",
    "load_systems",
    "main",
    "options_main",
    "resolve",
    "sbatch_command",
    "script_header",
    "site_environment",
]

#: field -> (sbatch option, environment variable that sets it).
FIELDS: dict[str, tuple[str, str]] = {
    "nodes": ("--nodes", "HPCAGENT_BENCH_JOB_NODES"),
    "partition": ("--partition", "SBATCH_PARTITION"),
    "account": ("--account", "SBATCH_ACCOUNT"),
    "ntasks": ("--ntasks", "HPCAGENT_BENCH_JOB_NTASKS"),
    "ntasks_per_node": ("--ntasks-per-node", "HPCAGENT_BENCH_JOB_NTASKS_PER_NODE"),
    "cpus_per_task": ("--cpus-per-task", "HPCAGENT_BENCH_JOB_CPUS_PER_TASK"),
    "gpus_per_node": ("--gpus-per-node", "HPCAGENT_BENCH_JOB_GPUS_PER_NODE"),
    "gpus_per_task": ("--gpus-per-task", "HPCAGENT_BENCH_JOB_GPUS_PER_TASK"),
    "time": ("--time", "HPCAGENT_BENCH_JOB_TIME"),
    "nice": ("--nice", "HPCAGENT_BENCH_NICE"),
}
#: The short form of a field's option, as a script's ``#SBATCH`` line may spell it.
SHORT = {"nodes": "-N", "partition": "-p", "account": "-A", "ntasks": "-n", "cpus_per_task": "-c", "time": "-t"}
GPU_FIELDS = ("gpus_per_node", "gpus_per_task")
TASK_FIELDS = ("ntasks_per_node", "ntasks")
#: Two fields that say one thing two ways: a pair comes whole from the highest source that sets either.
PAIRS = (GPU_FIELDS, TASK_FIELDS)
#: What the experiment job takes from this module. Its node count is the sum of its roles and its time limit is
#: computed from the tag, so neither is a field here; its tasks and cores are its own (one task per node).
JOB_FIELDS = ("partition", "account", "gpus_per_node")
#: Without these the job cannot run: the account bills it, the GPUs are what its roles share.
JOB_REQUIRED = ("account", "gpus_per_node")
#: Values a job needs that are no ``sbatch`` option: name -> (flag, environment variable).
EXTRAS: dict[str, tuple[str, str]] = {
    "hardware": ("--hardware", "HPCAGENT_BENCH_HARDWARE"),
    "max_time_hours": ("--max-time-hours", "HPCAGENT_BENCH_MAX_TIME_HOURS"),
}
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


def script_header(script: pathlib.Path) -> dict[str, str]:
    """The fields ``script`` pins with its leading ``#SBATCH`` lines; ``sbatch`` reads none past the first
    command."""
    words: list[str] = []
    for line in script.read_text(encoding="utf-8").splitlines():
        if line.startswith("#SBATCH"):
            words += shlex.split(line.removeprefix("#SBATCH"), comments=True)
        elif line.strip() and not line.startswith("#"):
            break
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for field in FIELDS:
        parser.add_argument(FIELDS[field][0], *([SHORT[field]] if field in SHORT else []), dest=field)
    known = parser.parse_known_args(words)[0]
    return {field: value for field, value in vars(known).items() if value is not None}


def image_partition(script: pathlib.Path, script_args: Sequence[str], environ: Mapping[str, str]) -> str:
    """The ``containers/images/images.env`` partition of an image job's role (``build_and_verify.sbatch <role>``,
    ``ROLE=<role> verify_image.sbatch``), else ``""``."""
    roles = {"build_and_verify.sbatch": next(iter(script_args), ""), "verify_image.sbatch": environ.get("ROLE", "")}
    role = roles.get(script.name, "")
    if not role:
        return ""
    images_env = paths.repo_root() / "containers" / "images" / "images.env"
    done = subprocess.run(
        ["bash", "-c", '. "$1" && ce_image "$2" partition', "_", str(images_env), role],
        capture_output=True,
        text=True,
        env=dict(environ),
        check=False,
    )
    return done.stdout.strip() if done.returncode == 0 else ""


def choose_system(
    name: str | None, systems: Mapping[str, Mapping[str, object]], environ: Mapping[str, str], partition: str = ""
) -> str:
    """``name``, else ``$HPCAGENT_BENCH_SYSTEM``, else the system of ``$SLURM_CLUSTER_NAME``, else ``""``. Unless
    ``name`` is given, a ``partition`` that another entry of the same cluster names picks that entry."""
    chosen = name or environ.get("HPCAGENT_BENCH_SYSTEM") or ""
    if not chosen:
        cluster = environ.get("SLURM_CLUSTER_NAME", "")
        chosen = next((key for key, entry in systems.items() if cluster and entry.get("cluster") == cluster), "")
    if chosen and chosen not in systems:
        raise SystemExit(f"job submit: no system {chosen!r}; known: {', '.join(sorted(systems))}")
    if name or not chosen or not partition or systems[chosen].get("partition") == partition:
        return chosen
    home = systems[chosen].get("cluster")
    siblings = (key for key, entry in systems.items() if home and entry.get("cluster") == home)
    return next((key for key in siblings if systems[key].get("partition") == partition), chosen)


class Resolved(NamedTuple):
    """What a job gets: the system it was resolved for (``""`` when none), its ``sbatch`` fields and its
    :data:`EXTRAS` (``""`` for one nothing sets)."""

    system: str
    values: dict[str, str]
    extras: dict[str, str]

    @property
    def hardware(self) -> str:
        return self.extras["hardware"]


def resolve(
    system: str | None,
    flags: Mapping[str, str | None],
    environ: Mapping[str, str],
    fields: Sequence[str] = tuple(FIELDS),
    required: Sequence[str] = (),
    *,
    pinned: Mapping[str, str] | None = None,
) -> Resolved:
    """The ``fields`` of a job, each from the highest source that sets it: its flag, Slurm's own ``SBATCH_*``
    variable, the script's ``pinned`` value, its other variable, the system's entry. Refuses a ``required`` field
    nothing sets, naming its flag and its variable."""
    native = [field for field in fields if FIELDS[field][1].startswith("SBATCH_")]
    sources = [
        {field: value for field in fields if (value := flags.get(field))},
        {field: value for field in native if (value := environ.get(FIELDS[field][1]))},
        {field: value for field, value in (pinned or {}).items() if field in fields},
        {field: value for field in fields if field not in native and (value := environ.get(FIELDS[field][1]))},
    ]
    systems = load_systems(environ)
    partition = next((source["partition"] for source in sources if "partition" in source), "")
    name = choose_system(system, systems, environ, partition)
    entry = systems.get(name, {})
    sources.append({field: str(value) for field, value in entry.items() if field in fields})
    merged: dict[str, str] = {}
    for source in sources:
        taken = {field for pair in PAIRS if not merged.keys().isdisjoint(pair) for field in pair} | merged.keys()
        merged |= {field: value for field, value in source.items() if field not in taken}
    values = {field: merged[field] for field in fields if field in merged}
    for field in required:
        if field not in values:
            option, variable = FIELDS[field]
            raise SystemExit(f"job submit: no {field.replace('_', ' ')}: pass {option} or set ${variable}")
    extras = {extra: str(flags.get(extra) or environ.get(EXTRAS[extra][1]) or entry.get(extra, "")) for extra in EXTRAS}
    return Resolved(name, values, extras)


def sbatch_command(
    script: str, values: Mapping[str, str], script_args: Sequence[str], logs: pathlib.Path | None = None
) -> list[str]:
    """``sbatch`` with every resolved field as an option, its Slurm output under ``logs`` when given (a
    script's own ``%x-%j.out`` lands wherever sbatch ran, the checkout included), then ``script`` and its
    arguments."""
    output = [f"--output={logs}/%x-%j.out", f"--error={logs}/%x-%j.err"] if logs is not None else []
    return [
        "sbatch",
        *(f"{FIELDS[field][0]}={value}" for field, value in values.items()),
        *output,
        script,
        *script_args,
    ]


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="hpcagent-bench job submit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--system", help="a system of systems.yaml (default: $HPCAGENT_BENCH_SYSTEM, else by cluster)")
    for field, (option, variable) in FIELDS.items():
        parser.add_argument(option, dest=field, metavar="VALUE", help=f"overrides #SBATCH, ${variable} and the system")
    parser.add_argument("--dry-run", action="store_true", help="print the sbatch command and submit nothing")
    parser.add_argument("script", type=pathlib.Path, help="the job script, e.g. docs/jobs/grade-under.sbatch")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="the script's own arguments")
    args = parser.parse_args(list(argv))
    if not args.script.is_file():
        parser.error(f"no job script {args.script}")
    environ = site_environment(os.environ)
    header = script_header(args.script)
    pinned = dict(header)
    if not args.system and (partition := image_partition(args.script, args.script_args, environ)):
        pinned["partition"] = partition
    resolved = resolve(args.system, vars(args), environ, pinned=pinned)
    options = {field: value for field, value in resolved.values.items() if header.get(field) != value}
    logs = paths.scratch_dir() / "logs"
    if not args.dry_run:
        logs.mkdir(parents=True, exist_ok=True)
    command = sbatch_command(str(args.script), options, args.script_args, logs)
    print(f"# system {resolved.system or 'none'}: {shlex.join(command)}", file=sys.stderr)
    return 0 if args.dry_run else subprocess.run(command, check=False).returncode


def options_main(argv: Sequence[str]) -> int:
    """``hpcagent-bench job options``: the ``sbatch`` options of the experiment job, one per line, or with ``--print``
    one of the other resolved values. The shell submitter reads them from here, so its flags, variables and
    ``systems.yaml`` resolve exactly as a helper job's do."""
    parser = argparse.ArgumentParser(prog="hpcagent-bench job options", description=options_main.__doc__)
    parser.add_argument("--system", help="a system of systems.yaml (default: $HPCAGENT_BENCH_SYSTEM, else by cluster)")
    for field in JOB_FIELDS:
        option, variable = FIELDS[field]
        parser.add_argument(option, dest=field, metavar="VALUE", help=f"overrides ${variable} and the system's value")
    for extra, (option, variable) in EXTRAS.items():
        parser.add_argument(option, dest=extra, metavar="VALUE", help=f"overrides ${variable} and the system's value")
    parser.add_argument(
        "--print", choices=(*JOB_FIELDS, *EXTRAS), help="print this value alone (empty when nothing sets it)"
    )
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        choices=JOB_FIELDS,
        help="fail naming its flag and variable when this field is unset (with --print; plain output requires "
        "every JOB_REQUIRED field)",
    )
    args = parser.parse_args(list(argv))
    environ = site_environment(os.environ)
    resolved = resolve(
        args.system, vars(args), environ, JOB_FIELDS, tuple(args.require) if args.print else JOB_REQUIRED
    )
    if args.print:
        print({**resolved.values, **resolved.extras}.get(args.print, ""))
    else:
        print("\n".join(f"{FIELDS[field][0]}={value}" for field, value in resolved.values.items()))
    return 0
