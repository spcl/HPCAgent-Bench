# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The server arguments of a served model: overlays for a benchmark, and the rules a configuration must meet.

``[bench.<name>]`` of ``experiments/experiments.toml`` is one benchmark configuration: a model, flags to set or drop in
the model's ``SGLANG_EXTRA_ARGS`` and other env keys to override. ``serve-only.sbatch`` applies it under
``SERVE_BENCH=<name>`` (``hpcagent-bench serve-bench submit``).

    python -m hpcagent_bench.cluster.serve_args check ENV_FILE      # the rules, on a staged setup env (submit refuses)
    python -m hpcagent_bench.cluster.serve_args bench NAME --base experiment:MODEL   # the overlay's env lines

The rules (:data:`RULES`) hold for a launch and for a benchmark alike; a benchmark that hunts a crash on purpose skips
them with ``--force-rules``.
"""

import argparse
import pathlib
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated

import pydantic

from hpcagent_bench.cluster import env_spec

__all__ = [
    "ARGS_KEY",
    "INT4_CHECKPOINT",
    "MOE_CHECKPOINT_KEY",
    "RULES",
    "BenchConfig",
    "apply",
    "bench_env",
    "main",
    "parse",
    "refusal",
]

#: The env key holding a model's SGLang arguments, one string.
ARGS_KEY = "SGLANG_EXTRA_ARGS"
#: The model key naming how its MoE weights are stored (``[model.<model>]``); a rule keys on it, never on a model name.
MOE_CHECKPOINT_KEY = "MOE_CHECKPOINT"
#: The int4 pack-quantized checkpoint: on ROCm its MoE runs the Triton runner, which indexes local expert ids.
INT4_CHECKPOINT = "int4-wna16"

type Flags = dict[str, str]
type Rule = Callable[[Flags, Mapping[str, str]], str]


class BenchConfig(pydantic.BaseModel):
    """One ``[bench.<name>]`` table."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    #: The ``[model.<model>]`` whose arguments the overlay edits.
    model: str
    #: Flags to set (``"--ep-size" = "4"``; ``""`` is a bare flag), then flags to remove with their value.
    set: dict[Annotated[str, pydantic.StringConstraints(pattern=r"^--[a-z0-9-]+$")], str] = {}
    drop: tuple[Annotated[str, pydantic.StringConstraints(pattern=r"^--[a-z0-9-]+$")], ...] = ()
    #: Other env keys to override (``AITER_USE_FLYDSL_MOE_SORTING = "1"``).
    env: env_spec.Env = {}


def unquoted(value: str) -> tuple[str, bool]:
    """``value`` without the pair of double quotes a rendered env wraps it in, and whether it had them."""
    quoted = len(value) >= 2 and value[0] == value[-1] == '"'
    return (value[1:-1] if quoted else value), quoted


def parse(value: str) -> Flags:
    """The flags of an argument string, in order: ``--flag value`` or a bare ``--flag`` (value ``""``)."""
    found: Flags = {}
    words = unquoted(value)[0].split()
    position = 0
    while position < len(words):
        flag = words[position]
        if not flag.startswith("--"):
            raise ValueError(f"{flag!r} is not a --flag in {value!r}")
        if flag in found:
            raise ValueError(f"{flag} twice in {value!r}")
        position += 1
        found[flag] = ""
        if position < len(words) and not words[position].startswith("--"):
            found[flag] = words[position]
            position += 1
    return found


def apply(value: str, set_flags: Mapping[str, str], drop: Sequence[str]) -> str:
    """``value`` with ``drop`` removed, then ``set_flags`` replaced in place or appended; quoted as it was."""
    flags = {flag: setting for flag, setting in parse(value).items() if flag not in drop}
    flags.update(set_flags)
    text = " ".join(word for flag, setting in flags.items() for word in (flag, setting) if word)
    return f'"{text}"' if unquoted(value)[1] else text


def ep_on_int4(flags: Flags, env: Mapping[str, str]) -> str:
    """Expert parallelism on the int4 checkpoint needs the Triton MoE runner: the dispatcher keeps global expert ids for
    the aiter runner, while the int4 MoE on ROCm runs ``CompressedTensorsWNA16TritonMoE``, which indexes local ids (an
    illegal memory access at CUDA-graph capture)."""
    if (
        env.get(MOE_CHECKPOINT_KEY) == INT4_CHECKPOINT
        and int(flags.get("--ep-size", "1")) > 1
        and flags.get("--moe-runner-backend") != "triton"
    ):
        return (
            f"--ep-size {flags['--ep-size']} on the {INT4_CHECKPOINT} checkpoint needs --moe-runner-backend triton "
            "(the aiter MoE runner indexes global expert ids; docs/serving/kimi27sglang.md)"
        )
    return ""


#: Every rule of a serving configuration, each returning why it is broken, else ``""``.
RULES: tuple[Rule, ...] = (ep_on_int4,)


def refusal(env: Mapping[str, str]) -> str:
    """Why the server arguments of ``env`` break a rule, ``""`` when they keep them all."""
    if ARGS_KEY not in env:
        return ""
    try:
        flags = parse(env[ARGS_KEY])
    except ValueError as exc:
        return f"{ARGS_KEY} is not a list of --flags: {exc}"
    return "; ".join(message for rule in RULES if (message := rule(flags, env)))


def bench_env(name: str, base: str) -> dict[str, str]:
    """``[bench.<name>]`` applied to the env ``base`` renders (``<experiment>:<model>``): the keys it changes."""
    tables = env_spec.TABLES.get("bench", {})
    if name not in tables:
        raise SystemExit(f"serve_args: no [bench.{name}] in experiments.toml; have {sorted(tables)}")
    config = BenchConfig.model_validate(tables[name])
    if not base.endswith(f":{config.model}"):
        raise SystemExit(f"serve_args: [bench.{name}] is for {config.model}, not {base}")
    rendered = env_spec.render(base)
    changed = {ARGS_KEY: apply(rendered[ARGS_KEY], config.set, config.drop)} | {
        key: str(value) for key, value in config.env.items()
    }
    return changed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="serve_args", description=__doc__.split("\n\n")[0] if __doc__ else None)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="refuse a staged setup env that breaks a rule")
    check.add_argument("env", type=pathlib.Path)
    bench = sub.add_parser("bench", help="print the env lines [bench.NAME] changes")
    bench.add_argument("name")
    bench.add_argument("--base", required=True, help="<experiment>:<model>")
    bench.add_argument("--force-rules", action="store_true", help="skip the rules: a crash hunt")
    args = parser.parse_args(argv)
    if args.command == "check":
        if message := refusal(env_spec.assignments(args.env)):
            print(f"serve_args: {args.env.name}: {message}", file=sys.stderr)
            return 2
        return 0
    changed = bench_env(args.name, args.base)
    if not args.force_rules and (message := refusal(env_spec.render(args.base) | changed)):
        print(f"serve_args: [bench.{args.name}]: {message} (--force-rules to run it anyway)", file=sys.stderr)
        return 2
    sys.stdout.write(env_spec.as_text(changed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
