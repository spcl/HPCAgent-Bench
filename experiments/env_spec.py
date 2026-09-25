#!/usr/bin/env python3
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render a base env flat from ``layers/*.env`` and ``arms.yaml``: ``KEY=VALUE`` lines.

    env_spec.py render campaign:qwen38          # campaign "campaign" for model qwen38
    env_spec.py render layers/model-qwen38.env  # an env file and its "# extends:" parents
    env_spec.py list                            # every campaign:model a submitter can render

A campaign:model renders, lowest precedence first: ``layers/common.env``, the campaign's ``env``,
the rest of ``layers/model-<model>.env``'s chain, the campaign's ``models.<model>``. A campaign that
``extends`` another applies the parent's ``env`` (and ``models`` entry) before its own. A later key
keeps its first position. Values are copied verbatim, so ``${SCRATCH:?}`` resolves where the job
sources the result.
"""

import argparse
import enum
import pathlib
import re
import sys
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr, StringConstraints, TypeAdapter

HERE = pathlib.Path(__file__).resolve().parent
SPEC = HERE / "arms.yaml"
LAYERS = HERE / "layers"
COMMON = LAYERS / "common.env"
ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
EXTENDS = re.compile(r"^# extends: (.+)$", re.MULTILINE)
MODEL_LAYER = re.compile(r"^model-([a-z0-9]+)\.env$")

#: One member per ``layers/model-<model>.env``: adding a model is adding its layer.
Model = enum.StrEnum(
    "Model", sorted((m[1], m[1]) for path in LAYERS.glob("model-*.env") if (m := MODEL_LAYER.match(path.name)))
)

type EnvKey = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
type CampaignName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]*$")]
type Env = dict[EnvKey, StrictStr | StrictInt]


class Campaign(BaseModel):
    """Keys one campaign sets on every model, and per model on top of that model's layer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    extends: CampaignName | None = None
    env: Env = {}
    models: dict[Model, Env] = {}


SPEC_ADAPTER = TypeAdapter(dict[CampaignName, Campaign])


def load_spec(path: pathlib.Path = SPEC) -> dict[str, Campaign]:
    """``arms.yaml`` validated: campaign names, model names, env key names and scalar values."""
    spec = SPEC_ADAPTER.validate_python(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    for name, campaign in spec.items():
        if campaign.extends is not None and campaign.extends not in spec:
            raise SystemExit(f"env_spec: campaign {name} extends unknown campaign {campaign.extends}")
    return spec


def layer_chain(path: pathlib.Path, seen: tuple[pathlib.Path, ...] = ()) -> list[pathlib.Path]:
    """``path`` and its ``# extends:`` parents (each relative to its child), parents first, each once."""
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"env_spec: no such layer {path}")
    if path in seen:
        raise SystemExit(f"env_spec: extends cycle through {path}")
    chain: list[pathlib.Path] = []
    for parent in EXTENDS.findall(path.read_text(encoding="utf-8")):
        chain += [p for p in layer_chain(path.parent / parent.strip(), (*seen, path)) if p not in chain]
    return [*chain, path]


def assignments(path: pathlib.Path) -> dict[str, str]:
    """The ``KEY=VALUE`` lines ``path`` itself sets, a later one winning."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return {match[1]: match[2] for line in lines if (match := ASSIGNMENT.match(line))}


def render_file(path: pathlib.Path) -> dict[str, str]:
    """``path`` flattened through its parents."""
    env: dict[str, str] = {}
    for layer in layer_chain(path):
        env.update(assignments(layer))
    return env


def campaign_chain(name: str, spec: dict[str, Campaign]) -> list[Campaign]:
    """Campaign ``name`` and the campaigns it extends, root first."""
    names: list[str] = []
    current: str | None = name
    while current is not None:
        if current not in spec:
            raise SystemExit(f"env_spec: no campaign {current} in {SPEC.name}")
        if current in names:
            raise SystemExit(f"env_spec: extends cycle through campaign {current}")
        names.insert(0, current)
        current = spec[current].extends
    return [spec[n] for n in names]


def render_campaign(name: str, model: str, spec: dict[str, Campaign]) -> dict[str, str]:
    """Campaign ``name`` for ``model`` flattened (precedence in the module docstring)."""
    try:
        member = Model(model)
    except ValueError:
        raise SystemExit(f"env_spec: no model {model}: no layers/model-{model}.env") from None
    campaigns = campaign_chain(name, spec)
    chain = layer_chain(LAYERS / f"model-{member}.env")
    if COMMON.resolve() not in chain:
        raise SystemExit(f"env_spec: layers/model-{member}.env does not extend common.env")
    env: dict[str, str] = {}
    for layer in chain:
        env.update(assignments(layer))
        if layer == COMMON.resolve():
            for campaign in campaigns:
                env.update((key, str(value)) for key, value in campaign.env.items())
    for campaign in campaigns:
        env.update((key, str(value)) for key, value in campaign.models.get(member, {}).items())
    return env


def render(target: str, spec: dict[str, Campaign] | None = None) -> dict[str, str]:
    """``target`` rendered flat: ``<campaign>:<model>``, else a path to an env file."""
    if ":" in target:
        name, model = target.split(":", 1)
        return render_campaign(name, model, load_spec() if spec is None else spec)
    if not pathlib.Path(target).is_file():
        raise SystemExit(f"env_spec: {target} is neither <campaign>:<model> nor an env file")
    return render_file(pathlib.Path(target))


def targets(spec: dict[str, Campaign] | None = None) -> list[str]:
    """Every ``<campaign>:<model>`` that renders."""
    return [f"{name}:{model}" for name in (load_spec() if spec is None else spec) for model in Model]


def as_text(env: dict[str, str]) -> str:
    """One ``KEY=VALUE`` line per key, the form a job sources."""
    return "".join(f"{key}={value}\n" for key, value in env.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("render", help="print one campaign:model or env file flat").add_argument("target")
    sub.add_parser("list", help="print every campaign:model")
    args = parser.parse_args()
    match args.command:
        case "render":
            sys.stdout.write(as_text(render(args.target)))
        case "list":
            sys.stdout.write("".join(f"{target}\n" for target in targets()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
