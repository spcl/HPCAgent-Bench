#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Where a setup's inference comes from (``INFERENCE_SOURCE``): a server this job starts (``node``, default) or a
hosted endpoint (``service``, the ``INFERENCE_SERVICE_*`` block). Both resolve to a base URL, model and key
(:func:`launcher_env`); the key travels by variable NAME only. Runs on the batch host under the host python."""

import argparse
import dataclasses
import enum
import os
import pathlib
import shlex
import sys
from collections.abc import Mapping

import pydantic
from hpcagent_agent import env_aliases
from hpcagent_agent.coded import Coded

__all__ = [
    "CLAUDE_KEY_VARIABLE",
    "CLAUDE_MODEL_PINS",
    "FREE_ONLY_KEY",
    "HARNESSES_BY_API",
    "RECORD_NAME",
    "REQUIRED",
    "Api",
    "Auth",
    "NodeProvenance",
    "PricingListing",
    "Provenance",
    "Service",
    "ServiceProvenance",
    "TunerProvenance",
    "Source",
    "check_free",
    "claude_key_variable",
    "from_environ",
    "launcher_env",
    "main",
    "not_free",
    "pricing_url",
    "provenance",
    "record",
    "required_value",
    "shell_block",
    "source",
]


class Source(Coded):
    """Where a setup's inference comes from (``INFERENCE_SOURCE``, ``inference.json`` ``source``)."""

    #: A server this job starts on a GPU node.
    NODE = enum.auto()
    #: A hosted endpoint (the ``INFERENCE_SERVICE_*`` block).
    SERVICE = enum.auto()
    #: No model: a tuner setup (``TUNER`` names the optimizer).
    NONE = enum.auto()


class Api(Coded):
    """The wire format a hosted service speaks (``INFERENCE_SERVICE_API``)."""

    #: ``/v1/chat/completions``.
    OPENAI = enum.auto()
    #: ``/v1/messages``.
    ANTHROPIC = enum.auto()


class Auth(Coded):
    """How a hosted service takes its key (``INFERENCE_SERVICE_AUTH``)."""

    SPELLINGS = enum.nonmember({"KEY_HEADER": "x-api-key"})

    #: ``Authorization: Bearer <key>``.
    BEARER = enum.auto()
    #: ``x-api-key: <key>``.
    KEY_HEADER = enum.auto()


#: Harnesses that speak each wire format (by name: the agent payload is not staged on the batch host).
HARNESSES_BY_API: dict[Api, tuple[str, ...]] = {
    Api.OPENAI: ("miniswe", "optimas", "openai"),
    Api.ANTHROPIC: ("claude", "anthropic"),
}

#: Which variable the claude CLI's key belongs in, per auth spelling.
CLAUDE_KEY_VARIABLE: dict[Auth, str] = {Auth.BEARER: "ANTHROPIC_AUTH_TOKEN", Auth.KEY_HEADER: "ANTHROPIC_API_KEY"}

#: What the run records about its inference, beside the judge databases.
RECORD_NAME = "inference.json"

#: Every variable through which the claude CLI picks a model on its own; a service setup pins each to its model (a
#: router would otherwise answer with an undeclared, possibly billed model).
CLAUDE_MODEL_PINS = (
    "ANTHROPIC_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
)

#: ``INFERENCE_SERVICE_FREE_ONLY=1``: run only while the provider's own listing prices the model at zero (checked
#: at launch; a price can appear at any time).
FREE_ONLY_KEY = "INFERENCE_SERVICE_FREE_ONLY"

#: The block's keys, all required (a missing one is named, not read as "").
REQUIRED = (
    "INFERENCE_SERVICE_PROVIDER",
    "INFERENCE_SERVICE_BASE_URL",
    "INFERENCE_SERVICE_MODEL",
    "INFERENCE_SERVICE_TIER",
    "INFERENCE_SERVICE_API",
    "INFERENCE_SERVICE_AUTH",
    "INFERENCE_SERVICE_KEY_ENV",
)


@dataclasses.dataclass(frozen=True, slots=True)
class Service:
    """One hosted inference service, as a setup declares it. Never holds the key."""

    provider: str
    base_url: str
    model: str
    tier: str
    api: Api
    auth: Auth
    key_env: str


def source(environ: Mapping[str, str]) -> Source:
    """Which source this setup runs against. An unknown spelling ends the launch."""
    declared = environ.get("INFERENCE_SOURCE", "").strip()
    try:
        return Source.parse(declared) if declared else Source.NODE
    except ValueError as exc:
        raise SystemExit(f"INFERENCE_SOURCE: {exc}") from exc


def required_value(environ: Mapping[str, str], key: str) -> str:
    value = environ.get(key, "").strip()
    if not value:
        raise SystemExit(f"a service setup must set {key}")
    return value


def from_environ(environ: Mapping[str, str]) -> Service:
    """The setup's service block, fully checked. Raises SystemExit with the offending key."""
    provider, base_url, model, tier, api, auth, key_env = (required_value(environ, key) for key in REQUIRED)
    try:
        wire = Api.parse(api)
    except ValueError as exc:
        raise SystemExit(f"INFERENCE_SERVICE_API={api!r} is not a wire format: {exc}") from exc
    try:
        spelling = Auth.parse(auth)
    except ValueError as exc:
        raise SystemExit(f"INFERENCE_SERVICE_AUTH={auth!r} is not an auth spelling: {exc}") from exc
    resolved = Service(provider, base_url, model, tier, wire, spelling, key_env)
    # A service setup must declare zero inference nodes.
    nodes = environ.get("INFERENCE_NODES", "0").strip() or "0"
    if nodes != "0":
        raise SystemExit(f"a service setup starts no server; set INFERENCE_NODES=0, not {nodes}")
    harness = env_aliases.resolve("HARNESS", "", environ).strip() or "claude"
    accepted = HARNESSES_BY_API[resolved.api]
    if harness not in accepted:
        raise SystemExit(
            f"HARNESS={harness} cannot speak the {resolved.api.text} wire format this service serves; "
            f"expected one of {accepted}"
        )
    # Checked before any agent starts: a run of 401s leaves no measurement.
    if not environ.get(resolved.key_env, "").strip():
        raise SystemExit(f"{resolved.key_env} is not set in the launching environment; export the service key there")
    return resolved


def claude_key_variable(service: Service) -> str:
    """The environment variable the claude CLI's key belongs in for this service."""
    return CLAUDE_KEY_VARIABLE[service.auth]


def launcher_env(service: Service) -> dict[str, str]:
    """The endpoint values run_cluster.sh exports, in the names every consumer reads: one replica, and
    VLLM_MASTER_HOST emptied (no node serves this setup)."""
    pins = dict.fromkeys(CLAUDE_MODEL_PINS, service.model) if service.api is Api.ANTHROPIC else {}
    return {
        **pins,
        "VLLM_MASTER_HOST": "",
        "VLLM_BASE_URL": service.base_url,
        "VLLM_REPLICA_URLS": service.base_url,
        "VLLM_SERVED_MODEL": service.model,
        "INFERENCE_KEY_ENV": service.key_env,
        "INFERENCE_CLAUDE_KEY_VARIABLE": claude_key_variable(service),
    }


class Endpoint(pydantic.BaseModel):
    """One provider serving the model in a :class:`PricingListing`."""

    model_config = pydantic.ConfigDict(extra="ignore", frozen=True)

    provider_name: str = "?"
    #: Metered unit -> price, as the provider spells it (a string number, or a discount number).
    pricing: dict[str, str | float] = pydantic.Field(default_factory=dict)


class ListingData(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="ignore", frozen=True)

    id: str
    endpoints: list[Endpoint] = pydantic.Field(default_factory=list)


class PricingListing(pydantic.BaseModel):
    """The body of :func:`pricing_url` (OpenRouter's ``/models/<id>/endpoints``)."""

    model_config = pydantic.ConfigDict(extra="ignore", frozen=True)

    data: ListingData | None = None


def pricing_url(service: Service) -> str:
    """The OpenRouter-style listing of the providers serving ``service.model`` and their prices."""
    return f"{service.base_url.rstrip('/')}/models/{service.model}/endpoints"


def not_free(listing: PricingListing, model: str) -> str | None:
    """Why ``listing`` does not show ``model`` as free, or None: every endpoint must price EVERY metered unit at
    zero; no endpoints or another model's listing is refused."""
    data = listing.data
    if data is None or data.id != model:
        return f"the provider's listing does not describe {model}"
    if not data.endpoints:
        return f"the provider lists no endpoint serving {model}"
    for endpoint in data.endpoints:
        if not endpoint.pricing:
            return f"an endpoint for {model} publishes no pricing"
        for unit, price in endpoint.pricing.items():
            if unit == "discount":
                continue
            try:
                charged = float(price)
            except (TypeError, ValueError):
                return f"{model} prices {unit} as {price!r}, which is not a number"
            if charged != 0.0:
                return f"{model} is not free: {endpoint.provider_name} charges {price} per {unit}"
    return None


def check_free(service: Service, timeout: float = 30.0) -> None:
    """Refuse the launch unless the provider currently lists ``service.model`` as free."""
    import urllib.request

    try:
        with urllib.request.urlopen(pricing_url(service), timeout=timeout) as response:
            listing = PricingListing.model_validate_json(response.read())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot confirm {service.model} is free ({pricing_url(service)}): {exc}") from exc
    reason = not_free(listing, service.model)
    if reason is not None:
        raise SystemExit(f"{FREE_ONLY_KEY}=1 and {reason}; refusing to launch")


def shell_block(exported: Mapping[str, str]) -> str:
    """``exported`` as :func:`shlex.quote`-d assignments for the launcher to eval."""
    return "\n".join(f"{key}={shlex.quote(value)}" for key, value in exported.items())


class NodeProvenance(pydantic.BaseModel):
    """``inference.json`` of a setup whose job started its own server."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    source: Source = Source.NODE
    engine: str
    ce_env: str
    model: str


class ServiceProvenance(pydantic.BaseModel):
    """``inference.json`` of a setup served by a hosted service; names the key's variable, never the key."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    source: Source = Source.SERVICE
    provider: str
    model: str
    tier: str
    api: Api
    base_url: str
    key_env: str


class TunerProvenance(pydantic.BaseModel):
    """``inference.json`` of a tuner setup: no model, the optimizer ``TUNER`` names."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    source: Source = Source.NONE
    tuner: str


#: ``inference.json``, any source (told apart by ``source``).
Provenance = NodeProvenance | ServiceProvenance | TunerProvenance


def provenance(environ: Mapping[str, str]) -> Provenance:
    """What produced this run's tokens: engine, image and checkpoint, or provider, model id and TIER (unrecoverable
    afterwards)."""
    declared = source(environ)
    if declared is Source.NONE:
        return TunerProvenance(tuner=environ.get("TUNER", "").strip())
    if declared is Source.NODE:
        return NodeProvenance(
            engine=environ.get("INFERENCE_ENGINE", "").strip() or "vllm",
            ce_env=environ.get("INFERENCE_CE_ENV", "").strip(),
            model=environ.get("VLLM_MODEL", "").strip(),
        )
    service = from_environ(environ)
    return ServiceProvenance(
        provider=service.provider,
        model=service.model,
        tier=service.tier,
        api=service.api,
        base_url=service.base_url,
        key_env=service.key_env,
    )


def record(run_dir: pathlib.Path, environ: Mapping[str, str]) -> Provenance:
    """Write the run's inference provenance and return it (holds no key by construction)."""
    written = provenance(environ)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RECORD_NAME).write_text(written.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return written


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Resolve a setup's inference source.")
    parser.add_argument("--export", action="store_true", help="print the endpoint block for the launcher to eval")
    parser.add_argument(
        "--check-free",
        action="store_true",
        help=f"when the setup sets {FREE_ONLY_KEY}=1, fail unless its model is free",
    )
    parser.add_argument("--record", type=pathlib.Path, help="write the run's inference provenance into this directory")
    args = parser.parse_args(argv)
    if args.check_free and os.environ.get(FREE_ONLY_KEY, "").strip() == "1":
        check_free(from_environ(os.environ))
    if args.record is not None:
        record(args.record, os.environ)
    if args.export:
        print(shell_block(launcher_env(from_environ(os.environ))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
