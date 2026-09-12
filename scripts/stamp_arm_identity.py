# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Repair a generated arm ``.env`` that predates the identity columns, ONCE.

Every launcher stamps the full identity today (``experiments/record_identity.sh``), so a freshly
submitted arm needs nothing from this. What it repairs is the arms on disk from an EARLIER
generation: 18 of them carry ``HPCAGENT_BENCH_RECORD_EXPERIMENT`` and nothing else, under an
experiment name that was really a device and a packet (``gpu-llr-focus40``, ``cpf-llr-focus40``).
A half-stamped arm is worse than an unstamped one -- its rows join on experiment and then group
into a NULL model, which draws as an extra model in every per-model figure.

THIS IS THE ONE PLACE ALLOWED TO PARSE AN ARM NAME, and it shares
:func:`scripts.migrate_db.parse_arm` with the database migration so the env and the rows written
under it can never disagree. It parses so that nothing at runtime ever has to: the result is
written down as columns, and the query groups on those.

Idempotent. An env that already carries a value keeps it -- the launcher that wrote it knew more
than a name does -- and only missing keys are added.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

from hpcagent_bench.experiment_tags import canonical

from migrate_db import UNATTRIBUTED, parse_arm

#: Generator SEEDS, not arms: a template a launcher copies and then stamps.
SEED = re.compile(r"^\.env\.(base|llrbase)-")

KEYS = ("EXPERIMENT", "MODEL", "LANGUAGE", "DEVICE", "PACKET", "ARM")


def stamped(text: str) -> dict[str, str]:
    """The identity keys an env already sets. ``ENABLED`` shares the prefix and is not a column."""
    found = re.findall(r"^HPCAGENT_BENCH_RECORD_([A-Z]+)=(.*)$", text, re.MULTILINE)
    return {key: value.strip() for key, value in found if key != "ENABLED"}


def repair(env: pathlib.Path, apply: bool) -> str:
    """Add whatever identity ``env`` is missing. Returns a one-line report."""
    arm = env.name.removeprefix(".env.")
    if SEED.match(env.name):
        return f"seed     {arm}"
    if arm in UNATTRIBUTED:
        # A smoke proves the plumbing, not a hypothesis. Recording an experiment for it puts a
        # phantom condition in every figure, so any stamp it carries is REMOVED rather than
        # completed -- migrate_db drops its rows for the same reason.
        text = env.read_text(encoding="utf-8")
        stale = [
            line
            for line in text.splitlines()
            if line.startswith("HPCAGENT_BENCH_RECORD_") and not line.startswith("HPCAGENT_BENCH_RECORD_ENABLED")
        ]
        if stale and apply:
            keep = [ln for ln in text.splitlines() if ln not in stale]
            env.write_text("\n".join(keep) + "\n", encoding="utf-8")
        return f"smoke    {arm}" + (f" (dropped {len(stale)} stale stamp(s))" if stale else "")
    try:
        experiment, model, language, device, packet = parse_arm(arm)
    except ValueError as exc:
        return f"UNMAPPED {arm}: {exc}"

    text = env.read_text(encoding="utf-8")
    have = stamped(text)
    want = dict(zip(KEYS, (experiment, model, language, device, packet, arm)))
    # PACKET is the one key whose correct value can be empty, so "missing" is absence of the key
    # rather than a falsy value -- otherwise every control arm is rewritten on every run.
    add = {k: v for k, v in want.items() if k not in have}

    # EXPERIMENT is the one key REWRITTEN rather than only filled in. The stale stamps are
    # `gpu-llr-focus40` and `cpf-llr-focus40` -- a device and a packet worn as an experiment name,
    # which is the thing this whole schema exists to stop. The registry's aliases say what each
    # resolves to, so an already-correct value is left exactly as it is.
    fix = {}
    if "EXPERIMENT" in have and canonical("experiments", have["EXPERIMENT"]) != have["EXPERIMENT"]:
        fix["EXPERIMENT"] = canonical("experiments", have["EXPERIMENT"])

    if not add and not fix:
        return f"ok       {arm}"
    if apply:
        for key, value in fix.items():
            text = re.sub(
                rf"^HPCAGENT_BENCH_RECORD_{key}=.*$", f"HPCAGENT_BENCH_RECORD_{key}={value}", text, flags=re.MULTILINE
            )
        if not text.endswith("\n"):
            text += "\n"
        text += "".join(f"HPCAGENT_BENCH_RECORD_{k}={v}\n" for k, v in add.items())
        env.write_text(text, encoding="utf-8")
    changes = " ".join(f"{k}={v!r}" for k, v in {**fix, **add}.items())
    return f"{'stamped ' if apply else 'would   '} {arm}: {changes}"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("envs", nargs="*", help="arm .env files (default: every .env.* beside the launchers)")
    ap.add_argument("--apply", action="store_true", help="write; without it nothing is changed")
    args = ap.parse_args(argv)

    targets = [pathlib.Path(e) for e in args.envs] or sorted(
        (pathlib.Path(__file__).resolve().parents[1] / "experiments").glob(".env.*")
    )
    reports = [repair(env, args.apply) for env in targets if env.is_file()]
    for line in reports:
        print(line)
    unmapped = [r for r in reports if r.startswith("UNMAPPED")]
    print(f"\n{len(targets)} env(s), {len(unmapped)} unmapped")
    return 1 if unmapped else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
