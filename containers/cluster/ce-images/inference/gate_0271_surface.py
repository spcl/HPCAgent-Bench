"""Static checks that vLLM 0.27.1 still exposes the surface the llr4 campaign drives.

Runs inside the 0.27.1 image with no weights and no decode. Every check answers a question that
0.27.1 would otherwise answer SILENTLY at campaign scale -- an ignored env var serves untuned, a
renamed serve flag kills the head at startup, a moved internal API breaks the pp collective split
in every rank. Exit status is the gate: 0 iff every check passes.
"""

import inspect
import os
import subprocess
import sys
from pathlib import Path

import vllm.envs

FAILURES: list[str] = []


def report(name: str, ok: bool, detail: str) -> None:
    """Record one check; a False lands in FAILURES and therefore in the exit status."""
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILURES.append(name)


def class_attr(cls: type, name: str) -> object | None:
    """The attribute defined anywhere on ``cls``'s MRO, or None.

    Walks ``vars()`` per class rather than getattr, which the house rules ban: this is an
    existence probe on a third-party class, exactly the case where a descriptor or a property
    would make getattr lie about what is really defined.
    """
    for klass in cls.__mro__:
        found = vars(klass).get(name)
        if found is not None:
            return found
    return None


def serve_help() -> str:
    """The full `vllm serve` help, which is the only version-proof source of flags and choices.

    0.27.1 GROUPS the help: a bare --help prints a 4.5 KB summary that names no flag the campaign
    passes, so asking it alone reports every flag missing (600652 did exactly that). --help=all is
    the full listing; fall back to the bare form on versions that do not know the syntax.
    """
    for flag in ("--help=all", "--help"):
        done = subprocess.run(["vllm", "serve", flag], capture_output=True, text=True, check=False)
        text = done.stdout + done.stderr
        if "--tensor-parallel-size" in text:
            return text
    return text


def check_serve_flags(help_text: str) -> None:
    """Every flag the campaign passes must still be spelled the same way."""
    wanted = os.environ["CAMPAIGN_ARGS"].split()
    missing = [flag for flag in wanted if flag not in help_text]
    report(
        "serve-arg parity",
        not missing,
        f"{len(wanted) - len(missing)}/{len(wanted)} present{'; MISSING ' + ' '.join(missing) if missing else ''}",
    )


def check_parser_choices(help_text: str) -> None:
    """kimi_k2 decides whether the kimi arms can move; qwen3_xml whether the qwen arms can."""
    for parser in ("kimi_k2", "qwen3_xml"):
        report(
            f"parser {parser}",
            parser in help_text,
            "listed in serve --help"
            if parser in help_text
            else "NOT registered -- arms using it cannot move to 0.27.1",
        )


def check_tuned_config_env() -> None:
    """An unrecognised VLLM_* var only WARNS, so serving untuned looks identical to serving tuned."""
    recognised = "VLLM_TUNED_CONFIG_FOLDER" in vllm.envs.environment_variables
    report(
        "VLLM_TUNED_CONFIG_FOLDER recognised",
        recognised,
        "in envs.environment_variables" if recognised else "unknown to 0.27.1 -- the 1.71x tuning would be IGNORED",
    )

    folder = Path(os.environ["VLLM_TUNED_CONFIG_FOLDER"])
    configs = sorted(path.name for path in folder.glob("*MI300A*.json"))
    report("tuned MoE config present", bool(configs), ", ".join(configs) or f"no MI300A json under {folder}")


def check_kv_connectors() -> None:
    """The recipe's CPU KV offload names a connector by STRING, which vLLM resolves at engine
    start -- so a missing one is a crash after the whole checkpoint has loaded, not a serve-arg
    error. Ask the factory's registry instead."""
    # Deferred for the same reason as check_sibling_group_api: this drags in torch.distributed.
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    registry = vars(KVConnectorFactory).get("_registry", {})
    wanted = os.environ.get("KV_CONNECTORS", "SimpleCPUOffloadConnector").split(",")
    for name in wanted:
        name = name.strip()
        if not name:
            continue
        report(
            f"kv connector {name}",
            name in registry,
            "registered" if name in registry else f"NOT registered; have: {', '.join(sorted(registry)) or '<none>'}",
        )


def check_sibling_group_api() -> None:
    """The pp collective split calls make_sibling_device_group(group_desc=...) -- an internal API."""
    # Deferred: vllm.distributed pulls in torch.distributed, so it is imported per check rather
    # than at module scope, where an unrelated check would pay for it.
    from vllm.distributed.parallel_state import GroupCoordinator

    method = class_attr(GroupCoordinator, "make_sibling_device_group")
    if method is None:
        report("make_sibling_device_group", False, "absent -- the pp collective split cannot install")
        return
    accepts = "group_desc" in inspect.signature(method).parameters
    report(
        "make_sibling_device_group",
        accepts,
        "accepts group_desc" if accepts else f"signature changed: {inspect.signature(method)}",
    )


def check_patch_installs() -> None:
    """Import the shipped sitecustomize and install the split, the way a rank would.

    The check is that the three collectives are REBOUND, so it has to hold the originals before
    the call: after it, both the wrapper and the wrapped function answer to the same name.
    """
    # Loaded BY PATH, not by name: the interpreter imports its own sitecustomize at startup, so
    # `import sitecustomize` returns that one from sys.modules and never reads ours (600652 died
    # on the resulting AttributeError). Deferred for the same reason as above.
    import importlib.util

    from vllm.distributed.parallel_state import GroupCoordinator

    patch_path = Path(os.environ["PG_PATCH_DIR"]) / "sitecustomize.py"
    spec = importlib.util.spec_from_file_location("optarena_pg_patch", patch_path)
    sitecustomize = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sitecustomize)

    collectives = ("broadcast", "broadcast_object_list", "broadcast_tensor_dict")
    before = {name: class_attr(GroupCoordinator, name) for name in collectives}
    before_init = class_attr(GroupCoordinator, "__init__")

    sitecustomize.patch_pp_collectives()

    rebound = [name for name in collectives if class_attr(GroupCoordinator, name) is not before[name]]
    init_patched = class_attr(GroupCoordinator, "__init__") is not before_init
    report(
        "pp collective split installs",
        len(rebound) == len(collectives) and init_patched,
        f"{len(rebound)}/{len(collectives)} collectives rebound, __init__ patched={init_patched}",
    )


def main() -> int:
    help_text = serve_help()
    report("vllm serve --help", "--tensor-parallel-size" in help_text, f"{len(help_text)} chars")
    check_serve_flags(help_text)
    check_parser_choices(help_text)
    check_tuned_config_env()
    check_kv_connectors()
    check_sibling_group_api()
    check_patch_installs()

    verdict = "PASSED" if not FAILURES else "FAILED -- " + ", ".join(FAILURES)
    print(f"\nGATE 5: {verdict}", flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
