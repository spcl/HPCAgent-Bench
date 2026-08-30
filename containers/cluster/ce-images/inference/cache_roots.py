"""Every directory a serving rank writes a compiled artefact into, and the variable that moves it.

We set HF_HOME, VLLM_CACHE_ROOT, TRITON_CACHE_DIR and AITER_JIT_DIR and took that for the whole
set. It is not: 610165 compiled an aiter TEMPLATE op into ~/.aiter/build, which AITER_JIT_DIR does
not govern, in a HOME whose quota is inodes. This enumerates the rest before they are found the
same way, so the campaign can point all of them at one root.
"""
import os
import pathlib
import re
import sys

EXTRA_KNOBS = ("TRITON_CACHE_DIR", "TRITON_HOME", "TORCHINDUCTOR_CACHE_DIR", "TORCH_HOME", "TORCH_EXTENSIONS_DIR",
               "XDG_CACHE_HOME", "HF_HOME", "HUGGINGFACE_HUB_CACHE", "AITER_JIT_DIR", "AITER_ASM_DIR",
               "AMD_COMGR_CACHE_DIR", "CUDA_CACHE_PATH", "SGLANG_CACHE_DIR", "OUTLINES_CACHE_DIR",
               "FLASHINFER_WORKSPACE_DIR")

CACHE_HINT = re.compile(r"[^\n]*(?:AITER_[A-Z0-9_]*(?:DIR|HOME|ROOT|CACHE)|home\(\)|expanduser"
                        r"|\.aiter)[^\n]*")


def aiter_cache_sites() -> None:
    try:
        import aiter
    except Exception as exc:  # noqa: BLE001 -- absence is itself the answer for this image
        print("  aiter unavailable:", type(exc).__name__)
        return
    root = pathlib.Path(aiter.__file__).parent
    print("  aiter at", root)
    for path in sorted(root.rglob("*.py")):
        for hit in CACHE_HINT.findall(path.read_text(errors="ignore")):
            line = hit.strip()
            if line.startswith(("#", "import ", "from ")):
                continue
            print(f"    {path.relative_to(root)}: {line[:150]}")


def env_knobs() -> None:
    names = set(EXTRA_KNOBS)
    try:
        from vllm import envs as vllm_envs
        names |= {n for n in dir(vllm_envs) if "CACHE" in n or n.endswith("_DIR")}
    except Exception as exc:  # noqa: BLE001
        print("  vllm envs unavailable:", type(exc).__name__)
    for name in sorted(names):
        print(f"  {name} = {os.environ.get(name, 'UNSET')}")


def resolved_defaults() -> None:
    print("  HOME =", pathlib.Path.home())
    for probe in ("~/.aiter", "~/.triton", "~/.cache", "~/.config", "/tmp/aiter_configs"):
        print(f"  {probe:20s} exists={pathlib.Path(probe).expanduser().exists()}")
    try:
        import triton.runtime.cache as triton_cache
        print("  triton default cache =", triton_cache.default_cache_dir())
    except Exception as exc:  # noqa: BLE001
        print("  triton default cache : unavailable", type(exc).__name__)
    try:
        from torch._inductor import config as inductor_config
        print("  inductor cache_dir   =", vars(inductor_config).get("cache_dir", "UNSET"))
    except Exception as exc:  # noqa: BLE001
        print("  inductor             : unavailable", type(exc).__name__)


def main() -> int:
    for title, fn in (("where aiter puts its caches", aiter_cache_sites),
                      ("cache/dir env knobs these libraries read", env_knobs), ("resolved defaults with nothing set",
                                                                                resolved_defaults)):
        print(f"\n=== {title} ===", flush=True)
        fn()
    return 0


if __name__ == "__main__":
    sys.exit(main())
