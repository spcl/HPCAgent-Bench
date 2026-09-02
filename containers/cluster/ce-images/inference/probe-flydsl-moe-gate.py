"""Would aiter's FlyDSL MoE actually take our checkpoint, or silently fall through?

The checkpoint is compressed-tensors pack-quantized, weights num_bits=4 type=int, NO activation
quantization -- W4A16, 61 layers, 384 experts. gfx942 has INT4/INT8 MFMA but no native MXFP4, so
the MXFP4 branches in the AMD MI350X recipe are dead here by construction and A8W4 (SiTUv2) wants
int8 activations this checkpoint does not carry.

That leaves the W4A16 FlyDSL MoE path. Enabling a switch proves nothing if the dtype gate rejects
us three frames down, so print the gate before spending six nodes on a serving smoke.
"""

import inspect
import pathlib
import re
import sys


def show(text: str, needle: str, before: int = 10, after: int = 26) -> None:
    lines = text.splitlines()
    for n, line in enumerate(lines):
        if needle in line:
            lo, hi = max(0, n - before), min(len(lines), n + after)
            print("\n".join(f"{i:5d}| {l}" for i, l in enumerate(lines[lo:hi], lo + 1)))
            return
    print(f"  (no occurrence of {needle})")


def main() -> int:
    import aiter

    src = pathlib.Path(aiter.__file__).resolve().parent / "fused_moe.py"
    text = src.read_text()
    print("fused_moe.py:", src, len(text.splitlines()), "lines\n")
    for needle in ("AITER_USE_FLYDSL_MOE", "AITER_FLYDSL_FORCE", "AITER_SITUV2_A8W4"):
        print("=" * 78, "\n", needle, sep="")
        show(text, needle)
        print()
    # Which dtypes the flydsl moe entry points declare, which is the real gate.
    try:
        from aiter.ops.flydsl import moe_kernels

        print("=" * 78, "\nflydsl moe_kernels entry points")
        for name, obj in sorted(vars(moe_kernels).items()):
            if name.startswith("__") or not callable(obj):
                continue
            try:
                print(f"  {name}{inspect.signature(obj)}")
            except (TypeError, ValueError):
                print(f"  {name}(?)")
    except Exception as exc:
        print("flydsl moe_kernels import failed:", exc)
    # Any tuned MoE config shipped for a 384-expert model.
    print("=" * 78, "\ntuned MoE configs mentioning 384 experts")
    root = pathlib.Path(aiter.__file__).resolve().parents[1]
    hits = [p for p in root.rglob("*.json") if "moe" in p.name.lower() and re.search(r"\bE=?384\b|_384_", p.name)]
    print(" ", [str(p.relative_to(root)) for p in hits[:10]] or "none")
    cfg = list((root / "aiter" / "configs").glob("*moe*")) if (root / "aiter" / "configs").is_dir() else []
    print("  aiter/configs moe files:", [p.name for p in cfg[:12]] or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
