"""What CAN be tuned for an int4 (W4A16) MoE on gfx942, now that FlyDSL is ruled out?

aiter's FlyDSL MoE gate requires q_dtype_w == fp4x2 (MXFP4) or fp8. Our checkpoint is
compressed-tensors pack-quantized int4 -- num_bits=4, type=int -- so every FlyDSL MoE branch is
closed on this hardware, and AITER_FLYDSL_FORCE already defaults to "1" so it was never the switch
that mattered.

That leaves the CK / asm int4 kernels the model actually dispatches to. Find their tuners and
their tuned tables, the same way the GEMM one was found.
"""
import pathlib
import sys


def main() -> int:
    import aiter
    root = pathlib.Path(aiter.__file__).resolve().parents[1]
    print("aiter root:", root, "\n")

    print("=" * 74, "\nMoE tuners on disk")
    tuners = sorted(p for p in root.rglob("*tune*.py") if "moe" in p.name.lower())
    for p in tuners:
        print("  ", p.relative_to(root))
    if not tuners:
        print("   none")

    print("=" * 74, "\nMoE tuned tables shipped")
    cfgdir = root / "aiter" / "configs"
    tables = sorted(p for p in cfgdir.rglob("*") if p.is_file() and "moe" in p.name.lower())
    for p in tables[:25]:
        print(f"   {p.relative_to(cfgdir)}  ({p.stat().st_size} bytes)")
    if not tables:
        print("   none")

    print("=" * 74, "\nuntuned-collection env switches (the AITER_TUNE_* family)")
    import subprocess
    out = subprocess.run(["grep", "-rhoE", r"AITER_[A-Z0-9_]*TUNE[A-Z0-9_]*",
                          str(root)],
                         capture_output=True,
                         text=True).stdout.split()
    for name in sorted(set(out)):
        print("  ", name)

    print("=" * 74, "\nint4 / a16w4 MoE entry points")
    names = sorted(
        n for n in dir(aiter)
        if ("moe" in n.lower() or "fmoe" in n.lower()) and any(t in n.lower()
                                                               for t in ("int4", "a16", "g1u1", "cktile", "ck_moe")))
    for n in names:
        print("  ", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
