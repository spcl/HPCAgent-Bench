"""Which kernel does a pack-quantized int4 checkpoint actually reach, and can it ride an fp8 one?

Two open questions:
  1. Is there an OPTIMIZED int4 kernel (asm/CK w4 MoE, a16w4 int) we are simply not selecting?
  2. Is there an UPCAST path -- load int4, keep fp8 in memory -- so the tuned fp8/FlyDSL kernels
     apply without requantizing 555 GB on disk?

Both are answered by what sglang's compressed-tensors path maps our scheme onto, and by which w4
entry points aiter exposes.
"""
import inspect
import pathlib
import re
import sys


def grep(root: pathlib.Path, pattern: str, limit: int = 14) -> None:
    rx = re.compile(pattern)
    n = 0
    for path in sorted(root.rglob("*.py")):
        try:
            for i, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
                if rx.search(line):
                    print(f"   {path.name}:{i}: {line.strip()[:120]}")
                    n += 1
                    if n >= limit:
                        return
        except OSError:
            continue
    if not n:
        print("   (none)")


def main() -> int:
    import aiter
    aroot = pathlib.Path(aiter.__file__).resolve().parent
    import sglang
    sroot = pathlib.Path(sglang.__file__).resolve().parent

    print("=" * 76, "\n1. aiter entry points carrying w4 / int4")
    names = sorted(n for n in dir(aiter) if re.search(r"(a16w4|a8w4|w4a|int4|_w4|4bit)", n, re.I))
    print("  ", names or "none")

    print("=" * 76, "\n2. QuantType members (which the MoE gate switches on)")
    try:
        from aiter import QuantType
        print("  ", [m for m in dir(QuantType) if not m.startswith("_")])
    except Exception as exc:
        print("   QuantType unavailable:", exc)

    print("=" * 76, "\n3. sglang: how compressed-tensors pack-quantized int4 is handled")
    q = sroot / "srt" / "layers" / "quantization"
    if q.is_dir():
        grep(q, r"pack.?quantized|PackedvLLMParameter|num_bits.*4|W4A16|wNa16")

    print("=" * 76, "\n4. any int4 -> fp8 / upcast-at-load path")
    for root, label in ((sroot, "sglang"), (aroot, "aiter")):
        print(f"  -- {label}")
        grep(root, r"dequant.*fp8|to_fp8|upcast|repack.*fp8|int4.*fp8|fp8.*from.*int4", limit=8)

    print("=" * 76, "\n5. the MoE dispatch's own quant branches")
    text = (aroot / "fused_moe.py").read_text()
    for m in re.finditer(r"^\s*(el)?if .*q_(dtype_w|type).*$", text, re.M):
        print("   ", m.group(0).strip()[:130])
    return 0


if __name__ == "__main__":
    sys.exit(main())
