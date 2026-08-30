"""The a16w4 fused-MoE tuned-table format, and what a Kimi row would have to say.

aiter ships gptoss_a16w4_{un,}tuned_fmoe.csv -- A16W4 is int4 weights with 16-bit activations,
which is exactly our checkpoint's shape (compressed-tensors pack-quantized, num_bits=4, type=int,
no activation quant). There is a tuner for it, csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py,
and NO Kimi table. That is the untuned int4 MoE kernel, in the open.

Print the schema and the model geometry so a Kimi untuned CSV can be written against it.
"""
import json
import pathlib
import sys


def main() -> int:
    import aiter
    cfg = pathlib.Path(aiter.__file__).resolve().parent / "configs" / "model_configs"
    for name in ("gptoss_a16w4_untuned_fmoe.csv", "gptoss_a16w4_tuned_fmoe.csv"):
        p = cfg / name
        print("=" * 74, f"\n{name}")
        if not p.is_file():
            print("  missing")
            continue
        lines = p.read_text().splitlines()
        print("  rows:", len(lines) - 1)
        for line in lines[:6]:
            print("   ", line)

    print("=" * 74, "\nthe tuner's own arguments")
    tuner = pathlib.Path(
        aiter.__file__).resolve().parents[1] / "csrc" / "ck_gemm_moe_2stages_codegen" / "gemm_moe_tune.py"
    if tuner.is_file():
        text = tuner.read_text()
        for line in text.splitlines():
            if "add_argument" in line or line.strip().startswith(("-", '"--')):
                print("   ", line.strip()[:150])
    else:
        print("  tuner not at", tuner)

    print("=" * 74, "\nour model geometry")
    mp = pathlib.Path("/iopsstor/scratch/cscs/ybudanaz/hf/hub/models--moonshotai--Kimi-K2.7-Code/"
                      "snapshots/74797c9c62378b951a1f6fcf5c4631024e9b8bef/config.json")
    c = json.loads(mp.read_text())
    t = c.get("text_config") or c
    for k in ("num_hidden_layers", "hidden_size", "intermediate_size", "moe_intermediate_size", "n_routed_experts",
              "num_experts_per_tok", "n_shared_experts", "num_attention_heads"):
        print(f"   {k:24s} {t.get(k)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
