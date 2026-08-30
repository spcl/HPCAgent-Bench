"""Which of the MI350X/MI355X Kimi recipe switches does THIS gfx942 build honour?

A switch nothing greps for is a switch that does nothing here, whatever it does on gfx950.
Written as a file rather than a heredoc so the sbatch does not nest quotes three deep.
"""
import pathlib
import subprocess
import sys

VARS = [
    "SGLANG_USE_AITER",
    "HIP_FORCE_DEV_KERNARG",
    "SGLANG_EXPERT_PARALLEL_SIZE",
    "SGLANG_USE_DYNAMIC_MXFP4_LINEAR",
    "TORCH_BLAS_PREFER_HIPBLASLT",
    "TENSILE_STREAMK_DYNAMIC_GRID",
    "AITER_QUICK_REDUCE_QUANTIZATION",
    "AITER_USE_FLYDSL_MOE_SORTING",
    "AITER_AR_1STAGE_MAX_KB",
    "AITER_MXFP4_INTERMEDIATE",
    "ROCM_QUICK_REDUCE_QUANTIZATION",
    "AITER_USE_FLYDSL_MOE",
    "AITER_FLYDSL_FORCE",
    "AITER_SITUV2_A8W4",
]
FLAGS = [
    "enable_aiter_allreduce_fusion", "attention_backend", "disable_radix_cache", "expert_parallel_size",
    "enable_dp_attention"
]


def roots() -> list[pathlib.Path]:
    out = []
    for mod in ("aiter", "sglang"):
        try:
            imported = __import__(mod)
        except Exception as exc:  # a build without one of them is a finding, not a crash
            print(f"{mod}: import failed: {exc}")
            continue
        out.append(pathlib.Path(imported.__file__).resolve().parent)
    return out


def main() -> int:
    import torch
    print("gfx:", torch.cuda.get_device_properties(0).gcnArchName)
    trees = roots()
    print("trees:", [str(t) for t in trees], "\n")
    for name in VARS + FLAGS:
        hits = []
        for tree in trees:
            found = subprocess.run(["grep", "-rl", name, str(tree)], capture_output=True, text=True)
            hits += [h.replace(str(tree.parent) + "/", "") for h in found.stdout.split()]
        mark = "HONOURED" if hits else "ignored "
        print(f"{mark} {name:34s} {len(hits):3d}  {hits[:3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
