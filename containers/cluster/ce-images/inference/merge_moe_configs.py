"""Merge the per-batch-size fused_moe tuner outputs into the ONE file vLLM loads.

Each tuning job writes its own copy of the target JSON holding only the sizes it tuned, so the
campaign config is their union; a partial re-tune must not drop sizes an earlier job won.

Only run this once every source job has completed: the tuner checkpoints its best-so-far to the
same filename it writes at the end, so a partial result is byte-indistinguishable from a final
one, and an in-flight job can silently override a finished one. Check squeue first.
"""

import argparse
import glob
import json
import os
from pathlib import Path

TARGET = "E=384,N=512,device_name=AMD_Instinct_MI300A,dtype=int4_w4a16.json"
REPO_DIR = Path(__file__).resolve().parent / "moe-configs"
SCRATCH = os.environ["SCRATCH"]
RUNS_GLOB = f"{SCRATCH}/kimi-smoke/*/{TARGET}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write the merge; default is a dry run")
    args = ap.parse_args()

    out_path = REPO_DIR / TARGET
    merged: dict[str, dict] = {}
    versions: set[str] = set()
    provenance: dict[str, str] = {}

    sources = sorted(glob.glob(RUNS_GLOB))
    if out_path.exists():
        sources.insert(0, str(out_path))  # the committed config is the base; later jobs override it

    for path in sources:
        with open(path) as fh:
            data = json.load(fh)
        version = data.pop("triton_version", None)
        if version is not None:
            versions.add(version)
        for size, cfg in data.items():
            # SPLIT_K > 1 needs a reduction workspace the serving path may not allocate (silent
            # zeros, not an error); ship one only after a correctness run.
            if cfg.get("SPLIT_K", 1) != 1:
                print(
                    f"  SKIP bs={size} from {Path(path).parent.name}: "
                    f"SPLIT_K={cfg['SPLIT_K']} is unverified for serving"
                )
                continue
            merged[size] = cfg
            provenance[size] = Path(path).parent.name

    if len(versions) > 1:
        raise SystemExit(f"refusing to merge across triton versions: {sorted(versions)}")

    for size in sorted(merged, key=int):
        print(f"  bs={size:<6} from {provenance[size]}")
    print(f"{len(merged)} tuned sizes, triton {versions or 'unknown'}")

    if not args.write:
        print(f"DRY RUN -- would write {out_path}")
        return
    payload: dict = {"triton_version": versions.pop()} if versions else {}
    payload.update({k: merged[k] for k in sorted(merged, key=int)})
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=4)
        fh.write("\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
