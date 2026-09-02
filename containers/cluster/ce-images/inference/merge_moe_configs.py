"""Merge the per-batch-size fused_moe tuner outputs into the ONE file vLLM loads.

Each tuning job (tune-moe-int4-mi300a.sbatch) writes its own copy of the target JSON holding
only the sizes it tuned, so the campaign config is their union. vLLM picks the nearest tuned
size at run time, which is why merging rather than replacing matters: a partial re-tune must
not drop sizes an earlier job won.

ONLY RUN THIS ONCE EVERY SOURCE JOB HAS COMPLETED. The tuner checkpoints its best-so-far to
the same filename it writes at the end, so a partial result is byte-indistinguishable from a
final one -- and a later job id sorts after an earlier one, so an in-flight job silently
overrides a finished one. Check squeue first.

N=512 only. Job 595206 emitted an N=4608 file from the DENSE intermediate_size (the multimodal
wrapper defeats benchmark_moe's model-params helper); serving never looks that shape up.
"""

import argparse
import glob
import json
import os
from pathlib import Path

TARGET = "E=384,N=512,device_name=AMD_Instinct_MI300A,dtype=int4_w4a16.json"
REPO_DIR = Path(__file__).resolve().parent / "moe-configs"
SCRATCH = os.environ.get("SCRATCH", "/capstor/scratch/cscs/ybudanaz/x86_64")
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
            # SPLIT_K > 1 needs a reduction workspace the serving path may not allocate, and the
            # failure mode is silent zeros rather than an error. Exploratory TUNE_SPLIT_K runs
            # land in the same glob and sort late, so without this guard they would quietly
            # overwrite a servable size. Ship one only after a correctness run.
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
