#!/usr/bin/env python3

import argparse
import os
import statistics
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def log(message: str) -> None:
    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", "?")
    print(
        f"[rank={rank} local_rank={local_rank}] {message}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mb", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    # This must happen before process-group initialization.
    torch.cuda.set_device(device)

    log(
        f"selected device={device}, "
        f"current_device={torch.cuda.current_device()}"
    )
    log("initializing process group")

    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=args.timeout_seconds),
        device_id=device,
    )

    rank = dist.get_rank()
    world = dist.get_world_size()

    log(f"process group initialized, world_size={world}")

    count = args.size_mb * 1024 * 1024 // 4

    log(f"allocating {args.size_mb} MiB tensor")

    tensor = torch.full(
        (count,),
        float(rank + 1),
        device=device,
        dtype=torch.float32,
    )

    torch.cuda.synchronize(device)

    if rank == 0:
        print("starting warmups", flush=True)

    for iteration in range(args.warmup):
        tensor.fill_(float(rank + 1))
        dist.all_reduce(tensor)
        torch.cuda.synchronize(device)

        if rank == 0:
            print(
                f"warmup {iteration + 1}/{args.warmup}",
                flush=True,
            )

    dist.barrier(device_ids=[local_rank])

    if rank == 0:
        print("starting timed iterations", flush=True)

    times: list[float] = []

    for iteration in range(args.iters):
        tensor.fill_(float(rank + 1))
        torch.cuda.synchronize(device)

        start = time.perf_counter()
        dist.all_reduce(tensor)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        times.append(elapsed)

        if rank == 0 and (
            iteration == 0
            or (iteration + 1) % 10 == 0
            or iteration + 1 == args.iters
        ):
            print(
                f"iteration {iteration + 1}/{args.iters}: "
                f"{elapsed:.6f} seconds",
                flush=True,
            )

    expected = world * (world + 1) / 2

    expected_tensor = torch.tensor(
        expected,
        device=device,
        dtype=tensor.dtype,
    )

    correct = bool(
        torch.isclose(tensor[0], expected_tensor).item()
    )

    gathered: list[list[float] | None] | None
    gathered = [None] * world if rank == 0 else None

    dist.gather_object(times, gathered, dst=0)

    if rank == 0:
        assert gathered is not None

        flat = [
            elapsed
            for rank_times in gathered
            if rank_times is not None
            for elapsed in rank_times
        ]

        median = statistics.median(flat)
        payload_gb = args.size_mb / 1024
        algorithmic_bandwidth = payload_gb / median
        bus_bandwidth = (
            algorithmic_bandwidth
            * (2 * (world - 1) / world)
        )

        print(f"world_size={world}")
        print(f"size_mib={args.size_mb}")
        print(f"median_seconds={median:.6f}")
        print(
            f"algorithmic_GBps="
            f"{algorithmic_bandwidth:.3f}"
        )
        print(f"estimated_bus_GBps={bus_bandwidth:.3f}")
        print(f"correct={correct}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
