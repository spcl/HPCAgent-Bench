# DESIGN: job submission -- one allocation, three shapes

High-level summary of how a run reaches a cluster. Three submission shapes exist because three
things are being distributed, not because three scripts drifted apart.

| shape | what is distributed | ranks talk? | script |
|---|---|---|---|
| **corpus sweep** | the KERNEL LIST across ranks | no | `submit_deterministic.sbatch`, `cscs/submit_loop_level_reasoning_alps.sbatch` |
| **role deployment** | ROLES (inference / judge / optimizer) across nodes | via the launcher, not MPI | `submit_launch.sbatch` |
| **problem decomposition** | ONE KERNEL across ranks | yes, MPI | `submit_mpi_scaling.sbatch`, `cscs/submit_mpi_scaling_alps.sbatch` |

`submit_xl.sbatch` is not a fourth shape: it is the corpus sweep at the `XL` rung, with both
native tracks run back to back in one allocation under distinct `RUN_TAG`s. It resolves every
selector before spending allocation time and refuses a non-`XL` preset, because the four-ranks-
per-node reasoning below is about `XL` specifically.

## 1. Corpus sweep -- static round-robin, no coordination

`sbatch -N 8` gives a nodelist. One task per node. Each rank reads `SLURM_PROCID` and takes
`--shard rank/N`, which is `names[rank::N]` — a stride, so neighbours in the sorted list (similar
sizes) land on different ranks. Every rank computes the same partition alone: no master, no work
stealing, and the same job twice produces the same split, which matters because the results DB is
keyed by shard.

Each rank writes its OWN CSV and DB shard. One shared file is not an option: SQLite WAL needs a
`-shm` mapping no parallel filesystem provides.

Container: one per rank, `--environment=$EDF` on every `srun` step (Alps) or the exec wrapper
locally. A step without it silently runs on the bare node, which reads as a broken environment
rather than a missing flag — so the factory refuses instead.

**Ranks per node.** Non-agent mode runs NATIVE, so one rank is one deterministic optimizer that
compiles, runs and validates its own output -- no inference endpoint, no judge, nothing to place
beside it. Four per node is what the memory allows: `sizing.TRACK_XL_CEILING` caps an `XL` working
set at 4 GB for both `loop_level_reasoning` and `scientific_computing`, so four ranks hold ~16 GB
of live data. The 20 GB `sizing.kernel_memory_gb` floor each rank imposes on its children is a
`RLIMIT_AS` limit, not a reservation, and the ceiling that sized the data is five times tighter,
so four caps cannot bind at once.

**Planned change** (`DESIGN_static_workload_distribution.md`): replace the stride with an LPT
bin-pack now that per-kernel cost is known, subject to a per-node memory cap.

## 2. Role deployment -- rank number IS the role

One `srun` across the whole allocation, one task per node; `cluster_launch.py` maps rank to role.
Agentic: ranks `[0, I*K)` serve inference, `[I*K, I*K+J)` judge, rank 0 also drives. Traditional:
ranks `[0, O)` optimize, then judge. The allocation is checked up front — a rank that never gets
a role would otherwise wait out the entire time limit before anyone hears about it.

### vLLM

vLLM is **not** an MPI program and never enters the PMI story. It is a server: one container per
node, exposing one URL.

- One node per endpoint is the default. Tensor-parallel across that node's 4 GH200 GPUs.
- A model too big for one node sets `NODES_PER_VLLM=K`: K containers form a **ray** cluster
  behind ONE URL — tensor-parallel within each node, pipeline-parallel across the K. Ray, not
  MPI; the transport is NCCL over the fabric.
- No container ever spans a node. K nodes means K containers plus a ray head.
- What the container needs from the site: the GPUs (`--device nvidia.com/gpu=all` / the CE GPU
  hook) and, for K>1, the fabric — without it NCCL falls back to TCP over the management network
  and it reads as a slow model rather than a misconfigured launch.
- The driver only ever sees an HTTP endpoint, so a hosted API and a locally served model are the
  same thing to everything downstream.

## 3. Problem decomposition -- MPI inside containers

The part with real risk. Verified working at 2/4/8 ranks locally (`jacobi_2d`, `heat_3d`,
bit-exact against the 1-rank result); never yet run across nodes. This section is about getting
the ranks placed and connected; what a kernel does with them once connected (halo exchange, data
distribution) is [mpi_patterns.md](mpi_patterns.md) and
[`hpcagent_bench/docs/mpi_distributions.md`](../hpcagent_bench/docs/mpi_distributions.md).

**The model.** One container per rank, one rank per container. Containers do not cluster —
Slurm places them and MPI connects the processes inside them. Ranks discover each other through
the **PMI/PMIx** the site's `srun` provides: `srun --mpi=pmix` exports the PMIx server address
and rank/size into each container's environment, and the MPI inside the container attaches to it.
That is why a container never needs to see another container's filesystem or network namespace —
it only needs the PMI socket and the fabric device.

**The one hard requirement, and the one real failure mode.** The MPI inside the image must be
ABI-compatible with the site's PMI and fabric. It is not automatic:

- Open MPI and MPICH have *different ABIs*. An image built against one cannot attach to a launcher
  expecting the other. This is exactly what XaaS names as the barrier (arXiv:2401.04552,
  arXiv:2509.17914) — a compiled image is portable, not performance-portable.
- Two ways out, and a site picks one: **hybrid** (the image carries a matching MPI and uses the
  host's PMI), or **bind-mount** (the site's MPI and libfabric are injected into the container).
  On Alps the second is what the Cray OCI hooks do, which is why the MPI track must enable the
  fabric hook in its EDF `[annotations]` and the loop_level_reasoning track deliberately does not.
- The failure is quiet: without the hook, ranks fall back to TCP and the result reads as poor
  scaling. A scaling curve is exactly the measurement that failure corrupts, so this must be
  asserted, not assumed.

**Still unverified.** `submit_mpi_scaling.sbatch` now sizes the allocation by RANK count with
`RANKS_PER_NODE` explicit, launches with `srun --mpi=pmix`, and runs the P-rank == 1-rank gate
before any timing is believed; the Alps variant adds the container flag and the fabric hook. What
has NOT happened is a run on real Slurm: syntax, the EDF's TOML and the local multi-rank path are
checked, placement across nodes is not.

## Invariants across all three

- Ranks never coordinate to decide who does what. Every assignment is a pure function of
  `(work list, ranks, nodes)`, computed identically and independently by each rank.
- One container per rank, never one spanning ranks.
- Every `srun` step carries the container selector, or it is not in the image.
- Results outlive the allocation: per-rank shards in the repo, merged afterwards; the merged
  failure count IS the job status.
