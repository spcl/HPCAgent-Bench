# DESIGN: job launch -- two distributions, both static

## Two patterns, one launcher

- **corpus** -- one rank computes one whole kernel; P ranks cover P different kernels.
  This is `--shard i/P` off `SLURM_PROCID` (`submit_deterministic.sbatch:147`,
  `submit_foundation_alps.sbatch:147`). Ranks never talk.
- **problem** -- P ranks collectively compute ONE kernel; the problem is split.
  Plumbing exists (`Descriptor(ranks=P)`, strong/weak sizing, `mpi.rank_counts`) and
  has never been run above 1 rank.

One flag selects: `--distribute=corpus|problem`. Everything below is static -- the
assignment is a pure function of `(kernel list, cost vector, ranks, nodes)`, so every
rank computes the identical answer alone. No master, no work stealing, no
communication. That is not a performance choice, it is a reproducibility one: the
results DB is keyed by shard, so the same job must produce the same partition twice.

## corpus: round-robin was a guess, and it no longer has to be

`shard_names` (`support/collect/sweep.py:159`) keeps `names[index::total]`. Its own
comment gives the reason: neighbours in the sorted name list tend to be similar
sizes, so a stride spreads them. That was the right call when kernel cost was unknown.

It is known now. The preset ladder fitted every kernel against a work model and a
footprint, so each kernel has a predicted time and a predicted working set at every
rung. Replace the stride with **LPT bin-packing**: sort descending by predicted cost,
give each kernel to the least-loaded rank. Deterministic, same on every rank, no
coordination.

Keep the stride as the fallback for when no cost model resolves (opaque kernels --
`size_audit.py` already classifies those as `opaque` / `unresolved`). A kernel with no
prediction is packed last, round-robin, so an unknown cost cannot skew the packing.

## corpus: the second dimension is memory, and it is why ranks != nodes

XL is bounded at 16 GB. Four ranks on one node at XL is 64 GB of concurrent working
set. So the packer is two-dimensional: balance predicted TIME across ranks, subject to
`sum(concurrent footprint on a node) <= node RAM`.

That constraint cannot even be stated today, because the harness only ever requests a
RANK count and never a node count -- `RANKS="${SLURM_JOB_NUM_NODES:-1}"` in both
sbatch scripts takes the NODE count and calls it ranks, which is only correct while
`--ntasks-per-node=1`. Fix: carry both. `ranks` is how many workers; `nodes` is how
many machines they sit on; ranks-per-node is what the memory constraint needs.
`mpi.rank_counts` is already named correctly and stays.

## problem: what is missing

1. **Declaration** -- a kernel must say which axis is distributable. Today the
   descriptor carries `ranks` but the manifest has no way to say "this array is split
   along axis 0". Without it, "8 ranks compute one kernel" is unexpressible per kernel.
2. **Derivation** -- strong scaling holds the global size fixed and gives each rank
   `global / P`; weak holds the per-rank size fixed and grows the global by `P`. Both
   modes exist in the sizing code; neither has been exercised.
3. **Halo** -- a split that needs neighbour data (ICON velocity, any stencil) needs an
   exchange, and an exchange that is wrong is a MISCOMPILE that looks like a scaling
   result. Validation is not optional here: the P-rank output must equal the 1-rank
   output exactly, and that check is the gate for adding a kernel to this mode.

## Gate

- corpus, P=4: the four shards partition the kernel list exactly -- no overlap, no gap,
  no kernel dropped. Holds for both the packer and the fallback stride.
- corpus: predicted max-rank load under LPT is lower than under the stride, measured
  on the real corpus. If it is not, keep the stride and delete the packer.
- problem, P in {2,4,8}: output equals the 1-rank output bitwise, at preset S.
- The partition is byte-identical across two runs of the same job.
