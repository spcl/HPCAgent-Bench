# DESIGN: job launch -- two distributions, both static

This page explains the two ways a kernel selection is spread across ranks -- corpus
sharding (one rank per kernel) and problem decomposition (many ranks per kernel) --
and why the assignment is a pure function computed identically by every rank, with no
coordination at launch time.

## Two patterns, one launcher

- **corpus** -- one rank computes one whole kernel; P ranks cover P different kernels.
  This is `--shard i/P` off `SLURM_PROCID` (`submit_deterministic.sbatch`,
  `submit_loop_level_reasoning_alps.sbatch`). Ranks never talk.
- **problem** -- P ranks collectively compute ONE kernel; the problem is split.
  Plumbing exists (`Descriptor.from_submission(..., ranks=P)`, strong/weak sizing in
  `harness/mpi_sizing.py`, `mpi.rank_counts`) and has never been run above 1 rank.

Which pattern applies is a property of which submission script and CLI subcommand a
job uses (`run-framework --shard` for corpus, `submit_mpi_scaling.sbatch` for
problem), not a single shared flag. Everything below is static -- the assignment is a
pure function of `(kernel list, cost vector, ranks, nodes)`, so every rank computes the
identical answer alone. No master, no work stealing, no communication. That is not a
performance choice, it is a reproducibility one: the results DB is keyed by shard, so
the same job must produce the same partition twice.

## corpus: round-robin was a guess, and LPT bin-packing replaced it

`shard_names` (`support/collect/sweep.py`) used to keep `names[index::total]`, a pure
stride: neighbours in the sorted name list tend to be similar sizes, so a stride
spreads them. That was the right call when kernel cost was unknown.

It is known now. The preset ladder fits every kernel against a work model and a
footprint, so each kernel has a predicted time at every rung. `shard_names` passes that
preset into `sizing.pack_lpt`, which sorts kernels descending by predicted cost and
gives each to the least-loaded rank -- deterministic, same on every rank, no
coordination. This is the default path in `run-framework` whenever a preset is known.

The stride remains the fallback for when no cost model resolves at all (opaque
kernels -- `size_audit.py` classifies those as `opaque` / `unresolved`); a kernel with
no prediction is packed last, round-robin, so an unknown cost cannot skew the packing.

## corpus: the second dimension is memory, declared but not yet wired

XL is bounded at 4 GB (`sizing.XL_BYTE_CEILING`). Four ranks on one node at XL is 16 GB
of concurrent working set. `pack_lpt` already accepts `ranks_per_node` and
`node_ram_bytes` and raises when a packing would overrun the node's memory budget
(`sizing.node_footprint_violations`, exercised directly in `tests/test_corpus_packing.py`)
-- the two-dimensional packer (balance TIME across ranks, subject to
`sum(concurrent footprint on a node) <= node RAM`) exists as a function.

What is missing is wiring: the one production call site, `sweep.shard_names` inside
`run-framework`, calls `pack_lpt` with cost only, never with `ranks_per_node` /
`node_ram_bytes`, so the memory check never runs on a real job today. `submit_deterministic.sbatch`
already derives `RANKS` and `RANKS_PER_NODE` correctly (as distinct SLURM-provided values, not by
conflating a node count with a rank count), so both numbers `shard_names` would need are
already available at the call site; only the plumbing from there into `pack_lpt` is
missing.

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
  no kernel dropped. Holds for both the packer and the fallback stride
  (`tests/test_corpus_packing.py`).
- corpus: wiring `ranks_per_node` / `node_ram_bytes` through `shard_names` must not
  change the partition for a job whose packing already fits the node budget -- only a
  job that would have overrun it should see a different split or a refusal.
- problem, P in {2,4,8}: output equals the 1-rank output bitwise, at preset S.
- The partition is byte-identical across two runs of the same job.
