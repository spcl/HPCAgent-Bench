"""Guards for :func:`optarena.flags.ncores` -- the core count that sizes autopar and OMP.

This number is not advisory. ``-ftree-parallelize-loops=N`` BAKES N into the generated
``GOMP_parallel(fn, data, num_threads=N, flags)`` call, and an explicit num_threads argument
OVERRIDES ``OMP_NUM_THREADS``. Measured, same source, three builds, all run with
``OMP_NUM_THREADS=1``::

    -ftree-parallelize-loops=2  -> pool of 2
    -ftree-parallelize-loops=4  -> pool of 4
    -ftree-parallelize-loops=8  -> pool of 8

So whatever ncores() returns at BUILD time is the thread count at RUN time, and the runtime
env cannot walk it back. Getting it wrong oversubscribes the cores for the life of the
cached .so:

  * counting hyperthreads as cores oversubscribes 2x on any SMT host;
  * reading the machine instead of the rank's share oversubscribes 4x on one 288-core node
    running 4 ranks of 72.
"""
import os

import pytest

from optarena import flags


def test_reports_physical_cores_not_hyperthreads():
    """os.cpu_count() counts hyperthreads. Sizing autopar by it puts 2 threads on every
    physical core of an SMT host, which is oversubscription, not parallelism."""
    logical = os.cpu_count() or 1
    assert flags.ncores() <= logical
    assert flags.ncores() == flags.physical_cores(os.sched_getaffinity(0))


def test_never_returns_zero_or_negative():
    """A zero here becomes -ftree-parallelize-loops=0 / OMP_NUM_THREADS=0."""
    assert flags.ncores() >= 1


def test_the_explicit_override_wins(monkeypatch):
    monkeypatch.setenv("OPTARENA_NCORES", "3")
    assert flags.ncores() == 3


@pytest.mark.parametrize("bogus", ["0", "-4", "", "many"])
def test_a_bogus_override_is_ignored_rather_than_obeyed(monkeypatch, bogus):
    """OPTARENA_NCORES=0 must not size a thread pool to zero."""
    monkeypatch.setenv("OPTARENA_NCORES", bogus)
    assert flags.ncores() >= 1


def test_smt_siblings_collapse_to_one_core():
    """The heart of the physical-core count: two logical cpus sharing a core report the same
    thread_siblings_list, so they must count once."""
    affinity = sorted(os.sched_getaffinity(0))
    groups = {}
    for cpu in affinity:
        try:
            with open(flags.SIBLINGS.format(cpu=cpu)) as fh:
                groups.setdefault(fh.read().strip(), []).append(cpu)
        except OSError:
            pytest.fail("sysfs CPU topology is unreadable; ncores() cannot distinguish cores from threads")
    assert flags.physical_cores(set(affinity)) == len(groups)
    # If this host has SMT at all, prove the collapse actually happens rather than being a
    # no-op that the assertion above would pass either way.
    smt_pairs = [cpus for cpus in groups.values() if len(cpus) > 1]
    if smt_pairs:
        pair = set(smt_pairs[0])
        assert flags.physical_cores(pair) == 1, f"{sorted(pair)} share a core but counted as {len(pair)}"


def test_a_cpu_with_no_readable_topology_counts_as_its_own_core():
    """Containers that do not mount sysfs, and non-Linux hosts, have no sibling lists. Merging
    unknown cpus would UNDERCOUNT; counting each separately is the conservative reading."""
    assert flags.physical_cores({999999, 999998}) == 2


def test_respects_cpu_affinity_rather_than_the_whole_machine():
    """os.cpu_count() is affinity-blind -- under `taskset -c 0-3` it still reports the full
    machine. This is the 288-core/4-rank bug in miniature: a rank confined to its share must
    size autopar to that share, not to the node."""
    full = os.sched_getaffinity(0)
    if len(full) < 2:
        pytest.fail("need >= 2 cpus in the affinity mask to prove affinity is honoured")
    subset = set(sorted(full)[:1])
    try:
        os.sched_setaffinity(0, subset)
        confined = flags.ncores()
    finally:
        os.sched_setaffinity(0, full)
    assert confined == 1, f"confined to cpu {sorted(subset)} but ncores() said {confined}"
    assert flags.ncores() == flags.physical_cores(full), "affinity was not restored"


def test_a_bound_rank_ignores_slurm_cpus_per_task(monkeypatch):
    """When the rank IS bound, affinity is exact and SLURM must not override it.
    SLURM_CPUS_PER_TASK counts LOGICAL cpus, so believing it over an exact affinity reading
    would undercount an allocation made with --hint=nomultithread."""
    full = os.sched_getaffinity(0)
    if len(full) < 2:
        pytest.fail("need >= 2 cpus in the affinity mask")
    subset = set(sorted(full)[:1])
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "512")
    try:
        os.sched_setaffinity(0, subset)
        bound = flags.ncores()
    finally:
        os.sched_setaffinity(0, full)
    assert bound == 1, f"a bound rank let SLURM_CPUS_PER_TASK=512 inflate it to {bound}"


def test_an_unbound_rank_falls_back_to_the_slurm_allocation(monkeypatch):
    """The user's case: one node, 288 cpus, 4 domains, 4 ranks -> 72 per rank. When SLURM
    allocates a share but does not confine us to it, affinity still spans the node, so the
    allocation is the only remaining signal."""
    total = os.cpu_count() or 1
    smt = max(1, total // max(1, flags.physical_cores(set(range(total)))))
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", str(smt))  # exactly one core's worth
    assert flags.ncores() == 1, "an unbound rank ignored its SLURM allocation"


def test_the_slurm_fallback_never_inflates_beyond_the_machine(monkeypatch):
    """A SLURM_CPUS_PER_TAST larger than the host (a stale/misconfigured env) must not size a
    pool bigger than the cores that exist."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "100000")
    assert flags.ncores() <= flags.physical_cores(os.sched_getaffinity(0))


def test_omp_num_threads_is_not_a_source(monkeypatch):
    """OMP_NUM_THREADS is a REQUEST, and it is the variable cpu_env() itself sets. If ncores()
    read it, a parent's OMP_NUM_THREADS=1 would bake -ftree-parallelize-loops=1 into the
    cached .so -- and since the baked count overrides the env, EVERY later multi-core run
    would silently reuse a single-threaded library."""
    real = flags.ncores()
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert flags.ncores() == real, "ncores() read OMP_NUM_THREADS; a single-core parent can now poison the .so"


def test_cpu_env_sizes_multi_core_from_ncores():
    """The consumer contract: MULTI_CORE pins the OMP knobs to the physical core count, and
    SINGLE_CORE pins them to 1."""
    multi = flags.cpu_env(flags.Mode.MULTI_CORE)
    assert multi["OMP_NUM_THREADS"] == str(flags.ncores())
    single = flags.cpu_env(flags.Mode.SINGLE_CORE)
    assert set(single.values()) == {"1"}
